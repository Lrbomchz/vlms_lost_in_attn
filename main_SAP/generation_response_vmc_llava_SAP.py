import os
import argparse
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoProcessor
from SAP_models.my_modeling_llava import LlavaForConditionalGeneration
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
import gc
from utils.SAP_utils import *
import json

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import yaml

import traceback
from PIL import Image

def _is_cuda_oom(e: BaseException) -> bool:
    oom_types = tuple(t for t in [getattr(torch.cuda, "OutOfMemoryError", None)] if t is not None)
    if oom_types and isinstance(e, oom_types):
        return True
    msg = str(e).lower()
    return ("out of memory" in msg) and ("cuda" in msg or "cublas" in msg or "cudnn" in msg)

def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_dict

def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches

def preprocess_all_attentions(attns, head_id: int = -1):
    """
    attns: list of length = batch_size
      每个元素 ats 是一个长度为 num_layers 的列表，
      ats[l] 的 shape ≈ [1, num_heads, seq_len, seq_len]

    返回:
      head_id == -1: [bs, num_layers, num_heads, seq_len]
      head_id >= 0 : [bs, num_layers, seq_len]
    """
    # 用第一个样本推断层数和 seq_len（key 的长度）
    num_layers = len(attns[0])
    seq_len = attns[0][0].shape[-1]

    if head_id == -1:
        # 所有 head
        # 外层 stack: batch 维；内层 stack: layer 维
        atten_via_inputs = torch.stack([
            torch.stack([
                ats[layer_idx][0, :, -1, :seq_len]   # [num_heads, seq_len]
                for layer_idx in range(num_layers)
            ], dim=0)  # [num_layers, num_heads, seq_len]
            for ats in attns
        ], dim=0)      # [bs, num_layers, num_heads, seq_len]
    else:
        # 指定 head
        atten_via_inputs = torch.stack([
            torch.stack([
                ats[layer_idx][0, head_id, -1, :seq_len]  # [seq_len]
                for layer_idx in range(num_layers)
            ], dim=0)  # [num_layers, seq_len]
            for ats in attns
        ], dim=0)      # [bs, num_layers, seq_len]

    return atten_via_inputs

def load_model_and_processor(model_path, device='cuda:0'):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device
    )
    model.eval()
    return model, processor


def extract_thinking_tags(text, start_tag="<think>", end_tag="</think>"):
    match = re.search(f"({re.escape(start_tag)}.*?{re.escape(end_tag)})", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        return thinking
    return ""

import re

def vmcbench_doc_to_text(doc, specific_kwargs=None):
    # 1. 获取问题
    question = doc["question"]

    # 2. 提取选项 A-D
    options = {cand: doc[cand] for cand in "ABCD"}
    options_prompt = "Options:\n"
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"

    # 3. 初步构造 Prompt
    prompt = f"Question: {question}\n{options_prompt}"

    # 4. 拼接可选的前后缀
    if specific_kwargs:
        pre = specific_kwargs.get("pre_prompt", "")
        post = specific_kwargs.get("post_prompt", "")
        if pre:
            prompt = f"{pre}{prompt}"
        if post:
            prompt = f"{prompt}{post}"

    return prompt


def url_to_local_path(url: str, root="coco_images"):
    # 取出 url 里的相对路径
    rel = url.split("images.cocodataset.org/")[-1]
    return str(Path(root) / rel)


def generate_response(model, processor, image, question, final_answer_tokens=1024, idx="test"):
    prompt = f'USER: <image>\n{question} Answer ONLY option\'s letter from the given choices.\nASSISTANT:'
    # print(prompt)
    inputs = processor(
        text=prompt,
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # print(inputs)

    input_ids_1d = inputs["input_ids"][0]
    image_token_id = model.config.image_token_id  # 或者 model.model.config.image_token_id

    img_pos = (input_ids_1d == image_token_id).nonzero(as_tuple=False).view(-1)

    #print("img_pos:", img_pos)

    if img_pos.numel() == 0:
        raise RuntimeError("No image tokens found in input_ids. Check prompt/processor.")
    pos = int(img_pos[0].item())
    pos_end = int(img_pos[-1].item()) + 1  # [pos, pos_end)

    #print(f"num_image_tokens: {pos_end-pos}")

    # pixel_values: [B,3,H,W]
    pv = inputs["pixel_values"]
    H, W = pv.shape[-2], pv.shape[-1]

    # patch_size: 不同 vision_tower 实现字段可能不同，做个兼容
    vt = model.model.vision_tower
    patch = getattr(vt, "patch_size", None)
    if patch is None and hasattr(vt, "config"):
        patch = getattr(vt.config, "patch_size", None)
    if patch is None:
        raise RuntimeError("Cannot infer vision patch_size from vision_tower.")
    output_shape = torch.tensor([H // patch, W // patch], dtype=torch.int32, device="cpu")

    resized_image = image.resize((W, H))

    #print(image_inputs[0])
    #print(type(image_inputs[0]))

    # compute vis_weight
    vis_weight = torch.randn(pos_end-pos).to(model.device)
    # patches =
    if args.mode == "complexity":
        patches = split_image_into_patches(resized_image, output_shape)
        #print(f"infered num_image_tokens: {len(patches)}")
        vis_weight = patch_complexity_grad_var(patches).to(model.device)
    elif args.mode == "key":
        pass

    model.model.set_visual_guidance_config(
        visual_token_range=(pos, pos_end),
        vis_weight=vis_weight,
        apipe_mode=args.mode,
        align_lambda=args.align_lambda,
        head_percentile_min=args.head_percentile_min,
        head_percentile_max=args.head_percentile_max,
        replace_layer=args.replace_layer
    )

    need_vision_attn = (args.mode == "vis_enc_attn")

    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=final_answer_tokens,
            do_sample=True,
            output_attentions=True,  # decoder 的
            return_dict_in_generate=True,
            output_vision_attentions=need_vision_attn,  #，往下传到 vision encoder
        )
        gen_ids = gen_output.sequences
        gen_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
        ]
        output_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    return output_text


def process_dataset(model, processor, pope_path, output_json, num_samples=100):
    ds = load_dataset(pope_path)  # 返回 DatasetDict
    data = ds["test"]  # data=data["test"] 也可以

    results = []

    num_samples = min(len(data), num_samples)
    data = data.select(range(num_samples))

    idx = 0

    for row in tqdm(data, total=len(data)):
        image_id = str(row['index'])
        image = row['image'].convert("RGB")

        question = vmcbench_doc_to_text(row)
        # print(question)
        answer = row['answer']

        #response = generate_response(model, processor, image, prompt, idx=idx)
        try:
            response = generate_response(model, processor, image, question, idx=idx)
            pass
            #response = generate_response(model, processor, image, prompt, idx=idx)
        except Exception as e:
            if _is_cuda_oom(e):
                print(f"[OOM] occurs in {image_id} -> retry with 336x336")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # 释放可回收的缓存
                    img = image
                    img_336 = img.resize((336, 336), resample=Image.Resampling.BICUBIC)  # Pillow>=9.1
                    response = generate_response(model, processor, img_336, question, idx=idx)
                except Exception as e2:
                    print(f"[ERROR] retry after OOM failed in {image_id}: {e2}")
                    traceback.print_exc()
                    response = "error"
            else:
                print(f"[ERROR] occurs in {image_id}: {e}")
                traceback.print_exc()
                response = "error"

        result = {
            "index": idx,
            "question": question,
            "correct_answer": answer,
            "model_response": response,
        }
        results.append(result)

        #print(response)
        #print(row["answer"])

        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

        idx += 1

        #if idx > num_samples:
        #    break

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/PATH/TO/YOUR/DATA_DIR/VMCBench/")
    parser.add_argument('--model_path', type=str, default="/PATH/TO/YOUR/MODEL_DIR/llava-1.5-7b-hf/")
    parser.add_argument('--out_dir', type=str, default="/PATH/TO/YOUR/RESULTS_DIR/")
    parser.add_argument('--model_name', type=str, default="llava-1.5-7b")
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--mode', type=str, default="gaussian_noise")
    parser.add_argument('--replace_layer', type=str, default="21,22,23,24,25,26,27")
    parser.add_argument('--align_lambda', type=float, default=0.0)
    parser.add_argument('--head_percentile_min', type=float, default=0.3)
    parser.add_argument('--head_percentile_max', type=float, default=0.6)
    args = parser.parse_args()

    #args.config = load_yaml(args.data_config_path)

    output = f"{args.out_dir}/SAP_results/output_openvl_output_{args.model_name}_hmin{args.head_percentile_min}_hmax{args.head_percentile_max}_rlayer{args.replace_layer}_vmcbench_SAP.json"
    model, processor = load_model_and_processor(args.model_path, args.device)
    process_dataset(model, processor, args.data_path, output, args.num_samples)
