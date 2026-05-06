import os
import argparse
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoProcessor
from SAP_models.my_modeling_llava_onevision import LlavaOnevisionForConditionalGeneration
from datasets import load_dataset, concatenate_datasets
from pathlib import Path
import gc
from utils.SAP_utils import *
import json
from transformers.utils import logging
logging.set_verbosity_error()

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

def process_single_sample(data):
    question = data['question']
    o_imgs_paths = []
    for option in data['options']:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:  # multiple images in options, used for random selection
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
             'image': None, 'question_type': data['question_type']}
    else:
        return {'id': data['id'], 'question': question, 'options': data['options'], 'answer': data['answer'],
             'image': data['image_1'], 'question_type': data['question_type']}

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

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="eager",  # 注入 q*k 需要 eager
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
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
    # 取出 url 里的相对路径，比如 train2017/000000557944.jpg
    rel = url.split("images.cocodataset.org/")[-1]
    return str(Path(root) / rel)


def generate_response(model, processor, image, question, final_answer_tokens=1024, idx="test"):
    # ---------- 1) OneVision prompt ----------
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{question}  Answer ONLY option\'s letter from the given choices."},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # ---------- 2) processor: OneVision 会产出 image_sizes / 可能还有 batch_num_images ----------
    inputs = processor(
        text=prompt,
        images=image,          # 单图：直接给 PIL
        padding=True,
        return_tensors="pt",
    )

    inputs = move_inputs_for_onevision(inputs, model.device)

    # ---------- 3) 找 image token span ----------
    input_ids_1d = inputs["input_ids"][0]
    image_token_id = model.config.image_token_id
    img_pos = (input_ids_1d == image_token_id).nonzero(as_tuple=False).view(-1)
    if img_pos.numel() == 0:
        raise RuntimeError("No image tokens found in input_ids. Use apply_chat_template + processor(images=...) to fix.")
    pos = int(img_pos[0].item())
    pos_end = int(img_pos[-1].item()) + 1
    num_img_tokens = pos_end - pos

    # ---------- 4) 计算 vis_weight ----------
    if args.mode == "complexity":
        vis_weight = build_onevision_anyres_complexity_weight(
            model=model,
            image_pil=image,
            num_img_tokens=num_img_tokens,
        ).to(model.device)
    else:
        vis_weight = torch.randn(num_img_tokens, device=model.device)

    # ---------- 5) 注入配置（my_modeling_llava_onevision 里需要有对应接口/字段） ----------
    #  llava1.5 的调用方式：model.model.set_visual_guidance_config(...)
    model.model.set_visual_guidance_config(
        visual_token_range=(pos, pos_end),
        vis_weight=vis_weight,
        apipe_mode=args.mode,
        align_lambda=args.align_lambda,
        head_percentile_min=args.head_percentile_min,
        head_percentile_max=args.head_percentile_max,
        replace_layer=args.replace_layer
    )

    # ---------- 6) generate ----------
    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=True,
            output_attentions=True,
            return_dict_in_generate=True,
            output_vision_attentions=(args.mode == "aligned"),
        )

    gen_ids = gen_output.sequences
    gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], gen_ids)]
    output_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    # print(output_text)
    return output_text


def process_dataset(model, processor, pope_path, output_json, num_samples=100):
    ds = load_dataset(pope_path)  # 返回 DatasetDict
    data = ds["test"]  # data=data["test"] 也可以

    results = []

    idx = 0

    for row in tqdm(data, total=len(data)):
        image_id = str(row['index'])
        image = row['image'].convert("RGB").resize((672, 672), resample=Image.Resampling.BICUBIC)

        question = vmcbench_doc_to_text(row)
        # print(question)
        answer = row['answer']

        #response = generate_response(model, processor, image, prompt, idx=idx)
        try:
            with torch.no_grad():
                response = generate_response(model, processor, image, question, idx=idx)
            pass
            #response = generate_response(model, processor, image, prompt, idx=idx)
        except Exception as e:
            is_oom = _is_cuda_oom(e)
            if is_oom:
                print(f"[OOM] occurs in {image_id} -> retry with 336x336")
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # 释放可回收的缓存（不保证一定解决，但常用于重试） [oai_citation:0‡PyTorch Forums](https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530?utm_source=chatgpt.com)
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
            "index": image_id,
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
    parser.add_argument('--model_path', type=str, default="/PATH/TO/YOUR/MODEL_DIR/llava-onevision-qwen2-7b-ov-hf/")
    parser.add_argument('--model_name', type=str, default="llava-onevision-qwen2-7b-ov")
    parser.add_argument('--out_dir', type=str, default="/PATH/TO/YOUR/RESULTS_DIR/")
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--mode', type=str, default="border")
    parser.add_argument('--replace_layer', type=str, default="21,22,23,24,25,26,27")
    parser.add_argument('--align_lambda', type=float, default=0.0)
    parser.add_argument('--head_percentile_min', type=float, default=0.1)
    parser.add_argument('--head_percentile_max', type=float, default=0.11)
    args = parser.parse_args()

    #args.config = load_yaml(args.data_config_path)

    output = f"{args.out_dir}/vmc_onevision_res/output_openvl_output_{args.model_name}replace_lambda{args.align_lambda}_hmin{args.head_percentile_min}_hmax{args.head_percentile_max}_wmode{args.mode}_replacelayer{args.replace_layer}_vmc_apipe.json"
    model, processor = load_model_and_processor(args.model_path, args.device)
    process_dataset(model, processor, args.data_path, output, args.num_samples)
