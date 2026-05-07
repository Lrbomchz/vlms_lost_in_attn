import os
import argparse
import re
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoProcessor
from SAP_models.my_modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from datasets import load_dataset
from pathlib import Path
from qwen_vl_utils import process_vision_info
import gc
from utils.SAP_utils import *
from PIL import Image
import traceback
import json

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

import torch

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

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        attn_implementation="eager",  # keep eager
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device
    )
    model.eval()
    return model, processor

# for reasoning models
def extract_thinking_tags(text, start_tag="<think>", end_tag="</think>"):
    match = re.search(f"({re.escape(start_tag)}.*?{re.escape(end_tag)})", text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        return thinking
    return ""

import re


def generate_response(model, processor, image, question, final_answer_tokens=1024, add_prompt="", idx="test"):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{question} {add_prompt}"}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # print(inputs)

    input_ids = inputs['input_ids'][0].tolist()
    orig_input_ids = inputs["input_ids"]
    # print(inputs.keys())
    question_ids = processor.tokenizer(question, add_special_tokens=False).input_ids
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    pos = input_ids.index(vision_start_token_id) + 1
    pos_end = input_ids.index(vision_end_token_id)
    question_pos = pos_end + 1
    question_pos_end = question_pos + len(question_ids)

    image_inputs_aux = processor.image_processor(images=image_inputs)
    image_grid_thw = image_inputs_aux["image_grid_thw"]
    image_indices = list(range(pos, pos_end))
    question_indices = list(range(question_pos, question_pos_end))

    resized_image = image_inputs[0]
    output_shape = image_grid_thw.squeeze(0)[1:] / 2
    output_shape = output_shape.to(torch.int32)

    #print(image_inputs[0])
    #print(type(image_inputs[0]))

    # compute vis_weight, vis_weight=random by default (SAP mode: random)
    vis_weight = torch.randn(pos_end - pos).to(model.device)
    # patches =
    if args.mode == "complexity":
        patches = split_image_into_patches(resized_image, output_shape)
        vis_weight = patch_complexity_grad_var(patches).to(model.device)
    elif args.mode == "key":
        pass

    # 假设你的 visual tokens 是 [v_start, v_end) 这段
    model.model.set_visual_guidance_config(
        visual_token_range=(pos, pos_end),
        vis_weight=vis_weight,
        apipe_mode=args.mode,
        align_lambda=args.align_lambda,
        head_percentile_min=args.head_percentile_min,
        head_percentile_max=args.head_percentile_max,
        replace_layer=args.replace_layer
    )

    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=final_answer_tokens,
            do_sample=False,
            output_attentions=False,
            return_dict_in_generate=True,
            output_vision_attentions=False,  # for visualization and SAP mode
        )
        gen_ids = gen_output.sequences
        gen_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)
        ]
        output_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    return output_text

# vmc-bench processing tools
def vmcbench_doc_to_text(doc, specific_kwargs=None):
    question = doc["question"]

    # 2. extract A-D
    options = {cand: doc[cand] for cand in "ABCD"}
    options_prompt = "Options:\n"
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"

    # 3. construct Prompt
    prompt = f"Question: {question}\n{options_prompt}"

    # 4. optional
    if specific_kwargs:
        pre = specific_kwargs.get("pre_prompt", "")
        post = specific_kwargs.get("post_prompt", "")
        if pre:
            prompt = f"{pre}{prompt}"
        if post:
            prompt = f"{prompt}{post}"

    return prompt


def url_to_local_path(url: str, root="coco_images"):
    # Extract the relative path from the URL, e.g., train2017/000000557944.jpg.
    rel = url.split("images.cocodataset.org/")[-1]
    return str(Path(root) / rel)

def _think_instruction_suffix():
    return (
        " You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        "The reasoning process MUST BE enclosed within <think> </think> tags. "
        "The final answer must be an option letter (i.e. A, B, C or D) from the given choices, "
        "enclosed in <answer></answer> tags. Let's think more."
    )

def process_dataset(model, processor, pope_path, output_json, num_samples=100, answer_type=""):
    ds = load_dataset(pope_path)  # DatasetDict
    data = ds["test"] # test split: ~ 9000 samples

    results = []

    num_samples = min(len(data), num_samples)
    data = data.select(range(num_samples))

    add_prompt = "Answer ONLY option's letter from the given choices."
    if answer_type == "reasoning":
        add_prompt = _think_instruction_suffix()

    idx = 0

    for row in tqdm(data, total=num_samples):
        image_id = str(row['index'])
        image = row['image']

        question = vmcbench_doc_to_text(row)
        # print(question)
        answer = row['answer']

        response = generate_response(model, processor, image, question, idx=image_id, add_prompt=add_prompt)

        result = {
            "index": image_id,
            #"image_name": image_name,
            "question": question,
            "correct_answer": answer,
            "model_response": response,
        }
        results.append(result)
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
    parser.add_argument('--model_path', type=str, default="/PATH/TO/YOUR/MODEL_DIR/Qwen2.5-VL-7B-Instruct/")
    parser.add_argument('--out_dir', type=str, default="/PATH/TO/YOUR/RESULTS_DIR/")
    parser.add_argument('--model_name', type=str, default="qwen-2.5-vl-7b")
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--mode', type=str, default="gaussian_noise")
    parser.add_argument('--replace_layer', type=str, default="21,22,23,24,25,26,27")
    parser.add_argument('--align_lambda', type=float, default=0.0)
    parser.add_argument('--head_percentile_min', type=float, default=0.3)
    parser.add_argument('--head_percentile_max', type=float, default=0.9)
    parser.add_argument('--answer_type', type=str, default="")
    args = parser.parse_args()

    output = f"{args.out_dir}/SAP_results/output_openvl_output_{args.model_name}_hmin{args.head_percentile_min}_hmax{args.head_percentile_max}_rlayer{args.replace_layer}_{args.answer_type}_vmcbench_SAP.json"
    model, processor = load_model_and_processor(args.model_path, args.device)
    process_dataset(model, processor, args.data_path, output, args.num_samples, args.answer_type)
