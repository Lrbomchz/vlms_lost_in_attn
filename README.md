# ICML 2026: Large Vision–Language Models Get Lost in Attention

Official repo for paper: "Large Vision–Language Models Get Lost in Attention" (**ICML 2026**)

This repository provides code for:

- **SAP** experiments (including SAP implementations on Qwen- and LLaVA-style architectures where **visual attention is replaced by predefined values**).

  

  **Coming Soon:**

- **RQ3** experiments (computing **RID** and **MixIG**),

- **RQ3 Demos**: rq3_demos/ provides a lightweight subset (50 data points) to quickly reproduce the main RQ3 results.

---

## Repository Structure

- `main_sap/`  
  Code for **SAP** experiments.

- `SAP_models/`  
  SAP implementations based on **Qwen** and **LLaVA** architectures.  
  We replace the **visual attention** in these architectures with **predefined values**.

- `utils/`  
  Shared utilities.

---

## Usage

### 1) Installation

#### Requirements

We recommend using the following versions for reproducibility:

- python		3.10.12
- torch                    2.5.0  
- torchaudio               2.5.0  
- torchvision              0.20.0  
- tqdm                     4.67.1  
- transformers             4.56.2  
- timm                     0.9.10  
- tokenizers               0.22.0  
- qwen-vl-utils            0.0.11  
- seaborn                  0.13.2  
- Pillow                   10.1.0  
- scikit-learn             1.5.2  

#### Install this repo

From the project root:

```bash
cd lost_in_attention
pip install -e .
```

### **2) Models and Datasets**

To reproduce our results, please download the required **model checkpoints** (e.g., LLaVA, Qwen) and **datasets** (e.g., VMCBench, POPE).

All resources are publicly available on **Hugging Face**.

After downloading, **replace** the placeholder path in the scripts:

- Replace /PATH/to/Your/DIR with your actual local directory path.

Run (example):

```shell
# for qwen-2.5-VL
python main_SAP/generation_response_vmc_qwen_SAP.py --data_path /data/hdd1/xigongli/VMCBench/ --model_path /data/hdd1/xigongli/Qwen2.5-VL-7B-Instruct/ --out_dir /data/hdd1/xigongli/lost_in_attn_results --num_samples 10 --device "cuda:0" --mode "gaussian_noise" --replace_layer "21,22,23,24,25,26,27" --align_lambda 1

# for qwen-2.5-VL based reasoning model (e.g. ocean-r1)
python main_SAP/generation_response_vmc_qwen_SAP.py --data_path /data/hdd1/xigongli/VMCBench/ --model_path /data/hdd1/xigongli/weights/Ocean_R1_7B_Instruct/ --model_name "ocean-r1-7b" --out_dir /data/hdd1/xigongli/lost_in_attn_results --num_samples 10 --device "cuda:0" --mode "gaussian_noise" --replace_layer "21,22,23,24,25,26,27" --align_lambda 1 --answer_type "reasoning"

# for llava-1.5
python main_SAP/generation_response_vmc_llava_SAP.py --data_path /data/hdd1/xigongli/VMCBench/ --model_path /data/hdd1/xigongli/weights/llava-1.5-7b-hf/ --out_dir /data/hdd1/xigongli/lost_in_attn_results --num_samples 10 --device "cuda:0" --mode "gaussian_noise" --replace_layer "18,19,20,21,22,23" --align_lambda 1

# for llava-onevision
python main_SAP/generation_response_vmc_onevision_SAP.py --data_path /data/hdd1/xigongli/VMCBench/ --model_path /data/hdd1/xigongli/weights/llava-onevision-qwen2-7b-ov-hf/ --out_dir /data/hdd1/xigongli/lost_in_attn_results --num_samples 10 --device "cuda:0" --mode "gaussian_noise" --replace_layer "21,22,23,24,25,26,27" --align_lambda 1
```

## Citation

If you find our work helpful, please cite us:
```bibtex
@inproceedings{
xi2026large,
title={Large Vision-Language Models Get Lost in Attention},
author={Xi, Gongli and others},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
}
```
