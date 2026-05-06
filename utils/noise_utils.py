import math
from typing import Iterable
import torch

def _gaussian_like_(dst: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, seed: int = 42):
    """
    用 N(mean, std^2) 生成噪声，写入 dst（in-place）。
    注意：mean/std 允许是标量 tensor（float32），dst 可能是 bf16/fp16/fp32。
    """
    # 不同 device 用不同 generator，避免跨卡不确定
    g = torch.Generator(device=dst.device)
    g.manual_seed(int(seed))

    # 用 float32 生成，再 cast 回 dst.dtype，数值更稳
    noise_fp32 = torch.empty_like(dst, dtype=torch.float32, device=dst.device)
    noise_fp32.normal_(mean=float(mean.item()), std=float(std.item()), generator=g)
    dst.copy_(noise_fp32.to(dtype=dst.dtype))


def _replace_param_with_gaussian_(param: torch.nn.Parameter, seed: int = 42):
    """
    param: nn.Parameter
    用 param 自己的均值/方差生成高斯噪声，替换 param.data（in-place）。
    """
    with torch.no_grad():
        w = param.data
        # 均值/方差在 float32 上算（bf16 上 std 很粗糙）
        w_fp32 = w.detach().to(dtype=torch.float32)
        mean = w_fp32.mean()
        std = w_fp32.std(unbiased=False)

        # std=0 的极端情况兜底（避免全常数导致噪声全相同）
        if float(std.item()) == 0.0:
            std = torch.tensor(1e-6, device=w.device, dtype=torch.float32)

        _gaussian_like_(w, mean=mean, std=std, seed=seed)


def replace_all_qkv_with_gaussian_(
    model: torch.nn.Module,
    seed: int = 42,
    include_bias: bool = False,
    verbose: bool = True,
):
    """
    将模型中所有 attention 的 q/k/v “参数矩阵”替换为高斯噪声矩阵，
    噪声的均值/方差来自原始参数本身（逐参数张量统计）。

    覆盖模式：
      1) 有 q_proj/k_proj/v_proj
      2) fused qkv: 名字含 qkv / qkv_proj / Wqkv 等
      3) in_proj_weight（torch MultiheadAttention 风格）

    返回：替换了多少个参数张量（weight；如果 include_bias=True 也算 bias）
    """
    n_replaced = 0
    replaced_names = []

    with torch.no_grad():
        for module_name, module in model.named_modules():
            # -------- case 1: q_proj/k_proj/v_proj --------
            has_q = hasattr(module, "q_proj") and isinstance(getattr(module, "q_proj"), torch.nn.Module)
            has_k = hasattr(module, "k_proj") and isinstance(getattr(module, "k_proj"), torch.nn.Module)
            has_v = hasattr(module, "v_proj") and isinstance(getattr(module, "v_proj"), torch.nn.Module)

            if has_q and has_k and has_v:
                for proj_name in ["q_proj", "k_proj", "v_proj"]:
                    proj = getattr(module, proj_name)
                    if hasattr(proj, "weight") and isinstance(proj.weight, torch.nn.Parameter):
                        _replace_param_with_gaussian_(proj.weight, seed=seed)
                        n_replaced += 1
                        replaced_names.append(f"{module_name}.{proj_name}.weight")
                    if include_bias and hasattr(proj, "bias") and isinstance(proj.bias, torch.nn.Parameter) and proj.bias is not None:
                        _replace_param_with_gaussian_(proj.bias, seed=seed + 1)
                        n_replaced += 1
                        replaced_names.append(f"{module_name}.{proj_name}.bias")
                continue  # 这个 module 已处理，跳过后续 case

            # -------- case 2: fused qkv 参数（qkv / qkv_proj / Wqkv 等）--------
            # 常见：module.qkv.weight 或 module.qkv_proj.weight 或者参数名里直接含 "qkv"
            for attr in ["qkv_proj", "Wqkv"]:
                if hasattr(module, attr):
                    sub = getattr(module, attr)
                    if hasattr(sub, "weight") and isinstance(sub.weight, torch.nn.Parameter):
                        _replace_param_with_gaussian_(sub.weight, seed=seed)
                        n_replaced += 1
                        replaced_names.append(f"{module_name}.{attr}.weight")
                    if include_bias and hasattr(sub, "bias") and isinstance(sub.bias, torch.nn.Parameter) and sub.bias is not None:
                        _replace_param_with_gaussian_(sub.bias, seed=seed + 1)
                        n_replaced += 1
                        replaced_names.append(f"{module_name}.{attr}.bias")

            # -------- case 3: torch MultiheadAttention 风格 in_proj_weight --------
            if hasattr(module, "in_proj_weight") and isinstance(getattr(module, "in_proj_weight"), torch.nn.Parameter):
                _replace_param_with_gaussian_(module.in_proj_weight, seed=seed)
                n_replaced += 1
                replaced_names.append(f"{module_name}.in_proj_weight")
            if include_bias and hasattr(module, "in_proj_bias") and isinstance(getattr(module, "in_proj_bias"), torch.nn.Parameter):
                _replace_param_with_gaussian_(module.in_proj_bias, seed=seed + 1)
                n_replaced += 1
                replaced_names.append(f"{module_name}.in_proj_bias")

    if verbose:
        print(f"[QKV GAUSS] replaced {n_replaced} parameter tensors")
        # 太长就别全打印，给个头尾
        if len(replaced_names) <= 20:
            for n in replaced_names:
                print("  -", n)
        else:
            for n in replaced_names[:10]:
                print("  -", n)
            print("  ...")
            for n in replaced_names[-10:]:
                print("  -", n)

    return n_replaced

def replace_all_ffn_with_gaussian_(
    model: torch.nn.Module,
    seed: int = 42,
    include_bias: bool = False,
    verbose: bool = True,
):
    """
    将模型中所有 FFN（MLP）里的全连接层参数替换为高斯噪声，
    噪声的均值/方差来自原始参数本身（逐 Linear 层统计）。

    覆盖模式（按模块结构粗暴扫）：
      - 任何 module 里名为 "mlp" 的子模块：
          * 先找 gate_proj / up_proj / down_proj 这类常见名字
          * 如果没有这些，就把 mlp 里的所有 nn.Linear 全替换
      - 其他情况：跳过（只改 FFN，不动别的）
    """
    n_replaced = 0
    replaced_names = []

    with torch.no_grad():
        for module_name, module in model.named_modules():
            # 只针对名为 "mlp" 的子模块（Qwen2.*, LLaMA 系列基本都是这样）
            if module_name.endswith(".mlp") or module_name == "mlp":
                # 优先按常见命名来（gate_proj / up_proj / down_proj）
                had_any = False
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    if hasattr(module, proj_name):
                        sub = getattr(module, proj_name)
                        if hasattr(sub, "weight") and isinstance(sub.weight, torch.nn.Parameter):
                            _replace_param_with_gaussian_(sub.weight, seed=seed)
                            n_replaced += 1
                            had_any = True
                            replaced_names.append(f"{module_name}.{proj_name}.weight")
                        if include_bias and hasattr(sub, "bias") and isinstance(sub.bias, torch.nn.Parameter) and sub.bias is not None:
                            _replace_param_with_gaussian_(sub.bias, seed=seed + 1)
                            n_replaced += 1
                            replaced_names.append(f"{module_name}.{proj_name}.bias")

                # 如果没找到这些典型名字，就兜底：把 mlp 里的所有 Linear 都处理掉
                if not had_any:
                    for sub_name, sub in module.named_modules():
                        if isinstance(sub, torch.nn.Linear):
                            full_name = f"{module_name}.{sub_name}" if sub_name else module_name
                            if hasattr(sub, "weight") and isinstance(sub.weight, torch.nn.Parameter):
                                _replace_param_with_gaussian_(sub.weight, seed=seed)
                                n_replaced += 1
                                replaced_names.append(f"{full_name}.weight")
                            if include_bias and hasattr(sub, "bias") and isinstance(sub.bias, torch.nn.Parameter) and sub.bias is not None:
                                _replace_param_with_gaussian_(sub.bias, seed=seed + 1)
                                n_replaced += 1
                                replaced_names.append(f"{full_name}.bias")

    if verbose:
        print(f"[FFN GAUSS] replaced {n_replaced} parameter tensors")
        if len(replaced_names) <= 20:
            for n in replaced_names:
                print("  -", n)
        else:
            for n in replaced_names[:10]:
                print("  -", n)
            print("  ...")
            for n in replaced_names[-10:]:
                print("  -", n)

    return n_replaced