import torch
import torch.nn.functional as F
from typing import List, Tuple
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def Apipe_prob(mode: str, ref_weight: torch.Tensor) -> torch.Tensor:
    """
    根据不同模式返回一份对视觉 patch 的概率分布。

    参数：
      mode: "border"（随机）、"aligned"、"complexity"、"key"
      ref_weight: 1D tensor，长度 = len(visual_patches)

    返回：
      prob: 1D tensor，同 shape，为合法概率分布（非负且 sum=1）
    """
    if ref_weight.dim() != 1:
        raise ValueError(f"ref_weight must be 1D, got shape {tuple(ref_weight.shape)}")

    n = ref_weight.shape[0]

    mode = mode.lower()
    if mode == "border":
        # 方式1：用 Dirichlet 直接采样一个随机分布
        # concentration 全 1，相当于在 simplex 上均匀采样
        concentration = torch.ones(n, device=ref_weight.device, dtype=ref_weight.dtype)
        dist = torch.distributions.Dirichlet(concentration)
        prob = dist.sample()
        return prob

        # 如果你不想依赖 Dirichlet，可以改成：
        # noise = torch.rand_like(ref_weight)
        # prob = F.softmax(noise, dim=0)
        # return prob

    elif mode in ["aligned", "complexity", "key"]:
        # 把 ref_weight 变成概率分布（softmax）
        # 减去 max 做数值稳定
        x = ref_weight - ref_weight.max()
        prob = F.softmax(x, dim=0)
        return prob

    else:
        raise ValueError(f"Unknown mode: {mode}, must be one of "
                         f"'border', 'aligned', 'complexity', 'key'.")


def _to_gray(patch: torch.Tensor) -> torch.Tensor:
    """
    patch: Tensor[C,H,W] or [H,W,C]，RGB
    返回: Tensor[1,H,W] 灰度
    """
    if patch.dim() != 3:
        raise ValueError(f"Patch must be 3D, got shape {tuple(patch.shape)}")

    # 调整到 [C,H,W]
    if patch.shape[0] in (1, 3):  # [C,H,W]
        c_first = patch
    else:  # [H,W,C]
        c_first = patch.permute(2, 0, 1)

    if c_first.shape[0] == 1:
        gray = c_first
    elif c_first.shape[0] == 3:
        r, g, b = c_first[0], c_first[1], c_first[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        gray = gray.unsqueeze(0)  # [1,H,W]
    else:
        raise ValueError(f"Unsupported channel number: {c_first.shape[0]}")

    return gray


def _sobel_grad(gray: torch.Tensor) -> torch.Tensor:
    """
    gray: Tensor[1,H,W]
    返回梯度幅值 Tensor[1,H,W]
    """
    # Sobel kernels
    sobel_x = torch.tensor(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]]
    ).view(1, 1, 3, 3).to(gray.device)

    sobel_y = torch.tensor(
        [[-1., -2., -1.],
         [ 0.,  0.,  0.],
         [ 1.,  2.,  1.]]
    ).view(1, 1, 3, 3).to(gray.device)

    x = gray.unsqueeze(0)  # [1,1,H,W]
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2)  # [1,1,H,W]
    return grad_mag.squeeze(0)  # [1,H,W]


def _laplacian(gray: torch.Tensor) -> torch.Tensor:
    """
    gray: Tensor[1,H,W]
    返回 Laplacian 响应 Tensor[1,H,W]
    """
    kernel = torch.tensor(
        [[0.,  1., 0.],
         [1., -4., 1.],
         [0.,  1., 0.]]
    ).view(1, 1, 3, 3).to(gray.device)

    x = gray.unsqueeze(0)  # [1,1,H,W]
    lap = F.conv2d(x, kernel, padding=1)
    return lap.squeeze(0)  # [1,H,W]


def _entropy(gray: torch.Tensor, num_bins: int = 32) -> torch.Tensor:
    """
    gray: Tensor[1,H,W]，值范围 [0,1] 或 [0,255]
    返回: 标量 Tensor（熵）
    """
    g = gray.flatten()
    # 归一化到 [0,1]
    g = g - g.min()
    if g.max() > 0:
        g = g / g.max()

    # 量化到 num_bins
    idx = torch.clamp((g * (num_bins - 1)).long(), 0, num_bins - 1)
    hist = torch.bincount(idx, minlength=num_bins).float()
    p = hist / (hist.sum() + 1e-12)

    # 避免 log(0)
    p = p + 1e-12
    H = -(p * torch.log(p)).sum()
    return H


def Apipe_complexity(
    patch_list: List[torch.Tensor],
    method: str = "A",
    alpha_a: float = 1.0,
    beta_a: float = 1.0,
    gamma_a: float = 1.0,
    alpha_b: float = 1.0,
    beta_b: float = 1.0,
) -> torch.Tensor:
    """
    计算每个 patch 的复杂度分数。

    参数：
      patch_list: list of patch Tensor，元素为 [C,H,W] 或 [H,W,C] 的 RGB patch
      method:
        "A"  -> 使用方案 A: var + gradient
        "B"  -> 使用方案 B: entropy + Laplacian
        "AB" -> 将 A 和 B 的复杂度分数平均
      alpha_a, beta_a, gamma_a: 方案 A 中的 α, β, γ
      alpha_b, beta_b: 方案 B 中的 α, β

    返回：
      complexity: Tensor[num_patches]，每个元素是该 patch 的复杂度标量
    """
    if len(patch_list) == 0:
        return torch.empty(0)

    device = patch_list[0].device if isinstance(patch_list[0], torch.Tensor) else "cpu"

    # ---------- 先对所有 patch 计算 raw 指标 ----------
    var_list = []
    mean_grad_list = []
    edge_ratio_list = []

    entropy_list = []
    lap_energy_list = []

    for patch in patch_list:
        if not isinstance(patch, torch.Tensor):
            patch = torch.tensor(patch, device=device)

        patch = patch.to(device=device, dtype=torch.float32)
        gray = _to_gray(patch)  # [1,H,W]

        # 方案 A：var + gradient
        # intensity var
        intensity_var = gray.var()
        var_list.append(intensity_var)

        # Sobel gradient
        grad_mag = _sobel_grad(gray)  # [1,H,W]
        mean_grad = grad_mag.mean()
        mean_grad_list.append(mean_grad)

        # edge_ratio：grad > T 的比例，T = k * mean_grad
        k = 2.0
        T = k * mean_grad
        edge_ratio = (grad_mag > T).float().mean()
        edge_ratio_list.append(edge_ratio)

        # 方案 B：entropy + Laplacian
        H = _entropy(gray)
        entropy_list.append(H)

        lap = _laplacian(gray)
        lap_energy = (lap ** 2).mean()
        lap_energy_list.append(lap_energy)

    var_t = torch.stack(var_list)          # [N]
    mean_grad_t = torch.stack(mean_grad_list)
    edge_ratio_t = torch.stack(edge_ratio_list)

    entropy_t = torch.stack(entropy_list)
    lap_energy_t = torch.stack(lap_energy_list)

    eps = 1e-12

    # ---------- 对各指标做 0-1 归一化（在所有 patch 上） ----------
    def _norm(x: torch.Tensor) -> torch.Tensor:
        x_min = x.min()
        x_max = x.max()
        if (x_max - x_min) < eps:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)

    norm_var = _norm(var_t)
    norm_mean_grad = _norm(mean_grad_t)
    # edge_ratio 本身在 [0,1]，可以直接用
    norm_edge_ratio = edge_ratio_t.clamp(0.0, 1.0)

    norm_entropy = _norm(entropy_t)
    norm_lap_energy = _norm(lap_energy_t)

    # ---------- 方案 A / B 的复杂度标量 ----------
    C_a = alpha_a * norm_var + beta_a * norm_mean_grad + gamma_a * norm_edge_ratio
    C_b = alpha_b * norm_entropy + beta_b * norm_lap_energy

    method = method.upper()
    if method == "A":
        complexity = C_a
    elif method == "B":
        complexity = C_b
    elif method == "AB":
        complexity = 0.5 * (C_a + C_b)
    else:
        raise ValueError(f"Unknown method: {method}, must be 'A', 'B', or 'AB'.")

    return complexity

def patch_complexity_grad_var(patch_list_t, eps=1e-8, return_prob=False, temp=1.0):
    """
    patch_list_t: List[torch.Tensor]
        每个 patch 形状可以是 (H,W,3) 或 (3,H,W) 或 (H,W)
    return:
        torch.Tensor, shape (num_patches,)
    """
    scores = []
    for p in patch_list_t:
        x = p.to(torch.float32)

        # RGB -> gray
        if x.dim() == 3:
            if x.shape[-1] == 3:          # (H,W,3)
                gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
            else:                         # (3,H,W)
                gray = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
        else:
            gray = x  # (H,W)

        # 便宜差分梯度
        gx = (gray[:, 1:] - gray[:, :-1]).abs().mean()
        gy = (gray[1:, :] - gray[:-1, :]).abs().mean()
        grad = gx + gy

        var = gray.var(unbiased=False)
        scores.append(grad + var)

    s = torch.stack(scores)  # (num_patches,)

    # 归一化到 [0,1]
    s = (s - s.min()) / (s.max() - s.min() + eps)

    if return_prob:
        s = torch.softmax(s / max(float(temp), eps), dim=0)

    return s

def aggregate_vision_attns_max(
    vision_attns: Tuple[torch.Tensor, ...],
    group_size: int = 4,
) -> Tuple[torch.Tensor, ...]:
    """
    将 vision_attns 中每个 [B, H, L, L] 的 patch-level 注意力张量，
    按 group_size（默认 4）在序列维度上做 max 聚合，得到 [B, H, L', L']，
    其中 L' = L // group_size。

    这里假设 L 可以被 group_size 整除，比如 1380 -> 345。
    """
    merged_attns = []

    for att in vision_attns:
        # att: [B, H, L_q, L_k]
        assert att.dim() == 4, f"Expected 4D attention tensor, got shape {att.shape}"
        B, H, L_q, L_k = att.shape
        assert L_q % group_size == 0 and L_k % group_size == 0, \
            f"L_q={L_q}, L_k={L_k} must be divisible by group_size={group_size}"

        Lq_g = L_q // group_size
        Lk_g = L_k // group_size

        # 视作 [B, H, Lq_g, group_size, Lk_g, group_size]
        att_reshaped = att.view(B, H, Lq_g, group_size, Lk_g, group_size)

        # 先在 query 的 group 维上 max，再在 key 的 group 维上 max
        # 结果: [B, H, Lq_g, Lk_g]
        att_merged = att_reshaped.max(dim=3).values.max(dim=4).values

        merged_attns.append(att_merged)

    return tuple(merged_attns)


def aggregate_vision_attns_max_and_to_decoder_order(
    att_layer: torch.Tensor,        # [1, num_heads, 1380, 1380]
    window_index,
    spatial_merge_size: int = 2,    # Qwen2.5-VL 默认 2
    merge_mode: str = "max",        # "max" or "mean"
) -> torch.Tensor:
    """
    将 vision encoder 的 patch-level attention [1,H,1380,1380]
    映射到 decoder 使用的 group-level 坐标系，返回 [H,345,345]。

    返回的 345 顺序与 decoder 看到的视觉 token 完全对齐。
    """
    assert att_layer.dim() == 4
    B, H, Lq, Lk = att_layer.shape
    assert B == 1, "目前只处理 batch_size=1 的情况"

    spatial_merge_unit = spatial_merge_size ** 2  # 4
    assert Lq % spatial_merge_unit == 0 and Lk % spatial_merge_unit == 0
    Gq = Lq // spatial_merge_unit   # 345
    Gk = Lk // spatial_merge_unit   # 345

    # 1) 在 window 顺序下做 group 聚合
    att_reshaped = att_layer.view(B, H, Gq, spatial_merge_unit, Gk, spatial_merge_unit)
    if merge_mode == "max":
        att_group = att_reshaped.max(dim=3).values.max(dim=4).values  # [1,H,Gq,Gk]
    elif merge_mode == "mean":
        att_group = att_reshaped.mean(dim=3).mean(dim=4)              # [1,H,Gq,Gk]
    else:
        raise ValueError(f"Unsupported merge_mode: {merge_mode}")

    # window_index: [G]
    reverse_indices = torch.argsort(window_index)  # [G]

    # 3) 用 reverse_indices 在行 / 列上各重排一次，得到 decoder 顺序
    att_llm = att_group[:, :, reverse_indices, :]          # [1,H,G,G]
    att_llm = att_llm[:, :, :, reverse_indices]            # [1,H,G,G]

    return att_llm # [1,H,G,G]，这里 G=345

def visualize_vision_attention(
    inputs: dict,
    vision_attns: Tuple[torch.Tensor, ...],
    layer: int,
    head: int,
    image_id: str,
    save_root: str = "SAP_results/visual_attns",
    spatial_merge_size: int = 2,
    cmap: str = "bwr",
):
    """
    可视化指定 layer / head 的视觉注意力，并叠加到原始图像上。

    参数：
      inputs: processor(...) 的输出，需包含：
        - 'pixel_values': [B, C, H, W]
        - 'image_grid_thw': [num_images, 3]，例如 [[1, 30, 46]]
      vision_attns: 来自 aggregate_vision_attns_max 的输出
        - 每个元素: [1, num_heads, 345, 345]
      layer: 选择第几层视觉注意力 (索引)
      head:  选择第几个 head (索引)
      image_id: 当前图像的标识，用于构建保存路径
      save_root: 可视化图片保存的根目录
      spatial_merge_size: 与视觉编码器的 spatial_merge_size 保持一致（默认 2）
      cmap: 注意力热力图的 colormap，"bwr" 是红高蓝低
    """

    # 1. 取出该层该 head 的注意力矩阵
    att_layer = vision_attns[layer]            # [1, H, 345, 345]
    assert att_layer.dim() == 4, f"Expected [B,H,L,L], got {att_layer.shape}"
    att = att_layer[0, head]                  # [345, 345]

    # 2. 沿 query 维平均，得到每个 token 的“被关注程度”分数 [345]
    token_scores = att.mean(dim=0)            # [345]

    # 3. reshape 成 2D 网格（15x23）
    #    image_grid_thw: [T, H, W]，这里 T=1, H=30, W=46
    grid_thw = inputs["image_grid_thw"][0]    # tensor([1, 30, 46], device=...)
    t, h, w = grid_thw.tolist()
    h_merged = h // spatial_merge_size        # 30 -> 15
    w_merged = w // spatial_merge_size        # 46 -> 23
    assert h_merged * w_merged == token_scores.numel(), \
        f"Reshape mismatch: {h_merged} * {w_merged} != {token_scores.numel()}"

    heatmap = token_scores.view(h_merged, w_merged).detach().cpu().float()  # [15, 23]

    # 4. 获取原图（反归一化这里简单做 min-max 归一到 [0,1]）
    pixel_values = inputs["pixel_values"][0].detach().cpu()  # [C, H_img, W_img] or [H_img, W_img]
    if pixel_values.dim() == 3:
        # [C, H, W] -> [H, W, C]
        img = pixel_values
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = img.permute(1, 2, 0).numpy()   # [H, W, C]
    elif pixel_values.dim() == 2:
        # 灰度图 fallback: [H, W]
        img = pixel_values
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = img.unsqueeze(-1).numpy()      # [H, W, 1]
    else:
        raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

    H_img, W_img = img.shape[0], img.shape[1]

    # 5. 将 15x23 的 heatmap 双线性插值到原图大小
    heat_tensor = heatmap.unsqueeze(0).unsqueeze(0)            # [1,1,15,23]
    heat_resized = F.interpolate(
        heat_tensor,
        size=(H_img, W_img),
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()                                            # [H_img, W_img]

    # 归一化到 [0,1]，方便 colormap
    heat_resized = heat_resized - heat_resized.min()
    if heat_resized.max() > 0:
        heat_resized = heat_resized / (heat_resized.max() + 1e-6)

    # 6. 叠加可视化：50% 原图 + 50% 注意力 (通过 alpha=0.5 实现)
    save_dir = os.path.join(save_root, str(image_id))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"layer_{layer}_head_{head}.png")

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(heat_resized, cmap=cmap, alpha=0.5)  # 红高蓝低，“注意力”半透明叠加
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    return save_path, token_scores

def visualize_vision_attention(
    inputs: dict,
    vision_attns: Tuple[torch.Tensor, ...],
    layer: int,
    head: int,
    image_id: str,
    raw_image=None,   # 新增：原始 PIL.Image 或 numpy 数组
    save_root: str = "SAP_results/visual_attns",
    spatial_merge_size: int = 2,
    cmap: str = "bwr",
):
    """
    可视化指定 layer / head 的视觉注意力，并叠加到原始图像上。

    参数：
      inputs: processor(...) 的输出，需包含：
        - 'image_grid_thw': [num_images, 3]，例如 [[1, 30, 46]]
      vision_attns: 已经映射到 decoder 顺序的注意力，每层: [1, num_heads, 345, 345]
      layer: 视觉 encoder 的层索引（支持负索引）
      head:  head 索引；<0 表示对所有 head 取平均
      image_id: 当前图像 ID，用于文件名
      raw_image: 原始 PIL.Image 或 np.ndarray（H, W, C），优先用于做背景
    """

    # ------- 1. 处理 layer / head 索引 -------
    num_layers = len(vision_attns)
    if layer < 0:
        layer = num_layers + layer
    if not (0 <= layer < num_layers):
        raise IndexError(f"layer={layer} out of range for vision_attns length={num_layers}")

    att_layer = vision_attns[layer]        # [1, H, 345, 345]
    assert att_layer.dim() == 4, f"Expected [B,H,L,L], got {att_layer.shape}"
    B, H, Lq, Lk = att_layer.shape
    assert B == 1 and Lq == Lk, f"Unexpected attention shape: {att_layer.shape}"

    if head < 0:
        att = att_layer[0].mean(dim=0)    # [345,345] head 平均
    else:
        if not (0 <= head < H):
            raise IndexError(f"head={head} out of range for num_heads={H}")
        att = att_layer[0, head]          # [345,345]

    # ------- 2. 得到每个视觉 token 的分数 [345] -------
    token_scores = att.mean(dim=0)        # [345]

    # ------- 3. reshape -> 网格 [15,23] -------
    grid_thw = inputs["image_grid_thw"][0]  # tensor([1,30,46], ...)
    t, h, w = grid_thw.tolist()
    h_merged = h // spatial_merge_size      # 30 -> 15
    w_merged = w // spatial_merge_size      # 46 -> 23
    assert h_merged * w_merged == token_scores.numel(), \
        f"Reshape mismatch: {h_merged} * {w_merged} != {token_scores.numel()}"

    heatmap = token_scores.view(h_merged, w_merged).detach().cpu().float()  # [15,23]

    # ------- 4. 构造背景图像 img: [H_img,W_img,C] -------
    if raw_image is not None:
        # 用原始 PIL 图像 / numpy 数组
        if hasattr(raw_image, "convert"):  # PIL.Image
            img_np = np.array(raw_image.convert("RGB"))  # [H,W,3]
        else:
            img_np = np.asarray(raw_image)
            if img_np.ndim == 2:
                img_np = np.stack([img_np]*3, axis=-1)
        img = img_np.astype("float32")
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    else:
        # 回退到 processor 的 pixel_values（可能是 [B,C,H,W] 或 [N,1176]）
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 4:
            # [B,C,H,W] 常规情况
            pv = pixel_values[0].detach().cpu()
            pv = (pv - pv.min()) / (pv.max() - pv.min() + 1e-6)
            img = pv.permute(1, 2, 0).numpy()
        else:
            raise ValueError(
                f"pixel_values has shape {pixel_values.shape}, "
                "and raw_image is None; please pass raw_image to visualize_vision_attention."
            )

    H_img, W_img = img.shape[0], img.shape[1]

    # ------- 5. heatmap 插值到图像大小 -------
    heat_tensor = heatmap.unsqueeze(0).unsqueeze(0)  # [1,1,15,23]
    heat_resized = F.interpolate(
        heat_tensor,
        size=(H_img, W_img),
        mode="bilinear",
        align_corners=False
    )[0, 0].numpy()                                  # [H_img,W_img]

    heat_resized = heat_resized - heat_resized.min()
    if heat_resized.max() > 0:
        heat_resized = heat_resized / (heat_resized.max() + 1e-6)

    # ------- 6. 叠加并保存 -------
    save_dir = os.path.join(save_root, str(image_id))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"layer_{layer}_head_{head}.png")

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(heat_resized, cmap=cmap, alpha=0.5)  # 50% 原图 + 50% 注意力
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    return save_path

def visualize_vision_attention(
    inputs: dict,
    vision_attns: Tuple[torch.Tensor, ...],
    layer: int,
    head: int,
    image_id: str,
    raw_image=None,   # 建议传入原始 PIL.Image，用它做背景
    save_root: str = "SAP_results/visual_attns",
    spatial_merge_size: int = 2,
    cmap: str = "bwr",
    alpha: float = 0.5,
):
    """
    按“格子”方式可视化指定 layer / head 的视觉注意力，并叠加到原图上（50% 原图 + 50% 注意力）。

    参数：
      - inputs: processor(...) 的输出，需要包含：
          * 'image_grid_thw': [num_images, 3]，例如 [[1, 30, 46]]
      - vision_attns: 已经映射到 decoder 顺序的视觉注意力，每层:
          * vision_attns[l]: [1, num_heads, 345, 345]
      - layer: 视觉 encoder 的层索引（支持负索引）
      - head:  头索引；若 <0，则对所有 head 做平均
      - image_id: 当前图像 ID，用于构造保存路径
      - raw_image: 原始 PIL.Image 或 np.ndarray(H,W,C)，优先用于作为背景
      - save_root: 保存根目录
      - spatial_merge_size: 与视觉 encoder 的 spatial_merge_size 保持一致（默认 2）
      - cmap: 注意力 colormap，"bwr" = 蓝低红高
      - alpha: heatmap 叠加到原图的透明度（0~1）
    """

    # ---------------- 1. 取出指定 layer / head 的注意力 ----------------
    num_layers = len(vision_attns)
    if layer < 0:
        layer = num_layers + layer
    if not (0 <= layer < num_layers):
        raise IndexError(f"layer={layer} out of range for vision_attns length={num_layers}")

    att_layer = vision_attns[layer]  # [1, H, 345, 345]
    assert att_layer.dim() == 4, f"Expected [B,H,L,L], got {att_layer.shape}"
    B, H, Lq, Lk = att_layer.shape
    assert B == 1 and Lq == Lk, f"Unexpected attention shape: {att_layer.shape}"

    if head < 0:
        # 对所有 head 求平均
        att = att_layer[0].mean(dim=0)    # [345, 345]
    else:
        if not (0 <= head < H):
            raise IndexError(f"head={head} out of range for num_heads={H}")
        att = att_layer[0, head]          # [345, 345]

    # ---------------- 2. 沿 query 维平均，得到每个视觉 token 分数 ----------------
    # 你也可以改成 att.sum(0) 或 max(0)，这里继续用 mean 和你之前一致
    token_scores = att.mean(dim=0)        # [345]

    # ---------------- 3. reshape 成 patch 网格 [th, tw] ----------------
    # image_grid_thw: [T, H_grid, W_grid]，例如 [1, 30, 46]
    grid_thw = inputs["image_grid_thw"][0]   # tensor([1, 30, 46], ...)
    t, h_grid, w_grid = grid_thw.tolist()

    # decoder 看到的视觉 token 数 = (h_grid / s) * (w_grid / s)
    th = h_grid // spatial_merge_size       # 30 -> 15
    tw = w_grid // spatial_merge_size       # 46 -> 23
    assert th * tw == token_scores.numel(), \
        f"Reshape mismatch: th*tw={th*tw} != {token_scores.numel()}"

    grid = token_scores.view(th, tw).detach().cpu().float().numpy()  # [th, tw]

    # ---------------- 4. 取原图作为背景（不再用 pixel_values 插值） ----------------
    if raw_image is not None:
        # 优先使用原始图片
        if hasattr(raw_image, "convert"):  # PIL.Image
            img_arr = np.array(raw_image.convert("RGB"))  # [H,W,3]
        else:
            img_arr = np.asarray(raw_image)
            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr] * 3, axis=-1)
    else:
        # 退级方案：如果你确实只想用 processor 的 pixel_values，可以在外面传 raw_image 进来；
        # 这里给一个简单 fallback，但推荐总是传 raw_image。
        pixel_values = inputs["pixel_values"]
        if pixel_values.dim() == 4:
            pv = pixel_values[0].detach().cpu()     # [C,H,W]
            pv = (pv - pv.min()) / (pv.max() - pv.min() + 1e-6)
            img_arr = pv.permute(1, 2, 0).numpy()   # [H,W,C]
        else:
            raise ValueError(
                f"pixel_values has shape {pixel_values.shape}, "
                "and raw_image is None; please pass raw_image to visualize_vision_attention."
            )

    img_arr = img_arr.astype("float32")
    img_min, img_max = img_arr.min(), img_arr.max()
    if img_max > img_min:
        img_arr = (img_arr - img_min) / (img_max - img_min + 1e-6)

    H_img, W_img = img_arr.shape[0], img_arr.shape[1]

    # ---------------- 5. 用“格子铺满”的方式扩展 heatmap ----------------
    #    完全仿照你旧版：ph = H//th, pw = W//tw，然后做 Kronecker product
    ph = H_img // th
    pw = W_img // tw
    if ph <= 0 or pw <= 0:
        raise ValueError(
            f"Invalid patch size: H_img={H_img}, W_img={W_img}, th={th}, tw={tw} -> ph={ph}, pw={pw}"
        )

    # 用 Kronecker 乘积扩展到 patch 级别的“格子图”：形状约为 [th*ph, tw*pw]
    heatmap = np.kron(grid, np.ones((ph, pw), dtype=np.float32))  # [th*ph, tw*pw]

    # 有可能因为整除问题 heatmap 比原图略小/略大，这里裁剪成和原图一致
    heatmap = heatmap[:H_img, :W_img]

    # 归一化到 [0,1]
    hm_min, hm_max = heatmap.min(), heatmap.max()
    if hm_max > hm_min:
        heatmap = (heatmap - hm_min) / (hm_max - hm_min + 1e-6)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)

    # ---------------- 6. 叠加显示并保存 ----------------
    save_dir = os.path.join(save_root, str(image_id))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"layer_{layer}_head_{head}.png")

    plt.figure(figsize=(6, 6))
    # 背景图：1 - alpha
    plt.imshow(img_arr, alpha=1 - alpha)
    # 注意力：alpha，红高蓝低
    plt.imshow(heatmap, cmap=cmap, alpha=alpha, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    return save_path


# ---------- helpers ----------

def _letters_only_lower(s: str) -> str:
    """Keep only letters and spaces; lowercase; collapse multiple spaces."""
    keep = [ch.lower() if (ch.isalpha() or ch.isspace()) else " " for ch in s]
    return " ".join("".join(keep).split())

def _parse_image_id(img_source: str) -> int:
    """Extract the trailing integer id from strings like 'COCO_val2014_000000310196' -> 310196."""
    digits = re.findall(r"\d+", img_source)
    if not digits:
        raise ValueError(f"Cannot parse image id from: {img_source}")
    tail = digits[-1]
    return int(tail.lstrip("0") or "0")

def _get_th_tw(patchtify_thw) -> tuple[int, int]:
    """Accepts [th, tw] or any array-like; returns (th, tw) as ints."""
    arr = np.array(patchtify_thw).astype(int).flatten()
    if arr.size < 2:
        raise ValueError(f"patchtify_thw must have >=2 elements, got {patchtify_thw}")
    return int(arr[-2]), int(arr[-1])

def _rect_iou(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1) -> float:
    """IoU between two axis-aligned rectangles given by (x0,y0,x1,y1), inclusive-exclusive in pixels."""
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = a_area + b_area - inter
    return (inter / denom) if denom > 0 else 0.0

# convert_bbox_to_patchid

def convert_bbox_to_patchid(
    img_source: str,
    patchtify_thw,
    image_inputs,
    annotation_path: str,
    object: str,
    orig_image: Image.Image,
) -> torch.Tensor:
    """
    返回每个 patch 对关键区域(bbox) 的 IoU 向量:
      - shape: (th * tw,)
      - 多个 bbox 时：对同一个 patch 取 IoU 最大值
      - 若图/类别找不到标注：返回全 0 向量
    """

    # 0) 网格（保证无标注也能返回固定长度）
    th, tw = _get_th_tw(patchtify_thw)
    num_patches = th * tw
    iou_vec = torch.zeros(num_patches, dtype=torch.float32)

    # 1) 读 COCO
    with open(annotation_path, "r") as f:
        data = json.load(f)
    images = data.get("images", [])
    anns   = data.get("annotations", [])
    cats   = data.get("categories", [])
    cat_id2name = {c["id"]: c["name"] for c in cats}

    # 2) 找图像
    image_id = _parse_image_id(img_source)
    img_entry = next((im for im in images if (im.get("id") == image_id) or
                      (isinstance(im.get("id"), str) and im["id"].isdigit() and int(im["id"]) == image_id)), None)
    if img_entry is None:
        return iou_vec

    # 3) 归一化类别并筛标注
    obj_clean = _letters_only_lower(object)
    target_anns = []
    for a in anns:
        if a.get("image_id") != image_id:
            continue
        cname = cat_id2name.get(a.get("category_id"), "")
        if _letters_only_lower(cname) == obj_clean and a.get("bbox") is not None:
            target_anns.append(a)
    if not target_anns:
        return iou_vec

    # 4) 尺度对齐：原图 -> 已 resize 图像
    orig_W, orig_H = orig_image.size  # PIL: (W,H)
    resized_img = image_inputs
    W, H = resized_img.size

    ph, pw = H // th, W // tw
    sx = W / float(orig_W) if orig_W > 0 else 1.0
    sy = H / float(orig_H) if orig_H > 0 else 1.0

    # 5) 对每个 bbox：只在可能相交的 patch 上算 IoU，并写入 iou_vec（取 max）
    for a in target_anns:
        x, y, w, h = map(float, a["bbox"])  # COCO: [x,y,w,h] in orig coords
        bx0, by0 = x * sx, y * sy
        bx1, by1 = (x + w) * sx, (y + h) * sy

        # 粗筛 patch 范围
        c0 = max(0, int(np.floor(bx0 / pw)))
        c1 = min(tw - 1, int(np.floor((bx1 - 1) / pw))) if bx1 > 0 else c0
        r0 = max(0, int(np.floor(by0 / ph)))
        r1 = min(th - 1, int(np.floor((by1 - 1) / ph))) if by1 > 0 else r0

        for r in range(r0, r1 + 1):
            py0, py1 = r * ph, min((r + 1) * ph, H)
            for c in range(c0, c1 + 1):
                px0, px1 = c * pw, min((c + 1) * pw, W)

                iou = _rect_iou(bx0, by0, bx1, by1, px0, py0, px1, py1)
                pid = r * tw + c
                if iou > float(iou_vec[pid]):
                    iou_vec[pid] = float(iou)

    return iou_vec

def split_image_into_patches(image_inputs: Image.Image, patchtify_thw, device="cpu") -> List[Image.Image]:
    """
    按 (th, tw) 网格把 image_inputs 切成 th*tw 个 patch，row-major 顺序返回：
    patch_list[0] 是左上角 (r=0,c=0)，然后从左到右、从上到下。

    输入:
      - image_inputs: PIL.Image（模型实际看到的 resized 图）
      - patchtify_thw: 形如 [th, tw] 或 [t, th, tw] 等 array-like（用你已有的 _get_th_tw 取最后两维）

    输出:
      - patch_list: List[PIL.Image]，长度 = th*tw
    """
    th, tw = _get_th_tw(patchtify_thw)  # 复用你现有的函数
    W, H = image_inputs.size

    ph = H // th
    pw = W // tw

    patch_list: List[Image.Image] = []
    for r in range(th):
        y0 = r * ph
        y1 = (r + 1) * ph if r < th - 1 else H
        for c in range(tw):
            x0 = c * pw
            x1 = (c + 1) * pw if c < tw - 1 else W
            #patch_list.append(np.asarray(image_inputs.crop((x0, y0, x1, y1))))
            patch_pil = image_inputs.crop((x0, y0, x1, y1))
            patch = torch.from_numpy(np.array(patch_pil)).to(device)  # (H, W, C), uint8
            patch = patch.permute(2, 0, 1).contiguous()  # (C, H, W)

            patch_list.append(patch.float() / 255.0)

    return patch_list