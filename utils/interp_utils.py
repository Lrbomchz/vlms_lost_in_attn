from typing import Optional, List, Tuple, Any
import torch.nn.functional as F
import torch
import numpy as np


def token_entropy_over_seq(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    X: (seq_len, hidden_size) on model.device, dtype torch.float
    return: scalar entropy (0-dim) on same device, dtype torch.float
    """
    # w: (seq_len,) non-negative token "mass" via L2 norm
    w = torch.linalg.vector_norm(X, ord=2, dim=1)

    # p: (seq_len,) probability distribution over tokens via softmax
    p = F.softmax(w, dim=0)

    # Shannon entropy: -sum p log p
    H = -(p * torch.log(p + eps)).sum()
    return H


def compute_token_entropies_for_hiddens(hiddens_96: List[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """
    hiddens_96: list of (seq_len, hidden_size), length=3*nlayer
    return: (3*nlayer,) tensor on model.device
    """
    ents = []
    for X in hiddens_96:
        ents.append(token_entropy_over_seq(X, eps=eps))
    return torch.stack(ents, dim=0)

def effective_rank_from_svdvals(s: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    s: singular values, shape (k,), on device
    erank = exp( -sum_i p_i log p_i ),  p_i = s_i / sum_j s_j
    """
    total = s.sum()
    if total.item() == 0:  # 极端情况：全 0 矩阵（理论上 hidden 不会）
        return torch.zeros((), device=s.device, dtype=torch.float)

    p = s / total
    # 0*log(0) 处理：加 eps
    H = -(p * torch.log(p + eps)).sum()
    return torch.exp(H)


def effective_rank_matrix(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    X: (seq_len, hidden_size) float32 on model.device
    return: scalar tensor (0-dim) on same device, dtype float32
    """
    # torch.linalg.svdvals supports float (float32)
    s = torch.linalg.svdvals(X)
    return effective_rank_from_svdvals(s, eps=eps)


def compute_eranks_for_hiddens(hiddens_96: List[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """
    hiddens_96: List[(S,H)] length = 3*nlayer
    return: (3*nlayer,) float32 tensor on same device
    """
    eranks = []
    for X in hiddens_96:
        # X is (S,H) on device
        er = effective_rank_matrix(X, eps=eps)  # scalar on device
        eranks.append(er)
    return torch.stack(eranks, dim=0)  # (3*nlayer,)

def tme_hidden(
    X: torch.Tensor,
    *,
    mode: str = "shift",          # "shift" or "relu"
    agg: str = "last",            # "last" (fast) or "mean" (needs SxS)
    normalize: bool = True,       # divide by log(S) -> [0,1] roughly
    eps: float = 1e-12,
    alpha: float = 8.0,
) -> torch.Tensor:
    """
    X: (S, H) on model.device, dtype=torch.float
    return: scalar tensor on same device/dtype

    mode:
      - "shift": w = (cos+1)/2 in [0,1]
      - "relu" : w = relu(cos)  in [0,1]
    agg:
      - "last": only compute distribution for last token (O(SH), no SxS)
      - "mean": compute full SxS transition and average row entropies (O(S^2H))
    """
    assert X.dim() == 2, f"Expected (S,H), got {tuple(X.shape)}"
    S, H = X.shape
    if S <= 1:
        return torch.zeros((), device=X.device, dtype=X.dtype)

    # L2 normalize each token vector to remove scale effects
    norms = torch.linalg.vector_norm(X, ord=2, dim=1, keepdim=True)  # (S,1)
    Xn = X / (norms + eps)                                           # (S,H)

    def _weights_from_cos(cos_sim: torch.Tensor) -> torch.Tensor:
        if mode == "shift":
            w = 0.5 * (cos_sim + 1.0)  # [-1,1] -> [0,1]
        elif mode == "relu":
            w = torch.relu(cos_sim)
        else:
            raise ValueError(f"Unknown mode={mode}, use 'shift' or 'relu'")
        return w

    def _weights_from_cos(cos_sim: torch.Tensor):
        if mode == "shift":
            w = 0.5 * (cos_sim + 1.0)  # [0,1]
        elif mode == "relu":
            w = torch.relu(cos_sim)  # [0,1]
        else:
            raise ValueError

        if alpha != 1.0:
            w = torch.pow(w + 1e-12, alpha)  # sharpen
        return w

    if agg == "last":
        # last token as "query": similarities to all tokens (S,)
        q = Xn[-1, :]                              # (H,)
        cos = torch.matmul(Xn, q)                  # (S,)
        w = _weights_from_cos(cos)                 # (S,)
        p = (w + eps) / (w.sum() + eps * S)        # (S,)
        H_row = -(p * torch.log(p + eps)).sum()    # scalar
        H_out = H_row

    elif agg == "mean":
        # full token-token cosine (S,S)
        cos = torch.matmul(Xn, Xn.transpose(0, 1))  # (S,S)
        w = _weights_from_cos(cos)                  # (S,S)
        p = (w + eps) / (w.sum(dim=-1, keepdim=True) + eps * S)  # row-stochastic
        H_rows = -(p * torch.log(p + eps)).sum(dim=-1)           # (S,)
        H_out = H_rows.mean()

    else:
        raise ValueError(f"Unknown agg={agg}, use 'last' or 'mean'")

    if normalize:
        # normalize by log(S) so max entropy ~ 1 (natural log)
        logS = torch.log(torch.tensor(float(S), device=X.device, dtype=X.dtype))
        H_out = H_out / (logS + eps)

    return H_out

def tme_state_3xnlayer(
    hiddens_96: List[torch.Tensor],
    *,
    mode: str = "shift",
    agg: str = "last",
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    hiddens_96: list of (S,H), length = 3*nlayer
    return: (3*nlayer,) tensor on device
    """
    out = []
    for X in hiddens_96:
        out.append(tme_hidden(X, mode=mode, agg=agg, normalize=normalize, eps=eps))
    return torch.stack(out, dim=0)

def tme_delta_2xnlayer(
    hiddens_96: List[torch.Tensor],
    *,
    mode: str = "shift",
    agg: str = "last",
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    For each layer i:
      Δattn = post_attn - x_in
      Δffn  = post_ffn  - post_attn
    return: (2*nlayer,) tensor on device, order:
      [L0_Δattn, L0_Δffn, L1_Δattn, L1_Δffn, ...]
    """
    if len(hiddens_96) % 3 != 0:
        raise ValueError(f"Expected len multiple of 3, got {len(hiddens_96)}")
    nlayer = len(hiddens_96) // 3

    out = []
    for i in range(nlayer):
        x_in      = hiddens_96[3*i + 0]
        post_attn = hiddens_96[3*i + 1]
        post_ffn  = hiddens_96[3*i + 2]

        d_attn = post_attn - x_in
        d_ffn  = post_ffn  - post_attn

        out.append(tme_hidden(d_attn, mode=mode, agg=agg, normalize=normalize, eps=eps))
        out.append(tme_hidden(d_ffn,  mode=mode, agg=agg, normalize=normalize, eps=eps))

    return torch.stack(out, dim=0)


@torch.no_grad()
def novelty_from_hiddens_96(
    hiddens_96: List[torch.Tensor],
    *,
    # combine two residuals into one scalar per X': sqrt(col^2 + row^2) or sum
    combine: str = "l2",   # "l2" or "sum"
    # SVD truncation: keep singular values > tol; if None use default like matrix_rank
    tol: Optional[float] = None,
    eps: float = 1e-12,
) -> List[float]:
    """
    Input:
      hiddens_96: [X_in0, X_attn0, X_ffn0, X_in1, ...] each (S,H) on GPU/CPU (stay where they are)
    Output:
      list length = 2*nlayer:
        [layer0_attn_novelty, layer0_ffn_novelty, layer1_attn_novelty, layer1_ffn_novelty, ...]
      also stored into result[out_key]
    """
    if len(hiddens_96) % 3 != 0:
        raise ValueError(f"Expected len(hiddens_96) multiple of 3, got {len(hiddens_96)}")
    nlayer = len(hiddens_96) // 3

    out_vals_t = []  # collect scalar tensors on device

    for i in range(nlayer):
        X_in   = hiddens_96[3*i + 0]  # (S,H)
        X_attn = hiddens_96[3*i + 1]  # (S,H)
        X_ffn  = hiddens_96[3*i + 2]  # (S,H)

        if X_in.dim() != 2 or X_attn.dim() != 2 or X_ffn.dim() != 2:
            raise ValueError("Each hidden must be 2D (seq_len, hidden_size).")

        device = X_in.device
        dtype  = X_in.dtype

        # --- SVD of X_in to get U, V ---
        # Handle degenerate near-zero X_in: then "old subspace" is undefined -> treat novelty as ||X'||_F
        #norm_in = torch.linalg.vector_norm(X_in, ord="fro")
        #if norm_in <= eps:
        #    def novelty_degenerate(Xp: torch.Tensor) -> torch.Tensor:
        #        # both residuals ~= ||Xp||_F
        #        col = torch.linalg.vector_norm(Xp, ord="fro")
        #        row = col
        #        if combine == "l2":
        #            return torch.sqrt(col*col + row*row)
        #        elif combine == "sum":
        #            return col + row
        #        else:
        #            raise ValueError("combine must be 'l2' or 'sum'")
        #    out_vals_t.append(novelty_degenerate(X_attn))
        #    out_vals_t.append(novelty_degenerate(X_ffn))
        #    continue

        # economy SVD: X_in = U @ diag(S) @ Vh, with U:(S,k), S:(k,), Vh:(k,H), k=min(S,H)
        U, Svals, Vh = torch.linalg.svd(X_in, full_matrices=False)

        # choose truncation rank r by tol
        if tol is None:
            # similar to common matrix_rank heuristic
            finfo = torch.finfo(dtype) if dtype.is_floating_point else torch.finfo(torch.float32)
            tol_eff = max(X_in.shape) * finfo.eps * (Svals.max() + 0.0)
        else:
            tol_eff = torch.tensor(float(tol), device=device, dtype=dtype)

        r = int((Svals > tol_eff).sum().item())
        if r == 0:
            # effectively zero-rank
            def novelty_rank0(Xp: torch.Tensor) -> torch.Tensor:
                col = torch.sqrt((Xp * Xp).sum() + eps)
                row = col
                if combine == "l2":
                    return torch.sqrt(col*col + row*row)
                elif combine == "sum":
                    return col + row
                else:
                    raise ValueError("combine must be 'l2' or 'sum'")
            out_vals_t.append(novelty_rank0(X_attn))
            out_vals_t.append(novelty_rank0(X_ffn))
            continue

        U_r = U[:, :r]              # (S,r), orthonormal columns
        V_r = Vh[:r, :].transpose(0, 1)  # (H,r), orthonormal columns

        # --- helper: compute the two residual Fro norms for a given X' ---
        def two_residuals(Xp: torch.Tensor) -> torch.Tensor:
            fro2 = (Xp * Xp).sum()
            fro = torch.sqrt(fro2 + eps)

            UtXp = U_r.transpose(0, 1) @ Xp
            proj_col2 = (UtXp * UtXp).sum()
            res_col2 = torch.clamp(fro2 - proj_col2, min=0.0)
            res_col = torch.sqrt(res_col2 + eps)

            XpV = Xp @ V_r
            proj_row2 = (XpV * XpV).sum()
            res_row2 = torch.clamp(fro2 - proj_row2, min=0.0)
            res_row = torch.sqrt(res_row2 + eps)

            # normalize (dimensionless)
            r_col = res_col / (fro + eps)
            r_row = res_row / (fro + eps)

            if combine == "l2":
                return torch.sqrt(r_col * r_col + r_row * r_row + eps)
            elif combine == "sum":
                return r_col + r_row
            else:
                raise ValueError

        out_vals_t.append(two_residuals(X_attn))
        out_vals_t.append(two_residuals(X_ffn))

    out_tensor = torch.stack(out_vals_t, dim=0)  # (2*nlayer,)
    return out_tensor

