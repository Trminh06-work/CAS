"""
GPU-accelerated non-dominated filtering via torch, using the
block-based reduction strategy validated in nd_numpy_ref.py.

Processes F_all in blocks: within each block, drops dominated points
(vectorised block-vs-block comparison), then cross-compares the
survivors of that block against the running survivor set, keeping
the survivor set as small as possible at every step. This avoids
ever materialising a full (N,N) comparison matrix for large N --
only (block_size, block_size) and (block_size, n_survivors) matrices
are ever built, which is what makes this GPU-friendly for large N
(the GPU parallelises each block's comparisons, and the per-block
cost stays bounded instead of growing with the full archive size).

Usage:
    mask, F_nd = non_dominated_mask_gpu(F_all, block_size=1024)
    X_nd = X_all[mask]
"""

import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda") #ROCm as well
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def _pairwise_dominance(A, B):
    """
    A : (a, m) tensor
    B : (b, m) tensor
    Returns (a, b) boolean tensor: dom[i,j] = True if A[i] dominates B[j]
    (minimisation: A[i] <= B[j] elementwise, and A[i] < B[j] somewhere).
    """
    le = torch.all(A.unsqueeze(1) <= B.unsqueeze(0), dim=2)   # (a, b)
    lt = torch.any(A.unsqueeze(1) <  B.unsqueeze(0), dim=2)   # (a, b)
    return le & lt


def non_dominated_mask_gpu(F_all, block_size=1024, device=None, dtype=torch.float64):
    """
    F_all : (N, m) array-like (numpy array or torch tensor), minimisation
    block_size : size of blocks processed at a time
    device : torch device; defaults to 'cuda' if available, else 'cpu'
    dtype : float64 by default -- float32 rounding makes near-ties compare equal,
            which both drops genuinely non-dominated points and keeps dominated ones

    Returns:
        mask   : (N,) boolean numpy array, True = non-dominated
        F_nd   : (K, m) numpy array of the non-dominated objective values
    """
    if device is None:
        device = get_device()   #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dtype == torch.float64 and torch.device(device).type == "mps":
        device = torch.device("cpu")        # MPS has no float64

    F = torch.as_tensor(F_all, dtype=dtype, device=device)
    N, m = F.shape
    order = torch.arange(N, device=device)

    survivors_F = torch.empty((0, m), dtype=dtype, device=device)
    survivors_idx = torch.empty((0,), dtype=torch.int64, device=device)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        block_F = F[start:end]
        block_idx = order[start:end]

        # 1) filter dominance WITHIN the block
        dom = _pairwise_dominance(block_F, block_F)         # (b, b)
        dom.fill_diagonal_(False)
        block_dominated = dom.any(dim=0)                     # (b,)
        block_F = block_F[~block_dominated]
        block_idx = block_idx[~block_dominated]
        if block_F.shape[0] == 0:
            continue

        if survivors_F.shape[0] == 0:
            survivors_F = block_F
            survivors_idx = block_idx
            continue

        # 2) drop block points dominated by current survivors
        surv_dominates_block = _pairwise_dominance(survivors_F, block_F).any(dim=0)  # (b,)
        block_F = block_F[~surv_dominates_block]
        block_idx = block_idx[~surv_dominates_block]

        # 3) drop survivors dominated by the (filtered) surviving block points
        if block_F.shape[0] > 0:
            block_dominates_surv = _pairwise_dominance(block_F, survivors_F).any(dim=0)  # (S,)
            survivors_F = survivors_F[~block_dominates_surv]
            survivors_idx = survivors_idx[~block_dominates_surv]

            survivors_F = torch.cat([survivors_F, block_F], dim=0)
            survivors_idx = torch.cat([survivors_idx, block_idx], dim=0)

    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[survivors_idx] = True

    mask_np = mask.cpu().numpy()
    F_nd_np = F[mask].cpu().numpy()
    return mask_np, F_nd_np


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    N, m = 500000, 10
    F_all = rng.random((N, m)).astype(np.float32)

    mask, F_nd = non_dominated_mask_gpu(F_all, block_size=512)
    print(f"N={N}  non-dominated found: {mask.sum()}")