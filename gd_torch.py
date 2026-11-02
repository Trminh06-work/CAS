"""
GPU-accelerated GD and IGD via torch.

No KD-tree on GPU, so nearest-neighbour search is done via a
pairwise distance matrix (chunked over A or R to bound memory for
large point sets, the same "block" idea as the non-dominated filter).

GD(A,R)  = ( mean_a  min_r ||a-r||^p )^(1/p)   -- convergence
IGD(A,R) = ( mean_r  min_a ||r-a||^p )^(1/p)   -- convergence+coverage

Usage:
    gd  = gd_gpu(A, R, p=2)
    igd = igd_gpu(A, R, p=1)
"""

import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda") #ROCm as well
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def _min_dist_chunked(query, ref, chunk_size=2048, device=None):
    """
    For each row in `query`, find the minimum Euclidean distance to
    any row in `ref`, processing `query` in chunks to bound memory
    (avoids materialising the full (len(query), len(ref)) matrix at
    once for very large point sets).

    query : (Q, m)
    ref   : (R, m)
    returns: (Q,) tensor of min distances
    """
    if device is None:
        device = get_device()   #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        
    query = torch.as_tensor(query, dtype=torch.float32, device=device)
    ref = torch.as_tensor(ref, dtype=torch.float32, device=device)

    Q = query.shape[0]
    out = torch.empty(Q, dtype=torch.float32, device=device)

    for start in range(0, Q, chunk_size):
        end = min(start + chunk_size, Q)
        q_chunk = query[start:end]                      # (c, m)
        # pairwise distances: (c, R)
        d = torch.cdist(q_chunk, ref, p=2)
        out[start:end] = d.min(dim=1).values

    return out


def gd_gpu(A, R, p=2, chunk_size=2048, device=None):
    """
    Generational Distance: GD(A,R) = (mean_a min_r ||a-r||^p)^(1/p)
    """
    d = _min_dist_chunked(A, R, chunk_size=chunk_size, device=device)
    val = (d.pow(p).mean()).pow(1.0 / p)
    return val.item()


def igd_gpu(A, R, p=1, chunk_size=2048, device=None):
    """
    Inverted Generational Distance: IGD(A,R) = (mean_r min_a ||r-a||^p)^(1/p)
    (roles of A and R swapped relative to GD)
    """
    d = _min_dist_chunked(R, A, chunk_size=chunk_size, device=device)
    val = (d.pow(p).mean()).pow(1.0 / p)
    return val.item()


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    A = rng.random((50, 3)).astype(np.float32)
    R = rng.random((200, 3)).astype(np.float32)

    gd = gd_gpu(A, R, p=2, chunk_size=16)   # small chunk_size to exercise chunking logic
    igd = igd_gpu(A, R, p=1, chunk_size=16)

    print(f"torch GD  = {gd:.6f}")
    print(f"torch IGD = {igd:.6f}")