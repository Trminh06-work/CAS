import numpy as np
from liblipt import LibLipT
import torch
import pandas as pd

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SLipInterpolant():
    def __init__(self,
        X: np.array = None, y: np.array = None,
        type: str = "global", mon: np.array = None, num_k: int = 1,
        a = None, b = None, w = None
    ):
        if X is None or y is None:
            raise ValueError("No training data is given")

        if type not in ["global", "global_mon", "global_mix", "local", "local_mon", "local_mix", "smooth"]:
            raise ValueError("type must be `global`, `global_mon`, `global_mix`, `local`, `local_mon`, `local_mix` or `smooth`")

        if type not in ["global", "local"] and mon is None:
            raise ValueError(f"mon must be clearly specified since type = {type}")
        else:
            self.mon = mon

        # Construct the Slip interpolant from the training data given - contiguous float32,
        # for compatibility with the liblip codebase
        X = np.ascontiguousarray(X, dtype = np.float32)
        y = np.ascontiguousarray(y, dtype = np.float32).reshape(-1)
        self.dim = int(X.shape[1])
        self.npts = int(X.shape[0])
        self.type = type
        self.num_k = num_k

        self.a = np.zeros(self.dim, dtype = float) if a is None else a
        self.b = np.zeros(self.dim) if b is None else b
        self.w = np.full(self.num_k, 1.0 / self.num_k) if w is None else w

        self.sli = LibLipT(capacity = self.npts + 7, dim = self.dim, knn = self.num_k, device = get_device())
        self.sli.add(X, y)


    def fit(self):
        # Always applied: the knn weights w are used by every type whenever num_k > 1
        self.sli.setparams(self.mon, self.a, self.b, self.w)

        if self.type in ["global", "global_mon", "global_mix"]:
            if self.npts < 500_000:
                self.LipsConst = self.sli.lipschitz_constant() # ~42mins for ~500K samples, provided 10^8 ops = 1s
            else:
                self.LipsConst = self.sli.lipschitz_anchor_sampling()

        if self.type in ["local", "local_mon", "local_mix"]:
            self.sli.compute_local_lipschitz() # no fallback method for extensive data
            # predict() uses the per-point constants directly; LipsConst is only reported, so summarise
            # them by their mean - 0.0 when the library kept nothing
            spline = getattr(self.sli, "LipSplineY", None)
            self.LipsConst = float(spline.mean()) if spline is not None and spline.numel() else 0.0

        # Smooth version uses the global_mon with reduced LipsConst
        if self.type == "smooth":
            if self.npts < 500_000:
                self.LipsConst = self.sli.lipschitz_constant() # ~42mins for ~500K samples, provided 10^8 ops = 1s
            else:
                self.LipsConst = self.sli.lipschitz_anchor_sampling()
            self.smoothened_LipConst = self.LipsConst / 2


    def predict(self, F: np.array):
        if self.type == "global":
            preds = self.sli.values(
                Q = F, M = self.LipsConst, model = 0, k = self.num_k
            )

        if self.type == "global_mon":
            preds = self.sli.values(
                Q = F, M = self.LipsConst, model = 1, k = self.num_k
            )

        if self.type == "global_mix":
            global_preds = self.sli.values(
                Q = F, M = self.LipsConst, model = 0, k = self.num_k
            )
            global_mon_preds = self.sli.values(
                Q = F, M = self.LipsConst, model = 1, k = self.num_k
            )
            preds = (global_preds + global_mon_preds) / 2

        if self.type == "local":
            preds = self.sli.values_local(
                Q = F, model = 0, k = self.num_k
            )

        if self.type == "local_mon":
            preds = self.sli.values_local(
                Q = F, model = 1, k = self.num_k
            )

        if self.type == "local_mix":
            local_preds = self.sli.values_local(
                Q = F, model = 0, k = self.num_k
            )
            local_mon_preds = self.sli.values_local(
                Q = F, model = 1, k = self.num_k
            )
            preds = (local_preds + local_mon_preds) / 2

        if self.type == "smooth":
            preds = self.sli.values(
                Q = F, M = self.smoothened_LipConst, model = 1, k = self.num_k
            )

        return preds
