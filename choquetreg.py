import torch
import pyfmtools as fm
import math

import numpy as np

from scipy.optimize import minimize  # Nelder-Mead, or use cma package for CMA-ES


def estimate_a_lower_tail(X, y, quantile=0.2, eps=1e-8):
    """
    For each feature i, restrict to data points where x_i is in its
    OWN lower quantile (i.e. candidate for being a 'binding' min term),
    then estimate local sensitivity dy/dx_i there via simple linear fit.
    """
    N, d = X.shape
    # print("Current shape", X.shape)
    if N <= 5:
        a_est = torch.ones(d)
        # print("Get out here")
        return a_est
    a_est = torch.zeros(d)

    for i in range(d):
        thresh = torch.quantile(X[:, i], quantile)
        mask = X[:, i] <= thresh
        if mask.sum() < 5:
            a_est[i] = 1.0 / (X[:, i].std() + eps)
            continue

        xi = X[mask, i]
        yi = y[mask]

        # simple slope estimate: cov(xi,yi)/var(xi)
        xi_c = xi - xi.mean()
        yi_c = yi - yi.mean()
        slope = (xi_c * yi_c).sum() / (xi_c.pow(2).sum() + eps)
        a_est[i] = slope.abs().clamp(min=eps)

    # normalize so the a_i's are on a sensible overall scale
    a_est = a_est / a_est.mean()
    return a_est


"""
CHANGES:

    - I added `env` and `method` to your original class

"""
class ChoquetReg:
    def __init__(self, method, env):
        self.dim=0
        self.N=0
        self.method=method
        self.env=env


    def __del__(self):
        if self.env is not None:
            fm.fm_free(self.env)

    def fit_closed_form(self, X, y, use_intercept=True):
        """
       X: (N, d), y: (N, 1)
      Returns w (d,) and b (scalar) if use_intercept, else just w.
      """
        N, d = X.shape

        self.dim=d
        self.N=N

        if use_intercept:
            ones = torch.ones(N, 1, dtype=X.dtype, device=X.device)
            X_aug = torch.cat([X, ones], dim=1)   # (N, d+1)
        else:
            X_aug = X

    # Solve normal equations via least squares (numerically stable)
        solution = torch.linalg.lstsq(X_aug, y).solution   # (d[+1], 1)

        if use_intercept:
            w, b = solution[:-1].squeeze(-1), solution[-1].item()
            return w, b
        else:
            return solution.squeeze(-1), 0.0

    def transform_features(self,X, w, d=1):
        """
        X: (N, d)
        w: (d,)
        Returns: (N, d), elementwise x_i -> x_i * w_i (broadcast over rows)
        """
        return X * w *d   # broadcasting: (N,d) * (d,) -> (N,d)


    def choquet_scaled(self,x):
        maxx=np.max(x)
        x=(1./maxx )*x
        return maxx*fm.ChoquetMob(x, self.mob, self.env)

    # Return a single Choquet integral w.r.t. x - a vector (d,)
    def choquet_value(self, x):
        if self.dim==0:
            return 0
        xt=self.transform_features(x,self.w_vec)
        xt=xt.numpy().astype('float64')
        #print(xt)
        match self.method:
            case 0:
                #print(fm.ChoquetMob(xt, self.mob, self.env))
                return self.dim * fm.Choquet(xt, self.v, self.env) + self.b_int
            case 1:
                return self.dim* self.choquet_scaled(xt) + self.b_int
            case 2:
                return self.dim*fm.Choquet2addMob(xt, self.v, self.dim) + self.b_int
            case 3:
                return self.dim*fm.ChoquetKinter(xt, self.v, self.kint, self.env) + self.b_int
            case 4:
                return self.dim*fm.OWA(xt, self.v, self.env) + self.b_int

    # Return a batch of Choquet integrals w.r.t. X - a matrix (n, d)
    def choquet_value_t(self, x):
        y=[]
        for xi in x:
            z=self.choquet_value(xi)
            y.append(z)
        return torch.tensor(y, dtype=x.dtype)


    def fit_choquet_inner(self, X, y, use_intercept=True, kadd=2, method=0):

        #self.w_vec=1./(self.w_vec*self.dim)
        X_transformed = self.transform_features(X, self.w_vec)

        y_col = y.view(-1, 1)
        X_with_target = torch.cat([X_transformed, (1./self.dim)*(y_col - self.b_int)], dim=1)   # (N, d+1)/self.dim
        X_with_target = X_with_target.contiguous()
        self.method=method

        match self.method:
            case 0:
                self.v= fm.FuzzyMeasureFit(self.N, kadd, self.env, X_with_target)
                self.mob=fm.Mobius(self.v, self.env)
            case 1:
                self.v= fm.FuzzyMeasureFitMob(self.N, kadd, self.env, X_with_target)
                self.mob=fm.Mobius(self.v, self.env)
            case 2: #2 additive
                self.v=fm.FuzzyMeasureFit2Additive(self.N, self.dim, 0, None,  None, 0, None, X_with_target)
            case 3:
                self.v=fm.FuzzyMeasureFitLPKinteractiveAutoK(self.N, kadd, self.env, 0.3, 100, X_with_target)
                self.kint=kadd

            case 4: #OWA
                self.v,_= fm.fittingOWA(self.N, self.env, X_with_target)
                print(self.v)
            case _:
                self.v=None
                raise ValueError(f"Unknown method: {self.method}")

    def fit_choquet(self, X, y, use_intercept=True, kadd=2, method=0):
        self.w_vec, self.b_int = self.fit_closed_form(X, y, use_intercept)

        # """
        #     estimate_a_lower_tail(X,y) causes errors for DTLZ1 since X.shape[0] is insufficient -> already fixed
        # """
        # print("Before: ", self.w_vec)
        self.w_vec=estimate_a_lower_tail(X,y)
        # print("After: ", self.w_vec)
        self.env=fm.fm_init(self.dim)
        self.fit_choquet_inner( X, y, use_intercept, kadd, method)

        #start iterations
        def objective(a_np):
            self.w_vec = torch.tensor(a_np, dtype=X.dtype)
            self.fit_choquet_inner( X, y, use_intercept, kadd, method)
            loss=torch.sum((y-self.choquet_value_t(X))**2)
            return loss.item()

        # result = minimize(objective, self.w_vec.numpy(), method='Nelder-Mead',
        #            options={'maxiter': 100, 'xatol': 1e-3})
        result = minimize(objective, self.w_vec.numpy(), method='Nelder-Mead',
                           options={'maxiter': 20, 'xatol': 1e-3})
        self.w_vec = torch.tensor(result.x, dtype=X.dtype)
        # print(self.w_vec)
        return self.w_vec




# Heuristic 2: inverse std normalization (more robust to outliers)
#    a_init = 1.0 / (X.std(dim=0) + 1e-8)




#result = minimize(objective, a_init.numpy(), method='Nelder-Mead',
#                   options={'maxiter': 100, 'xatol': 1e-3})
#a_best = torch.tensor(result.x, dtype=X.dtype)




def estimate_a_commensurate(X, y, eps=1e-8):
    """
    X: (N, d), y: (N,)
    Returns a: (d,) heuristic initial guess for scaling factors,
    combining commensurability (equal spread) with marginal
    relevance to y (features that matter more get scaled to be
    'competitive' in the min/max comparisons).
    """
    N, d = X.shape

    # 1. Commensurability: normalize each x_i to unit spread (rank-based, robust to outliers)
    x_std = X.std(dim=0) + eps
    a_scale = 1.0 / x_std

    # 2. Marginal relevance: how much does y vary with x_i alone?
    #    (simple correlation-based weighting)
    X_centered = X - X.mean(dim=0)
    y_centered = y - y.mean()
    corr = (X_centered * y_centered.unsqueeze(-1)).sum(dim=0) / (
        X_centered.norm(dim=0) * y_centered.norm() + eps
    )
    relevance = corr.abs().clamp(min=0.05)   # avoid zero-ing out a feature entirely

    a_init = a_scale * relevance
    return a_init

#a_init = estimate_a_commensurate(X, y)
#print(a_init)



