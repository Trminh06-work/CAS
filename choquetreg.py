import pyfmtools as fm
import math

import numpy as np

from scipy.optimize import minimize  # Nelder-Mead, or use cma package for CMA-ES

from scipy.optimize import linprog
from itertools import combinations

def estimate_a_lower_tail(X, y, quantile=0.2, eps=1e-8):
    """
    For each feature i, restrict to data points where x_i is in its
    OWN lower quantile (i.e. candidate for being a 'binding' min term),
    then estimate local sensitivity dy/dx_i there via simple linear fit.
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    N, d = X.shape
    if N <= 5:
        return np.ones(d)
    a_est = np.zeros(d)

    for i in range(d):
        thresh = np.quantile(X[:, i], quantile)
        mask = X[:, i] <= thresh
        if mask.sum() < 5:
            a_est[i] = 1.0 / (X[:, i].std(ddof=1) + eps)   # ddof=1: the sample std, as in the original version
            continue

        xi = X[mask, i]
        yi = y[mask]

        # simple slope estimate: cov(xi,yi)/var(xi)
        xi_c = xi - xi.mean()
        yi_c = yi - yi.mean()
        slope = (xi_c * yi_c).sum() / ((xi_c ** 2).sum() + eps)
        a_est[i] = max(abs(slope), eps)

    # normalize so the a_i's are on a sensible overall scale
    a_est = a_est / a_est.mean()
    return a_est



def solve_lp(Xy, p, enforceone=True):
    """
    Xy : array (K, n+1) where
         Xy[:, :n] = X  (features)
         Xy[:, -1] = y  (targets)
    p  : scalar penalty parameter
    """
    Xy_np = np.asarray(Xy, dtype=np.float64)
    X = Xy_np[:, :-1]   # (K, n)
    y = Xy_np[:, -1]    # (K,)


    """
    X : (K, n) array of x_ki values, k=1..K
    y : (K,)   array of y_k values
    p : scalar penalty parameter

    Variables layout:
    [t+_1..t+_K | t-_1..t-_K | m_1..m_n | m+_ij.. | m-_ij..]
      K            K             n          P          P
    where mij = m+_ij - m-_ij
    """
    K, n = X.shape
    pairs = list(combinations(range(n), 2))
    P = len(pairs)

    # variable index slices
    idx_tp = slice(0,          K)            # t+_k
    idx_tm = slice(K,          2*K)          # t-_k
    idx_m  = slice(2*K,        2*K+n)        # m_i (singletons)
    idx_mp = slice(2*K+n,      2*K+n+P)      # m+_ij
    idx_mm = slice(2*K+n+P,    2*K+n+2*P)   # m-_ij
    N = 2*K + n + 2*P

    # --------------------------------------------------
    # objective: min sum(t+) + sum(t-) + p*sum(m+ + m-)
    # --------------------------------------------------
    c_obj = np.zeros(N)
    c_obj[idx_tp] = 1.0
    c_obj[idx_tm] = 1.0
    c_obj[idx_mp] = p
    c_obj[idx_mm] = p

    # --------------------------------------------------
    # equality constraints
    # --------------------------------------------------
    A_eq_rows = []
    b_eq_rows = []

    # 1) for each k=1..K:
    # t+_k - t-_k - sum_i x_ki*m_i
    #   - sum_{i<j} (m+_ij - m-_ij)*min(x_ki, x_kj) = -y_k
    for k in range(K):
        row = np.zeros(N)
        row[idx_tp.start + k] =  1.0    # t+_k
        row[idx_tm.start + k] = -1.0    # t-_k
        for i in range(n):
            row[idx_m.start + i] = -X[k, i]
        for p_idx, (i, j) in enumerate(pairs):
            v = min(X[k, i], X[k, j])
            row[idx_mp.start + p_idx] = -v   # -m+_ij * min(...)
            row[idx_mm.start + p_idx] =  v   #  m-_ij * min(...)
        A_eq_rows.append(row)
        b_eq_rows.append(-y[k])

    if enforceone:
        # 2) sum_i m_i + sum_{i<j} (m+_ij - m-_ij) = 1
        row = np.zeros(N)
        row[idx_m]  =  1.0
        row[idx_mp] =  1.0
        row[idx_mm] = -1.0
        A_eq_rows.append(row)
        b_eq_rows.append(1.0)

    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq_rows)

    # --------------------------------------------------
    # inequality constraints
    # --------------------------------------------------
    A_ub_rows = []
    b_ub_rows = []

    # 3) m_i - sum_{j!=i} m-_ij >= 0
    #    → -m_i + sum_{j!=i} m-_ij <= 0
    for i in range(n):
        row = np.zeros(N)
        row[idx_m.start + i] = -1.0
        for p_idx, (a, b_) in enumerate(pairs):
            if a == i or b_ == i:
                row[idx_mm.start + p_idx] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    # --------------------------------------------------
    # bounds:
    # t+, t- >= 0
    # 0 <= m_i <= 1
    # m+_ij, m-_ij >= 0  (mij = m+ - m- in [-1,1] implicitly)
    # --------------------------------------------------
    bounds = (
        [(0, None)] * K  +   # t+
        [(0, None)] * K  +   # t-
        [(0, 1)]    * n  +   # m_i in [0,1]
        [(0, None)] * P  +   # m+_ij
        [(0, None)] * P      # m-_ij
    )

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds,
                  method='highs')

    if res.success:
        # reconstruct m array: singletons then pairs
        m_singletons = res.x[idx_m]
        m_pairs      = res.x[idx_mp] - res.x[idx_mm]  # mij = m+ - m-

        # combined m vector: [m_1..m_n, m_12, m_13, ...]
        m_all = np.concatenate([m_singletons, m_pairs])

        return {
            'success':      True,
            'obj':          res.fun,
            't+':           res.x[idx_tp],
            't-':           res.x[idx_tm],
            'm':            m_all,        # singletons then pairs
            'm_singletons': m_singletons,
            'm_pairs':      dict(zip(pairs, m_pairs)),
            'pairs':        pairs
        }
    else:
        return {'success': False, 'message': res.message}


    # reconstruct m array: singletons then pairs
    # m_singletons = res.x[idx_m]
    # m_pairs      = res.x[idx_mp] - res.x[idx_mm]  # mij = m+ - m-

    # # combined m vector: [m_1..m_n, m_12, m_13, ...]
    # m_all = np.concatenate([m_singletons, m_pairs])

    # return {
    #     'success':      True,
    #     'obj':          res.fun,
    #     't+':           res.x[idx_tp],
    #     't-':           res.x[idx_tm],
    #     'm':            m_all,        # singletons then pairs
    #     'm_singletons': m_singletons,
    #     'm_pairs':      dict(zip(pairs, m_pairs)),
    #     'pairs':        pairs
    # }


def fit(Xy, p, enforceone=True):
    """
    Xy : array (K, n+1)
    p  : scalar penalty
    returns: m as a numpy array (n + C(n,2),) or None if failed
    """
    result = solve_lp(Xy, p, enforceone)
    if result['success']:
        return result['m']
    else:
        #here you can go back and retry
        print(f"LP failed: {result['message']}")
        # print("Retry and ignore the failure message from LP solver")
        # return solve_lp(Xy, p, enforceone)["m"]
        return None

def Choquet2add(n,x,v):
    t=0
    for i in range(n):
        t+=v[i]*x[i]
    k=n
    for i in range(n):
        for j in range(i+1,n):
            t+=min(x[i],x[j]) * v[k]
            k+=1
    return t

class ChoquetReg:
    def __init__(self, env):
        self.dim=0
        self.N=0
        self.method=-1
        self.env=env


    def __del__(self):
        if self.env is not None:
            fm.fm_free(self.env)

    def fit_closed_form(self, X, y, use_intercept=True):
        """
       X: (N, d), y: (N,)
      Returns w (d,) and b (scalar) if use_intercept, else just w.
      """
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        N, d = X.shape

        self.dim=d
        self.N=N

        if use_intercept:
            ones = np.ones((N, 1))
            X_aug = np.concatenate([X, ones], axis=1)   # (N, d+1)
        else:
            X_aug = X

    # Solve normal equations via least squares (numerically stable)

        #> No `solution` member in output of np.linalg.lstsq - it returns a tuple
        solution = np.linalg.lstsq(X_aug, y, rcond=None)[0].reshape(-1)   # (d[+1],)
        if use_intercept:
            w, b = solution[:-1], float(solution[-1])
            return w, b
        else:
            return solution, 0.0

    def transform_features(self,X, w, d=1):
        """
        X: (N, d)
        w: (d,)
        Returns: (N, d), elementwise x_i -> x_i * w_i (broadcast over rows)
        """
        return X * w *d   # broadcasting: (N,d) * (d,) -> (N,d)


    def make_pair_indices(self,n):
        pairs = list(combinations(range(n), 2))
        idx_i = np.asarray([i for i,j in pairs], dtype=np.intp)
        idx_j = np.asarray([j for i,j in pairs], dtype=np.intp)
        return idx_i, idx_j

    def Choquet2add(self,n,x,v):
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
            # Singleton part
        t = np.dot(v[:n], x)
            # Pair part: build index arrays for i < j
        #    i_idx, j_idx = np.triu_indices(n, k=1)
        pair_values = np.minimum(x[self.i_idx], x[self.j_idx])
        pair_mobius = v[n:]
        t += np.dot(pair_values, pair_mobius)
        return t

    def Choquet2add_batch(self, n, X, v):
        """
        X : (batch, n) array
        v : (n + C(n,2),) array
        returns: (batch,) array
        """
        # print(f"X.shape = {X.shape} and v.shape = {v.shape}")
        singletons = X @ v[:n]                              # (batch,)
        mins       = np.minimum(X[:, self.idx_i], X[:, self.idx_j])  # (batch, P)
        pairs      = mins @ v[n:]                           # (batch,)
        return singletons + pairs

    def choquet_scaled(self,x):
        maxx=np.max(x)
        x=(1./maxx )*x
        return maxx*fm.ChoquetMob(x, self.mob, self.env)

    def choquet_value(self, x):
        if self.dim==0:
            return 0
        xt=self.transform_features(x,self.w_vec)
        xtnp=np.asarray(xt, dtype=np.float64)
        #print(xt)
        match self.method:
            case 0:
                #print(fm.ChoquetMob(xt, self.mob, self.env))
                return self.dim * fm.Choquet(xtnp, self.v, self.env) + self.b_int
            case 1:
                return self.dim* self.choquet_scaled(xtnp) + self.b_int
            case 2:
                return self.dim*self.Choquet2add(self.dim,xtnp,self.v)+ self.b_int # self.dim*fm.Choquet2addMob(xt, self.v, self.dim) + self.b_int
            case 3:
                return self.dim*fm.ChoquetKinter(xtnp, self.v, self.kint, self.env) + self.b_int
            case 4:
                return self.dim*fm.OWA(xtnp, self.v, self.env) + self.b_int

    def choquet_value_t(self, x):

        #here for method=2 I can use the whole array at once
        if self.method==2:
            xt=self.transform_features(np.asarray(x, dtype=np.float64),self.w_vec)
            # print(f"xt = \n {xt} \n v = \n {self.v}")
            return self.dim* self.Choquet2add_batch(self.dim,xt,self.v)+ self.b_int

        y=[]
        for xi in x:
            z=self.choquet_value(xi)
            y.append(z)
        return np.asarray(y, dtype=np.float64)


    def fit_choquet_inner(self, X, y, use_intercept=True, kadd=2, method=0, enforceone=True):

        #self.w_vec=1./(self.w_vec*self.dim)
        X_transformed = self.transform_features(np.asarray(X, dtype=np.float64), self.w_vec)

        y_col = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        X_with_target = np.concatenate([X_transformed, (1./self.dim)*(y_col - self.b_int)], axis=1)   # (N, d+1)/self.dim
        X_with_target = np.ascontiguousarray(X_with_target, dtype=np.float64)
        self.method=method

        match self.method:
            case 0:
                self.v= fm.FuzzyMeasureFit(self.N, kadd, self.env, X_with_target)
                self.mob=fm.Mobius(self.v, self.env)
            case 1:
                self.v= fm.FuzzyMeasureFitMob(self.N, kadd, self.env, X_with_target)
                self.mob=fm.Mobius(self.v, self.env)
            case 2: #2 additive
                self.v=fit(X_with_target, 1.0/self.dim,enforceone=enforceone) # fm.FuzzyMeasureFit2Additive(self.N, self.dim, 0, None,  None, 0, None, X_with_target)
            case 3:
                self.v=fm.FuzzyMeasureFitLPKinteractiveFree(self.N, kadd, self.env, K = 1, R = 1, dataset = X_with_target)
                self.kint=kadd
            case 4: #OWA
                self.v = fm.fittingOWA(self.N, self.env, X_with_target, ensureone=0 )
                # print(self.v)
            case _:
                self.v=None
                raise ValueError(f"Unknown method: {self.method}")

    def fit_choquet(self, X, y, use_intercept=True, kadd=2, method=0, enforceone=True):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        self.w_vec, self.b_int = self.fit_closed_form(X, y, use_intercept)

        # print(self.w_vec)
        self.w_vec=estimate_a_lower_tail(X,y)
        # print(self.w_vec)
        self.env=fm.fm_init(self.dim)

        self.i_idx, self.j_idx = np.triu_indices(self.dim, k=1)
        self.idx_i, self.idx_j=self.make_pair_indices(self.dim)
        self.fit_choquet_inner( X, y, use_intercept, kadd, method,enforceone=enforceone)

        #start iterations
        def objective(a_np):
            self.w_vec = np.asarray(a_np, dtype=np.float64)
            self.fit_choquet_inner( X, y, use_intercept, kadd, method,enforceone=enforceone)
            loss=np.sum((y-self.choquet_value_t(X))**2)
            return float(loss)

        result = minimize(objective, self.w_vec.copy(), method='Nelder-Mead',
                   options={'maxiter': 100, 'xatol': 1e-3})
        self.w_vec = np.asarray(result.x, dtype=np.float64)
        return self.w_vec




# Heuristic 2: inverse std normalization (more robust to outliers)
#    a_init = 1.0 / (X.std(axis=0, ddof=1) + 1e-8)




#result = minimize(objective, a_init, method='Nelder-Mead',
#                   options={'maxiter': 100, 'xatol': 1e-3})
#a_best = np.asarray(result.x, dtype=np.float64)




def estimate_a_commensurate(X, y, eps=1e-8):
    """
    X: (N, d), y: (N,)
    Returns a: (d,) heuristic initial guess for scaling factors,
    combining commensurability (equal spread) with marginal
    relevance to y (features that matter more get scaled to be
    'competitive' in the min/max comparisons).
    """
    X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
    N, d = X.shape

    # 1. Commensurability: normalize each x_i to unit spread (rank-based, robust to outliers)
    x_std = X.std(axis=0, ddof=1) + eps                  # ddof=1: the sample std, as in the original version
    a_scale = 1.0 / x_std

    # 2. Marginal relevance: how much does y vary with x_i alone?
    #    (simple correlation-based weighting)
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()
    corr = (X_centered * y_centered[:, None]).sum(axis=0) / (
        np.linalg.norm(X_centered, axis=0) * np.linalg.norm(y_centered) + eps
    )
    relevance = np.maximum(np.abs(corr), 0.05)   # avoid zero-ing out a feature entirely

    a_init = a_scale * relevance
    return a_init

#a_init = estimate_a_commensurate(X, y)
#print(a_init)
