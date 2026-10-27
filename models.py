from abc import ABC
from contextlib import contextmanager
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.optimize import direct
from tests import Problem

import pyfmtools as fm
from choquetreg import ChoquetReg
from slip_interpolant import SLipInterpolant
import time


class _BudgetSpent(Exception):
    "raised by F() when max_eval is reached - the one place evaluations are counted"


class BaseModel(ABC):
    def __init__(self,
        problem: Problem = None, random_state: int = 42, w0: np.ndarray = None,
        max_iter: int = 50, max_eval: int = 10_000, tol: float = 1e-3, solver: str = "L-BFGS-B"
    ):
        if problem is None:
            raise ValueError("Problem is not specified")

        self.problem: Problem = problem
        self.PF_dim: int = problem.n_obj        # number of single objectives
        self.PS_dim: int = problem.n_dim        # dimensions of space of Pareto set
        self.random_state: int = random_state
        self.solver: str = solver
        self.rng = np.random.default_rng(random_state)

        self.num_func_eval: int = 0
        self.y: np.ndarray  = np.empty(0)       # will be set to vector of ones  - (n_eval,)
        self.PF: np.ndarray = None              # Pareto front - objective space - (n_eval, PF_dim)
        self.PS: np.ndarray = None              # Pareto set   - decision space  - (n_eval, PS_dim)
        self.w: np.ndarray = np.ones(self.PF_dim) / self.PF_dim if w0 is None else np.asarray(w0, dtype = float)       # weight vector for scalarisation

        self._worst: np.ndarray = None          # worst point of the current archive - the utility offset
        self.state_hist: dict = {}              # iteration -> what scalarise() needs besides w

        self.w_hist: np.ndarray = None          # weight used at each evaluation - (n_eval, PF_dim)
        self.obj_hist: np.ndarray = None        # the full history of the OBJECTIVER vectors found
        self.dec_hist: np.ndarray = None        # the full history of the DECISION vectors found
        # self.n_filtered: int = 0                # evaluations already folded into the archive

        self.iter_hist: np.ndarray = np.empty(0, dtype = int)   # iteration index of each evaluation
        self.scal_hist: list = []               # scalarised value at each evaluation, as the solver saw it
        self.pf_hist: dict = {}                 # iteration -> mask of obj_hist that filter_PF_PS() kept
        self.iteration: int = 0                 # current outer iteration
        self.wallclock_time: float = np.nan     # seconds in solve(); stays nan unless it finished

        # Termination criteria
        self.max_iter: int = max_iter           # outer iterations threshold
        self.max_eval: int = max_eval           # number of function evaluations threshold
        self.tol: float = tol                   # solvers' tolerance threshold


    def learn_weight(self):
        raise NotImplementedError("The learning mechanism is not defined")


    def scalarise(self, F: np.ndarray, w: np.ndarray):
        raise NotImplementedError("The scalarisation method is not defined")


    def _utility(self, F):
        """
            Turn the objectives into a maximisation problem, so that higher is better and the archive's
            own values are non-negative - what the Choquet integral and its capacity estimation expect.

            Returns None before the first archive exists, i.e. during step 0.
        """
        return None if self._worst is None else - F + self._worst


    def _state(self):
        """
            Everything scalarise() reads besides w. The model keeps only its latest fit, so without
            this a plot of iteration k would be drawn with the function learnt at the last one.
        """
        return dict(worst = self._worst, w = np.array(self.w, dtype = float))


    def _set_state(self, state):
        self._worst, self.w = state["worst"], state["w"]


    @contextmanager
    def as_of(self, iteration):
        """
            Put the model back into the scalarising state it had at `iteration` for the duration of
            the block - an iteration that was never recorded leaves it as it is.
        """
        live = self._state()
        try:
            self._set_state(self.state_hist.get(int(iteration), live))
            yield
        finally:
            self._set_state(live)


    def F(self, x, w):
        if self.num_func_eval >= self.max_eval:     # the budget is spent - abandon the solver call
            raise _BudgetSpent

        # Ensure the current point is within the problem's feasible region
        # Assume the feasible region is box constraints
        xl, xu = self.problem.xl, self.problem.xu
        x = np.clip(x, -np.inf if xl is None else xl, np.inf if xu is None else xu)

        # Ensure w is a np.ndarry -> vectorisation for better runtime
        w = np.asarray(w, dtype = float)

        obj_vec = np.asarray(self.problem.evaluate(x), dtype = float)   # Evaluate the multi-objectives
        F_val = float(self.scalarise(obj_vec, w))                       # Scalarisation subject to the weight vector

        # Save the history
        self.obj_hist = obj_vec[None, :] if self.obj_hist is None else np.vstack([self.obj_hist , obj_vec])
        self.dec_hist = x[None, :] if self.dec_hist is None else np.vstack([self.dec_hist, x])
        self.w_hist = w[None, :] if self.w_hist is None else np.vstack([self.w_hist, w])

        self.iter_hist = np.append(self.iter_hist, self.iteration)
        self.scal_hist.append(F_val)
        self.num_func_eval += 1     # Save total function evaluations
        return F_val


    @staticmethod
    def non_dominated(obj, kind: str = "strongly", eps: float = 1e-5):
        """
            Boolean mask of the non-dominated rows of obj (minimisation).

            kind:
                "strongly"  the exact Pareto front
                "eps"       the eps-non-dominated points: x dominates y iff it beats y by more than eps in
                            every objective - a superset of the front that grows with eps
                "weakly"    the weakly Pareto optimal front
        """
        keep, front = np.zeros(len(obj), bool), np.empty((0, obj.shape[1]))

        match kind:
            case "strongly":
                # in lexicographic order a point can only be dominated by one before it
                # the lexsort here is to speed-up the filtering
                for i in np.lexsort(obj.T[::-1]):
                    if not np.any(np.all(front <= obj[i], 1)):
                        keep[i], front = True, np.vstack([front, obj[i]])
            case "eps":
                if np.any(np.asarray(eps) < 0):
                    raise ValueError("eps must be non-negative")
                front = obj[BaseModel.non_dominated(obj, "strongly")]
                first = np.zeros(len(obj), bool)                    # one copy per objective vector, as "strongly" keeps
                first[np.unique(obj, axis = 0, return_index = True)[1]] = True
                for i in np.flatnonzero(first):
                    keep[i] = not np.any(np.all(front < obj[i] - eps, 1))
            case "weakly":
                for i in range(len(obj)):
                    front = np.delete(obj, i, 0)
                    keep[i] = not np.any(np.all(front < obj[i], 1))
            case _:
                raise ValueError("Unknown Pareto optimality kind")
        return keep


    def filter_PF_PS(self):
        """
            Fold the evaluations made from every evaluation made so far.
        """
        # Follow the Pareto optimality definition
        keep = self.non_dominated(self.obj_hist)
        self.pf_hist[self.iteration] = keep     # what learn_weight() is handed at the next iteration
        self.PF, self.PS = self.obj_hist[keep], self.dec_hist[keep]
        self._worst = np.max(self.PF, axis = 0)


    def _search(self, call):
        try:
            call()
        except Exception:
            if self.num_func_eval < self.max_eval:
                raise


    def solve(self, callback = None):
        # keep the search inside the box the test problem is defined on
        bounds = list(zip(self.problem.xl, self.problem.xu)) if self.problem.xl is not None else None

        # the box the random multi-start points are drawn from - the problem's own, not [0, 1)
        lo, hi = ((np.zeros(self.PS_dim), np.ones(self.PS_dim)) if self.problem.xl is None else
                  (np.asarray(self.problem.xl, float), np.asarray(self.problem.xu, float)))

        start = time.perf_counter()

        # Step 0: Initialisation
        self.iteration = 0
        self.state_hist[0] = self._state()      # before any fit - the plain weighted sum
        if callback is not None:
            callback(self, "learn")
        # Other NLP solvers
        # x0 = self.rng.random(self.PS_dim)
        # minimize(self.F, x0, args = (self.w,), method = self.solver, bounds = bounds, tol = self.tol)

        # DIRECT solver
        # Use the given weight vector first
        self._search(lambda: direct(self.F, args = (self.w,),  # solve using scalarised F
                                    bounds = bounds, maxiter = self.PS_dim * 20))
        # Then apply multi-start weight vectors
        # for _ in range(self.max_iter * 100):
        #     w = self.rng.dirichlet(np.ones(self.PF_dim))     # no repeated vector, \sum_{w_i} = 1 and w_i > 0
        #     direct(self.F, args = (w,),  # solve using scalarised F
        #             bounds = bounds, maxiter = self.max_iter * 10)

        # Filter out the non-dominated points obtained -> current PF and PS approximation
        self.filter_PF_PS()
        if callback is not None:
            callback(self, "eval")

        # Adaptive Scalarisation
        for it in range(1, int(self.max_iter + 1)):
            self.iteration = it

            # Step 1: Learn weight
            # X is the objective space -> self.PF
            # y is the vector of ones y = [1, 1, ..., 1] whose size is len(self.PF)
            self.y = np.ones(len(self.PF))
            self.w = self.learn_weight()
            if callback is not None:
                callback(self, "learn")
            self.state_hist[it] = self._state()

            # Step 2: Optimise subject to the learnt weight
            if it % 5 == 0:
                # DIRECT solver
                # -> no need multi-start
                self._search(lambda: direct(self.F, args = (self.w,),  # solve using scalarised F
                                            bounds = bounds, maxiter = self.PS_dim * 20))
            else:
                # Other NLP solvers
                for _ in range(5 * self.PS_dim):
                    x = lo + (hi - lo) * self.rng.random(self.PS_dim)            # random intial point for solver
                    self._search(lambda: minimize(self.F, x, args = (self.w,), method = self.solver,  # solve using scalarised F
                                                bounds = bounds, tol = self.tol, options={'eps':0.001,'maxiter': 50}))
                    if self.num_func_eval >= self.max_eval:
                        break

            # Filter out the non-dominated points obtained -> current PF and PS approximation
            self.filter_PF_PS()
            if callback is not None:
                callback(self, "eval")
            if self.num_func_eval >= self.max_eval: # stop when the # func. eval. budget is reached
                break

        self.wallclock_time = time.perf_counter() - start


    def visualise(self, *args, **kwargs):
        """
            Plot one view of this run - see viz.visualise() for the arguments.
        """
        from utils import visualise
        return visualise(self, *args, **kwargs)




# ===================================================================
class LinearModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def scalarise(self, F, w):
        # the same utility the Choquet model aggregates: -w.u = w.F - w.(worst point), so the minimiser
        # is the plain weighted sum's, but the weights now mean what they mean in a capacity
        utility = self._utility(F)
        return np.dot(F, w) if utility is None else -np.dot(utility, w)


    def learn_weight(self):
        # Adaptively learn the weights from the current Pareto front, on the utilities the scalarisation
        w, _ = nnls(self._utility(self.PF), self.y)    # this solver gives solution s.t. w >= 0

        # argmin of w.F is scale–invariant for w >= 0
        # as the F -> the pf (near the origin 0), the ||w|| diverges
        n = np.linalg.norm(w)
        return w / n if n > 0 else np.ones(self.PF_dim) / self.PF_dim


# ===================================================================





# ===================================================================
class ChoquetModel(BaseModel):
    def __init__(self, method: int = 0,  **kwargs):
        """
            method: for ChoquetReg(), default: 0
            kadd: 2 - assume 2-additive measure/capacity
        """

        super().__init__(**kwargs)

        # Init environment and Choquet aggregator
        self.env = fm.fm_init(self.problem.n_obj)
        self.choquet_reg = ChoquetReg(self.env)
        self.choquet_reg.w_vec = np.asarray(self.w, dtype = float)
        self.choquet_reg.dim = self.problem.n_obj
        self.choquet_reg.idx_i, self.choquet_reg.idx_j = self.choquet_reg.make_pair_indices(self.choquet_reg.dim)
        self.method = method
        self.choquet_reg.b_int = 0.0
        # n = self.problem.n_obj
        # self.choquet_reg.v = np.array([bin(i).count('1') / n for i in range(2 ** n)], dtype = 'float64')    # This raises error for some configurations of DTLZ2


    def scalarise(self, F, w):
        # F: evaluated values of {self.problem.n_job} objectives in MOP -> x in Choquet integral
        # w is the Choquet capacities -> already stoed in `self.choquet_reg.w_vec` -> unused herein
        if self.choquet_reg.method == -1:
            return np.dot(F, w)
        else:
            return -self.choquet_reg.choquet_value(self._utility(F))


    def _state(self):
        # choquet_value() reads v (the capacity), w_vec (the input scaling), b_int and method - the
        # weight in w_hist is w_vec, which is not on its own enough to rebuild the integral
        reg, state = self.choquet_reg, super()._state()
        v = getattr(reg, "v", None)
        state.update(method = reg.method, b_int = reg.b_int,
                     v = None if v is None else np.array(v, dtype = float),
                     w_vec = np.array(reg.w_vec, dtype = float))
        return state


    def _set_state(self, state):
        super()._set_state(state)
        reg = self.choquet_reg
        reg.method, reg.b_int, reg.v, reg.w_vec = state["method"], state["b_int"], state["v"], state["w_vec"]


    def learn_weight(self):
        # Learn the new Choquet capacities
        if self.PF.shape[0] <= 5:
            w, _ = nnls(self._utility(self.PF), self.y)    # this solver gives solution s.t. w >= 0
            n = np.linalg.norm(w)
            return w / n if n > 0 else np.ones(self.PF_dim) / self.PF_dim

        w = self.choquet_reg.fit_choquet(
            self._utility(self.PF) , self.y,
            use_intercept = False, kadd = 2, method = self.method, enforceone = False
        )
        return w


# ===================================================================


# ===================================================================
class LipschitzModel(BaseModel):
    """
        This model utilises the Lipschitz-continuity-based interpolant as a scalarising function
        Therefore:
            - scalarise(): predict() - Lipchitz-continuity-based interpolation
            - learnt_weight(): fit() - Lipschitz constant estimation from non–dominated PF
    """
    def __init__(self,
        type: str = "global_mon", mon: np.array = None, num_k: int = 1,
        a = None, b = None, w = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.type = type
        self.mon = np.ones(self.problem.n_obj) if mon is None else mon
        if len(self.mon) != self.problem.n_obj:
            raise ValueError("mon size must be equal to number of objectives")
        self.num_k = num_k
        self.a = a
        self.b = b
        self.knn_w = w          # knn weights for the interpolant - not BaseModel's weight vector
        self.slip = None


    def scalarise(self, F, w):
        # The `w` for scalarisation is the LipConst, which is inhenrently
        # stored in `self.slip`. Step 0 runs before the first fit, so fall back to the weighted sum.
        if self.slip is None:
            return np.dot(F, w)
        # return -np.asarray(self.slip.predict(F), dtype = float).reshape(-1)[0]
        return -np.asarray(self.slip.predict(self._utility(F)), dtype = float).reshape(-1)[0]


    def _state(self):
        # the constant in w_hist cannot rebuild the interpolant - keep the fitted object itself,
        # which learn_weight() replaces rather than mutates
        state = super()._state()
        state["slip"] = self.slip
        return state


    def _set_state(self, state):
        super()._set_state(state)
        self.slip = state["slip"]


    def learn_weight(self):
        """
            Return the Lipchitz constant estimated from non-dominated PF and 1
        """
        if self.PF.shape[0] <= 5:
            w, _ = nnls(self._utility(self.PF), self.y)    # this solver gives solution s.t. w >= 0
            n = np.linalg.norm(w)
            return w / n if n > 0 else np.ones(self.PF_dim) / self.PF_dim

        def worst_point():
            # worst point of all current non-dominated solutions
            # we shall refer to as reference nadir point
            return np.max(self.PF, axis = 0)

        # We must add the nadir point and 0 to the training data for SLip
        # Otherwise the data is subject to a flat function -> yield nothing
        X = np.concatenate([self._utility(self.PF), self._utility(worst_point())[None, :]])   # (K + 1, n_obj)
        y = np.concatenate([self.y, [0.0]])                     # (K + 1,)
        self.slip = SLipInterpolant(
            X, y, self.type, self.mon, self.num_k, self.a, self.b, self.knn_w
        )
        self.slip.fit()

        # BaseModel keeps one weight row per evaluation, so hand the constant back as a vector
        return np.full(self.PF_dim, float(self.slip.LipsConst))
