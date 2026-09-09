from abc import ABC
import numpy as np
from scipy.optimize import minimize, nnls
from pymoo_tests import Problem

import pyfmtools as fm
from choquetreg import ChoquetReg
import torch
import time

class BaseModel(ABC):
    def __init__(self,
        problem: Problem = None, random_state: int = 42, w0: np.ndarray = None,
        max_iter: int = 50, tol: float = 1e-6, solver: str = "L-BFGS-B"
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

        self.w_hist: np.ndarray = None          # weight used at each evaluation - (n_eval, PF_dim)
        self.obj_hist: np.ndarray = None        # the full history of the OBJECTIVER vectors found
        self.dec_hist: np.ndarray = None        # the full history of the DECISION vectors found
        self.n_filtered: int = 0                # evaluations already folded into the archive

        self.iter_hist: np.ndarray = np.empty(0, dtype = int)   # iteration index of each evaluation
        self.iteration: int = 0                 # current outer (active-learning) iteration

        # Termination criteria
        self.max_iter: int = max_iter
        self.tol: float = tol


    def learn_weight(self):
        raise NotImplementedError("The learning mechanism is not defined")


    def scalarise(self, F: np.ndarray, w: np.ndarray):
        raise NotImplementedError("The scalarisation method is not defined")


    def F(self, x, w):
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
        self.num_func_eval += 1     # Save total function evaluations
        self.wallclock_time = 0
        return F_val


    @staticmethod
    def non_dominated(obj):
        """
            Boolean mask of the non-dominated rows of obj (minimisation).
        """
        keep, front = np.zeros(len(obj), bool), np.empty((0, obj.shape[1]))

        # in lexicographic order a point can only be dominated by one before it
        for i in np.lexsort(obj.T[::-1]):
            if not np.any(np.all(front <= obj[i], 1)):
                keep[i], front = True, np.vstack([front, obj[i]])
        return keep


    def filter_PF_PS(self):
        """
            Fold the evaluations made since the last call into the non-dominated archive.
        """
        # Avoid looping again the whole history to filter the PF, PS
        obj, dec = self.obj_hist[self.n_filtered:], self.dec_hist[self.n_filtered:]
        self.n_filtered = len(self.obj_hist)

        # Consider the prior PF, PS -> possibly discard some in the current iteration
        if self.PF is not None:
            obj, dec = np.vstack([self.PF, obj]), np.vstack([self.PS, dec])

        # Follow the Pareto optimality definition
        keep = self.non_dominated(obj)
        self.PF, self.PS = obj[keep], dec[keep]


    def solve(self):
        # keep the search inside the box the test problem is defined on
        bounds = list(zip(self.problem.xl, self.problem.xu)) if self.problem.xl is not None else None

        start = time.perf_counter()

        # Step 0: Initialisation
        self.iteration = 0
        x0 = self.rng.random(self.PS_dim)
        minimize(self.F, x0, args = (self.w,), method = self.solver, bounds = bounds, tol = self.tol)
        # Filter out the non-dominated points obtained -> current PF and PS approximation
        self.filter_PF_PS()

        # Adaptive Scalarisation
        for it in range(1, self.max_iter + 1):
            self.iteration = it

            # Step 1: Learn weight
            # X is the objective space -> self.PF
            # y is the vector of ones y = [1, 1, ..., 1] whose size is len(self.PF)
            self.y = np.ones(len(self.PF))
            self.w = self.learn_weight()

            # Step 2: Optimise subject to the learnt weight
            x = self.rng.random(self.PS_dim)                             # random intial point for solver
            minimize(self.F, x, args = (self.w,), method = self.solver,  # solve using scalarised F
                     bounds = bounds, tol = self.tol, options={'maxiter': 50})
            # Filter out the non-dominated points obtained -> current PF and PS approximation
            self.filter_PF_PS()

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
        return np.dot(F, w)


    def learn_weight(self):
        # Adaptively learn the weights from the current Pareto front
        w, _ = nnls(self.PF, self.y)    # this solver gives solution s.t. w >= 0

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
        self.method = method
        self.choquet_reg = ChoquetReg(method, self.env)


    def scalarise(self, F, w):
        # F: evaluated values of {self.problem.n_job} objectives in MOP -> x in Choquet integral
        # w is the Choquet capacities -> already stoed in `self.choquet_reg.w_vec` -> unused herein
        return self.choquet_reg.choquet_value(torch.from_numpy(F))


    def learn_weight(self):
        # Learn the new Choquet capacities
        w = self.choquet_reg.fit_choquet(
            torch.from_numpy(self.PF), torch.from_numpy(self.y),
            use_intercept = False, kadd = 2, method = self.method
        )
        return w.numpy()


# ===================================================================
