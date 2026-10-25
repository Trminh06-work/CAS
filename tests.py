from typing import List
import numpy as np
from pymoo.problems import get_problem
from pymoo.core.problem import ElementwiseProblem
from agg import AggFunc



# ========================================================================================
# ======================================= Helpers ========================================
# ========================================================================================
def take_first(x, n_required, name):
    """
        The first n_required coordinates of x. A problem with a fixed number of variables ignores
        any extra ones, but cannot run with fewer.
    """
    x = np.asarray(x, dtype = float).ravel()
    if x.size < n_required:
        raise ValueError(f"{name} needs at least {n_required} variables, got {x.size}")
    return x[:n_required]


def to_domain(x, lo, hi):
    """
        x in [0, 1] -> the function's own domain [lo, hi].
    """
    return lo + (hi - lo) * np.asarray(x, dtype = float)



# ========================================================================================
# ====================================== Base Class ======================================
# ========================================================================================
class Problem(ElementwiseProblem):
    def __init__(self, name: str = None, **kwargs):
        if name is None:
            raise ValueError("The problem's name is not specified")

        self.inner = get_problem(name, **kwargs)
        super().__init__(n_var = self.inner.n_var, n_obj = self.inner.n_obj,
                         xl = self.inner.xl, xu = self.inner.xu)
        self.n_dim = self.n_var         # pymoo calls it n_var


    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.inner.evaluate(np.asarray(x, dtype = float))


    def pareto_front(self, n_points = 300):
        return self.inner.pareto_front()


    def pareto_set(self, n_points = 300):
        raise NotImplementedError("The true Pareto set of this problem is not known")



class ExtendedProblem(ElementwiseProblem):
    """
        This class offers a means to combine different test problems in any possible permutation.
        The objectives of every problem given are concatenated, in the order the list has them, and
        `order` then picks and reorders that vector, or a subset of the objectives.

        Note: Not all test problems use the same number of decision variables, we shall use the first few decision variables
        if the problem requires fewer. However, we do not permit passing a decision vector whose dimensionality is lower than
        the minimum of the test problems in the given list

        Parameters:
            problems (list[ElementwiseProblem | AggFunc]): Optimization problems used to construct
                the decision space. Every one of them is searched on the unit box. An aggregation
                function from agg.py, e.g. QAM([...], gen, inv_gen), stands for the one objective it
                makes of its own problems - see Aggregated.

            order (list, optional): which objectives of the combined vector to keep, and in which
                order. Defaults to all of objectives in the given problems.
    """
    def __init__(self, problems: List[ElementwiseProblem | AggFunc] | None = None, order: List | None = None, **kwargs):
        if problems is None:
            raise ValueError("No problems are given")

        if not list(problems):
            raise ValueError("No problems are given")
        self.problems = self._build(problems)

        # the decision space has to serve the highest dimensional problem; the others take the first few
        self.n_dim = max(p.n_var for p in self.problems)
        n_obj = sum(p.n_obj for p in self.problems)     # the combined objective vector

        self.order = np.arange(n_obj) if order is None else np.asarray(order, dtype = int)  # any permutation of 0..n_obj-1
        if np.any(self.order < 0) or np.any(self.order >= n_obj):
            raise ValueError(f"order entries must be in [0, {n_obj})")
        if len(set(self.order.tolist())) != len(self.order):
            raise ValueError("order must not repeat indices")
        if len(self.order) < 2:
            raise ValueError("order must have at least 2 indices")

        super().__init__(n_var = self.n_dim, n_obj = len(self.order), xl = 0.0, xu = 1.0)


    @staticmethod
    def _build(problems):
        """
            Turn the list into instances. A class stands for an instance of itself only when its
            own definition fixes its size; anything else has to be constructed, so that a decision
            space is never set by default. An aggregation function becomes the problem it defines.
        """
        out = []
        for p in problems:
            if isinstance(p, AggFunc):              # one objective, aggregated from its own problems
                out.append(Aggregated(p))
            elif not isinstance(p, type):           # already an instance, left alone
                out.append(p)
            elif getattr(p, "N_VAR", None):         # its own definition fixes the size
                out.append(p())
            else:
                raise ValueError(
                    f"{p.__name__}'s decision space is not fixed by its definition - construct it "
                    f"with a size, as {p.__name__}(n_dim = ...)")
        return out


    def _objectives(self, x):
        "one point of the unit box -> every problem's objectives, concatenated and then reordered"
        x = np.asarray(x, dtype = float).ravel()
        F = [np.asarray(p.evaluate(take_first(x, p.n_var, type(p).__name__)), float).ravel()
             for p in self.problems]
        return np.concatenate(F)[self.order]


    def _evaluate(self, x, out):
        if len(x) < self.n_dim:
            raise ValueError(f"The given decision vector must have higher dimensionality than {self.n_dim}")
        out["F"] = self._objectives(x)


    def pareto_front(self, n_points = 300):
        raise NotImplementedError("Extended problem - no analytical PF")



class Aggregated(ElementwiseProblem):
    """
        One objective, made by an aggregation function (agg.py) from the objectives of its problems, which share
        one decision vector as in ExtendedProblem. An aggregation function acts on [0, 1]^n and is non-decreasing,
        whereas the objectives are minimised, each on its own scale. So every objective is
            1. normalised to [0, 1], with bounds fixed here, once, and clipped
            2. turned into a satisfaction degree, s = 1 - normalised (1 = best)
        and the aggregate is returned as 1 - A(s), to be minimised. Aggregating the normalised objectives as they are
        would make a conjunctive A read "one objective at its best is enough": T_P would be 0 wherever ZDT2's f1 is.

        ExtendedProblem wraps an aggregation function in this by itself; construct one directly to give own bounds.

        Parameters:
            func (AggFunc): the aggregation function, which holds the problems it aggregates

            bounds (list of (lo, hi), optional): one pair per aggregated objective. Defaults to lo, the problem's
                ideal point where its front is known, else the least sampled value, and hi, the largest.

            n_samples (int): random points of the unit box the default bounds are taken from

            seed (int): seed of those points
    """
    def __init__(self, func: AggFunc, bounds = None, n_samples: int = 10_000, seed: int = 0):
        problems = ExtendedProblem._build(func.problems)
        n_args = sum(p.n_obj for p in problems)
        if n_args < 2:
            raise ValueError(f"{type(func).__name__} needs at least 2 objectives to aggregate, got {n_args}")

        self.func = func
        self.inner = ExtendedProblem(problems = problems)      # every objective of every problem, in order
        func.set_arity(n_args)

        super().__init__(n_var = self.inner.n_var, n_obj = 1, xl = 0.0, xu = 1.0)
        self.n_dim = self.n_var

        if bounds is None:
            self.lo, self.hi = self._bounds(n_samples, seed)
        else:
            bounds = np.asarray(bounds, dtype = float)
            if bounds.shape != (n_args, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
                raise ValueError(f"bounds needs one (lo, hi) pair, lo < hi, for each of the {n_args} objectives")
            self.lo, self.hi = bounds[:, 0], bounds[:, 1]


    def _bounds(self, n_samples, seed):
        "lo: each problem's ideal point where its front is known, else the least sample; hi: the largest sample"
        F = self.inner.evaluate(np.random.default_rng(seed).random((n_samples, self.n_var)))
        lo, hi = F.min(axis = 0), F.max(axis = 0)

        j = 0
        for p in self.inner.problems:
            try:
                front = p.pareto_front()
                if front is not None:
                    lo[j:j + p.n_obj] = np.min(np.atleast_2d(front), axis = 0)
            except NotImplementedError:
                pass
            j += p.n_obj

        hi = np.where(hi > lo, hi, lo + 1.0)        # a flat objective would normalise to 0/0
        return lo, hi


    def _evaluate(self, x, out, *args, **kwargs):
        F = self.inner._objectives(x)
        s = 1 - np.clip((F - self.lo) / (self.hi - self.lo), 0, 1)

        # log(0), 1 / 0, ... of the generators are expected at the bounds, s_i = 0 or 1
        with np.errstate(divide = "ignore", over = "ignore"):
            out["F"] = np.array([1 - self.func.fun(s)], dtype = float)


    def pareto_front(self, n_points = 300):
        raise NotImplementedError("Aggregated problem - no analytical PF")



# =========================================================================================
# =============================== Single-objective Problems ===============================
# =========================================================================================
# Aim: combine with other test problems to form a MOP
# Source: https://arxiv.org/abs/1308.4008
# adopted from scipy: https://github.com/scipy/scipy/tree/e1a7f5e4239c32a0415c3368f7969b93feddfee2/benchmarks/benchmarks/go_benchmark_functions

class SOP(ElementwiseProblem):
    """
        Base for the single-objective problems.

        A subclass sets
            self.lo, self.hi        the function's own domain, applied in _objectives
            self.global_optimum     its known minimisers, in its own domain (as scipy lists them)
            self.fglob              its global minimum
        and implements fun(z) on its own domain.
    """
    def __init__(self, n_dim: int = 10):
        self.n_dim = n_dim
        super().__init__(n_var = n_dim, n_obj = 1,
                         xl = np.zeros(n_dim), xu = np.ones(n_dim))


    def fun(self, z):
        raise NotImplementedError("The benchmark function is not defined")


    def _objectives(self, x):
        "one point of the unit box -> a one-objective vector"
        z = to_domain(take_first(x, self.n_var, type(self).__name__), self.lo, self.hi)
        return np.array([self.fun(z)], dtype = float)


    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._objectives(x)


    def pareto_set(self, n_points = 300):
        # with one objective the set is the global minimisers; their pre-image in the unit box
        return (np.asarray(self.global_optimum, dtype = float) - self.lo) / (self.hi - self.lo)


    def pareto_front(self, n_points = 300):
        return np.full((len(self.global_optimum), 1), self.fglob)



class Ackley01(SOP):
    """
    Ackley 1, No. 1 in Jamil & Yang. Adopted from scipy's Ackley01
    Continuous, differentiable, non-separable, scalable, multimodal

        f(z) = -20 exp(-0.2 sqrt(sum(zi^2) / n)) - exp(sum(cos(2 pi zi)) / n) + 20 + e,   zi in [-35, 35]

    Global optimum: f = 0 at z = (0, ..., 0), the centre of the unit box
    Note: the paper prints -0.02 in the first exponent; scipy and Ackley's original use -0.2
    """
    def __init__(self, n_dim: int = 10):
        super().__init__(n_dim)
        self.lo, self.hi = -35.0, 35.0
        self.global_optimum = [[0.0] * n_dim]
        self.fglob = 0.0


    def fun(self, z):
        return (-20.0 * np.exp(-0.2 * np.sqrt(np.sum(z ** 2) / self.n_dim))
                - np.exp(np.sum(np.cos(2 * np.pi * z)) / self.n_dim) + 20.0 + np.e)



class Schwefel04(SOP):
    """
    Schwefel 2.4, No. 120 in Jamil & Yang. Adopted from scipy's Schwefel04
    Continuous, differentiable, non-separable, scalable

        f(z) = sum_{i=1}^{n} [(zi - 1)^2 + (z1 - zi^2)^2],   zi in [0, 10]

    Global optimum: f = 0 at z = (1, ..., 1), i.e. 0.1 in the unit box
    """
    def __init__(self, n_dim: int = 10):
        super().__init__(n_dim)
        self.lo, self.hi = 0.0, 10.0
        self.global_optimum = [[1.0] * n_dim]
        self.fglob = 0.0


    def fun(self, z):
        return np.sum((z - 1.0) ** 2 + (z[0] - z ** 2) ** 2)



class Gulf(SOP):
    """
    Gulf Research Problem, No. 60 in Jamil & Yang. Adopted from scipy's Gulf
    Continuous, differentiable, non-separable, non-scalable, multimodal

        f(z) = sum_{i=1}^{99} [exp(-|ui - z2|^z3 / z1) - i / 100]^2,   zi in [0, 50]
        ui   = 25 + (-50 ln(i / 100))^(2/3)

    Global optimum: f = 0 at z = (50, 25, 1.5), i.e. (1, 0.5, 0.03) in the unit box
    Note: scipy takes |ui - z2|
    """
    N_VAR = 3

    def __init__(self, n_dim: int = 3):
        if n_dim < self.N_VAR:
            raise ValueError(f"Gulf needs at least {self.N_VAR} variables, got {n_dim}")

        # always the 3 variable problem - a larger n_dim is dropped by take_first() in _objectives
        super().__init__(self.N_VAR)
        self.lo, self.hi = 0.0, 50.0
        self.global_optimum = [[50.0, 25.0, 1.5]]
        self.fglob = 0.0

        self.t = np.arange(1.0, 100.0) / 100                 # ti = i / 100, i = 1, ..., 99
        self.u = 25 + (-50 * np.log(self.t)) ** (2 / 3)


    def fun(self, z):
        a = np.abs(self.u - z[1]) ** z[2]
        # at z1 = 0 take the limit z1 -> 0+: a / z1 -> inf, but stays 0 where a = 0
        r = a / z[0] if z[0] > 0 else np.where(a > 0, np.inf, 0.0)
        return np.sum((np.exp(-r) - self.t) ** 2)



# ========================================================================================
# =============================== Multi-objective Problems ===============================
# ========================================================================================

# =============================== ZDT Problems ===============================
class ZDT1(Problem):
    def __init__(self, n_dim = 30):
        super().__init__("zdt1", n_var = n_dim)


    def pareto_set(self, n_points = 300):
        # x0 sweeps f1; every other coordinate sits at 0 so that g = 1
        X = np.zeros((n_points, self.n_dim))
        X[:, 0] = np.linspace(0.0, 1.0, n_points)
        return X



class ZDT2(Problem):
    def __init__(self, n_dim = 30):
        super().__init__("zdt2", n_var = n_dim)


    def pareto_set(self, n_points = 300):
        X = np.zeros((n_points, self.n_dim))
        X[:, 0] = np.linspace(0.0, 1.0, n_points)
        return X



class ZDT3(Problem):
    def __init__(self, n_dim = 30):
        super().__init__("zdt3", n_var = n_dim)


    def pareto_set(self, n_points = 300):
        # x0 runs over the disconnected segments, every other coordinate at 0
        f1 = self.pareto_front(n_points)[:, 0]
        idx = np.linspace(0, len(f1) - 1, min(n_points, len(f1))).astype(int)
        X = np.zeros((len(idx), self.n_dim))
        X[:, 0] = f1[idx]
        return X



# =============================== DTLZ Problems ===============================
class DTLZ1(Problem):
    def __init__(self, n_dim = 10, n_obj = 3):
        super().__init__("dtlz1", n_var = n_dim, n_obj = n_obj)


    def pareto_set(self, n_points = 300):
        # the tail sits at 0.5 so that g = 0; the first n_obj - 1 coordinates are free
        rng = np.random.default_rng(0)
        X = np.full((n_points, self.n_dim), 0.5)
        X[:, :self.n_obj - 1] = rng.random((n_points, self.n_obj - 1))
        return X



class DTLZ2(Problem):
    def __init__(self, n_dim = 10, n_obj = 3):
        super().__init__("dtlz2", n_var = n_dim, n_obj = n_obj)


    def pareto_set(self, n_points = 300):
        rng = np.random.default_rng(0)
        X = np.full((n_points, self.n_dim), 0.5)
        X[:, :self.n_obj - 1] = rng.random((n_points, self.n_obj - 1))
        return X



# =============================== Kursawe Problems ===============================
class Kursawe(Problem):
    """
    The PF can only be obtained analytically iff #dec_variables = 3
    See: https://al-roomi.org/benchmarks/multi-objective/unconstrained-list/322-kursawe-s-function-kur
    """
    N_VAR = 3
    def __init__(self, n_dim: int = 3):
        if n_dim < self.N_VAR:
            raise ValueError(f"Kursawe needs at least {self.N_VAR} variables, got {n_dim}")

        # always the 3 variable problem - a larger n_dim is dropped by take_first() below
        super().__init__("kursawe")

        # searched on the unit box; kursawe's own [-5, 5] is applied in _evaluate
        self.lo, self.hi = float(self.inner.xl[0]), float(self.inner.xu[0])
        self.xl, self.xu = np.zeros(self.N_VAR), np.ones(self.N_VAR)


    def _evaluate(self, x, out, *args, **kwargs):
        z = to_domain(take_first(x, self.N_VAR, "Kursawe"), self.lo, self.hi)
        out["F"] = self.inner.evaluate(z)



# =============================== Schaffer Problems ===============================
class Sch1(ElementwiseProblem):
    """
    Adopted from: https://al-roomi.org/benchmarks/multi-objective/unconstrained-list/319-schaffer-s-function-no-1-sch1
    The PF can only be obtained analytically iff #dec_variables = 1

    A: user-defined bound
    """
    N_VAR = 1

    def __init__(self, n_dim = 1, A: float = 1e5):
        if n_dim < self.N_VAR:
            raise ValueError(f"Sch1 needs at least {self.N_VAR} variable, got {n_dim}")
        if A <= 0:
            raise ValueError("A must be a positive real number")

        # always the 1 variable problem - a larger n_dim is dropped by take_first() below
        super().__init__(n_var = self.N_VAR, n_obj = 2, xl = 0.0, xu = 1.0)
        self.n_dim = self.N_VAR
        self.lo, self.hi = -A, A        # the function's own domain, applied in _objectives


    def _objectives(self, x):
        "one point of the unit box -> one objective vector"
        z = float(to_domain(take_first(x, self.N_VAR, "Sch1"), self.lo, self.hi)[0])
        return np.array([z ** 2, (z - 2) ** 2], dtype = float)


    def _evaluate(self, x, out):
        out["F"] = self._objectives(x)


    def pareto_set(self, n_points = 300):
        X = np.zeros((n_points, self.n_dim))
        X[:, 0] = (np.linspace(0.0, 2.0, n_points) - self.lo) / (self.hi - self.lo)
        return X


    def pareto_front(self, n_points = 300):
        return np.array([self._objectives(x) for x in self.pareto_set(n_points)])



class Sch2(ElementwiseProblem):
    """
    Adopted from: https://al-roomi.org/benchmarks/multi-objective/unconstrained-list/320-schaffer-s-function-no-2-sch2
    The PF can only be obtained analytically iff #dec_variables = 1
    """
    N_VAR = 1

    def __init__(self, n_dim = 1):
        if n_dim < self.N_VAR:
            raise ValueError(f"Sch2 needs at least {self.N_VAR} variable, got {n_dim}")

        # always the 1 variable problem - a larger n_dim is dropped by take_first() below
        super().__init__(n_var = self.N_VAR, n_obj = 2, xl = 0.0, xu = 1.0)
        self.n_dim = self.N_VAR
        self.lo, self.hi = -5.0, 10.0   # the function's own domain, applied in _objectives


    def _objectives(self, x):
        x = float(to_domain(take_first(x, self.N_VAR, "Sch2"), self.lo, self.hi)[0])
        if x <= 1:   f1 = -x
        elif x < 3:  f1 = x - 2
        elif x <= 4: f1 = 4 - x
        else:        f1 = x - 4
        return np.array([f1, (x - 5) ** 2], dtype = float)


    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._objectives(x)


    def pareto_set(self, n_points = 300):
        # the set is the union of [1, 2] and [4, 5]
        half = n_points // 2
        z = np.concatenate([np.linspace(1.0, 2.0, half), np.linspace(4.0, 5.0, n_points - half)])
        X = np.zeros((n_points, self.n_dim))
        X[:, 0] = (z - self.lo) / (self.hi - self.lo)       # pre-image in the unit box
        return X


    def pareto_front(self, n_points = 300):
        return np.array([self._objectives(x) for x in self.pareto_set(n_points)])



# =============================== Fonseca–Fleming Problems ===============================
class Fon(ElementwiseProblem):
    """
    Adopted from: https://al-roomi.org/benchmarks/multi-objective/unconstrained-list/321-fonseca-fleming-s-function-fon
    """
    def __init__(self, n_dim: int = 10):
        self.n_dim = n_dim
        super().__init__(n_var = n_dim, n_obj = 2,
                         xl = np.zeros(n_dim), xu = np.ones(n_dim))
        self.lo, self.hi = -4.0, 4.0    # the function's own domain, applied in _objectives


    def _objectives(self, x):
        z = to_domain(np.asarray(x, dtype = float).ravel(), self.lo, self.hi)
        tmp = 1 / np.sqrt(self.n_dim)
        return np.array([1 - np.exp(-np.sum(np.square(z - tmp))),
                         1 - np.exp(-np.sum(np.square(z + tmp)))], dtype = float)


    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._objectives(x)


    def pareto_set(self, n_points = 300):
        # zi in [-1/sqrt(n), 1/sqrt(n)]; the set is its pre-image in the unit box
        t = 1 / np.sqrt(self.n_dim)
        z = np.linspace(-t, t, n_points)
        return np.repeat(((z - self.lo) / (self.hi - self.lo))[:, None], self.n_dim, axis = 1)


    def pareto_front(self, n_points = 300):
        return np.array([self._objectives(x) for x in self.pareto_set(n_points)])



# =============================== Self-constructed Problems ===============================

class ZDT1_Shifted(ElementwiseProblem):
    """
        ZDT1(x + C) on ZDT1's own box, 0 <= x <= 1, so the decision space is unchanged
    """
    def __init__(self, n_dim: int = 10, C: float = 0.2):
        if C < 0:
            raise ValueError("C must be non-negative")

        self.inner = ZDT1(n_dim = n_dim).inner
        super().__init__(n_var = self.inner.n_var, n_obj = self.inner.n_obj,
                        xl = np.zeros(n_dim), xu = np.ones(n_dim))
        self.n_dim = n_dim
        self.C = C


    def _evaluate(self, x, out):
        f = self.inner.evaluate(np.asarray(x + self.C, float))
        out["F"] = f


    def pareto_front(self, n_points = 300):
        raise NotImplementedError("Self-constructed problem - no analytical PF")


class ZDT2_Shifted(ElementwiseProblem):
    """
        ZDT2(x + C) on ZDT1's own box, 0 <= x <= 1, so the decision space is unchanged
    """
    def __init__(self, n_dim: int = 10, C: float = 0.2):
        if C < 0:
            raise ValueError("C must be non-negative")

        self.inner = ZDT2(n_dim = n_dim).inner
        super().__init__(n_var = self.inner.n_var, n_obj = self.inner.n_obj,
                        xl = np.zeros(n_dim), xu = np.ones(n_dim))
        self.n_dim = n_dim
        self.C = C


    def _evaluate(self, x, out):
        f = self.inner.evaluate(np.asarray(x + self.C, float))
        out["F"] = f


    def pareto_front(self, n_points = 300):
        raise NotImplementedError("Self-constructed problem - no analytical PF")


class ZDT2_Combined(ElementwiseProblem):
    """
        ZDT2(x + C) on ZDT1's own box, 0 <= x <= 1, so the decision space is unchanged
    """
    def __init__(self, n_dim: int = 10, order = None, C: float = 0.2):
        if C < 0:
            raise ValueError("C must be non-negative")
        self.order = np.arange(4) if order is None else np.asarray(order, dtype = int)  # any permutation of 0..4
        if np.any(self.order < 0) or np.any(self.order >= 4):
            raise ValueError("order entries must be in [0, 4)")
        if len(set(self.order.tolist())) != len(self.order):
            raise ValueError("order must not repeat indices")

        self.inner = ZDT2(n_dim = n_dim).inner
        super().__init__(n_var = self.inner.n_var, n_obj = len(self.order),
                        xl = np.zeros(n_dim), xu = np.ones(n_dim))
        self.n_dim = n_dim
        self.C = C


    def _evaluate(self, x, out):
        shifted = self.inner.evaluate(np.asarray(x + self.C, float))
        origin = self.inner.evaluate(np.asarray(x, float))
        out["F"] = np.concat([origin, shifted])[self.order]


    def pareto_front(self, n_points = 300):
        raise NotImplementedError("Self-constructed problem - no analytical PF")