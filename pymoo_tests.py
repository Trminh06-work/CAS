import numpy as np
from pymoo.problems import get_problem
from pymoo.core.problem import ElementwiseProblem


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


# =============================== Base Class ===============================
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