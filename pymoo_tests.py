import numpy as np
from pymoo.problems import get_problem
from pymoo.core.problem import ElementwiseProblem



class Problem(ElementwiseProblem):
    """
        A pymoo benchmark, optionally reported through an ORDER-PRESERVING affine rescaling.
        All of these are MINIMISATION problems and stay minimisation problems.

        The stored objectives are

            z = (f - lo) / (hi - lo),       hi > lo componentwise

        which is z = a * f + b with a > 0. Because a is positive and diagonal, z_i <= z'_i
        holds exactly when f_i <= f'_i, so the dominance relation, the Pareto set and the
        Pareto front are the same objects as in raw units - only the axis labels move.
        true() is the exact inverse, so nothing is lost.

        Contrast with the K - f posing this replaces: that one is order-REVERSING, so it
        minimises K - f, i.e. it maximises f, and returns the maximal set - a different set
        of decision vectors, not the minimal one in different units.

        normalise selects where (lo, hi) come from, and the choice is not cosmetic:

            None       raw pymoo units, no rescaling.

            "bounds"   per-objective range over the whole feasible box, derived analytically
                       from the problem definition. Uses NO knowledge of the front, so it is
                       legitimate on a problem whose answer you do not have.

            "front"    ideal/nadir of the analytical front. This is an ORACLE: it needs the
                       answer in advance and is therefore only valid as a diagnostic, to
                       measure how much of the gap is conditioning. Never report it as a
                       method that would work on an unsolved problem.
    """
    def __init__(self, name: str = None, normalise: str = None, **kwargs):
        if name is None:
            raise ValueError("The problem's name is not specified")

        self.inner = get_problem(name, **kwargs)
        super().__init__(n_var = self.inner.n_var, n_obj = self.inner.n_obj,
                         xl = self.inner.xl, xu = self.inner.xu)
        self.n_dim = self.n_var         # pymoo calls it n_var

        self.normalise = normalise
        self.lo = self.hi = self.span = None
        if normalise is not None:
            self.lo, self.hi = self.scale_from(normalise)
            self.span = self.hi - self.lo
            if np.any(self.span <= 0):
                raise ValueError(f"degenerate objective range: lo = {self.lo}, hi = {self.hi}")


    def scale_from(self, mode: str):
        """
            (lo, hi) for the affine map - see the class docstring for what each mode assumes.
        """
        if mode == "bounds":
            return self.obj_bounds()
        if mode == "front":
            F = np.atleast_2d(self.inner.pareto_front())
            return F.min(0), F.max(0)
        raise ValueError(f"unknown normalise mode: {mode!r} - use None, 'bounds' or 'front'")


    def obj_bounds(self):
        """
            Per-objective (lo, hi) over the feasible box, from the problem definition alone.
        """
        raise NotImplementedError("The analytic objective bounds of this problem are not known")


    def stored(self, f):
        """
            Raw objective values -> the units this class reports and every model works in.
        """
        if self.lo is None:
            return f
        return (np.asarray(f, dtype = float) - self.lo) / self.span


    def true(self, Z):
        """
            Exact inverse of stored(): the units pymoo reports in. Identity when not normalised.
        """
        if self.lo is None:
            return Z
        return self.lo + np.asarray(Z, dtype = float) * self.span


    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.stored(self.inner.evaluate(np.asarray(x, dtype = float)))


    def pareto_front(self, n_points = 300):
        return self.stored(self.inner.pareto_front())


    def pareto_set(self, n_points = 300):
        raise NotImplementedError("The true Pareto set of this problem is not known")



class _ZDT(Problem):
    """
        f1 = x0 in [0, 1]; f2 = h(f1, g) with g = 1 + 9 * mean(x1..x_{n-1}) in [1, 10].
        d f2 / d g > 0 on the box, so the extremes of f2 sit at g = 1 and g = 10 and a scan
        over f1 at those two values of g gives the exact range.
    """
    def obj_bounds(self):
        f1 = np.linspace(0.0, 1.0, 200001)
        lo2, hi2 = self.h(f1, 1.0).min(), self.h(f1, 10.0).max()
        return np.array([0.0, lo2]), np.array([1.0, hi2])


    def h(self, f1, g):
        raise NotImplementedError



class ZDT1(_ZDT):
    def __init__(self, n_dim = 30, **kwargs):
        super().__init__("zdt1", n_var = n_dim, **kwargs)


    def h(self, f1, g):
        return g - np.sqrt(f1 * g)


    def pareto_set(self, n_points = 300):
        # x0 sweeps f1; every other coordinate sits at 0 so that g = 1
        X = np.zeros((n_points, self.n_dim))
        X[:, 0] = np.linspace(0.0, 1.0, n_points)
        return X



class ZDT2(_ZDT):
    def __init__(self, n_dim = 30, **kwargs):
        super().__init__("zdt2", n_var = n_dim, **kwargs)


    def h(self, f1, g):
        return g - f1**2 / g


    def pareto_set(self, n_points = 300):
        X = np.zeros((n_points, self.n_dim))
        X[:, 0] = np.linspace(0.0, 1.0, n_points)
        return X



class ZDT3(_ZDT):
    def __init__(self, n_dim = 30, **kwargs):
        super().__init__("zdt3", n_var = n_dim, **kwargs)


    def h(self, f1, g):
        return g - np.sqrt(f1 * g) - f1 * np.sin(10 * np.pi * f1)


    def pareto_set(self, n_points = 300):
        # x0 runs over the disconnected segments, every other coordinate at 0
        f1 = self.true(self.pareto_front(n_points))[:, 0]
        idx = np.linspace(0, len(f1) - 1, min(n_points, len(f1))).astype(int)
        X = np.zeros((len(idx), self.n_dim))
        X[:, 0] = f1[idx]
        return X



class DTLZ1(Problem):
    """
        f_i = 0.5 * (product of x) * (1 + g), so 0 <= f_i <= 0.5 * (1 + g_max), where
        g = 100 * [k + sum(t^2 - cos(20 pi t))] over the k = n_dim - n_obj + 1 tail
        variables, t = x - 0.5. The cos term is what makes this range enormous next to the
        front, which sits at g = 0 - see the analysis in the branch notes.
    """
    def __init__(self, n_dim = 7, n_obj = 3, **kwargs):
        super().__init__("dtlz1", n_var = n_dim, n_obj = n_obj, **kwargs)


    def obj_bounds(self):
        t = np.linspace(-0.5, 0.5, 200001)
        term = t**2 - np.cos(20 * np.pi * t)
        k = self.n_dim - self.n_obj + 1
        g_max = 100 * (k + k * term.max())
        return np.zeros(self.n_obj), np.full(self.n_obj, 0.5 * (1 + g_max))


    def pareto_set(self, n_points = 300):
        # the tail sits at 0.5 so that g = 0; the first n_obj - 1 coordinates are free
        rng = np.random.default_rng(0)
        X = np.full((n_points, self.n_dim), 0.5)
        X[:, :self.n_obj - 1] = rng.random((n_points, self.n_obj - 1))
        return X



class DTLZ2(Problem):
    """
        f_i = (1 + g) * (trig products in [0, 1]), g = sum (x - 0.5)^2 over the k tail
        variables, so 0 <= f_i <= 1 + 0.25 * k.
    """
    def __init__(self, n_dim = 12, n_obj = 3, **kwargs):
        super().__init__("dtlz2", n_var = n_dim, n_obj = n_obj, **kwargs)


    def obj_bounds(self):
        k = self.n_dim - self.n_obj + 1
        return np.zeros(self.n_obj), np.full(self.n_obj, 1 + 0.25 * k)


    def pareto_set(self, n_points = 300):
        rng = np.random.default_rng(0)
        X = np.full((n_points, self.n_dim), 0.5)
        X[:, :self.n_obj - 1] = rng.random((n_points, self.n_obj - 1))
        return X
