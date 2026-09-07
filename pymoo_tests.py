import numpy as np
from pymoo.problems import get_problem
from pymoo.core.problem import ElementwiseProblem


def _pareto_filter_2d(F):
    """Non-dominated rows of a 2-objective set (minimisation), via a sweep on f1."""
    order = np.argsort(F[:, 0], kind = "stable")
    keep, best_f2 = [], np.inf
    for i in order:
        if F[i, 1] < best_f2:
            keep.append(i)
            best_f2 = F[i, 1]
    return F[np.sort(keep)]



class Problem(ElementwiseProblem):
    K = None            # None -> plain minimisation; an array -> maximise f by minimising K - f
    normalise = False   # divide the stored objectives by K so they live in [0, 1]

    def __init__(self, name: str = None, **kwargs):
        if name is None:
            raise ValueError("The problem's name is not specified")

        self.inner = get_problem(name, **kwargs)
        super().__init__(n_var = self.inner.n_var, n_obj = self.inner.n_obj,
                         xl = self.inner.xl, xu = self.inner.xu)
        self.n_dim = self.n_var


    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.stored(self.inner.evaluate(np.asarray(x, dtype = float)))


    def stored(self, f):
        """
            True objective values -> the units this class reports and every plot works in.

            K - f, so that minimising it maximises f; divided by K when normalise is set, which
            is a positive per-objective scaling - it leaves the dominance relation untouched but
            keeps the numbers, and the weights nnls has to fit, near 1.
        """
        if self.K is None:
            return f
        return (self.K - f) / self.K if self.normalise else self.K - f


    def true(self, F):
        """
            Undo the minimisation flip, keeping whatever scale this problem reports in: real
            objective values f, or f / K when normalise is set so the axes stay in [0, 1].
            Identity for a problem that is plain minimisation.

            This is what the plots show. For real units out of a normalised problem, the full
            inverse is f = K * (1 - stored).
        """
        if self.K is None:
            return F
        return 1 - F if self.normalise else self.K - F


    def pareto_set(self, n_points = 300):
        raise NotImplementedError("The true Pareto set of this problem is not known")



class ZDT1(Problem):
    """
        Posed as MAXIMISATION of f = (f1, f2).
    """
    K = np.array([1.0, 10.0])           # f1 <= 1 and f2 <= g_max = 10, so K - f >= 0

    def __init__(self, n_dim = 30):
        super().__init__("zdt1", n_var = n_dim)


    def pareto_front(self, n_points = 300):
        # maximising puts the front at g = 10, where f2 = 10 - sqrt(10 * f1)
        f1 = np.linspace(0.0, 1.0, n_points)
        return self.K - np.column_stack([f1, 10 - np.sqrt(10 * f1)])


    def pareto_set(self, n_points = 300):
        # x0 sweeps f1, every other coordinate sits at its upper bound so that g = 10
        X = np.ones((n_points, self.n_dim))
        X[:, 0] = np.linspace(0.0, 1.0, n_points)
        return X



class ZDT2(Problem):
    """
        Posed as MAXIMISATION of f = (f1, f2).
    """
    K = np.array([1.0, 10.0])

    def __init__(self, n_dim = 30):
        super().__init__("zdt2", n_var = n_dim)


    def pareto_front(self, n_points = 300):
        # maximising puts the front at g = 10, where f2 = 10 - f1**2 / 10
        f1 = np.linspace(0.0, 1.0, n_points)
        return self.K - np.column_stack([f1, 10 - f1**2 / 10])


    def pareto_set(self, n_points = 300):
        X = np.ones((n_points, self.n_dim))
        X[:, 0] = np.linspace(0.0, 1.0, n_points)
        return X



class ZDT3(Problem):
    """
        Posed as MAXIMISATION of f = (f1, f2).
    """
    K = np.array([1.0, 10.0])

    def __init__(self, n_dim = 30):
        super().__init__("zdt3", n_var = n_dim)


    def pareto_front(self, n_points = 300):
        f1 = np.linspace(0.0, 1.0, max(20 * n_points, 4000))
        f2 = 10 - np.sqrt(10 * f1) - f1 * np.sin(10 * np.pi * f1)
        return _pareto_filter_2d(self.K - np.column_stack([f1, f2]))


    def pareto_set(self, n_points = 300):
        # x0 runs over the disconnected segments, every other coordinate at its upper bound
        f1 = self.K[0] - self.pareto_front(n_points)[:, 0]
        idx = np.linspace(0, len(f1) - 1, min(n_points, len(f1))).astype(int)
        X = np.ones((len(idx), self.n_dim))
        X[:, 0] = f1[idx]
        return X



class DTLZ1(Problem):
    """
        Posed as MAXIMISATION of f, reported normalised: every objective lands in [0, 1].
        Undo it with f = K * (1 - stored), not K - stored.
    """
    normalise = True

    def __init__(self, n_dim = 7, n_obj = 3):
        super().__init__("dtlz1", n_var = n_dim, n_obj = n_obj)

        t = np.linspace(-0.5, 0.5, 200001)
        term = t**2 - np.cos(20 * np.pi * t)
        k, i = self.n_dim - self.n_obj + 1, term.argmax()
        self.x_max, self.g_max = 0.5 + t[i], 100 * k * (1 + term[i])
        self.K = np.full(self.n_obj, 0.5 * (1 + self.g_max))    # f_i <= 0.5 * (1 + g_max)

    def pareto_front(self, n_points = 300):
        # pymoo's front is the simplex sum(f) = 0.5 at g = 0; at g = g_max it scales by 1 + g_max
        return self.stored((1 + self.g_max) * self.inner.pareto_front())
        # return -self.inner.pareto_front()    # PF of minimisation


    def pareto_set(self, n_points = 300):
        # the first n_obj - 1 coordinates are free; the tail sits where each g term is maximal
        rng = np.random.default_rng(0)
        X = np.full((n_points, self.n_dim), self.x_max)
        X[:, :self.n_obj - 1] = rng.random((n_points, self.n_obj - 1))
        return X



class DTLZ2(Problem):
    """
        Posed as MAXIMISATION of f, reported normalised: every objective lands in [0, 1].
        Undo it with f = K * (1 - stored), not K - stored.
    """
    normalise = True

    def __init__(self, n_dim = 12, n_obj = 3):
        super().__init__("dtlz2", n_var = n_dim, n_obj = n_obj)

        self.R = 1 + 0.25 * (self.n_dim - self.n_obj + 1)       # every (x - 0.5)^2 <= 0.25
        self.K = np.full(self.n_obj, self.R)                    # f_i <= ||f|| = R


    def pareto_front(self, n_points = 300):
        # pymoo's front is the unit sphere; at g = g_max it has radius R
        return self.stored(self.R * self.inner.pareto_front())
        # return -self.inner.pareto_front()    # PF of minimisation


    def pareto_set(self, n_points = 300):
        # the first n_obj - 1 coordinates are free; the tail sits at its bound, where g is maximal
        rng = np.random.default_rng(0)
        X = np.ones((n_points, self.n_dim))
        X[:, :self.n_obj - 1] = rng.random((n_points, self.n_obj - 1))
        return X
