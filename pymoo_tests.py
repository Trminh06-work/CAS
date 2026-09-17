import numpy as np
from pymoo.problems import get_problem
from pymoo.core.problem import ElementwiseProblem



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


class ZDT6(Problem):
    def __init__(self, n_dim = 30):
        super().__init__("zdt6", n_var = n_dim)

    def pareto_set(self, n_points=300):
        # every variable except x0 at 0 so that g = 1; every x0 then lies on f2 = 1 - f1**2
        X = np.zeros((n_points, self.n_dim))
        X[:, 0] = np.linspace(0.0, 1.0, n_points)
        return X



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
