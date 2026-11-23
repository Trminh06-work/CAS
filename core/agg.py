"""
    Aggregation Functions: A Guide for Practitioners (https://link.springer.com/book/10.1007/978-3-540-73721-6)

    This script implements certain classes of aggregation functions. We restrict to aggregation function as the following functional mapping:
        f: [0, 1]^n -> [0, 1]
    One can generalise to any sub-intervals, [a, b]^n in R^n, by:
        x in [a, b]^n -> [0, 1]^n : (x - a) / (b - a)
        x in [0, 1]^n -> [a, b]^n : a + (b - a) * x
"""
from abc import ABC, abstractmethod
from typing import Callable, List
import numpy as np
from pymoo.core.problem import ElementwiseProblem



# ========================================================================================
# ====================================== Base Class ======================================
# ========================================================================================
class Aggregator:
    @staticmethod
    def _compatible_check(x: np.ndarray):
        pass

    @staticmethod
    def add(x: np.ndarray):
        return np.sum(x)



class AggFunc(ABC):
    """
        Aggregation functions require the following axioms:
            1. Monotonicity (non-decreasing) : x, y in [0, 1]^n : x <= y => f(x) <= f(y) - element-wise comparision
            2. Preservation of bounds        : f(0, ..., 0) = 0 and f(1, ..., 1) = 1
                                                  n-times              n-times
    """
    def __init__(self, problems: List[ElementwiseProblem]):
        self.problems = problems
        self.neutral = None     # short as, e
        self.absorb = None      # short as, a
        self.n_args = len(problems)     # arity; set_arity() replaces it once the problems' objectives are counted


    def set_arity(self, n_args: int):
        """
            The number of values aggregated: every objective of every problem
        """
        self.n_args = n_args


    @abstractmethod
    def fun(self, x: np.ndarray):
        raise NotImplementedError("Aggregation function is not specified")



class ConjunctiveFunc(AggFunc):
    """
        x in [0, 1]^n: f(x) <= min(x)
    """
    def __init__(self, problems):
        super().__init__(problems)
        self.absorb = 0         # any conjunctive function, Example 1.31, (p. 13)




class DisjunctiveFunc(AggFunc):
    """
        x in [0, 1]^n: f(x) >= max(x)
    """
    def __init__(self, problems):
        super().__init__(problems)
        self.absorb = 1         # any disjunctive function, Example 1.31, (p. 13)



class AveragingFunc(AggFunc):
    """
        x in [0, 1]^n: min(x) <= f(x) <= max(x)
    """
    def __init__(self, problems):
        super().__init__(problems)



class MixedFunc(AggFunc):
    """
        A mixed aggregation function -> neither conjunctive nor disjunctive or averaging
    """
    def __init__(self, problems):
        super().__init__(problems)



# ====================================== Triangular norms/conorms ======================================
class TNorm(ConjunctiveFunc):
    """
        Implementation of Triangular norm (t-norm): associative, symmetric, e = 1 and a = 0
    """
    def __init__(self, problems):
        super().__init__(problems)
        self.neutral = 1
        self.absorb = 0
        self.is_archimedean = False


    def _func(self, x: float, y: float):
        raise NotImplementedError("Function is not specified")


    def gen(self, t: float):
        raise NotImplementedError("Additive generator is not specified")


    def inv_gen(self, t: float):
        raise NotImplementedError("Pseudo-inverse of the additive generator is not specified")


    def fun(self, x: np.ndarray):
        """
            Compute in (for multivariate evaluations):
                1. Iterative manner, due to associativity (can be recursively)
                2. Utilisation of univariate, due to Archimedean continuity

            Note: Additive generators are used, owing to numerical stability, instead of multiplicative generators
        """
        if (len(x)) != self.n_args:
            raise ValueError(f"The shape of x ({len(x)}) does not match with predefined dimensions ({self.n_args})")
        if self.is_archimedean:
            s = 0.0
            for i in range(len(x)):
                s += self.gen(x[i])
            return self.inv_gen(s)
        else:
            s = x[0]
            for i in range(1, len(x)):
                s = self._func(s, x[i])
            return s


    def dual(self, N: Callable | None = None):
        """
            The N-dual of this t-norm, i.e. t-conorm
        """
        return TConorm(self, N)



class TConorm(DisjunctiveFunc):
    """
        Implementation of Triangular conorm (t-conorm): associative, symmetric, e = 0 and a = 1

        The t-conorm is the N-dual of the corresponding t-norm, where N is a strong negation (default: N(t) = 1 - t), i.e.:
            t-conorm(x) = N(t-norm(N(x_1), ..., N(x_n)))
    """
    def __init__(self, t_norm: TNorm, N: Callable | None = None):
        super().__init__(t_norm.problems)
        self.neutral = 0
        self.absorb = 1
        self.t_norm = t_norm
        self.N = N if N is not None else (lambda t: 1 - t)


    def set_arity(self, n_args: int):
        super().set_arity(n_args)
        self.t_norm.set_arity(n_args)       # evaluated through its t-norm


    def fun(self, x: np.ndarray):
        if (len(x)) != self.n_args:
            raise ValueError(f"The shape of x ({len(x)}) does not match with predefined dimensions ({self.n_args})")
        return self.N(self.t_norm.fun(self.N(x)))



class Product(TNorm):
    """
        Product t-norm T_P, (p. 127). Product(problems).dual(), is the probabilistic sum S_P
    """
    def _func(self, x: float, y: float):
        return x * y



class Drastic(TNorm):
    """
        Drastic product (t-norm) T_D, (p. 127). Drastic(problems).dual(), is the drastic sum S_D
    """
    def _func(self, x: float, y: float):
        if max(x, y) == 1:
            return min(x, y)
        return 0.0



class Lukasiewicz(TNorm):
    """
        Lukasiewicz t-norm T_L, (p. 127), Archimedean with additive generator g(t) = 1 - t.
        Lukasiewicz(problems).dual(), is the Lukasiewicz t-conorm S_L
    """
    def __init__(self, problems):
        super().__init__(problems)
        self.is_archimedean = True


    def _func(self, x: float, y: float):
        return max(0.0, x + y - 1)


    def gen(self, t: float):
        return 1 - t


    def inv_gen(self, t: float):
        # pseudo-inverse, g^(-1)(min(t, g(0)))
        return 1 - min(t, 1)



class SchweizerSklar(TNorm):
    """
        Family of Schweizer-Sklar t-norms, Archimedean (p. 150)
    """
    def __init__(self, problems, lambda_param: float):
        super().__init__(problems)
        self.is_archimedean = np.isfinite(lambda_param)     # min and T_D (limiting cases)
        self.lambda_param = lambda_param


    def _func(self, x, y):
        if self.lambda_param == -np.inf:
            return min(x, y)
        elif self.lambda_param == 0:
            return Product(self.problems)._func(x, y)
        elif self.lambda_param == np.inf:
            return Drastic(self.problems)._func(x, y)
        else:
            x = x ** self.lambda_param
            y = y ** self.lambda_param
            return max((x + y - 1), 0) ** (1 / self.lambda_param)


    def gen(self, t: float):
        if self.lambda_param == 0:
            return - np.log(t)
        else:
            return (1 - t ** self.lambda_param) / self.lambda_param


    def inv_gen(self, t: float):
        if self.lambda_param == 0:
            return np.exp(-t)
        else:
            # pseudo-inverse, g^(-1)(min(t, g(0))), as g(0) = 1 / lambda for lambda > 0
            return max(1 - self.lambda_param * t, 0) ** (1 / self.lambda_param)



class Frank(TNorm):
    """
        Family of Frank t-norms, Archimedean, (p. 154)
    """
    def __init__(self, problems, lambda_param: float):
        super().__init__(problems)
        self.is_archimedean = (lambda_param != 0)     # min (limiting case)
        self.lambda_param = lambda_param


    def _func(self, x, y):
        if self.lambda_param == 0:
            return min(x, y)
        elif self.lambda_param == 1:
            return Product(self.problems)._func(x, y)
        elif self.lambda_param == np.inf:
            return Lukasiewicz(self.problems)._func(x, y)
        else:
            x = self.lambda_param ** x - 1
            y = self.lambda_param ** y - 1
            a = 1 + x * y / (self.lambda_param - 1)
            return np.log(a) / np.log(self.lambda_param)


    def gen(self, t):
        if self.lambda_param == 1:
            return -np.log(t)
        elif self.lambda_param == np.inf:
            return 1 - t
        else:
            # log of the (positive) ratio, split as |.| since both terms are negative for lambda < 1
            a = np.log(abs(self.lambda_param - 1))
            b = np.log(abs(self.lambda_param ** t - 1))
            return a - b


    def inv_gen(self, t):
        if self.lambda_param == 1:
            return np.exp(-t)
        elif self.lambda_param == np.inf:
            # pseudo-inverse, g^(-1)(min(t, g(0)))
            return 1 - min(t, 1)
        else:
            # log_l ( [l - 1] / e^t + 1 ) => log_l k = ln k / ln l
            k = (self.lambda_param - 1) / np.exp(t) + 1
            return np.log(k) / np.log(self.lambda_param)




# ====================================== Quasi-arithmetic means ======================================
class QAM(AveragingFunc):
    """
        Implementation of (Weighted) Quasi-arithmetic means
    """
    def __init__(self, problems, gen: Callable | None = None, inv_gen: Callable | None = None, w_vec: np.ndarray | None = None):
        super().__init__(problems)
        self.w_given = w_vec is not None
        self.w_vec = np.ones(self.n_args) / self.n_args if w_vec is None else np.asarray(w_vec, dtype = float)
        if gen is None or inv_gen is None:
            raise ValueError("A generating function and its inverse are not specified")
        self.gen = gen
        self.inv_gen = inv_gen

        # no neutral element; absorbing element a = 0 (resp. a = 1) only if all w_i > 0 and g(0) = +-inf (resp. g(1) = +-inf), (p. 49)
        # if both are infinite, a depends on the convention for -inf + inf (Note 2.19), hence left unset
        with np.errstate(divide = "ignore"):
            g0, g1 = np.isinf(gen(np.float64(0))), np.isinf(gen(np.float64(1)))
        if np.all(self.w_vec > 0) and g0 != g1:
            self.absorb = 0 if g0 else 1


    def set_arity(self, n_args: int):
        super().set_arity(n_args)
        if not self.w_given:
            self.w_vec = np.ones(n_args) / n_args       # equal weights, over every objective aggregated
        elif len(self.w_vec) != n_args:
            raise ValueError(f"w_vec has {len(self.w_vec)} weights, but {n_args} objectives are aggregated")


    def fun(self, x: np.ndarray):
        if (len(x)) != self.n_args:
            raise ValueError(f"The shape of x ({len(x)}) does not match with predefined dimensions ({self.n_args})")
        s = 0.0
        for i in range(len(x)):
            s += self.w_vec[i] * self.gen(x[i])
        return self.inv_gen(s)



# ================================================== Uninorms ==================================================
class Uninorm(MixedFunc):
    """
        Implementation of Uninorms: associative, symmetric and has neutral element, e, in (0, 1). Uninorms exhibit:
            1. T-norm   (conjunctive) in [0, e]^n
            2. T-conorm (disjunctive) in [e, 1]^n
            3. Averaging function elsewhere
    """
    def __init__(self, problems, is_representable: bool = False):
        super().__init__(problems)
        self.is_representable = is_representable


    def _func(self, x: np.ndarray):
        raise NotImplementedError("Function is not specified")


    def gen(self, t: float):
        raise NotImplementedError("Additive generator is not specified")


    def inv_gen(self, t: float):
        raise NotImplementedError("Pseudo-inverse of the additive generator is not specified")


    def fun(self, x: np.ndarray):
        """
            Compute in (for multivariate evaluations):
                1. Directly via _func(x), if the function admits
                2. Utilisation of univariate, if representable
        """
        if (len(x)) != self.n_args:
            raise ValueError(f"The shape of x ({len(x)}) does not match with predefined dimensions ({self.n_args})")
        if self.absorb is not None and np.any(x == self.absorb):
            return self.absorb      # absorbing element, resolves -inf + inf of representable uninorms
        if self.is_representable:
            s = 0.0
            # resolve egde cases (p. 204), i.e. 0 and 1 simultaneously appearring in x
            if np.isin([0, 1], x).all():   # Note 4.13 (p. 205) for n-ary representable uninorms
                if self.absorb == 0:    # conjunctive uninorm (p. 201)
                    return 0
                elif self.absorb == 1:  # disjunctive uninorm (p. 202)
                    return 1

            # do not fall into edge cases
            for i in range(len(x)):
                s += self.gen(x[i])
            return self.inv_gen(s)
        else:
            return self._func(x)




class ConjunctiveThreePi(Uninorm):
    """
        Three-Pi function, representable uninorm, (p. 209)
    """
    def __init__(self, problems):
        super().__init__(problems, is_representable=True)
        self.neutral = 0.5      # Example 1.81, (p. 28) and Example 4.19 (p. 209)
        self.absorb = 0         # convention 0/0 = 0, i.e. conjunctive uninorm, (pp. 201, 209)


    def _func(self, x: np.ndarray):
        prod = np.prod(x)
        neg_prod = np.prod(1 - x)
        return prod / (prod + neg_prod)


    def gen(self, t: float):
        return np.log(t) - np.log(1 - t)


    def inv_gen(self, t: float):
        # more numerically stable, t -> inf => 1, t -> -inf => 0
        return 1 / (1 + np.exp(-t))     # = e^t / (e^t + 1), divides num and denom by e^x




# ================================================== Visualisation ==================================================
def visualise_agg(funcs, title: str | None = None, n_grid: int = 101, show: bool = True):
    """
        Bivariate aggregation functions, f: [0, 1]^2 -> [0, 1], one column each: the surface on top and its level sets,
        f(x) = 0, 0.1, ..., 1, below.

        funcs       {name: AggFunc}, each of arity 2, e.g. {r"$T_P$": Product([None, None])}, or one AggFunc
        title       of the figure
        n_grid      grid points per axis of [0, 1]^2
        show        render the figure before returning
    """
    import matplotlib.pyplot as plt

    if isinstance(funcs, AggFunc):
        funcs = {type(funcs).__name__: funcs}

    t = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(t, t)
    levels = np.linspace(0, 1, 11)
    fig = plt.figure(figsize = (4 * len(funcs), 8))
    for k, (name, f) in enumerate(funcs.items()):
        # log(0), 1 / 0, ... are expected at the bounds of [0, 1]^2
        with np.errstate(divide = "ignore", invalid = "ignore", over = "ignore"):
            Z = np.array([[f.fun(np.array([x, y])) for x, y in zip(xr, yr)] for xr, yr in zip(X, Y)], dtype = float)

        # surface, viewed from the (0, 0) corner
        ax = fig.add_subplot(2, len(funcs), k + 1, projection = "3d")
        ax.grid(False)
        ax.plot_wireframe(X, Y, Z, color = "grey", linewidth = 0.6)
        ax.set(title = f"{name}\ne = {f.neutral}, a = {f.absorb}", xlabel = "$x_1$", ylabel = "$x_2$", zlim = (0, 1))
        ax.view_init(elev = 25, azim = -135)

        # level sets, f(x) = 0, 0.1, ..., 1
        ax = fig.add_subplot(2, len(funcs), len(funcs) + k + 1)
        cs = ax.contour(X, Y, Z, levels = levels, colors = "grey", linewidths = 0.6)

        d = np.interp(levels, np.diag(Z), t)
        gap = np.minimum(np.diff(d)[:-1], np.diff(d)[1:])       # to the closer neighbour, for levels 0.1, ..., 0.9
        ax.clabel(cs, fontsize = 7, manual = [(p, p) for p in d[1:-1][gap >= 0.045]])
        ax.set(xlabel = "$x_1$", ylabel = "$x_2$", aspect = "equal")
    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if show:
        plt.show()
    return fig




# ====================================================== TESTING ======================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    problems = [None, None]
    groups = {
        # "Basic t-norms": {
        #     r"$T_{min}$": SchweizerSklar(problems, -np.inf),        # limiting case, lambda = -inf
        #     r"$T_P$": Product(problems),
        #     r"$T_L$": Lukasiewicz(problems),
        #     r"$T_D$": Drastic(problems),
        # },
        # "Basic t-conorms, N-duals with N(t) = 1 - t": {
        #     r"$S_{max}$": SchweizerSklar(problems, -np.inf).dual(),
        #     r"$S_P$": Product(problems).dual(),
        #     r"$S_L$": Lukasiewicz(problems).dual(),
        #     r"$S_D$": Drastic(problems).dual(),
        # },
        "Schweizer-Sklar t-norms (limiting cases)": {rf"$\lambda = {l}$": SchweizerSklar(problems, l) for l in (-np.inf, 0, np.inf)},
        "Schweizer-Sklar t-norms": {rf"$\lambda = {l}$": SchweizerSklar(problems, l) for l in (-100, -1, 1, 100)},
        "Frank t-norms (limiting cases)": {rf"$\lambda = {l}$": Frank(problems, l) for l in (0, 1, np.inf)},
        "Frank t-norms": {rf"$\lambda = {l}$": Frank(problems, l) for l in (1e-9, 1e-4, 1e4, 1e9)},
        # "Quasi-arithmetic means": {
        #     "arithmetic": QAM(problems, gen = lambda t: t, inv_gen = lambda t: t),
        #     "geometric, w = (0.3, 0.7)": QAM(problems, gen = np.log, inv_gen = np.exp, w_vec = np.array([0.3, 0.7])),
        #     "harmonic": QAM(problems, gen = lambda t: 1 / t, inv_gen = lambda t: 1 / t),
        #     "quadratic": QAM(problems, gen = lambda t: t ** 2, inv_gen = np.sqrt),
        # },
        # "Uninorms": {
        #     r"3-$\Pi$": ThreePi(problems),
        # },
    }

    for group, funcs in groups.items():
        visualise_agg(funcs, group, show = False)
    plt.show()      # every figure at once