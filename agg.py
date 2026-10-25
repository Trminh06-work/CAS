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
from utils import visualise
from pymoo.core.problem import ElementwiseProblem



# ========================================================================================
# ====================================== Base Class ======================================
# ========================================================================================
class Aggregator:
    def _compatible_check(x: np.ndarray):
        pass

    @staticmethod
    def add(x: np.ndarray):
        return np.sum(x)



class AggFunc(ABC):
    """
        Aggregation functions require the following axioms:
            1. Monotonicity (non-decreasing) : x, y in [0, 1]^n : x < y => f(x) < f(y) - element-wise comparision
            2. Preservation of bounds        : f(0, ..., 0) = 0 and f(1, ..., 1) = 1
                                                  n-times              n-times
    """
    def __init__(self, problems: List[ElementwiseProblem]):
        self.problems = problems
        self.neutral = None     # short as, e
        self.absorb = None      # short as, a
        self.n_dim = len(problems)


    @abstractmethod
    def fun(self, x: np.ndarray):
        raise NotImplementedError("Aggregation function is not specified")



class ConjunctiveFunc(AggFunc):
    """
        x in [0, 1]^n: f(x) <= min(x)
    """
    def __init__(self, problems):
        super().__init__(problems)




class DisjunctiveFunc(AggFunc):
    """
        x in [0, 1]^n: f(x) >= max(x)
    """
    def __init__(self, problems):
        super().__init__(problems)



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
        if (len(x)) != self.n_dim:
            raise ValueError(f"The shape of x ({len(x)}) does not match with predefined dimensions ({self.n_dim})")
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


    def fun(self, x: np.ndarray):
        if (len(x)) != self.n_dim:
            raise ValueError(f"The shape of x ({len(x)}) does not match with predefined dimensions ({self.n_dim})")
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
        self.is_archimedean = True
        self.lambda_param = lambda_param


    def _func(self, x, y):
        if self.lambda_param == -np.inf:
            return min(x, y)
        elif self.lambda_param == 0:
            return Product()._func(x, y)
        elif self.lambda_param == np.inf:
            return Drastic()._func(x, y)
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
            return (1 - self.lambda_param * t) ** (1 / self.lambda_param)



class Frank(TNorm):
    """
        Family of Frank t-norms, Archimedean, (p. 154)
    """
    def __init__(self, problems, lambda_param: float):
        super().__init__(problems)
        self.is_archimedean = True
        self.lambda_param = lambda_param


    def _func(self, x, y):
        if self.lambda_param == 0:
            return min(x, y)
        elif self.lambda_param == 1:
            return Product()._func(x, y)
        elif self.lambda_param == np.inf:
            return Lukasiewicz()._func(x, y)
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
            a = np.log(self.lambda_param - 1)
            b = np.log(self.lambda_param ** t - 1)
            return a - b


    def inv_gen(self, t):
        if self.lambda_param == 1:
            return np.exp(-t)
        elif self.lambda_param == np.inf:
            return 1 - t
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
        self.w_vec = np.ones(self.n_dim) if w_vec is None else w_vec
        if gen is None or inv_gen is None:
            raise ValueError("A generating function and its inverse are not specified")
        self.gen = gen
        self.inv_gen = gen


    def fun(self, x: np.ndarray):
        if (len(x)) != self.n_dim:
            raise ValueError(f"The shape of x ({len(x)}) does not match with predefined dimensions ({self.n_dim})")
        s = 0.0
        for i in range(len(x)):
            s += self.gen(self.w_vec[i] * x[i])
        return self.inv_gen(s)



# ================================================== Uninorms ==================================================
class Uninorm(MixedFunc):
    """
        Implementation of Uninorms: associative, symmetric and has neutral element, e, in (0, 1)^n. Uninorms exhibit:
            1. T-norm   (conjunctive) in [0, e]^n
            2. T-conorm (disjunctive) in [e, 1]^n
            3. Averaging function elsewhere
    """
    def __init__(self, problems, is_representable: bool = False):
        super().__init__(problems)
        self.is_representable = is_representable


    def _func(self, x: float, y: float):
        raise NotImplementedError("Function is not specified")


    def gen(self, t: float):
        raise NotImplementedError("Additive generator is not specified")


    def inv_gen(self, t: float):
        raise NotImplementedError("Pseudo-inverse of the additive generator is not specified")


    def fun(self, x: np.ndarray):
        """
            Compute in (for multivariate evaluations):
                1. Iterative manner, if the function admits
                2. Utilisation of univariate, if representable
        """
        if (len(x)) != self.n_dim:
            raise ValueError(f"The shape of x ({len(x)}) does not match with predefined dimensions ({self.n_dim})")
        if self.is_representable:
            s = 0.0
            for i in range(len(x)):
                s += self.gen(x[i])
            return self.inv_gen(s)
        else:
            return self._func(x)




class ThreePi(Uninorm):
    """
        Three-Pi function, representable uninorm, (p. 209)
    """
    def __init__(self, problems, is_representable):
        super().__init__(problems, is_representable)
        self.is_representable = True


    def _func(self, x: np.ndarray):
        prod = np.prod(x)
        neg_prod = np.prod(1 - x)
        return prod / (prod + neg_prod)


    def gen(self, t: float):
        return np.log(t) - np.log(1 - t)


    def inv_gen(self, t: float):
        return np.exp(t) / (np.exp(t) + 1)






# ====================================================== TESTING ======================================================
if __name__ == "__main__":
    pass