from typing import List, Tuple, Dict, Type, Any
import numpy as np
import itertools, pandas as pd
from models import LinearModel, ChoquetModel, LipschitzModel, BaseModel
from pymoo_tests import ZDT1, ZDT2, ZDT3, DTLZ1, DTLZ2
from pymoo_tests import Problem

import matplotlib.pyplot as plt
from utils import evaluate, visualise_weights




def many_models(models: List[Type[BaseModel]]) -> List[BaseModel]:
    result = []
    for model in models:
        print(f"Runnning {type(model).__name__}")
        model.solve()
        result.append(model)
    return result



def trace_log(model: BaseModel, max_rows: int = 100) -> str:
    """
        Write a readable trace of one solved model: the parameters of each outer iteration, then the
        evaluations it made and the value of the scalarising function there.

        as_of() puts back the fit of that iteration - the scalarised value is not stored during the
        run, so it has to be recomputed against the state the solver actually saw.
    """
    path = f"trace_{type(model).__name__}_{type(model.problem).__name__}.log"
    obj, dec = np.atleast_2d(model.obj_hist), np.atleast_2d(model.dec_hist)
    W = np.atleast_2d(model.w_hist)
    iters = np.asarray(model.iter_hist)
    names = [f"f{i + 1}" for i in range(model.PF_dim)]

    with open(path, "w") as fh:
        fh.write(f"{type(model).__name__} on {type(model.problem).__name__} - "
                 f"seed {model.random_state}, {model.num_func_eval} evaluations\n")

        for it in np.unique(iters):
            rows = (iters == it)
            with model.as_of(it):
                scal = [float(model.scalarise(f, w)) for f, w in zip(obj[rows], W[rows])]

            table = pd.DataFrame(obj[rows], columns = names)
            table["scalarised"] = scal
            table.insert(0, "x", [np.array2string(x, precision = 4, separator = ",",
                                                  max_line_width = np.inf) for x in dec[rows]])

            fh.write(f"\n=== iteration {int(it)}: {int(rows.sum())} evaluations, w = {W[rows][0]}\n")

            prev = obj[iters < it]
            if len(prev):
                pf_it = prev[model.non_dominated(prev)]
                u = model.state_hist[int(it)]["worst"] - pf_it
                fit = pd.DataFrame(np.hstack([pf_it, u]),
                                   columns = names + [f"u{i + 1}" for i in range(model.PF_dim)])
                fh.write(f"  learn_weight input: {len(pf_it)} non-dominated points\n")
                fh.write(fit.head(max_rows).to_string(index = False) + "\n")

            for key, value in model.state_hist.get(int(it), {}).items():
                if key == "slip" and value is not None:     # the object is not readable on its own
                    value = f"{value.type}, {value.npts} points,\n  LC = {value.LipsConst:.6g}"
                fh.write(f"  {key:10s}: {value}\n")
            fh.write(table.head(max_rows).to_string(index = False) + "\n")
    return path



# ====================================================== TESTING ======================================================

# model instantiation
"""
    If you want to test a single model -> put one single model

    For multiple models -> add more to it
"""
models = [
    LinearModel(problem = ZDT1(n_dim = 20), max_iter = 10, random_state = 42),
    ChoquetModel(problem = ZDT1(n_dim = 20), max_iter = 10, random_state = 42, method = 2),
    LipschitzModel(problem = ZDT1(n_dim = 20), max_iter = 10, random_state = 42,
                    type = "global_mon", num_k = 2, mon = [1, 1, 1])
]

# Run and get the result
m_res = many_models(models)
print()


# Obtain the results - be careful with the Hypervolume
# If the `models` you specified above include other test problems, the HV will be incomparable
# Since the HV calculation extracts the PF found by all models to get the reference nadir point
print(evaluate(m_res), end = "\n\n")



# Trace file per model: the parameters of each iteration, then the evaluations it made and the
# value of the scalarising function there. max_rows keeps it readable - drop it for the full history
for model in m_res:
    print("trace written to", trace_log(model), end = "\n")
print()


# Sanity check
print("Decision space --- within [lb, ub]?")
def check(m: BaseModel, lb: float, ub: float) -> bool:
    return np.all((m.dec_hist >= lb) & (m.dec_hist <= ub))

res = []

for model in m_res:
    # you can change the bound - assume box constraints
    res.append({
        "model": type(model).__name__,
        "ok": check(model, 0, 1)
    })

print(
    pd.DataFrame(res).set_index("model")
)
