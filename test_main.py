from typing import List, Tuple, Dict, Type, Any
import numpy as np
import itertools, pandas as pd
from models import LinearModel, ChoquetModel, LipschitzModel, BaseModel
from tests import ZDT1, ZDT2, ZDT3, DTLZ1, DTLZ2
from tests import Problem

import matplotlib.pyplot as plt
from utils import evaluate, visualise_weights, live_trace




def many_models(models: List[Type[BaseModel]]) -> List[BaseModel]:
    result = []
    for model in models:
        print(f"Runnning {type(model).__name__}")
        # the trace is written while it runs: the parameters of each iteration, then the
        # evaluations it made and the value of the scalarising function there
        trace = f"trace_{type(model).__name__}_{type(model.problem).__name__}.log"
        model.solve(live_trace(trace))
        print("trace written to", trace)
        result.append(model)
    return result



# ====================================================== TESTING ======================================================

# model instantiation
"""
    If you want to test a single model -> put one single model

    For multiple models -> add more to it
"""
models = [
    LinearModel(problem = ZDT2(n_dim = 20), max_iter = 50, random_state = 42),
    ChoquetModel(problem = ZDT2(n_dim = 20), max_iter = 50, random_state = 42, method = 2),
    LipschitzModel(problem = ZDT2(n_dim = 20), max_iter = 50, random_state = 42,
                    type = "global_mon", num_k = 2)      # mon defaults to one per objective
]

# Run and get the result
m_res = many_models(models)
print()


# Obtain the results - be careful with the Hypervolume
# If the `models` you specified above include other test problems, the HV will be incomparable
# Since the HV calculation extracts the PF found by all models to get the reference nadir point
print(evaluate(m_res), end = "\n\n")



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
