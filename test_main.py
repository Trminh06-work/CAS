from typing import List, Tuple, Dict, Type, Any
import sys
from pathlib import Path
import numpy as np
import itertools, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))        # models, tests and utils live one level up, in src/

from models import LinearModel, ChoquetModel, LipschitzModel, BaseModel
from tests import ZDT1, ZDT2, ZDT3, DTLZ1, DTLZ2, Kursawe, Sch2, Fon
from tests import Problem

import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from utils import evaluate, visualise_weights, live_trace




def many_models(models: List[Type[BaseModel]], folder: Path, solver: str) -> List[BaseModel]:
    result = []
    for model in models:
        print(f"Runnning {type(model).__name__}")
        # the trace is written while it runs: the parameters of each iteration, then the
        # evaluations it made and the value of the scalarising function there
        trace = folder / f"{type(model).__name__}_{solver}.log"
        model.solve(live_trace(trace))
        print("trace written to", trace)
        result.append(model)
    return result



# ====================================================== TESTING ======================================================

"""
    One cell is one (problem, solver) pair
"""
N_DIM, MAX_ITER, SEED, PLOTS = 5, 10, 42, True
SWEEP = False       # True -> reruns and False -> rescores the log files in problems/

problems = {
    "ZDT1":     lambda: ZDT1(n_dim = N_DIM),
    "ZDT2":     lambda: ZDT2(n_dim = N_DIM),
    "DTLZ1_2d": lambda: DTLZ1(n_dim = N_DIM, n_obj = 2),
    "DTLZ1_3d": lambda: DTLZ1(n_dim = N_DIM, n_obj = 3),
    "DTLZ2_2d": lambda: DTLZ2(n_dim = N_DIM, n_obj = 2),
    "DTLZ2_3d": lambda: DTLZ2(n_dim = N_DIM, n_obj = 3),
    "Kursawe":  lambda: Kursawe(n_dim = N_DIM),
    "Sch2":     lambda: Sch2(n_dim = N_DIM),
    "Fon":      lambda: Fon(n_dim = N_DIM)
}

# all solvers here are constrained optimisation
# exclude "trust-constr", it raises error for passing `eps` -> can test later
solvers = ["L-BFGS-B", "Powell", "Nelder-Mead", "TNC", "COBYLA", "COBYQA", "SLSQP"]


def build(problem: Problem, solver: str) -> List[BaseModel]:
    "the three models on one problem, all given the same solver"
    return [
        LinearModel(problem = problem, max_iter = MAX_ITER, random_state = SEED, solver = solver),
        ChoquetModel(problem = problem, max_iter = MAX_ITER, random_state = SEED, solver = solver,
                     method = 2),
        LipschitzModel(problem = problem, max_iter = MAX_ITER, random_state = SEED, solver = solver,
                       type = "global_mon", num_k = 2)
    ]


def plots(model: BaseModel, folder: Path, solver: str) -> None:
    "the objective view of one run, saved rather than shown, next to its trace"
    path = folder / f"{type(model).__name__}_{solver}.png"
    try:
        model.visualise("objective", show = False).savefig(path)
    except Exception as exc:
        print(f"  no plot: {type(exc).__name__}: {exc}")
    plt.close("all")


def in_box(model: BaseModel):
    "did every evaluation stay inside the problem's own box? None when it declares none"
    if model.problem.xl is None:
        return None
    return bool(np.all((model.dec_hist >= model.problem.xl) & (model.dec_hist <= model.problem.xu)))


def read_log(path: Path) -> np.ndarray:
    "the objective vector of every evaluation in one trace - the rows that start with x = [...]"
    F = np.array([line.split("]")[1].split()[:-1]          # f1 .. fm, dropping the scalarised value
                  for line in open(path) if line.lstrip().startswith("[")], dtype = float)
    return F[np.all(np.isfinite(F), 1)]


def hv_from_logs(name: str, problem: Problem) -> pd.DataFrame:
    pf = problem.pareto_front()
    ref = pf.max(0) + 0.1 * (pf.max(0) - pf.min(0))
    hv = HV(ref_point = ref)
    rows = []
    for path in (HERE / "problems" / name).glob("*.log"):
        model, solver = path.stem.split("_", 1)
        if solver in solvers:
            F = read_log(path)
            rows.append(dict(problem = name, solver = solver, model = model,
                             hypervolume = hv(F[BaseModel.non_dominated(F, "strongly")]),
                             ref_point = np.round(ref, 4)))
    return pd.DataFrame(rows)





if SWEEP:
    rows, boxes = [], {}
    for (name, make), solver in itertools.product(problems.items(), solvers):
        stem = f"{name}_{solver}"
        print(f"===== {stem}")
        folder = HERE / "problems" / name
        folder.mkdir(parents = True, exist_ok = True)
        try:
            fitted = many_models(build(make(), solver), folder, solver)
            table = evaluate(fitted)
            boxes[stem] = table.attrs.pop("nadir")          # pd.concat() compares attrs, and
                                                            # comparing two nadir arrays is ambiguous
            table["in_box"] = [in_box(m) for m in fitted]   # evaluate() keeps the models in order
            rows.append(table.assign(problem = name, solver = solver, status = "ok"))
            if PLOTS:
                for model in fitted:
                    plots(model, folder, solver)
        except Exception as exc:            # one cell must not take the rest of the sweep with it
            print(f"  failed: {type(exc).__name__}: {exc}")
            rows.append(pd.DataFrame([dict(problem = name, solver = solver,
                                           status = f"{type(exc).__name__}: {exc}"[:120])],
                                     index = ["-"]))
        print()


    results = (pd.concat(rows).rename_axis("model").reset_index()
                 .set_index(["problem", "solver", "model"]).sort_index())
    print(results, end = "\n\n")
    results.to_csv(HERE / "results_by_solver.csv")         # across problems, so beside problems/
    print("written to", HERE / "results_by_solver.csv", end = "\n\n")

    # the hypervolume of a cell was measured in this box, and in no other
    print("reference point per cell")
    for stem, nadir in boxes.items():
        print(f"  {stem:28s} {np.round(nadir, 4)}")


# one fixed reference point per problem, taken from its true front
common = (pd.concat([hv_from_logs(name, make()) for name, make in problems.items()])
            .set_index(["problem", "solver", "model"]).sort_index())
print(common)
common.to_csv(HERE / "hv_common_nadir.csv")
print("written to", HERE / "hv_common_nadir.csv")


# df = pd.read_csv("results_by_solver.csv").set_index(["problem", "solver"])
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# print(df)

df = pd.read_csv("hv_common_nadir.csv").set_index(["problem", "solver"])
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
print(df)