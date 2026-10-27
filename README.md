# Adaptive scalarisation for multi-objective optimisation

Two models solve a multi-objective problem by repeatedly **learning a scalarising function from the non-dominated points found so far**, then minimising it.

| Key | Class | Scalarisation | Learnt by |
|---|---|---|---|
| `baseline` | `LinearModel` | weighted sum | non-negative least squares |
| `cas` | `ChoquetModel` | 2-additive Choquet integral | capacity fitted by linear programming (`choquetreg.py`) |


## Files

| File | Role |
|---|---|
| `models.py` | `BaseModel` and the two models|
| `tests.py` | the test problems, all searched on the unit box |
| `choquetreg.py` | Choquet regressor: fits a capacity to the archive |
| `utils.py` | visualisations and logging |
| `lab.ipynb` | the workspace: build, solve, plot, compare |

## Install

```bash
pip install -r requirements.txt      # Python 3.11, versions are pinned
jupyter lab lab.ipynb
```

## The models

```python
from models import LinearModel, ChoquetModel
from tests import ZDT2

model = ChoquetModel(problem = ZDT2(n_dim = 10), max_iter = 50, max_eval = 10_000,
                     random_state = 42, solver = "L-BFGS-B", method = 2)
model.solve()
model.PF, model.PS          # the non-dominated archive, in objective and decision space
```

| Argument | Applies to | Meaning |
|---|---|---|
| `max_iter` | all | outer iterations: each one learns a scalarisation, then searches under it |
| `max_eval` | all | hard budget of objective evaluations; the run stops exactly on it |
| `solver` | all | any `scipy.optimize.minimize` method that takes bounds, e.g. `Powell`, `SLSQP` |
| `tol` | all | that solver's tolerance |
| `random_state` | all | seeds the multi-start points |
| `method` | `ChoquetModel` | capacity estimation method passed to `ChoquetReg` |
| `type`, `num_k`, `mon` | `LipschitzModel` | interpolant kind, nearest neighbours, monotonicity per objective |

## The problems

Every problem is defined on the unit box `[0, 1]^n_dim` and rescales to its own domain.

| Family | Classes | Objectives |
|---|---|---|
| ZDT | `ZDT1`, `ZDT2`, `ZDT3` | 2, fixed |
| DTLZ | `DTLZ1`, `DTLZ2` | any `n_obj < n_dim` |
| WFG | `WFG1` ... `WFG9` | any `n_obj` |
| classic 2-objective | `Kursawe`, `Sch1`, `Sch2`, `Fon` | 2, fixed |
| single-objective | `Ackley01`, `Schwefel04`, `Gulf` | 1 - to be combined |


**`ExtendedProblem`** joins problems into one, concatenating their objectives and optionally reordering them:

```python
from tests import ExtendedProblem, ZDT1, Fon
problem = ExtendedProblem(problems = [ZDT1(n_dim = 10), Fon(n_dim = 10)], order = [2, 0, 3])
```

## Hypervolume

Scores solved models against **one shared reference point**.

The reference point is the problem's own analytical front, widened by 10% of its range. For a problem with no analytical front, fallback to the worst value per objective over the pooled archives, which compares only within that group.

## Plotting
More than 3 objectives cannot be visualised, fallback to radar plot.

| Call | Shows |
|---|---|
| `model.visualise("objective")` | every evaluation, the archive, and the analytical front |
| `model.visualise("learning")` | an interactive slider over iterations: the archive|
| `model.visualise("learning", plot_scalarisation = True)` | same as above, with the level sets of the scalarisation |
| `model.visualise("true_pf")` | the analytical front alone |


