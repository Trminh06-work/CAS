# Adaptive scalarisation for multi-objective optimisation

Three models solve a multi-objective problem by repeatedly **learning a scalarising function from
the non-dominated points found so far**, then minimising it. They differ only in what they learn:

| Key | Class | Scalarisation | Learnt by |
|---|---|---|---|
| `baseline` | `LinearModel` | weighted sum | non-negative least squares |
| `cas` | `ChoquetModel` | 2-additive Choquet integral | capacity fitted by linear programming (`choquetreg.py`) |
| `las` | `LipschitzModel` | Lipschitz interpolant | `SLipInterpolant` (`slip_interpolant.py`, `liblipt.py`) |

All three fit their function to `y = 1` on the current archive, so the scalarised value of a
non-dominated point is pulled towards 1 and the search is pushed outward from the front it has.

## Files

| File | Role |
|---|---|
| `models.py` | `BaseModel` and the three models; the outer loop, the archive, the evaluation budget |
| `tests.py` | the test problems, all searched on the unit box; combinations and aggregations |
| `agg.py` | aggregation functions - t-norms, t-conorms, quasi-arithmetic means, uninorms |
| `choquetreg.py` | Choquet regressor: fits a capacity to the archive |
| `slip_interpolant.py`, `liblipt.py` | the Lipschitz interpolant and its library |
| `utils.py` | `evaluate()`, `visualise()`, `visualise_weights()`, `live_trace()` |
| `lab.ipynb` | the workspace: build, solve, plot, compare |
| `sanity_check.ipynb` | brute-force random search, and a check that the archive stays in the box |
| `test_main.py` | sweeps (problem x solver) from the shell and scores every model on each |
| `script/run_experiment.py` | one task of the cluster sweep, and `--aggregate` over the results |
| `script/job.conf` | the grid that sweep expands |
| `script/bash.sh` | the SLURM array that runs it |
| `script/plot_from_npz.py` | replot a finished task from its `.npz` |

## Install

```bash
pip install -r requirements.txt      # Python 3.11, versions are pinned
jupyter lab lab.ipynb
```

## The models

```python
from models import LinearModel, ChoquetModel, LipschitzModel
from tests import ZDT2

model = ChoquetModel(problem = ZDT2(n_dim = 10), max_iter = 50, max_eval = 10_000,
                     random_state = 42, solver = "L-BFGS-B", method = 2)
model.solve()
model.PF, model.PS          # the non-dominated archive, in objective and decision space
```

| Argument | Applies to | Meaning |
|---|---|---|
| `max_iter` | all | outer iterations: each one learns a scalarisation, then searches under it |
| `max_eval` | all | hard budget of objective evaluations; the run stops **exactly** on it |
| `solver` | all | any `scipy.optimize.minimize` method that takes bounds, e.g. `Powell`, `SLSQP` |
| `tol` | all | that solver's tolerance |
| `random_state` | all | seeds the multi-start points |
| `method` | `ChoquetModel` | capacity estimation method passed to `ChoquetReg` |
| `type`, `num_k`, `mon` | `LipschitzModel` | interpolant kind, nearest neighbours, monotonicity per objective |

**The outer loop.** Iteration 0 runs DIRECT on the plain weighted sum. Every iteration after that
learns a scalarisation from the archive, then searches: DIRECT on every fifth iteration, otherwise
`5 * n_dim` random restarts of `solver`. The archive is refiltered at the end of each iteration.
The run ends on `max_iter`, or on the evaluation that reaches `max_eval` - `F()` refuses to
evaluate past the budget and the solver call is abandoned where it stands.

`solve(callback)` takes an optional callback, called as `callback(model, phase)` with `phase` of
`"learn"` (just after the fit) and `"eval"` (just after the archive is refiltered).

## The problems

Every problem is defined on the unit box `[0, 1]^n_dim` and rescales to its own domain internally,
so one decision vector is valid for all of them.

| Family | Classes | Objectives |
|---|---|---|
| ZDT | `ZDT1`, `ZDT2`, `ZDT3` | 2, fixed |
| DTLZ | `DTLZ1`, `DTLZ2` | any `n_obj < n_dim` |
| WFG | `WFG1` ... `WFG9` | any `n_obj` |
| classic 2-objective | `Kursawe`, `Sch1`, `Sch2`, `Fon` | 2, fixed |
| single-objective | `Ackley01`, `Schwefel04`, `Gulf` | 1 - to be combined |
| self-constructed | `ZDT1_Shifted`, `ZDT2_Shifted`, `ZDT2_Combined` | 2 or 4 |

A class whose own definition fixes its size (`Kursawe`, `Sch1`, `Sch2`, `Gulf`) declares `N_VAR`
and ignores a larger `n_dim`; anything else must be constructed with one.

**`ExtendedProblem`** joins problems into one, concatenating their objectives and optionally
reordering them:

```python
from tests import ExtendedProblem, ZDT1, Fon
problem = ExtendedProblem(problems = [ZDT1(n_dim = 10), Fon(n_dim = 10)], order = [2, 0, 3])
```

**`Aggregated`** turns an aggregation function from `agg.py` into a single objective. It normalises
each objective to `[0, 1]` - with bounds from the problem's own front and 10 000 random samples,
or bounds you pass - and applies the function to them. Being one objective, it is used as a part
of a combination:

```python
from tests import Aggregated, ExtendedProblem, ZDT2
from agg import Product
ExtendedProblem(problems = [ZDT2(n_dim = 10), Product([ZDT2(n_dim = 10)])])
```

## Aggregation functions

`agg.py` implements `f: [0, 1]^n -> [0, 1]`, non-decreasing and preserving the bounds.

| Kind | Classes |
|---|---|
| t-norms (conjunctive) | `Product`, `Drastic`, `Lukasiewicz`, `SchweizerSklar(p)`, `Frank(p)` |
| t-conorms (disjunctive) | `.dual()` of any t-norm, with `N(t) = 1 - t` |
| averaging | `QAM(gen, inv_gen, w_vec)` - quasi-arithmetic means |
| mixed | `ConjunctiveThreePi` - a uninorm |

Running `python agg.py` plots each family over the unit square.

## Hypervolume

`utils.evaluate(models)` scores solved models against **one shared reference point** and returns a
table of `hypervolume`, `n_func_eval`, `n_non_dominated`, `n_iter`, `wallclock_time`, carrying that
reference in `table.attrs["nadir"]`.

The reference point is the problem's own analytical front, widened by 10% of its range. That makes
the number comparable across models, seeds and runs. A problem with no analytical front - an
`ExtendedProblem`, say - falls back to the worst value per objective over the pooled archives,
which compares only within that group. `script/run_experiment.py` uses the same rule.

> A hypervolume is meaningless without the box it was measured in. Never compare one across
> problems, dimensions, or runs that fell back to different references.

## Plotting

| Call | Shows |
|---|---|
| `model.visualise("objective")` | every evaluation, the archive, and the analytical front |
| `model.visualise("learning")` | a plotly slider over iterations: the archive as it stood at each, with the parameters fitted there |
| `model.visualise("learning", plot_scalarisation = True)` | the same, with the level sets of the scalarisation |
| `model.visualise("true_pf")` | the analytical front alone |
| `visualise_weights(model, kind = "symmetric")` | how the weights move between iterations; also `"absolute"`, `"norm"` |

More than 3 objectives cannot be scattered, so every kind falls back to pymoo's radar plot.

## Tracing a run

`live_trace(path)` returns a callback that writes the run to disk **as it goes** - per iteration:
the non-dominated points the fit was given, the parameters learnt from them, and every evaluation
the search then made with its scalarised value. A run that is killed still leaves the iteration it
died in.

```python
from utils import live_trace
model.solve(live_trace("trace.log"))
```

## Sweeps

### From the shell: `test_main.py`

One cell is one (problem, solver) pair, scored over all three models. Edit `problems`, `solvers`,
`N_DIM`, `MAX_ITER` and `SEED` at the top; `SWEEP = True` reruns, `SWEEP = False` rescores the
trace logs already in `problems/`. Writes `results_by_solver.csv` and `hv_common_nadir.csv`.

### On the cluster: `script/`

```bash
python run_experiment.py --conf job.conf --count        # how many tasks the grid expands to
python run_experiment.py --conf job.conf --list         # the task list, each line led by its index
python run_experiment.py --conf job.conf --index 7      # run task 7
python run_experiment.py --conf job.conf --index 7 --live   # ... and trace it as it goes
python run_experiment.py --conf job.conf --aggregate    # merge the rows -> results/summary.csv
sbatch bash.sh                                          # the whole array
```

One task is one `(model, problem, n_dim, n_obj, seed, solver)` and writes
`results/<model>_<problem>_d<n_dim>_m<n_obj>_s<seed>_<solver>.{csv,npz}`, plus
`logs/trace_<task>.log` under `--live`. `--aggregate` adds `hypervolume`, `hv_ref`, `hv_models`
and `hv_rank`, ranking the models within one (problem, sizes, seed, solver).

`job.conf` holds the grid: `models`, `problems`, `dec_dims`, `obj_dims`, `seeds`, `solvers`,
`max_iter`, `max_eval`, the model settings `cas_method` and `las_type`, and `results_dir`. A
combination is `combo_<name> = A, B` (with an optional `combo_<name>_order`), and an aggregation
is `agg_<name> = Function, A, B`, used as a part of a combination. A list may continue on the next
line. **After editing it, re-check `--count` and set `--array` in `bash.sh` to `0 .. count - 1`.**

Replot any finished task:

```bash
python plot_from_npz.py results/cas_ZDT2_d10_m2_s0_L-BFGS-B.npz --kind learning --level-sets
```

The `.npz` holds the archive and the history but not the fitted models, so the script refits each
iteration's scalarisation from the saved evaluations before drawing it.
