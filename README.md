# Repo description and usage guide

## Files

| File | Role |
|------|------|
| `pymoo_lab.ipynb` | Experiment workplace |
| `models.py` | The algorithm. |
| `pymoo_tests.py` | The test problems - ZDT1/2/3, DTLZ1/2 - wrapping pymoo's implementations in pymoo's own units. All are **minimisation** problems. |
| `choquetreg.py` | The Choquet regressor.|
| `utils.py` | `evaluate()` returns hypervolume, evaluations, archive size, wallclock time. `visualise()` draws the objective space, an animated view of the learning process, or the analytical front. `visualise_weights()` plots how the learnt weights move between iterations. |

## Running

```bash
pip install -r requirements.txt
jupyter lab pymoo_lab.ipynb
```

## Using `pymoo_lab.ipynb`

Run the sections in order: later cells reuse the models built in **Modelling**.

### 1. Modelling

Choose one problem, build both models on it, solve, and score them.

| Problem | Constructor | Objectives |
|---|---|---|
| ZDT1, ZDT2, ZDT3 | `ZDT1(n_dim = 10)` | fixed at 2 |
| DTLZ1, DTLZ2 | `DTLZ1(n_dim = 10, n_obj = 3)` | any `n_obj` < `n_dim`|

| Argument | Model | Meaning |
|---|---|---|
| `max_iter` | both | outer iterations; each learns a weight, then runs one L-BFGS-B solve |
| `random_state` | both | reproducibility |
| `method` | `ChoquetModel` | choice of capacity estimation methods |

`evaluate([linear_model, choquet_model])` scores both against one shared reference point.
Re-run the whole cell to start over, because `solve()` does not reset a model's history.

### 2. Model visualisation

| Call | Shows |
|---|---|
| `model.visualise("true_pf");` | the analytical front alone |
| `model.visualise("objective");` | every evaluation, the non-dominated archive, and the true front |
| `model.visualise("learning", show = False)` | a slider over iterations, showing the archive as it stood at each |
| `model.visualise("learning", plot_scalarisation = True, show = False)` | same as above, complement with level sets of the scalarised function |
| `visualise_weights(model, kind = "symmetric")` | how the learnt weight moves between iterations; also `"absolute"` and `"norm"` |

### 3. Benchmark

This is to conduct benchmark on different experimental settings, including: `dec_dims`, `obj_dims`, `max_iter`, `model_list`, `prob_list` - and run it.

It skips combinations that do not exist (ZDT at anything but 2 objectives; DTLZ with `n_dim <= n_obj`),

The next cell picks the hypervolume winner per configuration.