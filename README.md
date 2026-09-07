# Repo description and usage guide

## Files

| File | Role |
|------|------|
| `pymoo_lab.ipynb` | Experiment workplace |
| `models.py` | The algorithm. |
| `pymoo_tests.py` | The test problems - ZDT1/2/3, DTLZ1/2 - wrapping pymoo's implementations. All are transformed into **maximisation** problems. |
| `choquetreg.py` | The Choquet regressor.|
| `utils.py` | `evaluate()` returns hypervolume, evaluations, archive size, wallclock time. `visualise()` draws the objective space, an animated view of the learning process, or the analytical front. `visualise_weights()` plots how the learnt weights move between iterations. |

## Running

```bash
pip install -r requirements.txt
jupyter lab pymoo_lab.ipynb
```