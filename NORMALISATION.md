# Objective normalisation: what it is, and what it is not

Branch `normalise-objectives`. Replaces the old `K - f` maximisation posing in
`pymoo_tests.Problem` with an order-preserving affine rescaling.

```
z = (f - lo) / (hi - lo),      hi > lo componentwise
```

`normalise` selects where `(lo, hi)` come from:

| mode       | source                                   | oracle? |
|------------|------------------------------------------|---------|
| `None`     | raw pymoo units                          | no      |
| `"bounds"` | analytic per-objective range over the box | no     |
| `"front"`  | ideal/nadir of the analytical front       | **yes** |

## Is the resulting PF/PS real, or an artefact?

**Real, and this is verified, not asserted.**

* `z = a*f + b` with `a > 0` diagonal, so `z_i <= z'_i` exactly when `f_i <= f'_i`.
  Dominance matrices computed on 60 sampled points are **identical** in raw and
  normalised units, for every problem and both modes.
* `true(stored(f)) == f` to `<= 5.7e-14` over 200 random points per problem.
* `pareto_set()` is **bit-identical** across modes: normalising the objectives
  cannot change which decision vector is Pareto optimal.

So the reported front is the true front in different units, and the reported set
is the true set. Nothing is fabricated. Contrast `K - f`, which is order-
*reversing*: it returns the **maximal** set, a genuinely different set of `x`
(ZDT1: tail variables at 1, `g = 10`, versus 0 and `g = 1` for minimisation).

## But it is not neutral for the search

Normalisation cannot change what is optimal; it does change which optima a
scalarised search finds. `LinearModel`, 10 seeds, IGD measured in **raw** units
against pymoo's analytical front:

| problem | `None`          | `"bounds"`      | `"front"`       | verdict            |
|---------|-----------------|-----------------|-----------------|--------------------|
| ZDT1    | 0.0049 ± 0.0010 | 0.3654 ± 0.3361 | 0.0049 ± 0.0010 | **much worse**     |
| ZDT2    | 0.3376 ± 0.0284 | 0.3406 ± 0.0323 | 0.3376 ± 0.0284 | within noise       |
| ZDT3    | 0.0340 ± 0.0075 | 0.1253 ± 0.3095 | 0.0342 ± 0.0189 | within noise       |
| DTLZ1   | 31.01 ± 11.27   | 15.00 ± 8.33    | 31.60 ± 11.74   | **better**         |
| DTLZ2   | 0.2474 ± 0.0484 | 0.1810 ± 0.0339 | 0.2474 ± 0.0484 | **better**         |

Three things follow.

**1. `"front"` is worthless here, which defuses the oracle worry.** pymoo's
benchmark fronts are already built in `[0, 1]`, so `"front"` is the *identity*
on ZDT1, ZDT2 and DTLZ2, and a uniform `f/0.5` on DTLZ1 — its numbers are
identical to `None`. Only ZDT3 has a non-trivial front box. There is no hidden
oracle advantage to accidentally claim, but the mode stays a diagnostic only.

**2. `"bounds"` is not an improvement, it is a trade.** It halves IGD on DTLZ1
and DTLZ2 and costs a factor of 75 on ZDT1. The ZDT loss is explainable: the
map is *non-uniform* there (`span = [1, 10]`), so it genuinely reweights the
scalarised objective toward `f1`. The DTLZ1 gain is *not* explainable that way —
`span` is uniform (551.15) and a weighted sum should absorb it. Ruled out by
experiment: solver `tol` (1e-15 to 1e-1, no effect), inner `maxiter` (50 to
5000, bit-identical), `nnls` direction (scale-invariant after the `||w||`
normalisation in `models.py`), finite-difference gradient accuracy (4.4e-7 vs
4.0e-7, cosine 1.000000), and first-step length (1.43 vs 1.06). What is left is
that L-BFGS-B's trajectory on a function with ~11^5 local minima is not scale-
invariant, so the subproblems land in different basins. The effect is systematic
over 10 seeds, but it is a **search-trajectory** effect, not a truer front.

**3. It does not fix concavity.** ZDT2's front is concave, so the weighted sum
collapses to the extreme points in *every* mode — `|PF| = 4`, IGD ~0.34
throughout. No affine map can repair that; it is the standard limitation of
linear scalarisation and the actual motivation for the Choquet aggregator.
`ChoquetModel` on ZDT2 reaches IGD 0.037 with 186 archive points where
`LinearModel` gets 0.34 with 4.

## Recommendation

Report `None` as the default. Use `"bounds"` only with the per-problem result
above stated, never as a blanket "we normalise". Keep `"front"` out of results
entirely. If normalisation is wanted as a *method*, the defensible version is
adaptive: ideal/nadir estimated from the running archive, as MOEA/D and NSGA-III
do — no oracle, and it tracks the front as it is discovered.

## Known unrelated bug

`ChoquetModel` Step 0 is a no-op: `ChoquetReg.choquet_value` returns the
constant 0 while `self.dim == 0` (`choquetreg.py:104`), which is the state until
the first `fit_choquet`. L-BFGS-B on a constant terminates at iteration 0, so
the Choquet seed archive is one random point plus its finite-difference
neighbours, and iteration 1 fits a 2-additive capacity to `N ~ 1`. Not addressed
on this branch.
