"""
    Score adaptive scalarisation and the pymoo MOEAs on one footing: HV, GD and IGD per run.

        python benchmark.py                score every run not already in the csv
        python benchmark.py --build        run and cache the MOEA grid in parallel, score nothing
        python benchmark.py --summary      print the comparison from the csv
        python benchmark.py --refs-only    build the reference fronts and check the box

    SELF-CONTAINED BY DESIGN. The only thing this script reads from elsewhere is the sweep's own
    npz (--as-dir). Everything else it needs it makes and owns, under results/benchmark: the four
    MOEA runs, the GD/IGD reference fronts, and the csv.

    So it borrows nothing from ablation/moea, nor from results/analysis - two neighbours with
    their own questions, settings and lifetimes. The MOEAs are configured exactly as
    ablation/moea configures them and reproduce its npz bit for bit; the reference fronts are
    built the same way analysis/compute_gd_igd.py builds its own and come out bit-identical.
    Both are facts worth verifying and neither is a dependency worth having: a neighbour is free
    to be deleted, re-run with other settings, or published as it stands, and this comparison
    must not change when it is.

    Resumable twice over: a cached MOEA npz is never re-run, and a scored row is never
    re-scored, so an interrupted pass loses nothing. Delete the csv to rescore from scratch.

    WHAT MAKES THE TWO COMPARABLE

      budget      both families spend max_eval = 100_000 objective evaluations.
      archive     both store PF = the non-dominated set of EVERY point evaluated, not the final
                  population. A MOEA's pop_F is ~100-200 points and would flatter its convergence
                  while destroying its spread, so it is deliberately not used here. The filter is
                  core.nd_torch, the function BaseModel.non_dominated("strongly") already uses,
                  so the two families' fronts come out of one implementation and not merely out
                  of one definition.
      box         one hypervolume reference for everyone: 1.1 * (2, 4, ..., 2m). WFG is built with
                  f_i in [0, 2i] and its front's nadir at exactly 2i, so this is analytic - no
                  sampling, identical for all nine problems, and independent of what either
                  family happened to reach. The alternative (pooling the runs' own maxima) is
                  endogenous: it would move when the MOEAs spread wider than AS did.
      front       one reference front per (problem, m) for GD/IGD, drawn from WFG's analytic
                  Pareto SET with a fixed seed, and shared by both families.
      problem     the MOEAs search the UNIT CUBE and WFG's [0, 2i] is applied by to_domain inside
                  _evaluate, exactly as tests.WFG does it. See recorded().

    WHAT IS NOT SYMMETRIC, AND MUST BE SAID IN ANY WRITE-UP

      seeds       AS ran one seed; the MOEAs ran five. Do not rank the two families' runs against
                  each other as if they were peers. Compare AS's single run against each MOEA's
                  median-of-5, and state which side carries error bars. --summary does this.
      solvers     an AS "run" is (model, solver); a MOEA run is (algorithm, seed). AS therefore
                  has 7 configurations per model per cell and the MOEAs have 1.

    CONVENTIONS

      GD uses p = 1, which is pymoo's (the mean over the minima). The older
      analysis/gd_igd.csv used p = 2 and reads about 14% high; both exponents are recorded in
      the csv so the two are never silently mixed. GD is sensitive to |R| and IGD is not, so
      n_ref is recorded per row and must be held fixed across anything being compared.

      Nothing above numpy is imported at module level, so --summary needs neither pymoo nor
      torch. reference_front() duplicates the sampler in analysis/compute_gd_igd.py rather than
      importing it, because that script's published gd_igd.csv uses p = 2 and must stay free to
      keep doing so; the two caches are separate and verified equal, 36 of 36.

      THE ORDERING IS THE RANK, AND NOTHING HERE IS AN EFFECT SIZE. Ranks are reported because
      they survive a change of reference box - re-boxing the sweep moved a model's mean rank by
      under 0.1 and flipped 1 of 36 cell winners, where it moves hv itself by orders of
      magnitude - and because they are the only form in which a MOEA and one AS configuration
      are the same kind of thing.

      An hv RATIO between the two families is deliberately not reported. It looks like an effect
      size and is not one: most of the box is volume both families dominate, so the contested
      fraction is small and the typical cell reads as a 2-6% difference however far apart the
      fronts are, and a tighter box on WFG3 m5 moved that cell's apparent gap by 7x on its own.
      For how large a difference is, quote igd_norm LEVELS, which have no box at all - not their
      ratio, which blows up wherever the denominator nears zero (WFG2 m2 gives 102x off an
      igd_norm of 0.0003). The cost of dropping the ratio is that rank cannot distinguish a
      collapse from a near miss: AS's best rank is 5 of 7 on both WFG1 and WFG7 at every m,
      though it loses 64% of the dominated volume on the first and 0.2% on the second. Say so
      from the hv columns, which are still printed per cell.

      Reporting MOEA/D alone would be cherry-picking: it is the weakest of the four baselines,
      and SMS-EMOA ranks first in 25 of the 36 cells. --summary therefore always prints the
      whole field, with the MOEA/D grid as the pair that isolates adaptive against fixed weights.
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, SRC)             # the core package
sys.path.insert(0, HERE)            # run_experiment

import numpy as np

AS_DIR = os.path.join(HERE, "results", "results")                   # the sweep's per-task npz
OWN = os.path.join(HERE, "results", "benchmark")                    # everything this script makes
REF_DIR = os.path.join(OWN, "ref_fronts")                           # its own, not analysis/ref_fronts
MOEA_DIR = os.path.join(OWN, "moea_runs")                           # its MOEA runs
OUT = os.path.join(OWN, "benchmark.csv")
CONF = os.path.join(HERE, "job.conf")

N_REF = 50_000          # samples of the Pareto set, before the non-dominated filter
REF_SEED = 0            # makes the reference front reproducible, unlike pymoo's own
P_GD = 1                # pymoo's convention: the mean over the minima
P_IGD = 1
PAD = 0.10              # the box is the nadir widened by this much
CHUNK = 2048            # rows of the pairwise distance matrix held at once
ND_BLOCK = 1024         # rows per block in the non-dominated filter, nd_torch's own default

BUDGET = 100_000                            # job.conf's max_eval, so both families spend the same
N_DIM = 10
SEEDS = (0, 1, 2, 3, 4)
ALGORITHMS = ("moead", "nsga2", "nsga3", "sms")
PARTITIONS = {2: 99, 3: 12, 4: 7, 5: 6}     # das-dennis -> 100, 91, 120, 210 directions
POP = 100                                   # NSGA-II and SMS-EMOA, which take a size not directions
WORKERS = 4                                 # --build only; scoring stays in one process

FIELDS = ["family", "method", "problem", "n_dim", "n_obj", "seed", "solver", "archive",
          "hv", "gd", "igd", "gd_norm", "igd_norm", "n_ref", "ref_seed", "p_gd", "p_igd"]
KEY = ("family", "method", "problem", "n_obj", "seed", "solver")
PROBS = [f"WFG{i}" for i in range(1, 10)]
OBJS = (2, 3, 4, 5)


def nadir(n_obj):
    "WFG is built with f_i in [0, 2i] and its front's nadir at exactly 2i"
    return 2.0 * np.arange(1, n_obj + 1)


def box(n_obj, pad = PAD):
    "the hypervolume reference: the analytic nadir, widened"
    return (1.0 + pad) * nadir(n_obj)


def reference_front(problem, n_obj, n = N_REF, seed = REF_SEED):
    """
        The true front, from WFG's analytic Pareto set rather than pymoo's Monte-Carlo
        pareto_front() - which is unseeded below m = 4 and raises above m = 3.
    """
    file = os.path.join(REF_DIR, f"{problem}_m{n_obj}_n{n}_s{seed}.npz")
    if os.path.exists(file):
        with np.load(file) as handle:
            return handle["R"]

    import run_experiment
    from core.nd_torch import non_dominated_mask_gpu

    inner = run_experiment.build_problem(problem, N_DIM, n_obj, run_experiment.read_conf(CONF)).inner
    # _calc_pareto_set does not forward random_state, so the sampler is called directly
    K = inner._rand_optimal_position(n, random_state = np.random.default_rng(seed))
    X = np.vstack([inner._calc_pareto_set_extremes(), inner._positional_to_optimal(K)])
    F = run_experiment.finite(np.asarray(inner.evaluate(X), dtype = float))
    uniq = np.unique(F, axis = 0)
    R = uniq[non_dominated_mask_gpu(uniq, block_size = ND_BLOCK)[0]]

    os.makedirs(REF_DIR, exist_ok = True)
    np.savez_compressed(file, R = R)
    return R


def recorded(problem):
    """
        The sweep's own problem, wrapped for pymoo: vectorised, and logging every evaluation.

        Two things here are load-bearing. The search box is the UNIT CUBE and WFG's [0, 2i] is
        applied by to_domain inside _evaluate, exactly as tests.WFG does it - handing pymoo
        get_problem("wfg4", ...) directly would search a different space and the comparison
        would be meaningless. And the logging is inside _evaluate rather than in a Callback,
        because MOEA/D is steady-state: its algorithm.off is a single Individual, so a callback
        over pop + off captures about 14% of the budget.

        The class is local so that importing this module costs nothing but numpy.
    """
    from pymoo.core.problem import Problem
    from core.tests import to_domain

    class Recorded(Problem):
        def __init__(self):
            super().__init__(n_var = problem.n_dim, n_obj = problem.n_obj,
                             xl = problem.xl, xu = problem.xu)      # the unit box
            self.F, self.X, self.gen = [], [], []

        def _evaluate(self, X, out, *args, **kwargs):
            X = np.asarray(X, dtype = float)
            F = problem.inner.evaluate(to_domain(X, problem.lo, problem.hi))
            self.F.append(np.asarray(F, dtype = float))
            self.X.append(X.copy())
            # int32: MOEA/D calls this once per individual, so the counter reaches the budget
            self.gen.append(np.full(len(X), len(self.gen), np.int32))
            out["F"] = F

    return Recorded()


def algorithm_for(algo, n_obj):
    """
        One of the four baselines, at pymoo's own defaults but for what each one must be told.
    """
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.sms import SMSEMOA
    from pymoo.util.ref_dirs import get_reference_directions

    dirs = lambda: get_reference_directions("das-dennis", n_obj, n_partitions = PARTITIONS[n_obj])
    match algo:
        case "moead":                   # Tchebicheff at m=2 and PBI beyond, pymoo's own choice
            return MOEAD(ref_dirs = dirs(), n_neighbors = 20, prob_neighbor_mating = 0.9)
        case "nsga2":
            return NSGA2(pop_size = POP)
        case "nsga3":
            return NSGA3(ref_dirs = dirs())
        case "sms":
            return SMSEMOA(pop_size = POP)
    raise ValueError(f"unknown algorithm {algo}")


def moea_file(algo, problem, n_obj, seed):
    "where a cached MOEA run lives: <PROBLEM>_m<M>/<algo>_s<SEED>.npz"
    return os.path.join(MOEA_DIR, f"{problem}_m{n_obj}", f"{algo}_s{seed}.npz")


def run_moea(algo, problem, n_obj, seed, budget = BUDGET):
    """
        Run one MOEA on this problem and return its archive front, caching it as npz.
    """
    file = moea_file(algo, problem, n_obj, seed)
    if os.path.exists(file):
        with np.load(file) as handle:
            return handle["PF"]

    from pymoo.optimize import minimize
    import run_experiment
    from core.nd_torch import non_dominated_mask_gpu

    inner = run_experiment.build_problem(problem, N_DIM, n_obj, run_experiment.read_conf(CONF))
    if inner.n_obj != n_obj:
        raise ValueError(f"{problem} builds {inner.n_obj} objectives, asked for {n_obj}")

    rec = recorded(inner)
    t0 = time.perf_counter()
    minimize(rec, algorithm_for(algo, n_obj), ("n_evals", budget), seed = seed, verbose = False)
    wall = time.perf_counter() - t0

    # a generational algorithm checks the budget after the generation, so it overshoots by up to
    # pop_size - 1: cut to the budget so every run of either family is scored on exactly the same spend
    F, X, gen = (np.concatenate(a)[:budget] for a in (rec.F, rec.X, rec.gen))
    uniq, first = np.unique(F, axis = 0, return_index = True)
    idx = np.sort(first[non_dominated_mask_gpu(uniq, block_size = ND_BLOCK)[0]])

    os.makedirs(os.path.dirname(file), exist_ok = True)
    np.savez_compressed(file, PF = F[idx], PS = X[idx], made = gen[idx].astype(np.int32),
                        iter_hist = gen.astype(np.int32), wallclock = wall, n_eval = len(F))
    return F[idx]


def _build_one(job):
    "one --build task, at module level so a spawned worker can unpickle it"
    algo, problem, n_obj, seed = job
    t0 = time.perf_counter()
    PF = run_moea(algo, problem, n_obj, seed)
    return job, len(PF), time.perf_counter() - t0


def build(workers = WORKERS, algorithms = ALGORITHMS):
    """
        Run and cache the MOEA grid in parallel, scoring nothing.

        Separate from scoring because the search is the expensive half (SMS-EMOA at m=5 is about
        20 minutes a run against seconds for NSGA-II) and embarrassingly parallel, while scoring
        must stay in one process to append to one csv. Slowest first, so the m=5 SMS-EMOA runs
        start immediately and the cheap algorithms fill in around them.

        Scoring falls back to running a missing MOEA itself, so this mode is an optimisation and
        never a prerequisite.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # each worker is a fresh spawned process, so it imports numpy under these and does not
    # oversubscribe the cores the other workers are using
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    grid = [(a, p, m, s) for a in algorithms for p in PROBS for m in OBJS for s in SEEDS]
    todo = [j for j in grid if not os.path.exists(moea_file(*j))]
    todo.sort(key = lambda j: (j[0] != "sms", -j[2]))       # SMS-EMOA first, highest m first
    print(f"{len(grid)} MOEA runs, {len(todo)} to run at {BUDGET:,} evaluations each, "
          f"{workers} workers", flush = True)
    if not todo:
        return

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers = workers, mp_context = mp.get_context("spawn"),
                             max_tasks_per_child = 1) as pool:
        for k, (job, n_pf, wall) in enumerate(pool.map(_build_one, todo), 1):
            algo, problem, n_obj, seed = job
            print(f"  {k:>3}/{len(todo)}  {algo:6} {problem} m{n_obj} s{seed}: "
                  f"front {n_pf:>6}, {wall:7.1f}s   [{time.perf_counter() - t0:.0f}s total]",
                  flush = True)


def jobs(as_dir):
    """
        Every run to score: the sweep's npz, and the MOEA grid this script runs itself.

        Sorted by (problem, n_obj) so each of the 36 reference fronts is built once and reused.
    """
    import run_experiment

    # the sweep: <model>_<problem>_d<n_dim>_m<n_obj>_s<seed>_<solver>.npz
    pattern = re.compile(rf"({'|'.join(run_experiment.MODELS)})_(.+)_d(\d+)_m(\d+)_s(\d+)_(.+)")
    out = []
    for path in sorted(glob.glob(os.path.join(as_dir, "*.npz"))):
        match = pattern.fullmatch(os.path.basename(path).removesuffix(".npz"))
        if match:
            model, problem, n_dim, n_obj, seed, solver = match.groups()
            out.append(dict(family = "as", method = model, problem = problem, n_dim = int(n_dim),
                            n_obj = int(n_obj), seed = int(seed), solver = solver))

    out += [dict(family = "moea", method = a, problem = p, n_dim = N_DIM, n_obj = m,
                 seed = s, solver = "")
            for a in ALGORITHMS for p in PROBS for m in OBJS for s in SEEDS]

    out.sort(key = lambda j: (j["problem"], j["n_obj"], j["family"], j["method"], j["seed"]))
    return out


def done(out):
    "the keys already in the csv, so a second run resumes instead of repeating"
    if not os.path.exists(out):
        return set()
    with open(out, newline = "") as fh:
        return {tuple(str(row[k]) for k in KEY) for row in csv.DictReader(fh)}


def archive_of(job, as_dir):
    "the front of one run: read from the sweep's npz, or run (or read) the MOEA"
    import run_experiment

    if job["family"] == "moea":
        PF = run_moea(job["method"], job["problem"], job["n_obj"], job["seed"])
    else:
        file = os.path.join(as_dir, f"{job['method']}_{job['problem']}_d{job['n_dim']}"
                                    f"_m{job['n_obj']}_s{job['seed']}_{job['solver']}.npz")
        with np.load(file) as handle:
            PF = handle["PF"]
    return run_experiment.finite(PF)


def score(job, as_dir, R, scale, ref):
    "one run: its hypervolume in the common box, and its distances to the shared front"
    from core import gd_torch
    from core.indicators import hypervolume

    A = archive_of(job, as_dir)
    row = {k: job[k] for k in ("family", "method", "problem", "n_dim", "n_obj", "seed", "solver")}
    row.update(archive = len(A), n_ref = len(R), ref_seed = REF_SEED, p_gd = P_GD, p_igd = P_IGD)
    if not len(A):                      # a run whose whole front is non-finite
        row.update(hv = 0.0, gd = np.nan, igd = np.nan, gd_norm = np.nan, igd_norm = np.nan)
        return row
    row.update(hv = hypervolume(A, ref),
               gd = gd_torch.gd_gpu(A, R, p = P_GD, chunk_size = CHUNK),
               igd = gd_torch.igd_gpu(A, R, p = P_IGD, chunk_size = CHUNK),
               gd_norm = gd_torch.gd_gpu(A / scale, R / scale, p = P_GD, chunk_size = CHUNK),
               igd_norm = gd_torch.igd_gpu(A / scale, R / scale, p = P_IGD, chunk_size = CHUNK))
    return row


def main(as_dir, out):
    from core import gd_torch

    todo = jobs(as_dir)
    have = done(out)
    n_as = sum(j["family"] == "as" for j in todo)
    todo = [j for j in todo if tuple(str(j[k]) for k in KEY) not in have]

    print(f"device {gd_torch.get_device()} | |R| {N_REF} seed {REF_SEED} | "
          f"GD p={P_GD}, IGD p={P_IGD} | box = {1 + PAD:.2f} x nadir")
    print(f"{n_as} as runs found; {len(todo)} to score, {len(have)} already done", flush = True)
    if not todo:
        return

    os.makedirs(os.path.dirname(out) or ".", exist_ok = True)
    new = not os.path.exists(out)
    t0 = time.perf_counter()
    key = R = scale = ref = None

    with open(out, "a", newline = "") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        if new:
            writer.writeheader()
        for k, job in enumerate(todo, 1):
            if (job["problem"], job["n_obj"]) != key:
                key = (job["problem"], job["n_obj"])
                R, scale, ref = reference_front(*key), nadir(key[1]), box(key[1])
                print(f"  {key[0]} m{key[1]}: reference front {len(R)} points", flush = True)
            try:
                writer.writerow(score(job, as_dir, R, scale, ref))
                fh.flush()
            except Exception as exc:        # one bad run must not take the pass with it
                print(f"  failed {job['family']}/{job['method']} {key} s{job['seed']}: "
                      f"{type(exc).__name__}: {exc}", flush = True)
            if k % 25 == 0 or k == len(todo):
                print(f"  {k} / {len(todo)}   {time.perf_counter() - t0:.0f}s", flush = True)

    print(f"written to {os.path.relpath(out, HERE)}")


def ranked(t, per_solver = True):
    """
        One row per configuration, ranked within each (problem, m) cell.

        Each MOEA is reduced to its median over the 5 seeds FIRST, so it enters as one
        configuration - the same unit as one AS (model, solver) pair - rather than as five draws
        competing against AS's single one.

        per_solver = True  : AS contributes all 7 solvers of each model, so 25 rows a cell.
                             Per-method means are then diluted by each model's weak solvers.
        per_solver = False : each AS model enters at its best solver, 7 rows a cell. This is the
                             only form in which per-method mean ranks compare like with like.
    """
    import pandas as pd

    cell = ["problem", "n_obj"]
    AS = t[t.family == "as"][cell + ["method", "solver", "hv"]]
    if not per_solver:
        AS = AS.groupby(cell + ["method"], as_index = False).hv.max().assign(solver = "best")
    MO = (t[t.family == "moea"].groupby(cell + ["method"], as_index = False).hv.median()
                               .assign(solver = "median-of-5"))
    out = pd.concat([AS, MO], ignore_index = True)
    out["rank"] = out.groupby(cell).hv.rank(ascending = False, method = "average")
    return out


def summary(out):
    """
        The comparison, respecting the seed asymmetry: AS is one draw per (model, solver),
        each MOEA is five draws, so the MOEAs are reduced to a median before anything is compared.
    """
    import pandas as pd

    if not os.path.exists(out):
        raise SystemExit(f"no {out} yet - run the scoring pass first")
    t = pd.read_csv(out)
    print(f"{len(t)} runs: {(t.family == 'as').sum()} as, {(t.family == 'moea').sum()} moea\n")

    # per (problem, m): AS's best over its 21 configurations, each MOEA's median over 5 seeds
    cell = ["problem", "n_obj"]
    med = t[t.family == "moea"].groupby(cell + ["method"]).hv.median().groupby(cell)
    comp = pd.concat([t[t.family == "as"].groupby(cell).hv.max().rename("as_best"),
                      t[t.family == "as"].groupby(cell).hv.median().rename("as_median"),
                      med.max().rename("moea_best"),
                      med.idxmax().str[-1].rename("best_moea")], axis = 1)
    comp["winner"] = np.where(comp.as_best >= comp.moea_best, "as", "moea")
    print("hypervolume in the common box, per cell "
          "(as_best = best of 21 AS runs; moea_best = best MOEA's median of 5)")
    print(comp.round(3).to_string(), "\n")
    print(f"AS wins {int((comp.winner == 'as').sum())} of {len(comp)} cells\n")

    print("by objective count (count of cells AS wins):")
    print(comp.groupby("n_obj").winner.value_counts().unstack(fill_value = 0).to_string(), "\n")

    # the rank view. Robust to the reference box in a way the hv columns above are not, and the
    # only framing in which a MOEA and one AS configuration are comparable units.
    d = ranked(t)
    n = int(d.groupby(cell).size().iloc[0])
    print(f"RANK within each cell: {n} configurations (AS's 21, plus each MOEA's median-of-5); "
          f"chance rank {(n + 1) / 2:.1f}")
    print("  an AS model contributes 7 solver rows and a MOEA only 1, so these means are pulled")
    print("  toward the middle by each model's weak solvers - compare on the next table instead")
    r = d.groupby("method")["rank"].agg(["mean", "median"]).round(2).sort_values("mean")
    r["ranks_1st"] = (d[d["rank"] == 1].method.value_counts()
                       .reindex(r.index, fill_value = 0).astype(int))
    print(r.to_string(), "\n")

    fair = ranked(t, per_solver = False)
    n = int(fair.groupby(cell).size().iloc[0])
    print(f"the comparable form: each AS model at its best solver, each MOEA at its median - "
          f"{n} rows a cell, chance rank {(n + 1) / 2:.1f}")
    tab = fair.pivot_table(index = "method", columns = "n_obj", values = "rank", aggfunc = "mean")
    tab["all m"] = fair.groupby("method")["rank"].mean()
    print(tab.round(2).sort_values("all m").to_string(), "\n")

    # the pair that isolates what adaptive weights buy: AS's 21 configurations and MOEA/D alone
    pair_rank = ranked(t[t.family.eq("as") | t.method.eq("moead")])
    n = int(pair_rank.groupby(cell).size().iloc[0])
    md = pair_rank[pair_rank.method == "moead"]
    print(f"MOEA/D's rank among AS's 21 configurations and itself ({n} a cell), per problem and m")
    print(md.pivot(index = "problem", columns = "n_obj", values = "rank")
            .to_string(float_format = lambda v: f"{v:.0f}"))
    first = int((md["rank"] == 1).sum())
    print(f"  MOEA/D beats all 21 AS configurations in {first} of {len(md)} cells; "
          f"AS takes the other {len(md) - first}")
    print(f"  mean MOEA/D rank by m: "
          f"{md.groupby('n_obj')['rank'].mean().round(2).to_dict()}\n")

    # the load-bearing pair: MOEA/D is fixed-weight decomposition, cas is the adaptive one
    pair = t.method.isin(["cas", "moead"])
    print("cas (adaptive scalarisation) against MOEA/D (fixed-weight decomposition):")
    print(t[pair].groupby(["family", "method"])[["hv", "igd_norm", "gd_norm"]]
                 .median().round(4).to_string(), "\n")

    print("median by family and method (igd_norm is the coverage measure to quote):")
    print(t.groupby(["family", "method"])[["hv", "gd_norm", "igd_norm", "archive"]]
           .median().round(4).to_string())


def refs_only():
    """
        Build and cache every reference front, and check the box that the whole comparison
        rests on: no true-front point may lie outside 1.1 x the analytic nadir.
    """
    print(f"{'problem':8} {'m':>2} {'|R|':>7} {'worst max / 2i':>15}")
    for problem in PROBS:
        for n_obj in OBJS:
            R = reference_front(problem, n_obj)
            print(f"{problem:8} {n_obj:>2} {len(R):>7} {(R / nadir(n_obj)).max():>15.3f}",
                  flush = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as-dir", default = AS_DIR, help = "the sweep's per-task npz")
    parser.add_argument("--out", default = OUT)
    parser.add_argument("--build", action = "store_true", help = "run and cache the MOEA grid")
    parser.add_argument("--workers", type = int, default = WORKERS, help = "--build only")
    parser.add_argument("--summary", action = "store_true", help = "print the comparison and exit")
    parser.add_argument("--refs-only", action = "store_true", help = "build the reference fronts")
    args = parser.parse_args()

    if args.refs_only:
        refs_only()
    elif args.build:
        build(args.workers)
    elif args.summary:
        summary(args.out)
    else:
        main(args.as_dir, args.out)
