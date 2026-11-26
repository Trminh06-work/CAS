"""
    Score adaptive scalarisation and the pymoo MOEAs on one footing: HV, GD and IGD per run.

        python benchmark.py                score every run not already in the csv
        python benchmark.py --build        run and cache the MOEA grid in parallel, score nothing
        python benchmark.py --summary      print the comparison from the csv
        python benchmark.py --refs-only    build the reference fronts and check the box

    WHAT MAKES THE TWO COMPARABLE

      budget      both families spend max_eval = 100_000 objective evaluations.
      archive     both store PF = the non-dominated set of EVERY point evaluated, not the final population.
      box         one hypervolume reference for everyone: 1.1 * (2, 4, ..., 2m).
      front       one reference front per (problem, m) for GD/IGD, drawn from WFG's analytic
                  Pareto SET with a fixed seed, and shared by both families.
      problem     the MOEAs search the UNIT CUBE and WFG's [0, 2i] is applied by to_domain inside
                  _evaluate, exactly as tests.WFG does it. See recorded().

    WHAT IS NOT SYMMETRIC, AND MUST BE SAID IN ANY WRITE-UP

      seeds       AS ran one seed; each MOEA ran several and enters as the median over them.
      solvers     an AS "run" is (model, solver); a MOEA run is (algorithm, seed).

    CONVENTIONS

      GD and IGD use p = 1.
      THE ORDERING IS THE RANK, NOT RAW HV.
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
# ALGORITHMS = ("moead", "nsga2", "nsga3", "sms")
ALGORITHMS = ("moead",)
PARTITIONS = {2: 99, 3: 12, 4: 7, 5: 6}     # das-dennis -> 100, 91, 120, 210 directions
POP = 100                                   # NSGA-II and SMS-EMOA, which take a size not directions
WORKERS = 4                                 # --build only; scoring stays in one process

FIELDS = ["family", "method", "problem", "n_dim", "n_obj", "seed", "solver", "archive",
          "hv", "gd", "igd", "gd_norm", "igd_norm", "n_ref", "ref_seed", "p_gd", "p_igd"]
KEY = ("family", "method", "problem", "n_obj", "seed", "solver")
# the indicators the summary tabulates -> (rank ascending?, how to read it). hv must be present:
# it is what the cell rank is taken on. ascending = False puts the largest value at rank 1.
REPORTED = {"hv":       (False, "higher is better"),
            "igd_norm": (True,  "lower is better"),
            "gd_norm":  (True,  "lower is better"),
            "archive":  (False, "front size, not a score")}
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
        This mode is an optimisation to run MOEA in faster manner.
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


def ranked(t, metrics = REPORTED):
    """
        One row per configuration, ranked within each (problem, m) cell, carrying every reported
        indicator. Each MOEA is its median over its seeds, so the seed asymmetry is handled here
        and once only - everything downstream reads one row per (config, cell).
    """
    import pandas as pd

    cell = ["problem", "n_obj"]
    label = f"median-of-{t[t.family == 'moea'].seed.nunique()}"
    AS = (t[t.family == "as"][cell + ["family", "method", "solver"] + list(metrics)]
            .assign(config = lambda f: f.method + "/" + f.solver))
    MO = (t[t.family == "moea"].groupby(cell + ["family", "method"], as_index = False)[list(metrics)]
                               .median()
                               .assign(solver = label, config = lambda f: f.method))
    out = pd.concat([AS, MO], ignore_index = True)
    out["rank"] = out.groupby(cell).hv.rank(ascending = False, method = "average")
    return out


def summary(out):
    """
        The comparison, respecting the seed asymmetry: AS is one draw per (model, solver),
        each MOEA several, so the MOEAs are reduced to a median before anything is compared.

        ALGORITHMS decides which baselines take part.
    """
    import pandas as pd

    if not os.path.exists(out):
        raise SystemExit(f"no {out} yet - run the scoring pass first")
    full = pd.read_csv(out)
    t = full[full.family.eq("as") | full.method.isin(ALGORITHMS)]
    print(f"{len(t)} runs: {(t.family == 'as').sum()} as, {(t.family == 'moea').sum()} moea "
          f"({', '.join(ALGORITHMS)})")
    left = sorted(set(full.loc[full.family == "moea", "method"]) - set(ALGORITHMS))
    if left:
        print(f"  scored but outside ALGORITHMS, so no part of this comparison: {', '.join(left)}")
    print()

    # the rank view, and the only view: Robust to the reference box in a way raw hv is not
    cell = ["problem", "n_obj"]
    d = ranked(t)
    sizes = d.groupby(cell).size()
    n, n_as = int(sizes.iloc[0]), int(d[d.family == "as"].groupby(cell).size().iloc[0])
    print(f"RANK within each cell: {n} configurations - {n_as} AS (model, solver) pairs and "
          f"{n - n_as} MOEA medians; chance rank {(n + 1) / 2:.1f}")
    if sizes.nunique() > 1:
        print(f"  WARNING cells hold {sizes.min()}-{sizes.max()} configurations, so the ranks "
              f"are not on one scale and the chance rank is only indicative")
    r = d.groupby("config")["rank"].agg(["mean", "median"]).round(2).sort_values("mean")
    r["ranks_1st"] = (d[d["rank"] == 1].config.value_counts()
                       .reindex(r.index, fill_value = 0).astype(int))
    print(r.to_string(), "\n")

    print(f"mean rank by objective count, the same {n} configurations:")
    tab = d.pivot_table(index = "config", columns = "n_obj", values = "rank", aggfunc = "mean")
    tab["all m"] = d.groupby("config")["rank"].mean()
    print(tab.round(2).sort_values("all m").to_string(), "\n")

    # cell by cell for each baseline: a mean hides a MOEA that either wins the cell or places last
    best_as = d[d.family == "as"].groupby(cell)["rank"].min()
    for algo in ALGORITHMS:
        mo = d[d.method == algo]
        if mo.empty:
            continue
        beats = int((mo.set_index(cell)["rank"] < best_as.reindex(mo.set_index(cell).index)).sum())
        print(f"{algo}: ranks 1st in {int((mo['rank'] == 1).sum())} of {len(mo)} cells, and beats "
              f"every one of the {n_as} AS configurations in {beats}; AS takes the other "
              f"{len(mo) - beats}")
        print(f"  mean rank by m: {mo.groupby('n_obj')['rank'].mean().round(2).to_dict()}\n")

    for metric, (ascending, sense) in REPORTED.items():
        tab = d.pivot_table(index = "config", columns = "n_obj", values = metric,
                            aggfunc = "median")
        levels = list(tab.columns)
        tab[levels] = tab[levels].round(0 if metric == "archive" else 4)
        tab["mean rank"] = (d.groupby(cell)[metric].rank(ascending = ascending)
                             .groupby(d.config).mean().round(2))
        print(f"median {metric} per objective count, the same {n} configurations ({sense}):")
        print(tab.sort_values("mean rank").to_string(), "\n")


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
