"""
    One configuration of the benchmark - one model, one problem, one seed.

        python run_experiment.py --conf job.conf --count       how many tasks the grid expands to
        python run_experiment.py --conf job.conf --list        the task list, each line led by its index
        python run_experiment.py --conf job.conf --index 7     run task 7 and write its result
        python run_experiment.py --conf job.conf --aggregate   merge the rows -> results/summary.csv

    A task writes results/<model>_<problem>_d<n_dim>_m<n_obj>_s<seed>.{csv,npz} and the matching
    logs/trace_<...>.log - what each iteration learnt and the evaluations it made. The row carries
    hv_own, the hypervolume on that run's own front; the comparable one needs the reference point
    the models of a seed share, so --aggregate adds it afterwards. A task that raises still
    writes its row, with status = the exception, so one failure leaves no hole in the table.
"""
import argparse, io, contextlib, glob, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))        # the models live one level up

import numpy as np
import pandas as pd

MODELS = {"baseline": "LinearModel", "cas": "ChoquetModel", "las": "LipschitzModel"}

# Objectives fixed by a definition rather than by obj_dims; anything outside this table and outside
# the dtlz group has two. Fixed variable counts are not listed here - the classes declare N_VAR.
N_OBJ = {"ZDT2_Combined": 4}


def read_conf(path):
    "key = value lines; values are comma separated lists; # starts a comment"
    conf = {}
    with open(path) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, value = (part.strip() for part in line.split("=", 1))
            conf[key] = [v.strip() for v in value.split(",") if v.strip()]
    return conf


def combos(conf):
    """
        combo_<name> = A, B, C in the conf is one ExtendedProblem joining those problems in that
        order, and combo_<name>_order = 2, 0, 3 keeps and reorders its objectives. The key is the
        problem's name everywhere else - in --list, and in the file names.
    """
    return {k: v for k, v in conf.items() if k.startswith("combo_") and not k.endswith("_order")}


def build_problem(name, n_dim, n_obj, conf):
    "one problem instance: a combination, a DTLZ (the only group taking n_obj), or n_dim alone"
    import tests

    parts = combos(conf).get(name)
    if parts is None:
        build = getattr(tests, name)
        return build(n_dim = n_dim, n_obj = n_obj) if name in conf["dtlz"] else build(n_dim = n_dim)

    joined = []
    for part in parts:      # a part fixing its own size goes in bare, the rest take this n_dim
        cls = getattr(tests, part)
        joined.append(cls if getattr(cls, "N_VAR", None) else cls(n_dim = n_dim))
    return tests.ExtendedProblem(problems = joined,
                                 order = [int(v) for v in conf.get(name + "_order", [])] or None)


def tasks(conf):
    """
        Every (model, problem, n_dim, n_obj, seed) the grid covers, in a fixed order,
        extra and combined come last so that adding a problem renumbers nothing.
        Only DTLZ takes n_obj, and it needs n_dim > n_obj;
        a problem declaring N_VAR ignores the rest, so it is listed once,
        and a combination does the same when every part of it does.
    """
    import tests                    # for the sizes the problem classes fix themselves

    dec_dims, obj_dims = ([int(v) for v in conf[k]] for k in ("dec_dims", "obj_dims"))
    seeds = [int(v) for v in conf["seeds"]]
    problems = (conf["zdt"] + conf["dtlz"] + conf.get("extra", []) + conf.get("combined", [])
                + list(combos(conf)))

    out = []
    for problem in problems:
        parts = combos(conf).get(problem)
        sizes = [getattr(getattr(tests, p), "N_VAR", None) for p in (parts or [problem])]
        fixed = max(sizes) if all(sizes) else None
        for n_dim in ([fixed] if fixed else dec_dims):
            # a combination's objectives are its parts' put together, so they are counted by building it
            for n_obj in (obj_dims if problem in conf["dtlz"] else
                          [build_problem(problem, n_dim, None, conf).n_obj] if parts else
                          [N_OBJ.get(problem, 2)]):
                if problem in conf["dtlz"] and n_dim <= n_obj:
                    continue
                out += [(model, problem, n_dim, n_obj, seed)
                        for model in conf["models"] for seed in seeds]
    return out


def stem(task):
    return "{}_{}_d{}_m{}_s{}".format(*task)


def finite(A):
    "rows free of nan/inf - an unbounded problem can return either"
    A = np.atleast_2d(A)
    return A[np.all(np.isfinite(A), 1)]


def run(task, conf, results_dir, logs_dir = None, trace_rows = 50):
    import models as model_module
    from pymoo.indicators.hv import HV

    model_key, problem_name, n_dim, n_obj, seed = task
    max_iter = int(conf["max_iter"][0])

    problem = build_problem(problem_name, n_dim, n_obj, conf)
    extra = (dict(method = int(conf["cas_method"][0])) if model_key == "cas" else
             dict(type = conf["las_type"][0]) if model_key == "las" else {})
    model = getattr(model_module, MODELS[model_key])(
        problem = problem, max_iter = max_iter, random_state = seed, **extra)

    row = dict(model = model_key, problem = problem_name, n_dim = n_dim, n_obj = n_obj, seed = seed,
               max_iter = max_iter, status = "ok")
    if problem.n_obj != n_obj:      # the file name carries the grid's n_obj, so say so when it lies
        row["status"] = f"n_obj mismatch: grid says {n_obj}, {problem_name} builds {problem.n_obj}"

    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):      # the solvers print; keep the log clean
            model.solve()
    except Exception as exc:
        row["status"] = f"{type(exc).__name__}: {exc}"[:200]
        traceback.print_exc()

    row["wallclock"] = round(time.perf_counter() - started, 2)
    row["n_func_eval"], row["n_iter"] = int(model.num_func_eval), int(model.iteration)
    row["archive"], row["hv_own"] = 0, np.nan

    if model.PF is not None and len(model.PF):
        PF, row["archive"] = finite(model.PF), len(finite(model.PF))
        # obj_hist is every point the search looked at, which plot_front.py draws behind the front;
        # the hypervolume reference comes from the front itself, as utils.find_nadir does
        np.savez_compressed(os.path.join(results_dir, stem(task) + ".npz"),
                            PF = PF, PS = model.PS, obj_hist = finite(model.obj_hist),
                            iter_hist = model.iter_hist, w_hist = model.w_hist,
                            scal_hist = np.asarray(model.scal_hist, dtype = float))
        if len(PF):
            # this run alone, so it reads straight away; --aggregate adds the one that compares
            row["hv_own"] = float(HV(ref_point = PF.max(0))(PF))

    if logs_dir is not None and len(model.scal_hist):
        # per iteration: the non-dominated points learn_weight() was given, what it learnt from
        # them (w_vec, v, b_int, the Lipschitz constant) and the evaluations that followed
        from utils import trace_log
        row["trace"] = trace_log(model, max_rows = trace_rows or None,
                                 path = os.path.join(logs_dir, "trace_" + stem(task) + ".log"))
    return row


def aggregate(results_dir):
    """
        Merge the per-task rows and add the hypervolume, which is defined for every task, the DTLZ
        runs past 3 objectives included. Its reference point is the one utils.evaluate() uses - the
        worst value per objective over the non-dominated points, find_nadir(use_all = False) -
        pooled over the models of one (problem, n_dim, n_obj, seed).

        One seed is one experiment: its models share a reference point, and seed 1's hypervolume is
        not on the scale of seed 2's. Compare models within a seed and aggregate across seeds by
        rank, which is why hv_rank is here, rather than by averaging the raw numbers.
    """
    from pymoo.indicators.hv import HV

    rows = [pd.read_csv(path) for path in sorted(glob.glob(os.path.join(results_dir, "*.csv")))
            if os.path.basename(path) != "summary.csv"]
    if not rows:
        raise SystemExit(f"no per-task csv files in {results_dir}")
    table = pd.concat(rows, ignore_index = True)

    archives = {key: np.load(path)["PF"]
                for key, path in ((tuple(r), os.path.join(results_dir, stem(tuple(r)) + ".npz"))
                                  for r in table[["model", "problem", "n_dim", "n_obj", "seed"]].values)
                if os.path.exists(path)}

    table["hypervolume"] = np.nan
    for group_key, group in table.groupby(["problem", "n_dim", "n_obj", "seed"]):
        keys = [k for k in archives if k[1:] == group_key]
        if not keys:
            continue
        # one box for the seed's models - the worst point of the pool sits on it and scores nothing
        ref = np.vstack([archives[k] for k in keys]).max(0)
        hv = HV(ref_point = ref)
        table.loc[group.index, "hv_ref"] = " ".join(f"{v:.6g}" for v in ref)
        table.loc[group.index, "hv_models"] = len(keys)     # 3 when the seed is complete
        table.loc[group.index, "hypervolume"] = [
            float(hv(archives[k])) if k in archives else np.nan
            for k in ((m, *group_key) for m in group.model)]

    table["hv_rank"] = table.groupby(["problem", "n_dim", "n_obj", "seed"])["hypervolume"] \
                            .rank(ascending = False, method = "min")      # 1 = largest, within a seed

    out = os.path.join(results_dir, "summary.csv")
    table.to_csv(out, index = False)
    return table, out


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--conf", default = os.path.join(HERE, "job.conf"))
    parser.add_argument("--results", default = None, help = "defaults to the results_dir in the conf")
    parser.add_argument("--index", type = int, help = "which task to run (0 based)")
    parser.add_argument("--count", action = "store_true", help = "print how many tasks there are")
    parser.add_argument("--list", action = "store_true",
                        help = "print every task, one per line, led by its 0 based index")
    parser.add_argument("--trace-rows", type = int, default = 50,
                        help = "points listed per iteration in the trace log; 0 writes all of them")
    parser.add_argument("--aggregate", action = "store_true",
                        help = "after the sweep: merge the rows and add the per-seed hypervolume")
    args = parser.parse_args()

    conf = read_conf(args.conf)
    grid = tasks(conf)
    results_dir = args.results or os.path.join(HERE, conf.get("results_dir", ["results"])[0])

    if args.count:
        print(len(grid))
    elif args.list:
        # the leading number is the --index / SLURM array id of that row
        width = len(str(len(grid) - 1))
        print("\n".join(f"{i:>{width}} " + " ".join(str(part) for part in task)
                        for i, task in enumerate(grid)))
    elif args.aggregate:
        table, out = aggregate(results_dir)
        print(f"{len(table)} rows -> {out}")
        # the raw hypervolumes are on a per-seed scale, so only the rank averages over seeds
        print(table.groupby("model")[["hv_rank"]].mean().to_string())
    elif args.index is None:
        parser.error("give --index, --count, --list or --aggregate")
    elif not 0 <= args.index < len(grid):
        parser.error(f"index {args.index} is outside 0..{len(grid) - 1}")
    else:
        logs_dir = os.path.join(HERE, "logs")       # SLURM needs it too, for its own -o file
        for directory in (results_dir, logs_dir):
            os.makedirs(directory, exist_ok = True)

        task = grid[args.index]
        print(f"task {args.index}: {' '.join(str(part) for part in task)}", flush = True)
        row = run(task, conf, results_dir, logs_dir, args.trace_rows)
        pd.DataFrame([row]).to_csv(os.path.join(results_dir, stem(task) + ".csv"), index = False)
        print({k: row[k] for k in ("status", "hv_own", "archive", "n_func_eval", "wallclock")},
              flush = True)
        if "trace" in row:
            print("trace written to", row["trace"], flush = True)


if __name__ == "__main__":
    main()
