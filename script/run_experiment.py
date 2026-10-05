"""
    One configuration of the benchmark - one model, one problem, one seed - for a SLURM array task.

        python run_experiment.py --conf job.conf --count          how many tasks the grid expands to
        python run_experiment.py --conf job.conf --list           the expanded task list, one per line
        python run_experiment.py --conf job.conf --index 7        run task 7 and write its result

    Each task writes results/<model>_<problem>_d<n_dim>_m<n_obj>_s<seed>.csv (one row, with a header)
    and the archive it found as the matching .npz, so the sweep can be concatenated afterwards with

        python -c "import pandas as pd, glob; pd.concat(map(pd.read_csv, glob.glob('results/*.csv'))).to_csv('all.csv', index=False)"

    A task that raises still writes its row, with status = the exception, so one failure does not
    leave a hole in the table.
"""
import argparse, io, contextlib, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))        # the models live one level up

import numpy as np
import pandas as pd

MODELS = {"baseline": "LinearModel", "cas": "ChoquetModel", "las": "LipschitzModel"}


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


def tasks(conf):
    """
        Every (model, problem, n_dim, n_obj, seed) the grid covers, in a fixed order so that a task
        index always means the same thing. ZDT is 2-objective by construction, and DTLZ needs
        n_dim > n_obj to leave tail variables for g.
    """
    dec_dims = [int(v) for v in conf["dec_dims"]]
    obj_dims = [int(v) for v in conf["obj_dims"]]
    seeds = [int(v) for v in conf["seeds"]]

    out = []
    for problem in conf["zdt"] + conf["dtlz"]:
        for n_dim in dec_dims:
            for n_obj in ([2] if problem in conf["zdt"] else obj_dims):
                if problem in conf["dtlz"] and n_dim <= n_obj:
                    continue
                for model in conf["models"]:
                    for seed in seeds:
                        out.append((model, problem, n_dim, n_obj, seed))
    return out


def run(task, conf, results_dir):
    import models as model_module
    import pymoo_tests
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.hv import HV

    model_key, problem_name, n_dim, n_obj, seed = task
    max_iter = int(conf["max_iter"][0])

    build = getattr(pymoo_tests, problem_name)
    problem = build(n_dim = n_dim) if problem_name.startswith("ZDT") else build(n_dim = n_dim, n_obj = n_obj)

    extra = {}
    if model_key == "cas":
        extra = dict(method = int(conf["cas_method"][0]))
    elif model_key == "las":
        extra = dict(type = conf["las_type"][0])
    model = getattr(model_module, MODELS[model_key])(
        problem = problem, max_iter = max_iter, random_state = seed, **extra)

    row = dict(model = model_key, problem = problem_name, n_dim = n_dim, n_obj = n_obj, seed = seed,
               max_iter = max_iter, status = "ok")
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):      # the solvers print; keep the log clean
            model.solve()
    except Exception as exc:
        row["status"] = f"{type(exc).__name__}: {exc}"[:200]
        traceback.print_exc()

    row["wallclock"] = round(time.perf_counter() - started, 2)
    row["n_func_eval"] = int(model.num_func_eval)
    row["n_iter"] = int(model.iteration)

    row["archive"], row["igd"], row["hypervolume"] = 0, np.nan, np.nan
    if model.PF is not None and len(model.PF):
        PF = np.atleast_2d(model.PF)
        PF = PF[np.all(np.isfinite(PF), 1)]
        row["archive"] = len(PF)
        np.savez_compressed(os.path.join(results_dir, stem(task) + ".npz"), PF = PF, PS = model.PS)
        try:
            # the reference point and the front come from the problem, not from the run, so these
            # numbers compare across models and seeds
            true = problem.pareto_front()
            row["igd"] = float(IGD(true)(PF))
            row["hypervolume"] = float(HV(ref_point = 1.1 * true.max(0) + 0.1)(PF))
        except Exception as exc:
            # pymoo cannot build a DTLZ front beyond 3 objectives without reference directions; the
            # run itself is still fine, and --aggregate scores it against a reference taken from the
            # archives of that configuration
            row["status"] += f" (no analytical front: {type(exc).__name__})"

    return row


def aggregate(results_dir):
    """
        Collect the per-task rows and add a hypervolume every task can be compared on: for each
        (problem, n_dim, n_obj) the reference point is the worst corner over every archive of that
        configuration, so models and seeds share it even where no analytical front exists.
    """
    import glob
    from pymoo.indicators.hv import HV

    rows = [pd.read_csv(path) for path in sorted(glob.glob(os.path.join(results_dir, "*.csv")))
            if os.path.basename(path) != "summary.csv"]
    if not rows:
        raise SystemExit(f"no per-task csv files in {results_dir}")
    table = pd.concat(rows, ignore_index = True)

    archives = {}
    for _, r in table.iterrows():
        path = os.path.join(results_dir, f"{r.model}_{r.problem}_d{r.n_dim}_m{r.n_obj}_s{r.seed}.npz")
        if os.path.exists(path):
            archives[(r.model, r.problem, r.n_dim, r.n_obj, r.seed)] = np.load(path)["PF"]

    table["hv_shared"] = np.nan
    for (problem, n_dim, n_obj), group in table.groupby(["problem", "n_dim", "n_obj"]):
        pile = [archives[k] for k in archives if k[1:4] == (problem, n_dim, n_obj)]
        if not pile:
            continue
        ref = np.vstack(pile).max(0) * 1.1 + 0.1
        hv = HV(ref_point = ref)
        for i, r in group.iterrows():
            PF = archives.get((r.model, problem, n_dim, n_obj, r.seed))
            if PF is not None:
                table.loc[i, "hv_shared"] = float(hv(PF))

    out = os.path.join(results_dir, "summary.csv")
    table.to_csv(out, index = False)
    return table, out


def stem(task):
    model_key, problem_name, n_dim, n_obj, seed = task
    return f"{model_key}_{problem_name}_d{n_dim}_m{n_obj}_s{seed}"


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--conf", default = os.path.join(HERE, "job.conf"))
    parser.add_argument("--results", default = None, help = "defaults to the results_dir in the conf")
    parser.add_argument("--index", type = int, help = "which task to run (0 based)")
    parser.add_argument("--count", action = "store_true", help = "print how many tasks there are")
    parser.add_argument("--list", action = "store_true", help = "print every task, one per line")
    parser.add_argument("--aggregate", action = "store_true",
                        help = "after the sweep: merge the rows and add a shared-reference hypervolume")
    args = parser.parse_args()

    conf = read_conf(args.conf)
    grid = tasks(conf)

    if args.count:
        print(len(grid))
        return
    if args.list:
        print("\n".join(" ".join(str(part) for part in task) for task in grid))
        return
    if args.aggregate:
        results_dir = args.results or os.path.join(HERE, conf.get("results_dir", ["results"])[0])
        table, out = aggregate(results_dir)
        print(f"{len(table)} rows -> {out}")
        return
    if args.index is None:
        parser.error("give --index, --count, --list or --aggregate")
    if not 0 <= args.index < len(grid):
        parser.error(f"index {args.index} is outside 0..{len(grid) - 1}")

    results_dir = args.results or os.path.join(HERE, conf.get("results_dir", ["results"])[0])
    os.makedirs(results_dir, exist_ok = True)

    task = grid[args.index]
    print(f"task {args.index}: {' '.join(str(part) for part in task)}", flush = True)
    row = run(task, conf, results_dir)
    pd.DataFrame([row]).to_csv(os.path.join(results_dir, stem(task) + ".csv"), index = False)
    print({k: row[k] for k in ("status", "igd", "hypervolume", "n_func_eval", "wallclock")}, flush = True)


if __name__ == "__main__":
    main()
