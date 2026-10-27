"""
    One configuration of the benchmark - one model, one problem, one seed, one solver.

        python run_experiment.py --conf job.conf --count       how many tasks the grid expands to
        python run_experiment.py --conf job.conf --list        the task list, each line led by its index
        python run_experiment.py --conf job.conf --index 7     run task 7 and write its result
        python run_experiment.py --conf job.conf --index 7 --live   ... and trace it as it goes
        python run_experiment.py --conf job.conf --aggregate   merge the rows -> results/summary.csv

    A task writes results/<model>_<problem>_d<n_dim>_m<n_obj>_s<seed>_<solver>.{csv,npz}, under --live
    the matching logs/trace_<...>.log - what each iteration learnt and the evaluations it made.

    5 pieces of information about Hypervolume:
        hv_own      : the HV of that run alone, same reference point (after running a task, i.e. --index #)
        hypervolume : comparable HV using one reference nadir point  (after running --aggregate)
        hv_ref      : the reference nadir point for `hypervolume`    (after running --aggregate)
        hv_rank     : ranking of models according to `hypervolume`   (after running --aggregate)
        hv_models   : number of models involved in HV calculation    (after running --aggregate)
"""
import argparse, inspect, io, contextlib, glob, os, re, sys, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))        # the models live one level up

import numpy as np
import pandas as pd
import models

MODELS = {"baseline": "LinearModel", "cas": "ChoquetModel", "las": "LipschitzModel"}


def read_conf(path):
    conf, key = {}, None
    with open(path) as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" in line:
                key, value = (part.strip() for part in line.split("=", 1))
                conf[key] = []
            elif key is None:
                raise SystemExit(f"{path}: a value before any key: {line}")
            else:
                value = line
            conf[key] += [v.strip() for v in value.split(",") if v.strip()]
    return conf


def combos(conf):
    """
        combo_<name> = A, B, C in the conf is one ExtendedProblem joining those problems in that
        order, and combo_<name>_order = 2, 0, 3 keeps and reorders its objectives. A part may be
        an aggregation, agg_<name>.
    """
    return {k: v for k, v in conf.items() if k.startswith("combo_") and not k.endswith("_order")}


def aggs(conf):
    """
        agg_<name> = Function(p), A, B in the conf is that aggregation function (agg.py) of the
        objectives of A, B, ... - one objective, so it is used as a part of a combination.
    """
    return {k: v for k, v in conf.items() if k.startswith("agg_") and not k.endswith("_weights")}


def leaves(name, conf):
    "the problem classes behind a name, through the combinations and aggregations it is made of"
    if name in combos(conf):
        parts = combos(conf)[name]
    elif name in aggs(conf):
        parts = aggs(conf)[name][1:]        # past the function
    else:
        return [name]
    return [leaf for part in parts for leaf in leaves(part, conf)]


def takes_n_obj(name, conf):
    "whether the grid chooses this problem's objectives - true of DTLZ, false of a combination"
    import tests
    return (name not in combos(conf)
            and "n_obj" in inspect.signature(getattr(tests, name)).parameters)


def build_problem(name, n_dim, n_obj, conf):
    "one problem instance: a combination, one the grid sizes in both spaces, or n_dim alone"
    import tests

    parts = combos(conf).get(name)
    if parts is None:
        build = getattr(tests, name)
        return build(n_dim = n_dim, n_obj = n_obj) if takes_n_obj(name, conf) else build(n_dim = n_dim)

    return tests.ExtendedProblem(problems = [build_part(part, n_dim, conf) for part in parts],
                                 order = [int(v) for v in conf.get(name + "_order", [])] or None)


def build_part(part, n_dim, conf):
    "a part of a combination or an aggregation: an aggregation, or a problem that takes this n_dim"
    import tests

    if part in aggs(conf):
        return build_agg(part, n_dim, conf)
    cls = getattr(tests, part)      # a part fixing its own size goes in bare
    return cls if getattr(cls, "N_VAR", None) else cls(n_dim = n_dim)


def build_agg(name, n_dim, conf):
    """
        The function of agg_<name> is written Function, Function(p) or Function(p).dual:
            p       lambda_param of SchweizerSklar and Frank
            p       r of QAM, the weighted power mean: g(t) = t^r, or log(t) if r = 0, 1 by default (p. 48)
            .dual   the t-conorm, N-dual of the t-norm with N(t) = 1 - t
    """
    import agg

    spec, *parts = aggs(conf)[name]
    match = re.fullmatch(r"(\w+)(?:\((.+)\))?(\.dual)?", spec.replace(" ", ""))
    if match is None:
        raise SystemExit(f"{name}: cannot read the function {spec!r}")
    cls, param = getattr(agg, match[1]), None if match[2] is None else float(match[2])
    problems = [build_part(part, n_dim, conf) for part in parts]

    if cls is agg.QAM:
        r = 1.0 if param is None else param
        gen, inv_gen = (np.log, np.exp) if r == 0 else (lambda t: t ** r, lambda s: s ** (1 / r))
        w = conf.get(name + "_weights")
        func = cls(problems, gen = gen, inv_gen = inv_gen, w_vec = [float(v) for v in w] if w else None)
    else:
        func = cls(problems) if param is None else cls(problems, param)
    return func.dual() if match[3] else func


def tasks(conf):
    """
        Every (model, problem, n_dim, n_obj, seed, solver) the grid covers,
        in the order the conf lists the problems.

        - Problems with fixed n_dim: may take dec_dims in conf, but ignore during evaluations
        - Problems with fixed n_obj: ignore obj_dims in conf
    """
    import tests                    # for the sizes the problem classes fix themselves

    dec_dims, obj_dims = ([int(v) for v in conf[k]] for k in ("dec_dims", "obj_dims"))
    seeds, solvers = [int(v) for v in conf["seeds"]], conf["solvers"]

    out = []
    for problem in conf["problems"] + list(combos(conf)):
        if problem in aggs(conf):
            raise SystemExit(f"{problem} is one objective: use it in a combination, e.g. combo_<name> = ZDT2, {problem}")
        sizes = [getattr(getattr(tests, p), "N_VAR", None) for p in leaves(problem, conf)]
        fixed = max(sizes) if all(sizes) else None
        chosen = takes_n_obj(problem, conf)
        for n_dim in ([fixed] if fixed else dec_dims):
            # a combination's objectives are its parts' put together, so they are counted by building it
            for n_obj in (obj_dims if chosen else [build_problem(problem, n_dim, None, conf).n_obj]):
                if chosen and n_dim <= n_obj:
                    continue
                out += [(model, problem, n_dim, n_obj, seed, solver)
                        for model in conf["models"] for seed in seeds for solver in solvers]
    return out


def stem(task):
    # the solver goes last, so aggregate() can still match a key to its group by dropping the model
    return "{}_{}_d{}_m{}_s{}_{}".format(*task)


def finite(A):
    "rows free of nan/inf - an unbounded problem can return either"
    A = np.atleast_2d(A)
    return A[np.all(np.isfinite(A), 1)]


def ref_point(problem):
    """
        The box the hypervolume is measured in: the problem's own front, widened by 10% of its range.

        None when the problem has no analytical front.
    """
    try:
        pf = finite(problem.pareto_front())
        return pf.max(0) + 0.1 * (pf.max(0) - pf.min(0))        # +10% to widen the region
    except Exception:
        return None


def run(task, conf, results_dir, logs_dir = None, trace_rows = 50, live = False):
    from pymoo.indicators.hv import HV

    model_key, problem_name, n_dim, n_obj, seed, solver = task
    max_iter = int(conf["max_iter"][0])
    max_eval = int(conf["max_eval"][0])

    problem = build_problem(problem_name, n_dim, n_obj, conf)
    extra = (dict(method = int(conf["cas_method"][0])) if model_key == "cas" else
             dict(type = conf["las_type"][0]) if model_key == "las" else {})
    model = getattr(models, MODELS[model_key])(
        problem = problem, max_iter = max_iter, max_eval = max_eval, random_state = seed,
        solver = solver, **extra)

    row = dict(model = model_key, problem = problem_name, n_dim = n_dim, n_obj = n_obj, seed = seed,
               solver = solver, max_iter = max_iter, max_eval = max_eval, status = "ok")
    if problem.n_obj != n_obj:
        row["status"] = f"n_obj mismatch: grid says {n_obj}, {problem_name} builds {problem.n_obj}"

    # logs/trace_<model>_<problem>_d<n_dim>_m<n_obj>_s<seed>.log
    trace_path = None if logs_dir is None or not live else os.path.join(
        logs_dir, "trace_" + stem(task) + ".log")
    callback = None
    if trace_path is not None:
        from utils import live_trace
        callback = live_trace(trace_path, trace_rows or None)

    try:
        with contextlib.redirect_stdout(io.StringIO()):      # the solvers print; keep the log clean
            model.solve(callback)
    except Exception as exc:
        row["status"] = f"{type(exc).__name__}: {exc}"[:200]
        traceback.print_exc()

    row["wallclock"] = round(model.wallclock_time, 4)        # solve() times itself, failures too
    if trace_path is not None and os.path.exists(trace_path):
        row["trace"] = trace_path
    row["n_func_eval"], row["n_iter"] = int(model.num_func_eval), int(model.iteration)
    row["archive"], row["hv_own"] = 0, np.nan

    if model.PF is not None and len(model.PF):
        PF, row["archive"] = finite(model.PF), len(finite(model.PF))
        ref = ref_point(problem)        # saved only when it is the problem's, so shared by construction
        np.savez_compressed(os.path.join(results_dir, stem(task) + ".npz"),
                            PF = PF, PS = model.PS, obj_hist = finite(model.obj_hist),
                            iter_hist = model.iter_hist, w_hist = model.w_hist,
                            scal_hist = np.asarray(model.scal_hist, dtype = float),
                            **({} if ref is None else dict(ref = ref)))
        if len(PF):
            # hv_own: this run alone, in the problem's own box when it has one, else its own
            row["hv_own"] = float(HV(ref_point = PF.max(0) if ref is None else ref)(PF))

    return row


def aggregate(results_dir):
    """
        A task is: ("problem", "n_dim", "n_obj", "seed", "solver") -> comparable among models

        Merge the per-task rows and compute per-task hypervolume, against the reference point
        run() saved with the archive: the problem's own front widened by 10%, or - for a problem
        with no analytical front - the worst value per objective over the non-dominated points,
        pooled over the models of one (problem, n_dim, n_obj, seed, solver).

        One seed is one experiment: its models share a reference point.
        Compare models within a seed and aggregate across seeds by rank, i.e. hv_rank,
        rather than by averaging across the seeds.
    """
    from pymoo.indicators.hv import HV

    rows = [pd.read_csv(path) for path in sorted(glob.glob(os.path.join(results_dir, "*.csv")))
            if os.path.basename(path) != "summary.csv"]
    if not rows:
        raise SystemExit(f"no per-task csv files in {results_dir}")
    table = pd.concat(rows, ignore_index = True)

    def load(path):
        with np.load(path) as handle:       # the archive, and the box run() measured it in
            return handle["PF"], (handle["ref"] if "ref" in handle else None)

    saved = {key: load(path)
             for key, path in ((tuple(r), os.path.join(results_dir, stem(tuple(r)) + ".npz"))
                               for r in table[["model", "problem", "n_dim", "n_obj", "seed", "solver"]].values)
             if os.path.exists(path)}
    archives = {key: PF for key, (PF, _) in saved.items()}

    table["hypervolume"] = np.nan
    # group_key does not include the models
    # group_key: ("problem", "n_dim", "n_obj", "seed", "solver")
    for group_key, group in table.groupby(["problem", "n_dim", "n_obj", "seed", "solver"]):
        keys = [k for k in archives if k[1:] == group_key]
        if not keys:
            continue
        # every model of a group shares one problem, so they all saved the same reference
        ref = next((saved[k][1] for k in keys if saved[k][1] is not None),
                   np.vstack([archives[k] for k in keys]).max(0))
        hv = HV(ref_point = ref)
        table.loc[group.index, "hv_ref"] = " ".join(f"{v:.6g}" for v in ref)
        table.loc[group.index, "hv_models"] = len(keys)
        table.loc[group.index, "hypervolume"] = [
            float(hv(archives[k])) if k in archives else np.nan
            for k in ((m, *group_key) for m in group.model)]

    table["hv_rank"] = table.groupby(["problem", "n_dim", "n_obj", "seed", "solver"])["hypervolume"] \
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
    parser.add_argument("--live", action = "store_true",
                        help = "trace the run into logs/trace_<task>.log while it goes")
    parser.add_argument("--trace-rows", type = int, default = 0,
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
        logs_dir = os.path.join(HERE, "logs")
        for directory in (results_dir, logs_dir):
            os.makedirs(directory, exist_ok = True)

        task = grid[args.index]
        print(f"task {args.index}: {' '.join(str(part) for part in task)}", flush = True)
        row = run(task, conf, results_dir, logs_dir, args.trace_rows, args.live)
        pd.DataFrame([row]).to_csv(os.path.join(results_dir, stem(task) + ".csv"), index = False)
        print({k: row[k] for k in ("status", "hv_own", "archive", "n_func_eval", "wallclock")}, flush = True)
        if "trace" in row:
            print("trace written to", row["trace"], flush = True)


if __name__ == "__main__":
    main()
