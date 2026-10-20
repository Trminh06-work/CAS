"""
    Plot a task's result from its .npz.

        python plot_from_npz.py results/cas_ZDT1_d10_m2_s0.npz
        python plot_from_npz.py results/las_ZDT1_d10_m3_s0.npz --kind learning --level-sets
        python plot_from_npz.py results/baseline_ZDT1_d10_m2_s0.npz --kind true_pf
        python plot_from_npz.py results/cas_ZDT1_d10_m2_s0.npz --save front.png

    The .npz does not contain models' states, hence refitting the models.
"""
import argparse, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np
import models
from run_experiment import MODELS, read_conf, build_problem
from utils import visualise


def realign(model):
    if len(model.obj_hist) == len(model.iter_hist):
        return True
    keep = np.isfinite(model.scal_hist)
    if keep.sum() != len(model.obj_hist):
        return False
    model.iter_hist, model.w_hist, model.scal_hist = (a[keep] for a in
                                                      (model.iter_hist, model.w_hist, model.scal_hist))
    return True


def replay_states(model):
    """
        Refit the scalarising function of every iteration, as solve() did.
    """
    obj, steps, W, final = model.obj_hist, np.asarray(model.iter_hist), model.w_hist, model.PF
    model.state_hist = {0: model._state()}          # before any fit - the plain weighted sum
    gap = 0.0
    for it in range(1, int(steps.max()) + 1):
        Q = obj[steps <= it - 1]
        model.PF = Q[model.non_dominated(Q)]
        model._worst = model.PF.max(0)
        model.y = np.ones(len(model.PF))
        w = model.learn_weight()
        model.state_hist[it] = model._state()
        if np.any(steps == it):
            gap = max(gap, float(np.abs(w - W[steps == it][0]).max()))
    model.PF = final
    return gap


def main():
    parser = argparse.ArgumentParser(description = __doc__,
                                     formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", help = "a results/<task>.npz written by run_experiment.py")
    parser.add_argument("--kind", choices = ["objective", "learning", "true_pf"], default = "objective")
    parser.add_argument("--level-sets", action = "store_true",
                        help = "learning only: overlay each iteration's scalarising function")
    parser.add_argument("--conf", default = os.path.join(HERE, "job.conf"),
                        help = "the conf the task ran with - combo_ problems and model settings")
    parser.add_argument("--save", help = "write the figure here instead of opening a window "
                                         "(learning is plotly: .html, or an image if kaleido is installed)")
    args = parser.parse_args()

    # <model>_<problem>_d<n_dim>_m<n_obj>_s<seed>, as run_experiment.stem() writes it
    name = os.path.basename(args.npz).removesuffix(".npz")
    match = re.fullmatch(rf"({'|'.join(MODELS)})_(.+)_d(\d+)_m(\d+)_s(\d+)", name)
    if match is None:
        parser.error(f"{name}: not a <model>_<problem>_d<n>_m<n>_s<n> file name")
    model_key, problem_name, n_dim, n_obj, seed = match.groups()

    # the model as run_experiment.run() builds it - the replayed fit depends on these settings
    conf = read_conf(args.conf)
    extra = (dict(method = int(conf["cas_method"][0])) if model_key == "cas" else
             dict(type = conf["las_type"][0]) if model_key == "las" else {})
    problem = build_problem(problem_name, int(n_dim), int(n_obj), conf)
    model = getattr(models, MODELS[model_key])(problem = problem, random_state = int(seed), **extra)

    data = np.load(args.npz)
    for key in data.files:
        setattr(model, key, data[key])

    if args.kind != "true_pf" and not realign(model):
        if args.kind == "learning":
            parser.error(f"obj_hist has {len(model.obj_hist)} rows but iter_hist {len(model.iter_hist)},"
                         " and scal_hist does not tell which were dropped")
        model.iter_hist = np.zeros(len(model.obj_hist), int)   # the objective view never reads it

    if args.kind == "learning":
        print(f"replaying {int(np.max(model.iter_hist))} fits - may take long", flush = True)
        gap = replay_states(model)
        if gap > 1e-6:
            print(f"warning: the replayed weights differ from the recorded ones by up to {gap:.3g}")

    fig = visualise(model, args.kind, show = not args.save, plot_scalarisation = args.level_sets)

    if args.save:
        if hasattr(fig, "write_html"):          # plotly
            fig.write_html(args.save) if args.save.endswith(".html") else fig.write_image(args.save)
        else:
            fig.savefig(args.save, dpi = 150, bbox_inches = "tight")
        print("written to", args.save)


if __name__ == "__main__":
    main()
