import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
from typing import List
from models import BaseModel
import pandas as pd
# from pymoo.


def evaluate(models_arr: List[BaseModel]):
    """
        Compare solved models on one shared reference point.
    """
    if not models_arr:
        raise ValueError("No models to evaluate")
    if any(m.PF is None for m in models_arr):
        raise RuntimeError("Every model must be solved first - call solve()")
    if len({m.PF_dim for m in models_arr}) != 1:
        raise ValueError("Models must share the same number of objectives")

    def finite(A):
        "Rows of A that are free of nan/inf - an unbounded problem can return either"
        A = np.atleast_2d(A)
        return A[np.all(np.isfinite(A), 1)]


    def find_nadir(use_problem_nadir: bool = True, in_space: str = "objective", use_all: bool = False):
        """
            Either use problem's PF or model.obj_hist / model.dec_hist or model.PF / model.PS to find nadir point

            Parameters:
                    use_problem_nadir (bool, optional): use problem's PF to obtain nadir point, if available
                    in_space (str, optional): the space in which it operates, either `"objective"` or `"decision"`. Defaults to `"objective"`.
                    use_all (bool, optional): decide to select the nadir point from either all points found or only non-dominated points
        """
        # use problem's PF -> nadir point, if unavailable, fallback to default nadir selection
        if use_problem_nadir:
            try:
                if in_space == "objective":
                    H = np.vstack([finite(m.problem.pareto_front()) for m in models_arr])
                else:
                    H = np.vstack([finite(m.problem.pareto_set()) for m in models_arr])
                return H.max(0) + 0.1 * (H.max(0) - H.min(0))   # +10% to widen the regions
            except Exception:
                print("No available problem's PF, fallback to default nadir selection")

        # worst value per objective over every evaluation any model made, so the reference is shared and the hypervolumes compare
        if use_all:
            # Use all evaluated points to find the nadir
            if in_space == "objective":
                H = np.vstack([finite(m.obj_hist) for m in models_arr])
            else:
                H = np.vstack([finite(m.dec_hist) for m in models_arr])
        else:
            # Only use the non-dominated points to find the nadir
            if in_space == "objective":
                H = np.vstack([finite(m.PF) for m in models_arr])
            else:
                H = np.vstack([finite(m.PS) for m in models_arr])
        return H.max(0)


    def summary():
        """
            Return a pd.DataFrame, rows are models - columns are performance metrics

            Hypervolume - use pymoo
            Number of function evaluations
        """
        from pymoo.indicators.hv import HV
        nadir = find_nadir(in_space = "objective")
        hv = HV(ref_point = nadir)
        rows = [
            dict(
                hypervolume = hv(finite(m.PF)),
                n_func_eval = m.num_func_eval,
                # n_non_dominated = len(finite(m.PF)), n_iter = int(m.iteration),
                n_non_dominated = int(BaseModel.non_dominated(finite(m.obj_hist), "strongly").sum()),
                n_iter = int(m.iteration),
                wallclock_time = m.wallclock_time
            )
            for m in models_arr
        ]
        table = pd.DataFrame(rows, index = [type(m).__name__ for m in models_arr])
        # the hypervolumes above mean nothing without the box they were measured in - keep it with
        # them, so it can be saved and passed to HV(ref_point = ...) to compare a later run
        table.attrs["nadir"] = nadir
        return table


    return summary()



def trace_learn(model: BaseModel, it: int, max_rows: int = None) -> str:
    """
        The head of one iteration's block: the non-dominated points learn_weight() was handed and
        what it learnt from them.
    """
    names = [f"f{i + 1}" for i in range(model.PF_dim)]
    # live, solve() may not have snapshotted this iteration yet - the model itself is that state
    state = model.state_hist.get(int(it)) or model._state()
    out = [f"\n=== iteration {int(it)}"]

    keep = model.pf_hist.get(int(it) - 1)       # the archive learn_weight() was handed here
    if keep is not None:
        obj = np.atleast_2d(model.obj_hist)
        scal, iters = np.asarray(model.scal_hist, dtype = float), np.asarray(model.iter_hist)
        pf_it, rows = obj[:len(keep)][keep], slice(None, max_rows)
        u = state["worst"] - pf_it[rows]
        fit = pd.DataFrame(np.hstack([pf_it[rows], u]),
                           columns = names + [f"u{i + 1}" for i in range(model.PF_dim)])
        fit.insert(0, "it", iters[:len(keep)][keep][rows])      # the block this row is repeated from
        fit["scalarised"] = scal[:len(keep)][keep][rows]        # exactly as that block printed it
        out += [f"  learn_weight input: {len(pf_it)} non-dominated points",
                fit.to_string(index = False)]

    for key, value in state.items():
        if key == "slip" and value is not None:         # the object is not readable on its own
            value = f"{value.type}, {value.npts} points, LC = {value.LipsConst:.6g}"
        out.append(f"  {key:10s}: {value}")
    return "\n".join(out) + "\n"


def trace_evals(model: BaseModel, it: int, max_rows: int = None) -> str:
    """
        The tail of one iteration's block: the evaluations the solver made under that fit, with
        the value it scored them - recorded in F() at the time, so already that iteration's own.
    """
    obj, dec = np.atleast_2d(model.obj_hist), np.atleast_2d(model.dec_hist)
    iters, scal = np.asarray(model.iter_hist), np.asarray(model.scal_hist, dtype = float)
    rows = (iters == it)

    table = pd.DataFrame(obj[rows][:max_rows], columns = [f"f{i + 1}" for i in range(model.PF_dim)])
    table["scalarised"] = scal[rows][:max_rows]
    table.insert(0, "x", [np.array2string(x, precision = 4, separator = ",",
                                          max_line_width = np.inf) for x in dec[rows][:max_rows]])
    return f"  {int(rows.sum())} evaluations\n" + table.to_string(index = False) + "\n"


def trace_header(model: BaseModel) -> str:
    # no evaluation count: each block carries its own, and a live header is written before the end
    return f"{type(model).__name__} on {type(model.problem).__name__} - seed {model.random_state}\n"


def live_trace(path: str, max_rows: int = None):
    first = True

    def write(model, phase):
        nonlocal first
        with open(path, "w" if first else "a") as fh:
            if first:
                fh.write(trace_header(model))
            fh.write((trace_learn if phase == "learn" else trace_evals)(
                model, model.iteration, max_rows))
        first = False

    return write


def visualise_weights(model: BaseModel, kind = "symmetric", show = True):
    """
        How the learnt weight vector moves from one outer iteration to the next.

        kind:
            "symmetric"  the signed per-component change w_k - w_(k-1), on a symlog axis so that
                            the direction of each move stays visible

            "absolute"   |w_k - w_(k-1)| per component, on a log axis

            "norm"       ||w_k - w_(k-1)||, a single curve - the overall size of the step

        show             render the figure before returning
    """
    if kind not in ("symmetric", "absolute", "norm"):
        raise ValueError(f"unknown kind: {kind!r}")
    if model.w_hist is None:
        raise RuntimeError("Nothing to visualise yet - call solve() first")

    # one weight per outer iteration, in the order it was learnt; np.unique(axis = 0) would sort
    # the rows lexicographically and lose the sequence entirely
    W, it = np.atleast_2d(model.w_hist), np.asarray(model.iter_hist)
    iters = np.unique(it)
    w_seq = np.vstack([W[np.argmax(it == k)] for k in iters])
    d = np.diff(w_seq, axis = 0)
    if len(d) == 0:
        raise RuntimeError("Need at least two iterations to show a change in the weights")

    sns.set_theme(style = "whitegrid")
    fig, ax = plt.subplots(figsize = (8, 5))

    if kind == "norm":
        ax.plot(iters[1:], np.linalg.norm(d, axis = 1), marker = "o", ms = 3, color = "crimson")
        ax.set_yscale("log")
        ax.set_ylabel(r"$\|w_k - w_{k-1}\|$")
    else:
        y = d if kind == "symmetric" else np.abs(d)
        for i in range(y.shape[1]):
            ax.plot(iters[1:], y[:, i], marker = "o", ms = 3, label = f"$w_{{{i + 1}}}$")
        # symlog keeps the sign readable while still spanning several decades
        ax.set_yscale("symlog", linthresh = 1e-5) if kind == "symmetric" else ax.set_yscale("log")
        ax.set_ylabel(r"$w_k - w_{k-1}$" if kind == "symmetric" else r"$|w_k - w_{k-1}|$")
        ax.legend(title = "component", bbox_to_anchor = (1.02, 1), loc = "upper left")

    ax.set(xlabel = "iteration $k$",
           title = f"{type(model).__name__} on {type(model.problem).__name__} - weight movement")
    ax.grid(True, alpha = 0.3)
    fig.tight_layout()

    if show:
        plt.show()
    return fig



def visualise(model, kind = "objective", show_true = True, plot_scalarisation = False,
              zoom = True, show_params = True, show = True):
    """
        Plot one view of a solved model, in the problem's own objective units.

        model                any BaseModel, after solve()

        More than 3 objectives cannot be scattered, so EVERY kind falls back to pymoo's radar
        plot, whatever was asked for: one panel per solution, over 5 points taken evenly from
        the archive and 5 from the analytical front.

        kind:
            "objective"  every evaluation (obj_hist) with the non-dominated archive (PF) highlighted and the true front behind them

            "learning"   a slider over the outer iterations showing only the non-dominated archive as it
                         stood at that iteration - every one of its points is drawn, and the axes are
                         refitted to them, so the axis numbers show how the archive rescales

            "true_pf"    the analytical front on its own

        show_true            draw the analytical front (no effect if the problem has none)

        plot_scalarisation   "learning" only: overlay the level sets of scalarise(., w_k) - ignore n_obj > 3

        show_params          "learning" only: a panel beside the plot with the parameters that
                             iteration was fitted with - the capacity and scaling vector for
                             ChoquetModel, the interpolant and its Lipschitz constant for
                             LipschitzModel, the weights otherwise

        zoom                 frame the archive and the true front instead of
                             every evaluation, so the front is not squashed into a corner by the
                             unconverged cloud; the points left outside are counted in the legend.
                             "learning" never leaves a point off-frame: zoom refits the axes to each
                             iteration, zoom = False holds one box that spans every iteration

        show                 render the figure before returning
    """
    if kind not in ("objective", "learning", "true_pf"):
        raise ValueError(f"unknown kind: {kind!r}")

    sns.set_theme(style = "whitegrid")
    only_true = kind == "true_pf"
    dim = model.PF_dim
    names = [f"$f_{{{i + 1}}}$" for i in range(dim)]     # matplotlib renders the TeX
    plain = [f"f{i + 1}" for i in range(dim)]            # plotly does not

    try:            # analytical front, when the test problem defines one
        true = np.atleast_2d(model.problem.pareto_front())
    except Exception:   # pymoo raises a bare Exception for DTLZ past 3 objectives
        true = None
    if only_true and true is None:
        raise RuntimeError(f"{type(model.problem).__name__} has no analytical Pareto front")
    if not only_true and model.obj_hist is None:
        raise RuntimeError("Nothing to visualise yet - call solve() first")

    # stored units -> real objective values; identity when the problem is plain minimisation
    to_true   = getattr(model.problem, "true", None)   or (lambda F: F)
    to_stored = getattr(model.problem, "stored", None) or (lambda F: F)
    if true is not None:
        true = to_true(true)

    # every evaluation made, tagged with the iteration that made it
    hist = np.atleast_2d(model.obj_hist) if model.obj_hist is not None else np.empty((0, dim))
    ok = np.all(np.isfinite(hist), 1)         # a problem with no declared box can return nan/inf
    stored_hist, steps = hist[ok], np.asarray(model.iter_hist)[ok]
    hist = to_true(stored_hist)               # what gets drawn

    # the box every view frames on: the final archive plus the analytical front, not the whole
    # cloud - computed once here so "objective" and "learning" cannot disagree
    front = np.atleast_2d(model.PF) if model.PF is not None and not only_true else np.empty((0, dim))
    front = to_true(front[np.all(np.isfinite(front), 1)])
    box = np.vstack([front] + ([true] if true is not None and show_true else []))
    zlo = zhi = None
    if zoom and not only_true and len(box):
        zlo, zhi = box.min(0), box.max(0)
        pad = 0.05 * np.where(zhi > zlo, zhi - zlo, 1.0)
        zlo, zhi = zlo - pad, zhi + pad

    # ---- more than 3 objectives: pymoo's radar, whatever kind was asked for ----
    if dim > 3:
        from pymoo.visualization.radar import Radar
        from pymoo.visualization.util import get_circle_points

        def sample(Q, n = 5):
            "n points spread evenly through Q - radar draws one panel per solution"
            Q = np.atleast_2d(Q)
            return Q if len(Q) <= n else Q[np.linspace(0, len(Q) - 1, n).astype(int)]

        sets = [] if only_true else [("non-dominated", sample(front), "royalblue")]
        if true is not None and (show_true or only_true):
            sets.append(("true PF", sample(true), "crimson"))
        if not sets or not len(sets[0][1]):
            raise RuntimeError("Nothing to visualise yet - call solve() first")

        Q = np.vstack([q for _, q, _ in sets])
        lo, hi = Q.min(0), Q.max(0)
        hi = np.where(hi > lo, hi, lo + 1.0)        # a flat objective would normalise to 0/0

        # pymoo writes the same title on every panel, so name each panel instead and head the
        # figure once; the figure has to grow with the panel count or the labels collide
        n_cols = max(len(q) for _, q, _ in sets)
        titles = [t for label, q, _ in sets
                  for t in [f"{label} {i + 1}" for i in range(len(q))] + [""] * (n_cols - len(q))]
        radar = Radar(bounds = [lo, hi], labels = plain, title = titles,
                      figsize = (2.6 * n_cols, 3.0 * len(sets)), tight_layout = True)
        for _, q, colour in sets:
            radar.add(q, color = colour, alpha = 0.5)
        radar.do()

        V = get_circle_points(dim)
        for ax, t in zip(np.array(radar.ax).ravel(), titles):
            # pymoo's plot_axes_lines/plot_radar_line take an ax but draw with plt.plot, so every
            # panel's spokes land on the last axes - clear them and draw each panel's own
            for line in list(ax.lines):
                line.remove()
            for x, y in V:
                ax.plot([0, x], [0, y], color = "black", lw = 0.5, alpha = 0.75)
            ax.plot(*np.vstack([V, V[:1]]).T, color = "black", lw = 0.5, alpha = 0.75)

            # the panels must share one frame, and pymoo leaves the aspect on datalim, which
            # silently overrides any limit set here
            ax.set_aspect("equal", adjustable = "box")
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            for label in ax.texts:                  # pymoo sets the size itself, so shrink after
                label.set_fontsize(8)
            ax.set_title(t, fontsize = 9)
            if not t:                               # a set shorter than the widest one
                ax.axis("off")
        radar.fig.suptitle(f"{type(model.problem).__name__} - objective space ({dim} objectives)",
                           fontsize = 12, y = 1.02)

        if show:
            plt.show()
        return radar.fig


    # ---- the learning process, one plotly animation frame per iteration ----
    if kind == "learning":
        iters, W = np.unique(steps), np.atleast_2d(model.w_hist)
        all_it = np.asarray(model.iter_hist)
        res = 60 if dim == 2 else 20                     # level-set grid resolution
        n_iso = 5                                        # isosurfaces drawn when dim == 3

        # the non-dominated archive as it stood at the end of each iteration
        fronts = {}
        for it in iters:
            Q = stored_hist[steps <= it]                 # non_dominated() assumes minimisation
            fronts[int(it)] = to_true(Q[model.non_dominated(Q)])

        # the analytical front shares the frame when it is drawn, so convergence stays readable
        ref = true if (true is not None and show_true and dim <= 3) else np.empty((0, dim))

        def frame_box(it):
            """
                The axes for one iteration: every non-dominated point it holds, plus the analytical
                front when that is drawn. Fitting the box to the points rather than clipping them is
                what makes the rescaling visible - read it off the axis ticks as the slider moves.
            """
            Q = np.vstack([fronts[int(it)], ref]) if len(ref) else fronts[int(it)]
            lo, hi = Q.min(0), Q.max(0)
            # an archive of one point - or of that point and its finite-difference twins, which
            # differ in the 7th digit - has no spread to frame, so fall back to its own magnitude
            # instead of zooming the axes down onto floating-point noise
            spread = np.maximum(np.abs(hi), 1.0)
            pad = 0.05 * np.where(hi - lo > 1e-6 * spread, hi - lo, spread)
            return lo - pad, hi + pad

        # zoom: refit the axes to each iteration. zoom = False: one box that spans them all.
        # Either way no point is ever left outside the frame.
        boxes = {int(it): frame_box(it) for it in iters}
        if not zoom:
            lo = np.min([b[0] for b in boxes.values()], 0)
            hi = np.max([b[1] for b in boxes.values()], 0)
            boxes = {int(it): (lo, hi) for it in iters}


        def wrap(cells, per_line = 4):
            """One line per `per_line` entries, so the panel stays narrow whatever n_obj is."""
            return ("<br>&nbsp;&nbsp;&nbsp;").join(", ".join(cells[i:i + per_line])
                                                   for i in range(0, len(cells), per_line))

        def numbers(a, per_line = 6):
            return wrap([f"{v:.4g}" for v in np.atleast_1d(np.asarray(a, float)).ravel()], per_line)

        def param_text(it, w):
            """
                What the model was fitted to at iteration `it`, read from the state solve() recorded:
                the capacity and scaling vector for the Choquet model, the interpolant and its
                constant for the Lipschitz one, the weights otherwise.
            """
            state = getattr(model, "state_hist", {}).get(int(it))
            lines = [f"<b>{type(model).__name__}</b> - iteration {int(it)}"]
            if state is None:                       # solved before state_hist existed
                lines.append("no parameters recorded for this iteration")
                return "<br>".join(lines)

            if "v" in state:                        # ChoquetModel
                lines.append(f"scaling w_vec: {numbers(state['w_vec'])}")
                v = state["v"]
                if v is None:
                    lines.append("capacity: not fitted yet - plain weighted sum")
                else:
                    v = np.asarray(v, float).ravel()
                    pairs = [(i, j) for i in range(dim) for j in range(i + 1, dim)]
                    if len(v) == dim + len(pairs):  # 2-additive, in Mobius form
                        lines.append("capacity m_i: "
                                     + wrap([f"m{i + 1}={v[i]:.4g}" for i in range(dim)]))
                        if pairs:
                            lines.append("interactions: "
                                         + wrap([f"m{i + 1}{j + 1}={v[dim + k]:.4g}"
                                                 for k, (i, j) in enumerate(pairs)], 3))
                    else:                           # any other fitting method - show it as it is
                        lines.append(f"capacity ({len(v)} values): {numbers(v)}")
                lines.append(f"intercept b: {float(state['b_int']):.4g}")

            elif "slip" in state:                   # LipschitzModel
                sl = state["slip"]
                if sl is None:
                    lines.append("interpolant: not fitted yet - plain weighted sum")
                else:
                    lines.append(f"Lipschitz constant: {float(sl.LipsConst):.6g}")
                    lines.append(f"interpolant: {sl.type}, {sl.npts} training points, knn {sl.num_k}")

            else:                                   # LinearModel, and anything else w is enough for
                lines.append(f"weights w: {numbers(w)}")

            # worst = state.get("worst")
            # lines.append("utility offset: " + ("none - step 0 minimises w.F" if worst is None
            #                                    else numbers(worst)))
            return "<br>".join(lines)

        def panel(it):
            """The parameter box beside the plot - it rides in the frame layout, so it follows the slider."""
            if not show_params:
                return []
            return [dict(text = param_text(it, W[np.argmax(all_it == it)]),
                         xref = "paper", yref = "paper", x = 1.02, y = 1.0,
                         xanchor = "left", yanchor = "top", align = "left", showarrow = False,
                         font = dict(family = "Courier New, monospace", size = 11),
                         bordercolor = "#999999", borderwidth = 1, borderpad = 6,
                         bgcolor = "rgba(255,255,255,0.85)")]

        def axes_layout(it):
            """The axis ranges of one frame - they travel with the frame so the slider rescales them."""
            lo, hi = boxes[int(it)]
            if dim == 3:
                return go.Layout(annotations = panel(it),
                                 scene = dict(zip(("xaxis", "yaxis", "zaxis"),
                                                  (dict(range = [lo[k], hi[k]]) for k in range(3)))))
            if dim == 2:
                return go.Layout(annotations = panel(it),
                                 xaxis = dict(range = [lo[0], hi[0]]), yaxis = dict(range = [lo[1], hi[1]]))
            return go.Layout(annotations = panel(it),
                             yaxis = dict(range = [lo.min(), hi.max()]))    # parallel coordinates

        def traces(it):
            w = W[np.argmax(all_it == it)]               # weight learnt at iteration it
            Q, (lo, hi) = fronts[int(it)], boxes[int(it)]
            xyz = (dict(zip("xyz", Q.T)) if dim <= 3 else
                   dict(x = np.tile(np.arange(dim), len(Q)), y = Q.ravel()))
            # every point of the archive is drawn; the count rides in the legend label, so it follows the slider
            out = [(go.Scatter3d if dim == 3 else go.Scatter)(
                **xyz, mode = "markers", name = f"non-dominated ({len(Q)})",
                marker = dict(size = 5.5 if dim == 2 else 2, color = "royalblue"))]

            # the level sets need nothing model-specific - just scalarise() on a grid
            if plot_scalarisation and dim <= 3:
                axes = [np.linspace(lo[k], hi[k], res) for k in range(dim)]
                G = np.stack(np.meshgrid(*axes, indexing = "ij"), -1).reshape(-1, dim)
                with model.as_of(it):       # the fit of THIS iteration, not the one the run ended on
                    v = np.array([model.scalarise(f, w) for f in to_stored(G)])
                # plotly spaces isosurfaces inclusively over [isomin, isomax], and a level at the
                # field's own min or max is empty - no cell straddles it - so keep them inside
                iso = np.linspace(v.min(), v.max(), n_iso + 2)[1:-1]
                out.append(
                    go.Contour(x = axes[0], y = axes[1], z = v.reshape(res, res).T,
                               colorscale = "Greys", opacity = 0.35, showscale = False,
                               showlegend = False, contours = dict(showlabels = True))
                    if dim == 2 else
                    go.Isosurface(x = G[:, 0], y = G[:, 1], z = G[:, 2], value = v,
                                  isomin = iso[0], isomax = iso[-1], surface_count = n_iso,
                                  caps = dict(x_show = False, y_show = False, z_show = False),
                                  colorscale = "Greys", opacity = 0.2,
                                  showscale = False, showlegend = False))
            return out

        fig = go.Figure()
        if true is not None and show_true and dim <= 3:
            fig.add_trace((go.Scatter3d if dim == 3 else go.Scatter)(
                **dict(zip("xyz", true.T)), mode = "markers", name = "true PF",
                marker = dict(size = 3 if dim == 2 else 2, color = "crimson")))
        base = len(fig.data)
        for trace in traces(iters[-1]):
            fig.add_trace(trace)

        # the axis titles are fixed; the ranges come from each frame's own layout, so the contour
        # grid and the points are always framed by the same box and autorange never re-frames it
        if dim == 3:
            fig.update_scenes(**dict(zip(("xaxis_title", "yaxis_title", "zaxis_title"), plain)))
        elif dim == 2:
            fig.update_layout(xaxis_title = plain[0], yaxis_title = plain[1])
        fig.update_layout(axes_layout(iters[-1]))        # the iteration the slider opens on

        fig.frames = [go.Frame(name = str(it), data = traces(it), layout = axes_layout(it),
                               traces = list(range(base, len(fig.data)))) for it in iters]
        fig.update_layout(
            height = 600, width = 1020 if show_params else 700, showlegend = True,
            margin = dict(r = 330) if show_params else None,
            legend = dict(yanchor = "top", y = 0.99, xanchor = "right", x = 0.99,
                          bgcolor = "rgba(255,255,255,0.7)"),
            title = f"{type(model.problem).__name__} - learning process",
            sliders = [dict(active = len(iters) - 1, currentvalue = dict(prefix = "iteration "),
                            steps = [dict(method = "animate", label = str(it),
                                          args = [[str(it)], dict(mode = "immediate",
                                                                  frame = dict(redraw = True))])
                                     for it in iters])])
        if show:
            fig.show()
        return fig

    # ---- the sampled objective space, with the analytical front behind it ----
    def draw(Q, lines = False, **kw):
        """scatter, except the true front which stays joined up in parallel coordinates"""
        if dim <= 3:
            return ax.scatter(*Q.T[:3], **kw)
        if lines:
            kw.pop("s", None)
            ax.add_collection(LineCollection(
                [np.column_stack([np.arange(dim), q]) for q in Q], **kw))
            return ax.autoscale()
        kw["s"] = kw.get("s", 20) / 4              # one marker per coordinate, so many more
        return ax.scatter(np.tile(np.arange(dim), len(Q)), Q.ravel(), **kw)


    fig = plt.figure(figsize = (8, 7))
    ax = fig.add_subplot(111, projection = "3d") if dim == 3 else fig.add_subplot(111)
    mag = 2 if dim == 2 else 1          # a flat plot carries larger markers than a 3-D scene
    # if dim == 3:                      # f1 to the right, f2 to the left, both meeting at their max
    #     ax.view_init(elev = 22, azim = 45)

    if not only_true:
        off = 0 if zlo is None else int(np.sum(~np.all((hist >= zlo) & (hist <= zhi), 1)))
        draw(hist, color = "0.6", s = 5 * mag, alpha = 0.35, lw = 0.5,
             label = f"evaluated ({len(hist)}" + (f", {off} off-frame)" if off else ")"))
    if true is not None and (show_true or only_true):
        draw(true, lines = True, color = "crimson", s = 3 * mag, lw = 0.6, label = "true PF")
    if not only_true:
        draw(front, color = "royalblue", s = 3 * mag, lw = 1.2, label = f"non-dominated ({len(front)})")

    if dim <= 3:
        ax.set(xlabel = names[0], ylabel = names[1])
        if dim == 3:
            ax.set_zlabel(names[2])
    else:
        every = max(1, dim // 10)                 # 30 tick labels would be unreadable
        ax.set(xticks = range(0, dim, every), xticklabels = names[::every], ylabel = "value")
    if zlo is not None:
        if dim <= 3:
            ax.set_xlim(zlo[0], zhi[0])
            ax.set_ylim(zlo[1], zhi[1])
            if dim == 3:
                ax.set_zlim(zlo[2], zhi[2])
        else:                                     # parallel coordinates: only the value axis
            ax.set_ylim(zlo.min(), zhi.max())

    ax.set_title(f"{type(model.problem).__name__} - objective space")
    ax.legend(fontsize = 8)
    fig.tight_layout()

    if show:
        plt.show()
    return fig
