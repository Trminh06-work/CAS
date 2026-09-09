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


    def find_nadir(in_space: str = "objective"):
        """
            Use model.obj_hist / model.dec_hist to find nadir point

            Parameter
                in_space: str, default = "objective". Possible values: "objective", "decision"
        """
        # worst value per objective over every evaluation any model made, so the reference is shared and the hypervolumes compare
        if in_space == "objective":
            H = np.vstack([finite(m.obj_hist) for m in models_arr])
        else:
            H = np.vstack([finite(m.dec_hist) for m in models_arr])
        return H.max(0)


    def summary():
        """
            Return a pd.DataFrame, rows are models - columns are performance metrics

            Hypervolume - use pymoo
            Number of function evaluations
        """
        from pymoo.indicators.hv import HV
        hv = HV(ref_point = find_nadir(in_space = "objective"))
        rows = [
            dict(
                hypervolume = hv(finite(m.PF)),
                n_func_eval = m.num_func_eval,
                n_non_dominated = len(finite(m.PF)), n_iter = int(m.iteration),
                wallclock_time = m.wallclock_time
            )
            for m in models_arr
        ]
        return pd.DataFrame(rows, index = [type(m).__name__ for m in models_arr])


    return summary()



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
              zoom = True, show = True):
    """
        Plot one view of a solved model, in the problem's own objective units.

        The solver works on stored units (K - f, so that minimising maximises f); everything
        here is mapped back through problem.true() first, so a maximisation problem is drawn as
        a maximisation problem. Dominance is still decided on the stored values.

        model                any BaseModel, after solve()

        kind:
            "objective"  every evaluation (obj_hist) with the non-dominated archive (PF) highlighted and the true front behind them

            "learning"   a slider over the outer iterations showing only the non-dominated archive as it stood at that iteration

            "true_pf"    the analytical front on its own

        show_true            draw the analytical front (no effect if the problem has none)

        plot_scalarisation   "learning" only: overlay the level sets of scalarise(., w_k) - ignore n_obj > 3

        zoom                 frame the archive and the true front instead of
                             every evaluation, so the front is not squashed into a corner by the
                             unconverged cloud; the points left outside are counted in the legend

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
    except (AttributeError, NotImplementedError):
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
        seen = np.vstack(list(fronts.values()) + ([true] if true is not None and show_true else []))
        lo, hi = (zlo, zhi) if zlo is not None else (seen.min(0), seen.max(0))

        def traces(it):
            w = W[np.argmax(all_it == it)]               # weight learnt at iteration it
            Q = fronts[int(it)]
            xyz = (dict(zip("xyz", Q.T)) if dim <= 3 else
                   dict(x = np.tile(np.arange(dim), len(Q)), y = Q.ravel()))
            # the count rides in the legend label, so it follows the slider
            off = 0 if zlo is None else int(np.sum(~np.all((Q >= zlo) & (Q <= zhi), 1)))
            out = [(go.Scatter3d if dim == 3 else go.Scatter)(
                **xyz, mode = "markers",
                name = f"non-dominated ({len(Q)}" + (f", {off} off-frame)" if off else ")"),
                marker = dict(size = 5.5 if dim == 2 else 2, color = "royalblue"))]

            # the level sets need nothing model-specific - just scalarise() on a grid
            if plot_scalarisation and dim <= 3:
                axes = [np.linspace(lo[k], hi[k], res) for k in range(dim)]
                G = np.stack(np.meshgrid(*axes, indexing = "ij"), -1).reshape(-1, dim)
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
                marker = dict(size = 3 if dim == 2 else 2, color = "red")))
        base = len(fig.data)
        for trace in traces(iters[-1]):
            fig.add_trace(trace)

        # pin the axes to the same box whatever traces are present, so plot_scalarisation only
        # adds the level sets instead of also re-framing the plot: the contour grid spans every
        # frame, the points span one, and autorange would otherwise disagree between the two
        rpad = 0.03 * np.where(hi > lo, hi - lo, 1.0)
        rlo, rhi = (lo, hi) if zlo is not None else (lo - rpad, hi + rpad)

        if dim == 3:
            fig.update_scenes(**dict(zip(("xaxis_title", "yaxis_title", "zaxis_title"), plain)),
                              **{f"{a}axis_range": [rlo[k], rhi[k]] for k, a in enumerate("xyz")})
        elif dim == 2:
            fig.update_layout(xaxis_title = plain[0], yaxis_title = plain[1],
                              xaxis_range = [rlo[0], rhi[0]], yaxis_range = [rlo[1], rhi[1]])
        else:                                    # parallel coordinates: x is the coordinate index
            fig.update_layout(yaxis_range = [rlo.min(), rhi.max()])

        fig.frames = [go.Frame(name = str(it), data = traces(it),
                               traces = list(range(base, len(fig.data)))) for it in iters]
        fig.update_layout(
            height = 600, width = 700, showlegend = True,
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

    if true is not None and (show_true or only_true):
        draw(true, lines = True, color = "0.6", s = 10 * mag, lw = 0.6, label = "true PF")
    if not only_true:
        off = 0 if zlo is None else int(np.sum(~np.all((hist >= zlo) & (hist <= zhi), 1)))
        draw(hist, color = "C0", s = 5 * mag, alpha = 0.35, lw = 0.5,
             label = f"evaluated ({len(hist)}" + (f", {off} off-frame)" if off else ")"))
        draw(front, color = "crimson", s = 3 * mag, lw = 1.2, label = f"non-dominated ({len(front)})")

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
