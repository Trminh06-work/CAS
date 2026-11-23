import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # the library lives in src/core

from core.indicators import dominates_ref     # one definition, in the module the sweep imports


def wfg_counts_points_beyond_the_reference():
    from wfg import mainentryWFG
    from moocore import hypervolume
    from core.nd_torch import non_dominated_mask_gpu

    cases = [
        ("one point beyond the ref",      [[7., 3.]],                     [5., 5.]),
        ("both on the boundary",          [[1., 5.], [7., 5.]],           [5., 5.]),
        ("dominated point beyond ref",    [[.3, .9], [.9, .3], [2., 2.]], [1., 1.]),
        ("non-dominated, straddling ref", [[1., 6.], [6., 1.]],           [5., 5.]),
        ("every point dominates the ref", [[1., 4.], [2., 2.], [4., 1.]], [5., 5.]),
    ]
    print(f"  {'case':<32} {'N':>3} {'nd':>3} {'wfg raw':>9} {'wfg nd':>9} {'moocore nd':>11}")
    for name, pts, ref in cases:
        P, r = np.array(pts), np.array(ref)
        nd = P[dominates_ref(P, r)]                         # minimisation dominance, in float64
        print(f"  {name:<32} {len(P):>3} {len(nd):>3} {mainentryWFG(P, r):>9.4f} "
              f"{mainentryWFG(nd, r) if len(nd) else 0.0:>9.4f} {hypervolume(nd, ref = r):>11.4f}")


def gd_torch_loses_small_distances_in_float32():
    """
        gd_torch casts to float32, and torch.cdist in float32 cannot resolve distances
        near the rounding floor - which is the whole point of GD on a converged front.
        Its default p = 2 is also not pymoo's indicator: pymoo means over the minima, p = 1.
    """
    import torch
    from pymoo.indicators.gd import GD
    from core.gd_torch import gd_gpu

    rng = np.random.default_rng(3)
    R = rng.random((4000, 3)); R /= np.linalg.norm(R, axis = 1, keepdims = True)
    cpu = torch.device("cpu")                           # mps cannot hold float64 either way

    print(f"  {'front offset':<32} {'pymoo f64':>12} {'gd_torch f32':>14} {'ratio':>8}")
    for eps in (1e-2, 1e-4, 1e-6):
        A = R[:400] + eps
        truth, got = float(GD(R)(A)), gd_gpu(A, R, p = 1, device = cpu)
        print(f"  A = R[:400] + {eps:<19.0e} {truth:>12.3e} {got:>14.3e} {got / truth:>7.1f}x")

    B = rng.random((300, 3)); B /= np.linalg.norm(B, axis = 1, keepdims = True)
    base = float(GD(R)(B))
    print(f"\n  {'p convention, GD(B, R)':<32} {'value':>12} {'vs pymoo':>14}")
    print(f"  {'pymoo':<32} {base:>12.6f} {'-':>14}")
    for p in (1, 2):
        got = gd_gpu(B, R, p = p, device = cpu)
        print(f"  {'gd_torch p = ' + str(p):<32} {got:>12.6f} {got / base - 1:>13.2%}")


if __name__ == "__main__":
    # for test in (wfg_counts_points_beyond_the_reference, gd_torch_loses_small_distances_in_float32):
    #     print(f"\n{test.__name__}")
    #     try:
    #         test()
    #     except ImportError as e:
    #         print(f"  skipped: {e.name} not installed")
    wfg_counts_points_beyond_the_reference()
