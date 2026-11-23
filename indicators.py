"""
    Quality indicators for the sweep: the single import site for all of them.

    hypervolume() is implemented here; gd/igd/non_dominated are thin wrappers over gd_torch.py
    and nd_torch.py, which keep their own implementations. Every heavy import is function-local,
    so `import indicators` pulls neither torch nor pymoo - run_experiment.py reaches
    hypervolume() on every cluster task and must not pay for them.
"""
import numpy as np


def dominates_ref(F, ref_point, direction = 'minimize'):
    """
    Keep only the points of F that dominate ref_point - the only ones that add hypervolume.

    F         : (N, m) array of points
    ref_point : (m,) reference point
    direction : 'minimize' -> f dominates ref_point if f <= ref_point, with f < ref_point somewhere
                'maximize' -> f dominates ref_point if f >= ref_point, with f > ref_point somewhere

    Points incomparable with ref_point, dominated by it or equal to it are dropped: wfg's
    mainentryWFG maps every point to |f - ref_point|, so any of them left in is mirrored back
    inside the box and counted. Points on the boundary (f_i == ref_i somewhere) are kept - they
    add no volume.

    Returns boolean mask (True = keep). The mask can be all False: the HV is then 0, but
    mainentryWFG raises on an empty set.
    """
    if direction == 'minimize':
        keep = np.all(F <= ref_point, axis=1) & np.any(F < ref_point, axis=1)
    elif direction == 'maximize':
        keep = np.all(F >= ref_point, axis=1) & np.any(F > ref_point, axis=1)
    else:
        raise ValueError("direction must be 'minimize' or 'maximize'")

    return keep


def hypervolume(F, ref):
    """
        Exact hypervolume of F against ref, minimisation, via wfg. Returns a float; an empty F,
        or one with nothing inside the box, gives 0.0.

        Falls back to pymoo (which is moocore) only when wfg is not installed, so a cluster
        without the package still runs - slower at high m, but never wrong.
    """
    F = np.asarray(F, dtype = float)
    if not len(F):
        return 0.0

    try:
        from wfg import mainentryWFG
    except ImportError:
        from pymoo.indicators.hv import HV
        return float(HV(ref_point = ref)(F))

    # mandatory, not an optimisation: without this filter mainentryWFG counts points that lie
    # beyond the reference, which inflated one sanity case 7.7x (1.0000 against a true 0.1300)
    inside = F[dominates_ref(F, ref)]
    return float(mainentryWFG(inside, ref)) if len(inside) else 0.0


def gd(A, R, p = 1, chunk_size = 2048, device = None):
    """
        Generational distance: how close A got to the reference front R. gd_torch.gd_gpu.

        p = 1 is pymoo's convention. GD is sensitive to |R|, so fix the reference size and
        record it; IGD is not.
    """
    from gd_torch import gd_gpu
    return gd_gpu(A, R, p = p, chunk_size = chunk_size, device = device)


def igd(A, R, p = 1, chunk_size = 2048, device = None):
    "Inverted generational distance: how well A covers R. gd_torch.igd_gpu."
    from gd_torch import igd_gpu
    return igd_gpu(A, R, p = p, chunk_size = chunk_size, device = device)


def non_dominated(F, block_size = 1024, device = None):
    """
        Block-wise non-dominated filter, minimisation. nd_torch.non_dominated_mask_gpu.

        Returns (mask, F_nd). float64, because float32 rounding makes near-ties compare equal,
        which both drops non-dominated points and keeps dominated ones.
    """
    from nd_torch import non_dominated_mask_gpu
    return non_dominated_mask_gpu(F, block_size = block_size, device = device)
