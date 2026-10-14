"""Spectral pre-processing (row-wise, no fitted parameters, so no information leaks across folds)."""
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull


def snv(X):
    """Standard normal variate: removes additive offset and multiplicative scatter (grain size, illumination)."""
    return (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)


def derivative(X, n_vnir, window=7, order=2):
    """Savitzky-Golay first derivative, applied to VNIR and SWIR separately (no derivative across the seam)."""
    v = savgol_filter(X[..., :n_vnir], window, order, deriv=1, axis=-1)
    s = savgol_filter(X[..., n_vnir:], window, order, deriv=1, axis=-1)
    return np.concatenate([v, s], -1)


def continuum_removed(X, wl, n_vnir):
    """Divide each spectrum by its upper convex hull, per sensor: absorption features become dips below 1."""
    out = np.empty_like(X)
    for a, b in ((0, n_vnir), (n_vnir, X.shape[-1])):
        w = wl[a:b]
        for i, x in enumerate(X[:, a:b]):
            pts = np.c_[w, x]
            hull = ConvexHull(np.vstack([pts, [w[0], x.min() - 1], [w[-1], x.min() - 1]]))
            v = sorted(set(hull.vertices) - {len(w), len(w) + 1})
            out[i, a:b] = x / np.interp(w, w[v], x[v])
    return out


def apply(name, X, wl, n_vnir):
    if name == "raw":
        return X
    if name == "snv":
        return snv(X)
    if name == "d1":
        return derivative(snv(X), n_vnir)
    if name == "cr":
        return continuum_removed(X, wl, n_vnir)
    raise ValueError(name)
