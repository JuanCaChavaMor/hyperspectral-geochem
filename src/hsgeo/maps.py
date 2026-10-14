"""Pixel-level mineral maps of a crop: where in the image the absorption features are.

This is the image side of the problem: the composition model sees one spectrum per sample, but the cube
shows that a sample is a mixture of particles with different mineralogy.
"""
from pathlib import Path

import h5py
import numpy as np
from sklearn.cluster import KMeans

from .data import Featurizer
from .features import FEATURES, band_depth
from .preprocess import continuum_removed


def load_cube(crop_dir, fz=None):
    """(H, W, bands) reflectance of one crop on the Featurizer grid (VNIR binned + SWIR)."""
    fz = fz or Featurizer()
    crop_dir = Path(crop_dir)
    with h5py.File(crop_dir / "vnir_low.h5") as v, h5py.File(crop_dir / "swir_low.h5") as s:
        return fz(v["hsi_data"][:], s["hsi_data"][:]), fz


def band_depth_map(cube, wl, name):
    return band_depth(cube, wl, *FEATURES[name][0])


def valid_mask(cube, q=0.02):
    """Drop shadows / gaps between particles (the darkest q of the pixels) and dead pixels (zero counts)."""
    b = cube.mean(-1)
    return (b > np.quantile(b, q)) & np.all(cube > 0, -1)


def cluster_pixels(cubes, wl, n_vnir, k=4, seed=0):
    """k-means on continuum-removed SWIR spectra of all valid pixels of all cubes (shared cluster labels).

    Continuum removal leaves only the shape of the absorptions, so clusters follow mineralogy and not
    brightness. Returns per-cube label maps (-1 where invalid) and the cluster mean CR spectra.
    """
    sw = slice(n_vnir, None)
    feats, idx = [], []
    for c in cubes:
        m = valid_mask(c)
        px = c[m]
        cr = continuum_removed(px, wl, n_vnir)[:, sw]
        feats.append(cr); idx.append(m)
    F = np.concatenate(feats)
    km = KMeans(k, n_init=4, random_state=seed).fit(F)
    # order clusters by depth of the 1935 nm water feature, so colours are comparable across runs
    w = wl[sw]
    j = np.argmin(np.abs(w - 1935))
    order = np.argsort(km.cluster_centers_[:, j])[::-1]          # shallowest first
    rank = np.empty(k, int); rank[order] = np.arange(k)
    maps, a = [], 0
    for c, m in zip(cubes, idx):
        L = -np.ones(m.shape, int)
        L[m] = rank[km.labels_[a:a + m.sum()]]
        a += m.sum()
        maps.append(L)
    centres = km.cluster_centers_[order]
    share = np.array([[np.mean(L[L >= 0] == r) for r in range(k)] for L in maps])
    return maps, centres, w, share


def nearest_cluster(px, wl, n_vnir, centres):
    """Assign pixel spectra (n, bands) to the nearest cluster centre (continuum-removed SWIR). Dead pixels: -1."""
    ok = np.all(px > 0, -1)
    cr = continuum_removed(np.where(ok[:, None], px, 1.0), wl, n_vnir)[:, n_vnir:]
    lab = ((cr[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
    return np.where(ok, lab, -1)


def tray_index(centres, w):
    """The cluster with the deepest C-H overtone at ~1210 nm is the plastic sample tray."""
    return int(np.argmin(centres[:, np.argmin(np.abs(w - 1210))]))
