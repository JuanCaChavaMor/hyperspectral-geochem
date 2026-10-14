"""HIDSAG GEOCHEM: compact per-crop extracts, spectral features and patch construction.

The raw record (28 samples, 106 crops, 7.2 GB of HDF5 cubes) is reduced once by `scripts/extract_hidsag.py`
to one .npz per crop: mean/std spectra of the whole crop and a random subsample of 1000 pixel spectra,
for the co-registered VNIR (942 bands, 400-1000 nm) and SWIR (268 bands, 1000-2500 nm) cameras.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ELEMENTS = ["Al", "Si", "P", "S", "K", "Ca", "Ti", "Fe", "Cu", "Zn", "As", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Pb"]
# elements whose spread across samples is resolved by pXRF; the rest sit at or near the detection limit
MAJOR = ["Al", "Si", "P", "S", "K", "Ca", "Ti", "Fe", "Cu"]


def wavelengths():
    wl = json.loads((DATA / "wavelengths.json").read_text())
    return np.array(wl["wavelength_VNIR"]), np.array(wl["wavelength_SWIR"])


class Featurizer:
    """Concatenated VNIR + SWIR reflectance on a common, de-noised grid.

    VNIR (0.6 nm sampling, noisy) is averaged into 10 nm bins from 420 to 1000 nm; SWIR is kept at its native
    5.7 nm sampling between 1000 and 2450 nm (the edges are noisy). Raw counts are divided by 1e4.
    """

    def __init__(self, vnir_bin=10.0, vnir_range=(420, 1000), swir_range=(1000, 2450)):
        wv, ws = wavelengths()
        edges = np.arange(vnir_range[0], vnir_range[1], vnir_bin)
        self.vbins = [(wv >= a) & (wv < a + vnir_bin) for a in edges]
        self.smask = (ws > swir_range[0]) & (ws < swir_range[1])
        self.wl = np.r_[edges + vnir_bin / 2, ws[self.smask]]
        self.n_vnir = len(edges)

    def __call__(self, vnir, swir):
        vnir = np.asarray(vnir, np.float32)
        swir = np.asarray(swir, np.float32)
        vb = np.stack([vnir[..., m].mean(-1) for m in self.vbins], -1)
        return np.concatenate([vb, swir[..., self.smask]], -1) / 1e4


@dataclass
class Dataset:
    X: np.ndarray          # (n, bands) spectra (crop means or patch means)
    Y: np.ndarray          # (n, 18) composition [wt %]
    sample: np.ndarray     # sample id (GCH-00xx): the unit of independence
    crop: np.ndarray       # crop id (sample_crop)
    tag: np.ndarray        # "fine" / "coarse"
    wl: np.ndarray         # wavelength of each feature [nm]

    def __len__(self):
        return len(self.Y)


def load(level="crop", patch_px=25, extract_dir=DATA / "extract", seed=0):
    """Crop-level (one mean spectrum per crop, 106 rows) or patch-level dataset.

    Patches are means of `patch_px` random pixels of one crop (40 per crop with 1000 stored pixels): they
    average out pixel noise while keeping the within-sample variability that the crop mean discards.
    """
    fz = Featurizer()
    if level == "crop" and not Path(extract_dir).exists():
        return load_crop_means(fz)
    rng = np.random.default_rng(seed)
    X, Y, S, C, T = [], [], [], [], []
    for f in sorted(Path(extract_dir).glob("GCH-*.npz")):
        d = np.load(f)
        s, c, tag = f.stem[:8], f.stem, str(d["tag"])
        if level == "crop":
            X.append(fz(d["vnir_mean"], d["swir_mean"])[None]); n = 1
        else:
            px = fz(d["vnir_px"], d["swir_px"])
            idx = rng.permutation(len(px))
            n = len(px) // patch_px
            X.append(px[idx[:n * patch_px]].reshape(n, patch_px, -1).mean(1))
        Y.append(np.tile(d["y"], (n, 1))); S += [s] * n; C += [c] * n; T += [tag] * n
    return Dataset(np.concatenate(X), np.concatenate(Y), np.array(S), np.array(C), np.array(T), fz.wl)


def load_crop_means(fz=None):
    """Crop-level dataset from the small CSV shipped with the repository (no HDF5 / extract needed)."""
    import pandas as pd
    fz = fz or Featurizer()
    m = pd.read_csv(DATA / "crop_spectra.csv.gz")
    bands = [c for c in m.columns if c.startswith("b_")]
    return Dataset(m[bands].to_numpy(np.float32), m[ELEMENTS].to_numpy(float), m["sample"].to_numpy(),
                   m["crop"].to_numpy(), m["tag"].to_numpy(), fz.wl)
