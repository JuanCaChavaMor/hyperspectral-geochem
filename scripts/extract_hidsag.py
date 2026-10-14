"""Compact extract of a HIDSAG record: per crop mean/std spectra + a random subsample of pixel spectra.

    python scripts/extract_hidsag.py <record dir, e.g. .../GEOCHEM> <out dir> [sample names ...]

Without sample names, all samples of the record are extracted. One .npz per crop with
  y, names        numeric response variables of the sample (GEOCHEM: the 18 elements in fixed order)
  tag             crop tags joined by "|" (e.g. "fine", "P1|S3|coarse")
  vnir_mean, swir_mean, vnir_std, swir_std, vnir_px, swir_px (1000 random pixels, float16), px_idx,
  swir_bright     mean SWIR brightness image (H, W)
Only the co-registered low-resolution cubes (vnir_low, swir_low) are read.
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

GEOCHEM = ["Al", "Si", "P", "S", "K", "Ca", "Ti", "Fe", "Cu", "Zn", "As", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Pb"]

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
samples = sys.argv[3:] or sorted(p.name for p in root.iterdir() if (p / "metadata.json").exists())
rng = np.random.default_rng(0)
for name in samples:
    d = json.load(open(root / name / "metadata.json"))
    names = GEOCHEM if d["datarecord"] == "GEOCHEM" else [k for k, v in d["vars"].items()
                                                           if isinstance(v, (int, float))]
    y = np.array([d["vars"][k] for k in names], float)
    for crop in d["crops"]:
        key = [k for k in crop if k != "tags"][0]
        with h5py.File(root / name / crop[key]["swir_low"]["path_hsi"]) as fs, \
                h5py.File(root / name / crop[key]["vnir_low"]["path_hsi"]) as fv:
            s, v = fs["hsi_data"][:], fv["hsi_data"][:]
        assert s.shape[:2] == v.shape[:2], (name, key, s.shape, v.shape)
        H, W = s.shape[:2]
        S = s.reshape(-1, s.shape[2]).astype(np.float32)
        V = v.reshape(-1, v.shape[2]).astype(np.float32)
        idx = rng.choice(H * W, size=min(1000, H * W), replace=False)
        np.savez_compressed(out / f"{name}_{key}.npz", y=y, names=np.array(names), tag="|".join(crop["tags"]),
                            H=H, W=W, vnir_mean=V.mean(0), swir_mean=S.mean(0), vnir_std=V.std(0),
                            swir_std=S.std(0), vnir_px=V[idx].astype(np.float16), swir_px=S[idx].astype(np.float16),
                            px_idx=idx, swir_bright=S.mean(1).reshape(H, W).astype(np.float32))
        print(name, key, "|".join(crop["tags"]), H, W, flush=True)
