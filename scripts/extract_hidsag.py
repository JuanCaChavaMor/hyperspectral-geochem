"""Compact extract of HIDSAG GEOCHEM: per crop mean/std spectra + a random pixel subsample (VNIR low + SWIR low)."""
import h5py, json, sys, numpy as np
from pathlib import Path
root = Path(sys.argv[1]); out = Path(sys.argv[2]); samples = sys.argv[3:]
E = ["Al","Si","P","S","K","Ca","Ti","Fe","Cu","Zn","As","Rb","Sr","Y","Zr","Nb","Mo","Pb"]
rng = np.random.default_rng(0)
for name in samples:
    d = json.load(open(root / name / "metadata.json"))
    y = np.array([d["vars"][e] for e in E], float)
    for crop in d["crops"]:
        key = [k for k in crop if k != "tags"][0]
        s = h5py.File(root / name / crop[key]["swir_low"]["path_hsi"])["hsi_data"][:]
        v = h5py.File(root / name / crop[key]["vnir_low"]["path_hsi"])["hsi_data"][:]
        assert s.shape[:2] == v.shape[:2], (name, key, s.shape, v.shape)
        H, W = s.shape[:2]
        S = s.reshape(-1, s.shape[2]).astype(np.float32); V = v.reshape(-1, v.shape[2]).astype(np.float32)
        idx = rng.choice(H * W, size=min(1000, H * W), replace=False)
        np.savez_compressed(out / f"{name}_{key}.npz", y=y, tag=crop["tags"][0], H=H, W=W,
            vnir_mean=V.mean(0), swir_mean=S.mean(0), vnir_std=V.std(0), swir_std=S.std(0),
            vnir_px=V[idx].astype(np.float16), swir_px=S[idx].astype(np.float16), px_idx=idx,
            swir_bright=S.mean(1).reshape(H, W).astype(np.float32))
        print(name, key, crop["tags"][0], H, W, flush=True)
