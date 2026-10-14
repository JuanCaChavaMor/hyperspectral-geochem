"""Grain-size corrections: do they make the spectra size-invariant, and do they improve prediction?

    python scripts/grain_size.py      # ~5 min, crop level (data/crop_spectra.csv.gz only)

Invariance index of a representation Z (bands standardised over all crops):
    mean over samples of ||Z_coarse - Z_fine||  /  mean over sample pairs of ||Z_s - Z_t||
0 means the size fraction leaves no trace; 1 means the fine and coarse fraction of one sample are as far apart
as two different samples. Prediction: leave-one-sample-out skill, PCA + ridge and PLS.
Writes results/grain_size.csv.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from _path import ROOT
from hsgeo.data import MAJOR, ELEMENTS, load_crop_means
from hsgeo.evaluate import cross_predict, metrics
from hsgeo.grainsize import SCALE, TRANSFORMS, emsc, epo, hapke_log_absorption
from hsgeo.models import PCARidge, PLS

ds = load_crop_means()
X0 = ds.X.astype(float)
NV = 58
jm = [ELEMENTS.index(e) for e in MAJOR]


def invariance(Z):
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    d, means = [], []
    for s in np.unique(ds.sample):
        c, f = (ds.sample == s) & (ds.tag == "coarse"), (ds.sample == s) & (ds.tag == "fine")
        if c.any() and f.any():
            d.append(np.linalg.norm(Z[c].mean(0) - Z[f].mean(0)))
        means.append(Z[ds.sample == s].mean(0))
    return np.mean(d) / pdist(np.array(means)).mean()


def full_epo(k):
    """EPO fitted on all pairs: only for the (label-free) invariance index, never for prediction."""
    return lambda Z: epo(k)(Z, ds, np.ones(len(ds), bool))


rows = []
variants = [(name, f, None) for name, f in TRANSFORMS.items()]
variants += [(f"EPO k={k} on {base}", TRANSFORMS[base], epo(k)) for base in ("raw", "log(1/R)") for k in (1, 2, 3)]
variants += [("EMSC + size interferent on log(1/R)", TRANSFORMS["log(1/R)"], emsc())]
# sensitivity of the Hapke route to the unknown absolute reflectance level
for f in (0.8, 1.25):
    sc = {k: v * f for k, v in SCALE.items()}
    variants.append((f"Hapke log(alpha D), scale ×{f}", lambda X, nv, sc=sc: hapke_log_absorption(X, nv, sc), None))
for name, T, ft in variants:
    Z = T(X0, NV)
    inv_Z = Z if ft is None else ft(Z, ds, np.ones(len(ds), bool))
    r = dict(representation=name, invariance=invariance(inv_Z))
    for M in (PCARidge, PLS):
        _, P, Y = cross_predict(M, ds, Z, fold_transform=ft)
        sk = metrics(P, Y)["skill"]
        r[f"skill {M.name}"] = sk[jm].mean()
        if M is PCARidge:
            r.update({e: sk[ELEMENTS.index(e)] for e in ("Al", "K", "Ca", "Fe", "Cu")})
    rows.append(r)
    print(f"{name:38s} invariance {r['invariance']:.2f}   skill PCA+ridge {r['skill PCA + ridge']:+.3f}"
          f"   PLS {r['skill PLS']:+.3f}", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/grain_size.csv", index=False, float_format="%.3f")
