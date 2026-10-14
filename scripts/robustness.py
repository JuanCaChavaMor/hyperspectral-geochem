"""Does the model rely on artefacts, and does it profit from simple corrections? (Table 4 of the report)

    python scripts/robustness.py      # ~3 min; the tray-masking rows need data/extract and data/raw

Variants, each scored with leave-one-sample-out for the two best crop-level models:
  * baseline                  crop means as shipped (data/crop_spectra.csv.gz)
  * grain-size centring       spectra centred per fine/coarse class (fold-safe, see evaluate.tag_centre)
  * fine only / coarse only   one size fraction (the 25 samples that have both)
  * both fractions            same 25 samples, both fractions (reference for the two rows above)
  * 1000 px / tray masked     crop means recomputed from the 1000 stored pixels, without / with the pixels of
                              the plastic tray (nearest spectral cluster of scripts/mineral_maps.py)
"""
import numpy as np
import pandas as pd

from _path import ROOT
from hsgeo import preprocess as pp
from hsgeo.data import DATA, Dataset, Featurizer, load_crop_means
from hsgeo.evaluate import cross_predict, metrics
from hsgeo.models import PCARidge, PLS

MODELS = [(PCARidge, "raw"), (PLS, "d1")]
ds = load_crop_means()


def subset(ii, X=None):
    X = ds.X if X is None else X
    return Dataset(X[ii], ds.Y[ii], ds.sample[ii], ds.crop[ii], ds.tag[ii], ds.wl)


both = [s for s in np.unique(ds.sample) if set(ds.tag[ds.sample == s]) == {"fine", "coarse"}]
inb = np.isin(ds.sample, both)
variants = [("baseline", ds, False), ("grain-size centring", ds, True),
            ("both fractions (25 samples)", subset(inb), False),
            ("fine only (25 samples)", subset(inb & (ds.tag == "fine")), False),
            ("coarse only (25 samples)", subset(inb & (ds.tag == "coarse")), False)]

if (DATA / "extract").exists() and (DATA / "raw").exists():
    from hsgeo.maps import cluster_pixels, load_cube, nearest_cluster, tray_index
    cubes = [load_cube(DATA / "raw" / c)[0] for c in ("GCH-0015/01", "GCH-0015/03", "GCH-0019/01", "GCH-0019/03")]
    fz = Featurizer()
    _, centres, w, _ = cluster_pixels(cubes, fz.wl, fz.n_vnir)
    tray = tray_index(centres, w)
    Xa, Xm, frac = [], [], []
    for crop in ds.crop:
        z = np.load(DATA / "extract" / f"{crop}.npz")
        px = fz(z["vnir_px"], z["swir_px"])
        bad = np.isin(nearest_cluster(px, fz.wl, fz.n_vnir, centres), [tray, -1])
        Xa.append(px.mean(0)); Xm.append(px[~bad].mean(0)); frac.append(bad.mean())
    print(f"tray pixels per crop: median {np.median(frac):.1%}, max {np.max(frac):.1%}")
    allc = np.ones(len(ds), bool)
    variants += [("1000 px per crop", subset(allc, np.array(Xa)), False),
                 ("1000 px, tray masked", subset(allc, np.array(Xm)), False)]

rows = []
for name, d, centre in variants:
    for M, prep in MODELS:
        X = pp.apply(prep, d.X.astype(float), d.wl, 58)
        _, P, Y = cross_predict(M, d, X, centre_tags=centre)
        sk = metrics(P, Y)["skill"][:9]
        rows.append(dict(variant=name, model=f"{M.name} ({prep})", n_samples=len(np.unique(d.sample)),
                         mean_skill=sk.mean(), Ca=sk[5], Fe=sk[7], Cu=sk[8]))
        print(f"{name:30s} {M.name:12s} {prep:4s} mean skill {sk.mean():+.3f}", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/robustness.csv", index=False, float_format="%.3f")
