"""Leave-one-sample-out benchmark: models x data level x pre-processing (Table 2 of the report).

    python scripts/run_benchmark.py            # crop level, from data/crop_spectra.csv.gz (~5 min)
    python scripts/run_benchmark.py --patch    # patch level, needs data/extract (~40 min)
"""
import argparse
import time

import numpy as np
import pandas as pd

from _path import ROOT
from hsgeo import preprocess as pp
from hsgeo.data import ELEMENTS, load
from hsgeo.evaluate import cross_predict, metrics
from hsgeo.models import CNN1D, BestBand, ClusterPLS, MeanPredictor, PCARidge, PLS

ap = argparse.ArgumentParser()
ap.add_argument("--patch", action="store_true")
a = ap.parse_args()

if not a.patch:
    CONFIGS = [("crop", "raw", MeanPredictor), ("crop", "d1", BestBand), ("crop", "raw", PLS), ("crop", "d1", PLS),
               ("crop", "raw", PCARidge), ("crop", "d1", PCARidge), ("crop", "snv", CNN1D)]
else:
    CONFIGS = [("patch", "snv", PLS), ("patch", "snv", PCARidge), ("patch", "snv", ClusterPLS),
               ("patch", "snv", CNN1D)]

rows, preds = [], {}
cache = {}
for level, prep, M in CONFIGS:
    if level not in cache:
        cache[level] = load(level)
    ds = cache[level]
    X = pp.apply(prep, ds.X, ds.wl, n_vnir=58)
    t = time.time()
    s, P, Y = cross_predict(M, ds, X, protocol="sample")
    m = metrics(P, Y)
    key = f"{M.name} | {level} | {prep}"
    preds[key] = P
    for j, e in enumerate(ELEMENTS):
        rows.append(dict(model=M.name, level=level, prep=prep, element=e,
                         **{k: float(v[j]) for k, v in m.items()}))
    print(f"{key:40s} mean skill (Al..Cu) {m['skill'][:9].mean():+.3f}  [{time.time() - t:.0f} s]", flush=True)

out = ROOT / "results"
suffix = "_patch" if a.patch else ""
pd.DataFrame(rows).to_csv(out / f"benchmark{suffix}.csv", index=False)
np.savez(out / f"predictions{suffix}.npz", samples=s, Y=Y, **{k: v for k, v in preds.items()})
