"""Is the leave-one-sample-out skill better than chance? Labels are permuted between whole samples.

    python scripts/permutation_test.py --n 100      # crop level, two models, ~1 h
"""
import argparse

import pandas as pd

from _path import ROOT
from hsgeo import preprocess as pp
from hsgeo.data import ELEMENTS, load
from hsgeo.evaluate import permutation_pvalue
from hsgeo.models import PCARidge, PLS

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
ds = load("crop")
rows = []
for M, prep in ((PCARidge, "raw"), (PLS, "d1")):
    X = pp.apply(prep, ds.X, ds.wl, n_vnir=58)
    obs, p = permutation_pvalue(M, ds, X, n_perm=a.n)
    rows += [dict(model=M.name, prep=prep, element=e, skill=float(obs[j]), p_value=float(p[j]))
             for j, e in enumerate(ELEMENTS)]
    print(M.name, prep, {e: (round(float(obs[j]), 2), round(float(p[j]), 3)) for j, e in enumerate(ELEMENTS[:9])},
          flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "permutation.csv", index=False)
