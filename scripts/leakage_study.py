"""How much accuracy does a leaky split invent? Same model, three ways of splitting (Table 3 of the report).

    python scripts/leakage_study.py      # needs data/extract (patch level), ~25 min
"""
import time

import numpy as np
import pandas as pd

from _path import ROOT
from hsgeo import preprocess as pp
from hsgeo.data import ELEMENTS, load
from hsgeo.evaluate import cross_predict, metrics
from hsgeo.models import ClusterPLS, PLS

ds = load("patch")
X = pp.snv(ds.X)
rows = []
for name, M in (("PLS", PLS), ("cluster + PLS (HIDSAG-like)", ClusterPLS)):
    for protocol, label in (("random", "random patches (80/20)"), ("crop", "leave one crop out"),
                            ("sample", "leave one sample out")):
        t = time.time()
        _, P, Y = cross_predict(M, ds, X, protocol=protocol)
        m = metrics(P, Y)
        for j, e in enumerate(ELEMENTS):
            rows.append(dict(model=name, protocol=label, element=e, **{k: float(v[j]) for k, v in m.items()}))
        print(f"{name:28s} {label:24s} skill(Al..Cu) {m['skill'][:9].mean():+.3f}  MAPE Al {m['MAPE'][0]:.2f} %"
              f"  [{time.time() - t:.0f} s]", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results" / "leakage.csv", index=False)
