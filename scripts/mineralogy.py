"""Mineralogy route: spectra -> minerals (learned on HIDSAG MINERAL1) -> elements of GEOCHEM.

    python scripts/mineralogy.py      # ~4 min; needs data/ext/mineral1 (scripts/extract_hidsag.py on MINERAL1)

1. Spectra -> minerals on MINERAL1 (99 crops of 36 monthly composites, QEMSCAN modal mineralogy), validated by
   leaving out whole composites. Only minerals with a positive skill are carried on ("visible minerals").
2. The model trained on all of MINERAL1 predicts the mineralogy of every GEOCHEM crop; GEOCHEM labels are not
   used, so the predicted minerals are external features and leave-one-sample-out stays clean.
3. Elements of GEOCHEM from the predicted minerals:
   (A) stoichiometry, no GEOCHEM labels at all;
   (B) stoichiometry + a linear calibration per element, fitted leave-one-sample-out;
   (C) predicted minerals as features of a ridge regression (LOSO), alone and averaged with the spectral model.
Writes results/mineral_within.csv, results/geochem_minerals.csv, results/mineral_element_corr.csv,
results/mineralogy.csv.
"""
import numpy as np
import pandas as pd

from _path import ROOT
from hsgeo import preprocess as pp
from hsgeo.data import ELEMENTS, MAJOR, Dataset, load_crop_means, load_record
from hsgeo.evaluate import cross_predict, loo_mean_mae, metrics, per_sample
from hsgeo.minerals import elements_from_minerals
from hsgeo.models import PCARidge, PLS, RidgeCV

PREP = "snv"            # spectral shape only: brightness differs between HIDSAG campaigns
RES = ROOT / "results"
jm = [ELEMENTS.index(e) for e in MAJOR]

# ---------------------------------------------------------------- 1. spectra -> minerals on MINERAL1
m1, minerals = load_record("MINERAL1")
X1 = pp.apply(PREP, m1.X.astype(float), m1.wl, 58)
_, P1, Y1 = cross_predict(PLS, m1, X1)
mw = metrics(P1, Y1)
within = pd.DataFrame({"mineral": minerals, "mean_wt%": Y1.mean(0), "sd_wt%": Y1.std(0),
                       "skill": mw["skill"], "Q2": mw["Q2"]}).sort_values("skill", ascending=False)
within.to_csv(RES / "mineral_within.csv", index=False, float_format="%.3f")
visible = within[(within.skill > 0.1) & (within["mean_wt%"] > 0.5)].mineral.tolist()
print("visible minerals (MINERAL1, leave-one-composite-out):",
      ", ".join(f"{m} {s:+.2f}" for m, s in within.set_index("mineral").skill[visible].items()), flush=True)

# ---------------------------------------------------------------- 2. predicted mineralogy of GEOCHEM
gch = load_crop_means()
Xg = pp.apply(PREP, gch.X.astype(float), gch.wl, 58)
mineral_model = PLS().fit(X1, m1.Y, m1.sample)
Mg = mineral_model.predict(Xg)                                   # (crops, minerals)
s, Ms = per_sample(Mg, Dataset(Xg, Mg, gch.sample, gch.crop, gch.tag, gch.wl))[:2]
_, _, Ys = per_sample(gch.Y, gch)
pred_min = pd.DataFrame(Ms, index=s, columns=minerals)
pred_min.to_csv(RES / "geochem_minerals.csv", float_format="%.3f")
print("\npredicted mean mineralogy of GEOCHEM [wt %]:",
      ", ".join(f"{m} {v:.1f}" for m, v in pred_min[visible].mean().sort_values(ascending=False).items()), flush=True)

# correlation of the predicted (external) mineralogy with the measured elements, over the 28 samples
corr = pd.DataFrame({e: [np.corrcoef(pred_min[m], Ys[:, ELEMENTS.index(e)])[0, 1] for m in visible] for e in MAJOR},
                    index=visible)
corr.to_csv(RES / "mineral_element_corr.csv", float_format="%.2f")
print("\ncorrelation predicted mineral vs measured element:\n", corr.round(2), flush=True)

rows = []


def record(route, Ps, Ys):
    m = metrics(Ps, Ys)
    r = dict(route=route, mean_skill=m["skill"][jm].mean())
    r.update({e: m["skill"][ELEMENTS.index(e)] for e in MAJOR})
    rows.append(r)
    print(f"{route:52s} mean skill {r['mean_skill']:+.3f}  " +
          " ".join(f"{e} {r[e]:+.2f}" for e in MAJOR), flush=True)


# ---------------------------------------------------------------- 3A. stoichiometry, no GEOCHEM labels
E_st = elements_from_minerals(Ms, minerals, ELEMENTS)
r = [np.corrcoef(E_st[:, j], Ys[:, j])[0, 1] if E_st[:, j].std() > 0 else np.nan for j in range(len(ELEMENTS))]
print("\nstoichiometric estimate vs pXRF (28 samples):")
for e in MAJOR:
    j = ELEMENTS.index(e)
    print(f"  {e:3s} predicted mean {E_st[:, j].mean():6.2f} %   measured {Ys[:, j].mean():6.2f} %   r = {r[j]:+.2f}")
record("(A) stoichiometry, uncalibrated", E_st, Ys)

# ---------------------------------------------------------------- 3B. stoichiometry + LOSO linear calibration
Pc = np.zeros_like(Ys)
for i in range(len(Ys)):
    tr = np.arange(len(Ys)) != i
    for j in range(len(ELEMENTS)):
        if E_st[tr, j].std() > 1e-9:
            b, a = np.polyfit(E_st[tr, j], Ys[tr, j], 1)
            Pc[i, j] = a + b * E_st[i, j]
        else:
            Pc[i, j] = Ys[tr, j].mean()
record("(B) stoichiometry + linear calibration (LOSO)", Pc, Ys)

# ---------------------------------------------------------------- 3C. minerals as features (crop level, LOSO)
vis = [minerals.index(m) for m in visible]
_, P_min, Y = cross_predict(RidgeCV, gch, Mg[:, vis])
record("(C) ridge on predicted visible minerals", P_min, Y)
_, P_all, _ = cross_predict(RidgeCV, gch, Mg)
record("(C) ridge on all predicted minerals", P_all, Y)
_, P_spec, _ = cross_predict(PCARidge, gch, gch.X.astype(float))
record("spectra: PCA + ridge (baseline)", P_spec, Y)
record("(C) average of spectral model and mineral ridge", (P_spec + P_min) / 2, Y)
pd.DataFrame(rows).to_csv(RES / "mineralogy.csv", index=False, float_format="%.3f")
