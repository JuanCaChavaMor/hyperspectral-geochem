"""What the spectra can and cannot see (Figs. 1-3 of the report), from the crop-mean spectra alone.

    python scripts/spectral_analysis.py      # ~20 s, needs only data/crop_spectra.csv.gz

1. mean spectra of the low- and high-Ca end members with the absorption features marked;
2. correlation spectrum: Pearson r between continuum-removed reflectance and each element over the 28 samples;
3. where the variance of each band depth lives: between samples (composition) vs between the fine and
   coarse crops of one sample (grain size / acquisition).
Writes figures/spectra, figures/correlation, results/feature_variance.csv, results/feature_correlations.csv.
"""
import numpy as np
import pandas as pd

from _path import ROOT
from hsgeo.data import ELEMENTS, MAJOR, load_crop_means
from hsgeo.features import FEATURES, all_features
from hsgeo.plotting import C, INK, MUTED, mark_features, plt, save
from hsgeo.preprocess import continuum_removed

ds = load_crop_means()
wl, nv = ds.wl, 58
samples = np.unique(ds.sample)
Ys = pd.DataFrame([ds.Y[ds.sample == s][0] for s in samples], index=samples, columns=ELEMENTS)

# ---------------------------------------------------------------- 1. spectra
lo, hi = Ys["Ca"].idxmin(), Ys["Ca"].idxmax()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, height_ratios=[1, 1.2])
for s, c in ((lo, C[0]), (hi, C[1])):
    for tag, ls in (("fine", "-"), ("coarse", "--")):
        ii = (ds.sample == s) & (ds.tag == tag)
        if not ii.any():
            continue
        x = ds.X[ii].mean(0)
        lab = f"{s} (Ca {Ys.loc[s, 'Ca']:.2f} %), {tag}"
        a1.plot(wl[:nv], x[:nv], color=c, ls=ls, label=lab)
        a1.plot(wl[nv:], x[nv:], color=c, ls=ls)
        cr = continuum_removed(x[None], wl, nv)[0]
        a2.plot(wl[nv:], cr[nv:], color=c, ls=ls)
a1.set_ylabel("reflectance [counts / 1e4]")
a1.legend(fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
a2.set_ylabel("continuum removed (SWIR)")
a2.set_xlabel("wavelength [nm]")
a2.set_xlim(wl[0], wl[-1])
mark_features(a2, FEATURES)
save(fig, ROOT / "figures/spectra.png")

# ---------------------------------------------------------------- 2. correlation spectrum (sample means)
CR = continuum_removed(ds.X.astype(float), wl, nv)
Xs = np.array([CR[ds.sample == s].mean(0) for s in samples])
Z = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12)
on_hull = Xs.mean(0) > 0.99              # no absorption there: CR is ~1 and r is noise
show = ["Al", "K", "Ca", "S", "Fe", "Cu"]
fig, ax = plt.subplots(figsize=(8, 3.4))
for k, e in enumerate(show):
    y = (Ys[e] - Ys[e].mean()) / Ys[e].std()
    r = Z.T @ y.to_numpy() / len(samples)
    r = np.convolve(r, np.ones(3) / 3, mode="same")                    # light smoothing (3 bands)
    r[on_hull] = np.nan
    ax.plot(wl[nv:], r[nv:], color=(C + [INK])[k], lw=1.3, label=e)
r_crit = 0.374                                                         # |r| for p = 0.05, n = 28
ax.axhspan(-r_crit, r_crit, color=MUTED, alpha=0.1, lw=0)
ax.set_xlim(wl[nv], wl[-1]); ax.set_ylim(-0.9, 0.9)
ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("Pearson r (28 samples)")
mark_features(ax, FEATURES)
ax.legend(ncol=6, fontsize=8, loc="lower left")
save(fig, ROOT / "figures/correlation.png")

# ---------------------------------------------------------------- 3. feature correlations and variance split
F = pd.DataFrame(all_features(ds.X.astype(float), wl))
F["sample"], F["tag"] = ds.sample, ds.tag
Fs = F.groupby("sample")[list(F.columns[:-2])].mean()
corr = pd.DataFrame({e: Fs.corrwith(Ys[e]) for e in MAJOR}).round(2)
corr.to_csv(ROOT / "results/feature_correlations.csv")

rows = []
for f in F.columns[:-2]:
    by_tag = F.groupby(["sample", "tag"])[f].mean().unstack()
    paired = (by_tag["coarse"] - by_tag["fine"]).dropna()
    rows.append(dict(feature=f,
                     sd_between_samples=F.groupby("sample")[f].mean().std(),
                     sd_within_sample=np.sqrt(F.groupby("sample")[f].var().mean()),
                     coarse_minus_fine=paired.mean(), coarse_minus_fine_sd=paired.std(),
                     n_pairs=len(paired)))
var = pd.DataFrame(rows).set_index("feature")
var["within/between"] = var["sd_within_sample"] / var["sd_between_samples"]
var.to_csv(ROOT / "results/feature_variance.csv", float_format="%.4f")
print(corr, "\n")
print(var.round(4))
