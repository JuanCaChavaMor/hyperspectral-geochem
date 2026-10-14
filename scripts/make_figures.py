"""Result figures from results/*.csv (Figs. 5-7 of the report).

    python scripts/make_figures.py
"""
import numpy as np
import pandas as pd

from _path import ROOT
from hsgeo.data import ELEMENTS, MAJOR
from hsgeo.plotting import C, INK, MUTED, plt, save

RES, FIG = ROOT / "results", ROOT / "figures"
BEST = "PCA + ridge | crop | raw"
bench = pd.read_csv(RES / "benchmark.csv")
if (RES / "benchmark_patch.csv").exists():
    bench = pd.concat([bench, pd.read_csv(RES / "benchmark_patch.csv")])
bench["config"] = bench.model + " (" + bench.level + ", " + bench.prep + ")"

# ---------------------------------------------------------------- skill per model and per element
mean_skill = (bench[bench.element.isin(MAJOR)].groupby("config", sort=False).skill.mean().sort_values())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), width_ratios=[1, 1.15])
col = [C[1] if "crop, raw" in k and k.startswith("PCA") else C[0] for k in mean_skill.index]
a1.barh(range(len(mean_skill)), mean_skill.to_numpy(), color=col)
a1.set_yticks(range(len(mean_skill)), mean_skill.index, fontsize=8)
a1.axvline(0, color=INK, lw=0.8)
a1.set_xlabel("mean skill over Al ... Cu (leave one sample out)")
a1.grid(axis="y", visible=False)

b = bench[(bench.model == "PCA + ridge") & (bench.level == "crop") & (bench.prep == "raw")].set_index("element")
sk = b.loc[MAJOR, "skill"]
a2.bar(MAJOR, sk, color=C[1])
a2.axhline(0, color=INK, lw=0.8)
a2.set_ylabel("skill = 1 - MAE / MAE(mean)")
a2.set_title("PCA + ridge (crop means, raw)", loc="left", fontsize=9, color=INK)
if (RES / "permutation.csv").exists():
    perm = pd.read_csv(RES / "permutation.csv")
    p = perm[(perm.model == "PCA + ridge") & (perm.prep == "raw")].set_index("element").p_value
    for i, e in enumerate(MAJOR):
        n_perm = 100
        txt = f"p < {1 / (n_perm + 1) + 1e-4:.3f}" if p[e] <= 1 / (n_perm + 1) + 1e-9 else f"p = {p[e]:.3f}"
        a2.text(i, max(sk[e], 0) + 0.015, txt, ha="center", fontsize=6.5, color=MUTED, rotation=90, va="bottom")
    a2.set_ylim(top=max(sk.max() + 0.2, 0.6))
save(fig, FIG / "skill.png")

# ---------------------------------------------------------------- predicted vs measured (best model)
pr = np.load(RES / "predictions.npz", allow_pickle=True)
Y, P = pr["Y"], pr[BEST]
show = ["Al", "K", "S", "Ca", "Fe", "Cu"]
fig, axes = plt.subplots(2, 3, figsize=(9, 5.6))
for ax, e in zip(axes.ravel(), show):
    j = ELEMENTS.index(e)
    lo, hi = min(Y[:, j].min(), P[:, j].min()), max(Y[:, j].max(), P[:, j].max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, ls="--")
    ax.axhline(np.mean(Y[:, j]), color=C[0], lw=1, alpha=0.6)
    ax.scatter(Y[:, j], P[:, j], s=18, color=C[1], zorder=3)
    r = b.loc[e]
    ax.set_title(f"{e}: skill {r.skill:+.2f}, Q$^2$ {r.Q2:+.2f}", loc="left", fontsize=9, color=INK)
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
for ax in axes[1]:
    ax.set_xlabel("measured (pXRF) [wt %]")
for ax in axes[:, 0]:
    ax.set_ylabel("predicted [wt %]")
fig.tight_layout()
save(fig, FIG / "pred_vs_meas.png")

# ---------------------------------------------------------------- leakage
if (RES / "leakage.csv").exists():
    lk = pd.read_csv(RES / "leakage.csv")
    order = ["random patches (80/20)", "leave one crop out", "leave one sample out"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for k, (m, g) in enumerate(lk.groupby("model", sort=False)):
        s = g[g.element.isin(MAJOR)].groupby("protocol").skill.mean()[order]
        mape = g[g.element == "Al"].set_index("protocol").MAPE[order]
        x = np.arange(3) + (k - 0.5) * 0.38
        a1.bar(x, s, 0.38, color=C[k], label=m)
        a2.bar(x, mape, 0.38, color=C[k])
    base = bench[(bench.model == "mean") & (bench.element == "Al")].MAPE.iloc[0]
    a2.axhline(base, color=INK, lw=1, ls="--", label="mean predictor (LOSO)")
    for ax in (a1, a2):
        ax.set_xticks(range(3), ["random\npatches", "leave one\ncrop out", "leave one\nsample out"])
        ax.grid(axis="x", visible=False)
    a1.set_ylabel("mean skill (Al ... Cu)"); a1.legend(fontsize=8, loc="upper right")
    a2.set_ylabel("MAPE of Al [%]")
    paper = pd.read_csv(ROOT / "data/hidsag_paper_table5.csv", comment="#").set_index("element").loc["Al"]
    a2.axhline(paper.swir, color=C[2], lw=1.4, ls=":", label="HIDSAG paper, SWIR only")
    a2.legend(fontsize=7, loc="upper left")
    a2.set_ylim(0, base * 1.35)
    a2.set_title(f"HIDSAG paper, full range: {paper.full:.3f} % (not visible)", loc="left", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, FIG / "leakage.png")
print("figures written to", FIG)
