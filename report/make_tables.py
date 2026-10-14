"""Write the LaTeX tables of the report from the CSV files in results/.

    python report/make_tables.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)
MAJOR = ["Al", "Si", "P", "S", "K", "Ca", "Ti", "Fe", "Cu"]


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", name)


def tab(spec, header, rows):
    return f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) + \
        "\n\\bottomrule\n\\end{tabular}\n"


def f(x, d=2):
    return f"{x:+.{d}f}".replace("-", "$-$")


# ---------------------------------------------------------------- benchmark (crop and patch level)
b = pd.concat([pd.read_csv(RES / f) for f in ("benchmark.csv", "benchmark_patch.csv") if (RES / f).exists()])
PREP = {"raw": "raw", "snv": "SNV", "d1": "SNV + 1st deriv."}
rows, last = [], None
best = b[b.element.isin(MAJOR)].groupby(["model", "level", "prep"], sort=False).skill.mean().max()
for (m, lev, prep), g in b.groupby(["model", "level", "prep"], sort=False):
    g = g.set_index("element")
    if last is not None and lev != last:
        rows.append("\\midrule")
    last = lev
    ms = g.loc[MAJOR, "skill"].mean()
    cell = f(ms) if abs(ms - best) > 1e-9 else f"\\textbf{{{f(ms)}}}"
    rows.append(f"{m} & {lev} & {PREP[prep]} & {cell} & {f(g.loc[MAJOR, 'Q2'].mean())} & "
                + " & ".join(f(g.loc[e, "skill"]) for e in ["Al", "K", "S", "Ca", "Fe", "Cu"]) + " \\\\")
write("benchmark", tab("lllrr" + "r" * 6,
      "model & level & pre-proc. & skill & $Q^2$ & Al & K & S & Ca & Fe & Cu", rows))

# ---------------------------------------------------------------- permutation p-values
if (RES / "permutation.csv").exists():
    p = pd.read_csv(RES / "permutation.csv")
    rows = []
    for (m, prep), g in p.groupby(["model", "prep"], sort=False):
        g = g.set_index("element")
        rows.append(f"{m} ({PREP[prep]}) & skill & " + " & ".join(f(g.loc[e, 'skill']) for e in MAJOR) + " \\\\")
        rows.append(" & $p$ & " + " & ".join(
            f"{g.loc[e, 'p_value']:.3f}" if g.loc[e, "p_value"] > 0.0100 else "$<$0.01" for e in MAJOR) + " \\\\")
    write("permutation", tab("ll" + "r" * 9, "model & & " + " & ".join(MAJOR), rows))

# ---------------------------------------------------------------- leakage
if (RES / "leakage.csv").exists():
    lk = pd.read_csv(RES / "leakage.csv")
    rows = []
    for (m, prot), g in lk.groupby(["model", "protocol"], sort=False):
        g = g.set_index("element")
        rows.append(f"{m} & {prot} & {f(g.loc[MAJOR, 'skill'].mean())} & {g.loc['Al', 'MAPE']:.2f} & "
                    f"{g.loc['Ca', 'MAPE']:.1f} & {g.loc['Cu', 'MAPE']:.1f} \\\\")
    write("leakage", tab("llrrrr", "model & split & skill & \\multicolumn{3}{c}{MAPE [\\%]: Al, Ca, Cu}", rows))

# ---------------------------------------------------------------- MAPE vs the paper
paper = pd.read_csv(ROOT / "data/hidsag_paper_table5.csv", comment="#").set_index("element")
crop = pd.read_csv(RES / "benchmark.csv")
mean = crop[crop.model == "mean"].set_index("element").MAPE
ours = crop[(crop.model == "PCA + ridge") & (crop.prep == "raw")].set_index("element").MAPE
leaky = None
if (RES / "leakage.csv").exists():
    leaky = lk[(lk.model == "cluster + PLS (HIDSAG-like)") & (lk.protocol.str.startswith("random"))] \
        .set_index("element").MAPE
rows = []
for e in MAJOR:
    rows.append(f"{e} & {mean[e]:.1f} & {ours[e]:.1f} & " + (f"{leaky[e]:.1f}" if leaky is not None else "--")
                + f" & {paper.loc[e, 'swir']:.1f} & {paper.loc[e, 'full']:.3f} \\\\")
write("mape", tab("lrrrrr", r"& \multicolumn{3}{c}{this work} & \multicolumn{2}{c}{HIDSAG paper} \\ "
      r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}" "\nelement & mean pred. & LOSO & random patches & SWIR & full", rows))

# ---------------------------------------------------------------- robustness
if (RES / "robustness.csv").exists():
    r = pd.read_csv(RES / "robustness.csv")
    rows = [f"{v.variant} & {v.model} & {v.n_samples} & {f(v.mean_skill)} & {f(v.Ca)} & {f(v.Fe)} & {f(v.Cu)} \\\\"
            for v in r.itertuples()]
    write("robustness", tab("llrrrrr", "variant & model & samples & skill & Ca & Fe & Cu", rows))

# ---------------------------------------------------------------- band depths: correlations and variance split
c = pd.read_csv(RES / "feature_correlations.csv", index_col=0)
v = pd.read_csv(RES / "feature_variance.csv", index_col=0)
rows = [f"{k} & " + " & ".join(f(c.loc[k, e]) for e in ["Al", "K", "S", "Ca", "Fe", "Cu"])
        + f" & {v.loc[k, 'sd_between_samples']:.4f} & {f(v.loc[k, 'coarse_minus_fine'], 4)} \\\\" for k in c.index]
write("features", tab("l" + "r" * 8, r"& \multicolumn{6}{c}{correlation with element (28 samples)} & "
      r"\multicolumn{2}{c}{variation} \\ \cmidrule(lr){2-7}\cmidrule(lr){8-9}"
      "\nfeature & Al & K & S & Ca & Fe & Cu & between samples (sd) & coarse $-$ fine", rows))
