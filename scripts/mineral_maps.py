"""Pixel maps of absorption features and spectral clusters for a low-Ca and a high-Ca sample (Fig. 4).

    python scripts/mineral_maps.py      # needs the four crops in data/raw (see README), ~30 s

Writes figures/mineral_maps.{png,pdf} and results/cluster_shares.csv.
"""
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from _path import ROOT
from hsgeo.data import ELEMENTS
from hsgeo.features import FEATURES
from hsgeo.maps import band_depth_map, cluster_pixels, load_cube, tray_index, valid_mask
from hsgeo.plotting import C, INK, MUTED, image_axis, mark_features, plt, save

CROPS = ["GCH-0015/01", "GCH-0015/03", "GCH-0019/01", "GCH-0019/03"]
MAPS = ["H2O 1935", "gyp 1745", "AlOH 2205"]
K = 4

labels = pd.read_csv(ROOT / "data/labels.csv", index_col=0)
tags = pd.read_csv(ROOT / "data/crop_spectra.csv.gz").set_index("crop")["tag"]
raw = ROOT / "data/raw"
cubes = []
for c in CROPS:
    cube, fz = load_cube(raw / c)
    cubes.append(cube)
maps, centres, w, share = cluster_pixels(cubes, fz.wl, fz.n_vnir, k=K)

depth = {f: [np.where(valid_mask(c), band_depth_map(c, fz.wl, f), np.nan) for c in cubes] for f in MAPS}
lim = {f: np.nanpercentile(np.concatenate([d.ravel() for d in depth[f]]), [2, 98]) for f in MAPS}

fig = plt.figure(figsize=(10, 8.6))
gs = fig.add_gridspec(len(CROPS) + 1, len(MAPS) + 2, height_ratios=[1] * len(CROPS) + [1.5], hspace=0.35,
                      wspace=0.08)
cmap_k = ListedColormap(["#ffffff"] + C[:K])
for i, (c, cube) in enumerate(zip(CROPS, cubes)):
    s, n = c.split("/")
    crop = f"{s}_{n}"
    ax = fig.add_subplot(gs[i, 0]); image_axis(ax)
    ax.imshow(mpimg.imread(raw / c / "vnir_low.png"), aspect="auto")
    ax.set_ylabel(f"{crop}\n{tags[crop]}, Ca {labels.loc[s, 'Ca']:.2f} %", fontsize=8, color=INK)
    for j, f in enumerate(MAPS):
        ax = fig.add_subplot(gs[i, j + 1]); image_axis(ax)
        im = ax.imshow(depth[f][i], vmin=lim[f][0], vmax=lim[f][1], cmap="viridis", aspect="auto")
        ax.text(0.02, 0.04, f"median {np.nanmedian(depth[f][i]):.3f}", transform=ax.transAxes, fontsize=7,
                color="white")
        if i == 0:
            ax.set_title(f"depth {f} nm", fontsize=9, color=INK)
    ax = fig.add_subplot(gs[i, -1]); image_axis(ax)
    ax.imshow(maps[i] + 1, cmap=cmap_k, vmin=0, vmax=K, aspect="auto", interpolation="nearest")
    if i == 0:
        ax.set_title("spectral cluster", fontsize=9, color=INK)
fig.axes[0].set_title("VNIR rendering", fontsize=9, color=INK)

# the cluster with the C-H overtone at ~1210 nm (also 1730, 2310) is the plastic sample tray, not rock
tray = tray_index(centres, w)
names = [f"cluster {k + 1}" + (" = tray (C-H)" if k == tray else "") for k in range(K)]
ax = fig.add_subplot(gs[-1, :])
for k in range(K):
    ax.plot(w, centres[k], color=C[k], label=names[k])
ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("continuum-removed reflectance")
ax.set_xlim(w[0], w[-1])
mark_features(ax, FEATURES)
ax.legend(ncol=K, loc="lower left", fontsize=8)
save(fig, ROOT / "figures/mineral_maps.png")

out = pd.DataFrame(share, index=[c.replace("/", "_") for c in CROPS], columns=names)
out.to_csv(ROOT / "results/cluster_shares.csv", float_format="%.3f")
print(out.round(2))
