"""Absorption band depths: the classical remote-sensing features of mineral spectroscopy.

Band depth at centre c with shoulders (a, b):  BD = 1 - R(c) / R_cont(c), where R_cont is the straight line
between the reflectance at the shoulders. It is insensitive to brightness and slope, and points to minerals:
"""
import numpy as np

# name: (left shoulder, centre, right shoulder) [nm], mineral meaning
FEATURES = {
    "OH 1405":    ((1340, 1405, 1470), "hydroxyl and water in micas, clays and gypsum"),
    "gyp 1745":   ((1700, 1745, 1800), "gypsum (CaSO4·2H2O)"),
    "H2O 1935":   ((1880, 1935, 2050), "molecular water (gypsum, smectite)"),
    "AlOH 2205":  ((2160, 2205, 2235), "Al-OH of muscovite/sericite, kaolinite"),
    "Fe/MgOH 2300": ((2275, 2300, 2320), "Fe-OH / Mg-OH (chlorite, biotite)"),
}
# ratio index of the VNIR: ferric iron absorbs strongly below ~550 nm (charge transfer), so iron-rich
# material is "redder"
RATIOS = {"red/blue 700/480": ((700, 480), "ferric iron (charge-transfer edge)")}


def band_depth(R, wl, left, centre, right):
    """R: (..., bands) reflectance on wavelengths wl. Returns (...) band depths."""
    def at(w):
        i = np.argsort(np.abs(wl - w))[:3]                          # average of the 3 nearest bands
        return R[..., i].mean(-1)
    ra, rc, rb = at(left), at(centre), at(right)
    cont = ra + (rb - ra) * (centre - left) / (right - left)
    return 1.0 - rc / cont


def _at(R, wl, w):
    i = np.argsort(np.abs(wl - w))[:3]
    return R[..., i].mean(-1)


def all_features(R, wl):
    out = {k: band_depth(R, wl, *v[0]) for k, v in FEATURES.items()}
    out.update({k: _at(R, wl, v[0][0]) / _at(R, wl, v[0][1]) for k, v in RATIOS.items()})
    return out
