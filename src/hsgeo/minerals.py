"""Mineral -> element conversion by stoichiometry (ideal end-member formulas).

Element mass fractions are computed from the formulas with standard atomic weights. Solid solutions are
represented by a mid-composition (biotite, chlorite: Fe/(Fe+Mg) = 0.5; plagioclase by its An content);
QEMSCAN's "Anhydrite/Gypsum" by the mean of the two. Minor phases without a sensible ideal formula are omitted.
"""
import re

import numpy as np

ATOMIC = {"H": 1.008, "C": 12.011, "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982,
          "Si": 28.085, "P": 30.974, "S": 32.06, "K": 39.098, "Ca": 40.078, "Ti": 47.867, "Fe": 55.845,
          "Cu": 63.546, "Mo": 95.95}


def composition(formula):
    """Element counts of a flat formula like 'K1Al3Si3O12H2' (numbers may be decimal; no brackets)."""
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)([0-9.]*)", formula):
        out[el] = out.get(el, 0) + (float(n) if n else 1.0)
    return out


def mass_fractions(formula):
    c = composition(formula)
    m = {el: n * ATOMIC[el] for el, n in c.items()}
    tot = sum(m.values())
    return {el: v / tot for el, v in m.items()}


def _plag(an):
    """Plagioclase Ab(1-x) An(x): Na(1-x) Ca(x) Al(1+x) Si(3-x) O8."""
    return f"Na{1 - an:g}Ca{an:g}Al{1 + an:g}Si{3 - an:g}O8"


# QEMSCAN classes of MINERAL1 -> flat formulas
FORMULAS = {
    "Quartz": "Si1O2",
    "Muscovite/Sericite": "K1Al3Si3O12H2",             # KAl2(AlSi3O10)(OH)2
    "K-Feldspar": "K1Al1Si3O8",
    "Albite": _plag(0.0), "Oligoclasa_An20": _plag(0.2), "Andesina_An40": _plag(0.4),
    "Labradorite_An60": _plag(0.6), "Bytownite_An80": _plag(0.8), "Anorthite": _plag(1.0),
    "Biotite": "K1Mg1.5Fe1.5Al1Si3O12H2",              # K(Mg,Fe)3AlSi3O10(OH)2, Fe/(Fe+Mg) = 0.5
    "Chlorite": "Mg2.5Fe2.5Al2Si3O18H8",               # (Mg,Fe)5Al(AlSi3O10)(OH)8
    "Kaolinite": "Al2Si2O9H4",                          # Al2Si2O5(OH)4
    "Pyrite": "Fe1S2",
    "Chalcopyrite": "Cu1Fe1S2",
    "Bornite": "Cu5Fe1S4",
    "Chalcosite/Digenite": "Cu2S1",
    "Covelite": "Cu1S1",
    "Molybdenite": "Mo1S2",
    "Fe Oxides": "Fe3O4",
    "Rutile": "Ti1O2",
    "Calcite": "Ca1C1O3",
    "Siderite": "Fe1C1O3",
    "Apatite": "Ca5P3O12F1",
}
_GYP, _ANH = mass_fractions("Ca1S1O6H4"), mass_fractions("Ca1S1O4")
FRACTIONS = {k: mass_fractions(v) for k, v in FORMULAS.items()}
FRACTIONS["Anhydrite/Gypsum"] = {el: (_GYP.get(el, 0) + _ANH.get(el, 0)) / 2 for el in set(_GYP) | set(_ANH)}


def stoichiometric_matrix(minerals, elements):
    """A[i, j] = mass fraction of element j in mineral i (0 for minerals without formula)."""
    return np.array([[FRACTIONS.get(m, {}).get(e, 0.0) for e in elements] for m in minerals])


def elements_from_minerals(M, minerals, elements):
    """M: (n, minerals) in wt %  ->  (n, elements) in wt %."""
    return np.clip(M, 0, None) @ stoichiometric_matrix(minerals, elements)
