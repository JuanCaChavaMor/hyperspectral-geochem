"""Grain-size corrections for reflectance spectra of crushed ore.

Coarse crops of a sample absorb more than its fine crops (longer optical path per grain). Three families of
corrections are implemented, all usable inside a cross-validation fold:

1. fixed transforms (no parameters): pseudo-absorbance log(1/R), Kubelka-Munk F(R) = (1-R)^2 / 2R, and the
   transform u / log(1+u) with u = log(1/R) of Aucott et al. (1988);
2. paired corrections, fitted on the fine/coarse pairs of the *training* samples:
   EPO, external parameter orthogonalisation (Roger et al., 2003): remove the subspace spanned by the
   coarse - fine differences; EMSC (Martens & Stark, 1991) with the mean coarse - fine difference as interferent;
3. a physical model: Hapke's radiative transfer (isotropic scatterers, no opposition effect) converts reflectance
   to single-scattering albedo w; Hapke's equivalent-slab model relates w to the internal transmission
   Theta = exp(-alpha D) of a grain of size D. In log(alpha D) = log(alpha) + log(D) the grain size is an additive
   constant per spectrum, which per-spectrum centring removes (exactly under the model's assumptions).
"""
import numpy as np

from .preprocess import snv


# ---------------------------------------------------------------- 1. fixed transforms
def log_inv(R):
    return -np.log(np.clip(R, 1e-4, None))


def kubelka_munk(R):
    R = np.clip(R, 1e-4, 0.9999)
    return (1 - R) ** 2 / (2 * R)


def aucott(R):
    u = log_inv(R)
    return u / np.log1p(u)


# ---------------------------------------------------------------- 2. paired corrections
def _pair_differences(X, ds, rows):
    """coarse-mean minus fine-mean spectrum of every sample (among `rows`) that has both fractions."""
    D = []
    for s in np.unique(ds.sample[rows]):
        ii = rows & (ds.sample == s)
        c, f = ii & (ds.tag == "coarse"), ii & (ds.tag == "fine")
        if c.any() and f.any():
            D.append(X[c].mean(0) - X[f].mean(0))
    return np.array(D)


def epo(k=1):
    """Fold transform: X -> X (I - V_k V_k^T), V_k the first k right singular vectors of the training differences."""
    def f(X, ds, tr):
        D = _pair_differences(X, ds, tr)
        V = np.linalg.svd(D, full_matrices=False)[2][:k]
        return X - (X @ V.T) @ V
    return f


def emsc(degree=2):
    """Fold transform: EMSC with reference = mean training spectrum, polynomial baseline and the mean training
    coarse - fine difference as a known interferent; returns the corrected spectra (x - baseline - e g) / b."""
    def f(X, ds, tr):
        ref = X[tr].mean(0)
        g = _pair_differences(X, ds, tr).mean(0)
        t = np.linspace(-1, 1, X.shape[1])
        B = np.c_[ref, np.vander(t, degree + 1, increasing=True), g]   # columns: b, a, c, d, ..., e
        coef = np.linalg.lstsq(B, X.T, rcond=None)[0].T                 # (n, n_basis)
        b = coef[:, :1]
        rest = coef[:, 1:] @ B[:, 1:].T
        return (X - rest) / b
    return f


# ---------------------------------------------------------------- 3. Hapke
def _H(x, w):
    """Chandrasekhar H function, Hapke's (2002) approximation for isotropic scatterers."""
    g = np.sqrt(1 - w)
    r0 = (1 - g) / (1 + g)
    return 1 / (1 - w * x * (r0 + (1 - 2 * r0 * x) / 2 * np.log((1 + x) / x)))


def hapke_reflectance(w, mu0=np.cos(np.radians(30)), mu=1.0):
    """Reflectance factor (relative to a perfect Lambertian surface) of a half-space of isotropic scatterers."""
    return w / 4 / (mu0 + mu) * _H(mu0, w) * _H(mu, w)


_W = np.linspace(1e-4, 0.99999, 20000)


def single_scattering_albedo(R, **geom):
    """Invert hapke_reflectance numerically (monotone in w)."""
    Rw = hapke_reflectance(_W, **geom)
    return np.interp(np.clip(R, Rw[0], Rw[-1]), Rw, _W)


def internal_transmission(w, n=1.55):
    """Hapke's equivalent-slab model: w = Se + (1-Se)(1-Si) Theta / (1 - Si Theta), solved for Theta.
    Se, Si: external / internal surface reflection coefficients of a grain with refractive index n."""
    Se = (n - 1) ** 2 / (n + 1) ** 2 + 0.05
    Si = 1 - 4 / (n * (n + 1) ** 2)
    w = np.clip(w, Se + 1e-4, None)
    return np.clip((w - Se) / ((1 - Se) * (1 - Si) + Si * (w - Se)), 1e-6, 1 - 1e-6)


# HIDSAG values are relative to the camera's white reference and exceed 1 (VNIR up to 1.32, SWIR up to 1.12 at
# crop level), so their absolute reflectance is unknown. Assumption: the brightest crop of the data set has a
# reflectance factor of 1/1.1 = 0.91 in each sensor. scripts/grain_size.py checks the sensitivity to this choice.
SCALE = {"vnir": 1.45, "swir": 1.23}


def scaled(R, n_vnir, scale=SCALE):
    """Reflectance factor under the assumption above (both sensors in [0, 1))."""
    R = R.astype(float).copy()
    R[..., :n_vnir] /= scale["vnir"]
    R[..., n_vnir:] /= scale["swir"]
    return R


def hapke_log_absorption(R, n_vnir, scale=SCALE, **geom):
    """log(alpha D) per band, centred per spectrum and per sensor: the grain size D cancels."""
    R = scaled(R, n_vnir, scale)
    w = single_scattering_albedo(R, **geom)
    aD = -np.log(internal_transmission(w))
    L = np.log(aD)
    for a, b in ((0, n_vnir), (n_vnir, R.shape[-1])):
        L[..., a:b] -= L[..., a:b].mean(-1, keepdims=True)
    return L


TRANSFORMS = {
    "raw": lambda X, nv: X,
    "SNV": lambda X, nv: snv(X),
    "log(1/R)": lambda X, nv: log_inv(scaled(X, nv)),
    "SNV of log(1/R)": lambda X, nv: snv(log_inv(scaled(X, nv))),
    "Kubelka-Munk": lambda X, nv: kubelka_munk(scaled(X, nv)),
    "Aucott u/log(1+u)": lambda X, nv: aucott(scaled(X, nv)),
    "Hapke log(alpha D)": lambda X, nv: hapke_log_absorption(X, nv),
}
