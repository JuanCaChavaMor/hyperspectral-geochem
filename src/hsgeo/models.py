"""Regressors from spectra to the 18-element composition. All share fit(X, Y, groups) / predict(X).

Targets are standardised inside every model (elements differ by four orders of magnitude), and every
hyper-parameter is chosen by an inner group-wise cross-validation on the training samples only.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class _Scaled:
    def _fit_y(self, Y):
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-12
        return (Y - self.mu) / self.sd

    def _unscale(self, Z):
        return Z * self.sd + self.mu


class MeanPredictor:
    """Predicts the training mean: the floor every model has to beat."""
    name = "mean"

    def fit(self, X, Y, groups=None):
        self.m = Y.mean(0)
        return self

    def predict(self, X):
        return np.tile(self.m, (len(X), 1))


def _inner_select(make, grid, X, Z, groups, n_splits=4):
    """Pick the grid value with the lowest inner group-CV error (standardised targets)."""
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    err = []
    for p in grid:
        e = 0.0
        for tr, te in gkf.split(X, Z, groups):
            m = make(p).fit(X[tr], Z[tr])
            e += np.mean((m.predict(X[te]) - Z[te]) ** 2) * len(te)
        err.append(e)
    return grid[int(np.argmin(err))]


class PLS(_Scaled):
    """Partial least squares, the standard of chemometrics; number of components by inner CV."""
    name = "PLS"

    def __init__(self, grid=(1, 2, 3, 4, 6, 8, 10, 12, 15)):
        self.grid = grid

    def fit(self, X, Y, groups):
        Z = self._fit_y(Y)
        # inner folds train on ~3/4 of the rows: keep k below that
        grid = [k for k in self.grid if k < min(int(0.7 * len(X)), X.shape[1]) - 1]
        self.k = _inner_select(lambda k: PLSRegression(k, scale=True), grid, X, Z, groups)
        self.m = PLSRegression(self.k, scale=True).fit(X, Z)
        return self

    def predict(self, X):
        return self._unscale(self.m.predict(X))


class PCARidge(_Scaled):
    """Principal components + ridge regression; components and penalty by inner CV."""
    name = "PCA + ridge"

    def __init__(self, grid=tuple((k, a) for k in (5, 10, 20) for a in (0.1, 1.0, 10.0, 100.0))):
        self.grid = grid

    def _make(self, p):
        k, a = p
        return make_pipeline(StandardScaler(), PCA(k), Ridge(a))

    def fit(self, X, Y, groups):
        Z = self._fit_y(Y)
        grid = [p for p in self.grid if p[0] < min(int(0.7 * len(X)), X.shape[1]) - 1]
        self.p = _inner_select(self._make, grid, X, Z, groups)
        self.m = self._make(self.p).fit(X, Z)
        return self

    def predict(self, X):
        return self._unscale(self.m.predict(X))


class ClusterPLS(_Scaled):
    """Approximation of the HIDSAG validation pipeline: k-means clusters of patch spectra, one PLS per cluster."""
    name = "cluster + PLS (HIDSAG-like)"

    def __init__(self, k=6, n_comp=8):
        self.k, self.n_comp = k, n_comp

    def fit(self, X, Y, groups=None):
        Z = self._fit_y(Y)
        self.sc = StandardScaler().fit(X)
        Xs = self.sc.transform(X)
        self.km = KMeans(self.k, n_init=4, random_state=0).fit(Xs)
        self.global_ = PLSRegression(self.n_comp, scale=False).fit(Xs, Z)
        self.local = {}
        for c in range(self.k):
            ii = self.km.labels_ == c
            if ii.sum() > 3 * self.n_comp:
                self.local[c] = PLSRegression(self.n_comp, scale=False).fit(Xs[ii], Z[ii])
        return self

    def predict(self, X):
        Xs = self.sc.transform(X)
        lab = self.km.predict(Xs)
        Z = self.global_.predict(Xs)
        for c, m in self.local.items():
            if np.any(lab == c):
                Z[lab == c] = m.predict(Xs[lab == c])
        return self._unscale(Z)


class CNN1D(_Scaled):
    """Small 1D convolutional network on (patch) spectra, trained with early stopping on held-out samples."""
    name = "1D-CNN"

    def __init__(self, epochs=150, patience=25, lr=2e-3, weight_decay=1e-3, seed=0):
        self.epochs, self.patience, self.lr, self.wd, self.seed = epochs, patience, lr, weight_decay, seed

    def _net(self, n_in, n_out):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv1d(1, 8, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 16, 7, padding=3), nn.GELU(), nn.AdaptiveAvgPool1d(8),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(16 * 8, n_out))

    def fit(self, X, Y, groups):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)            # small network: threads only compete with each other
        Z = self._fit_y(Y)
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.xm) / self.xs
        # early-stopping split by sample (never by row: rows of one sample share the label)
        g = np.unique(groups)
        rng = np.random.default_rng(self.seed)
        val_g = rng.choice(g, max(2, len(g) // 6), replace=False)
        va = np.isin(groups, val_g)
        tX = torch.tensor(Xn[~va, None], dtype=torch.float32); tZ = torch.tensor(Z[~va], dtype=torch.float32)
        vX = torch.tensor(Xn[va, None], dtype=torch.float32); vZ = torch.tensor(Z[va], dtype=torch.float32)
        net = self._net(X.shape[1], Y.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        best, state, bs = np.inf, None, 64
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(len(tX))
            for i in range(0, len(tX), bs):
                b = perm[i:i + bs]
                loss = torch.nn.functional.mse_loss(net(tX[b]), tZ[b])
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(net(vX), vZ).item()
            if v < best:
                best, state, self.best_epoch = v, {k: t.clone() for k, t in net.state_dict().items()}, ep
            elif ep - self.best_epoch >= self.patience:
                break
        net.load_state_dict(state)
        self.net = net
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            Z = self.net(torch.tensor(((X - self.xm) / self.xs)[:, None], dtype=torch.float32)).numpy()
        return self._unscale(Z)


class BestBand:
    """Per element: one spectral feature + linear regression. The band (and its pre-processing) is the one with
    the largest |correlation| with the element on the *training* samples, i.e. it is re-selected in every fold.
    On tiny data sets this physically interpretable model is the natural competitor of PLS."""
    name = "best band"

    def __init__(self, n_bands=1):
        self.n_bands = n_bands

    def fit(self, X, Y, groups):
        # correlations on sample means (rows of one sample are not independent)
        g = np.unique(groups)
        Xs = np.array([X[groups == s].mean(0) for s in g]); Ys = np.array([Y[groups == s].mean(0) for s in g])
        Xc = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-12); Yc = (Ys - Ys.mean(0)) / (Ys.std(0) + 1e-12)
        R = Xc.T @ Yc / len(g)                                     # (bands, elements)
        self.bands, self.coef = [], []
        for j in range(Y.shape[1]):
            b = np.argsort(-np.abs(R[:, j]))[:self.n_bands]
            A = np.c_[Xs[:, b], np.ones(len(g))]
            c, *_ = np.linalg.lstsq(A, Ys[:, j], rcond=None)
            self.bands.append(b); self.coef.append(c)
        return self

    def predict(self, X):
        return np.stack([np.c_[X[:, b], np.ones(len(X))] @ c for b, c in zip(self.bands, self.coef)], 1)


class RidgeCV(_Scaled):
    """Standardised features + ridge regression, penalty by inner group CV (for low-dimensional features)."""
    name = "ridge"

    def __init__(self, grid=(0.1, 1.0, 10.0, 100.0, 1000.0)):
        self.grid = grid

    def _make(self, a):
        return make_pipeline(StandardScaler(), Ridge(a))

    def fit(self, X, Y, groups):
        Z = self._fit_y(Y)
        self.a = _inner_select(self._make, list(self.grid), X, Z, groups)
        self.m = self._make(self.a).fit(X, Z)
        return self

    def predict(self, X):
        return self._unscale(self.m.predict(X))
