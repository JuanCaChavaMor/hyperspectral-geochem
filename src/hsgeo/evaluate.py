"""Validation protocols and metrics.

Every crop and every pixel of a sample carries the *same* label (one pXRF analysis per sample). The unit of
independence is therefore the sample: a model must be tested on samples it has never seen. Splitting by
crop or by patch puts spectra of the test sample into the training set and measures recognition of a
sample, not prediction of a composition.
"""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

from .data import ELEMENTS


def tag_centre(X, tag, tr):
    """Grain-size correction: subtract from every row the mean spectrum of its tag (fine / coarse), computed on
    the training rows `tr` only. Coarse crops have systematically deeper absorptions than fine crops of the
    same sample (by ~3x the sample-to-sample spread), an effect that has nothing to do with composition."""
    Xc = X.copy()
    for t in np.unique(tag):
        Xc[tag == t] -= X[tr & (tag == t)].mean(0)
    return Xc


def cross_predict(make_model, ds, X, protocol="sample", seed=0, centre_tags=False):
    """Out-of-fold predictions, aggregated to one prediction per sample (mean over its rows).

    protocol: "sample" (leave-one-sample-out), "crop" (leave-one-crop-out) or "random" (80/20 row split,
    repeated so that every row is predicted once: 5 folds of a shuffled index).
    centre_tags: apply `tag_centre` inside every fold.
    """
    P = np.full_like(ds.Y, np.nan)
    if protocol in ("sample", "crop"):
        groups = ds.sample if protocol == "sample" else ds.crop
        folds = LeaveOneGroupOut().split(X, ds.Y, groups)
    elif protocol == "random":
        idx = np.random.default_rng(seed).permutation(len(X))
        folds = ((np.setdiff1d(idx, te), te) for te in np.array_split(idx, 5))
    else:
        raise ValueError(protocol)
    for tr, te in folds:
        Xf = X
        if centre_tags:
            m = np.zeros(len(X), bool); m[tr] = True
            Xf = tag_centre(X, ds.tag, m)
        P[te] = make_model().fit(Xf[tr], ds.Y[tr], ds.sample[tr]).predict(Xf[te])
    return per_sample(P, ds)


def per_sample(P, ds):
    s = np.unique(ds.sample)
    Ps = np.array([P[ds.sample == g].mean(0) for g in s])
    Ys = np.array([ds.Y[ds.sample == g][0] for g in s])
    return s, Ps, Ys


def loo_mean_mae(Ys):
    """MAE of the leave-one-sample-out mean predictor, per element."""
    n = len(Ys)
    return np.mean([np.abs(Ys[i] - np.delete(Ys, i, 0).mean(0)) for i in range(n)], axis=0)


def metrics(Ps, Ys):
    """Per element: MAE, MAPE [%], skill = 1 - MAE / MAE(mean predictor), Q^2 (predictive R^2)."""
    mae = np.abs(Ps - Ys).mean(0)
    base = loo_mean_mae(Ys)
    n = len(Ys)
    loo_mean = np.array([np.delete(Ys, i, 0).mean(0) for i in range(n)])
    q2 = 1 - ((Ps - Ys) ** 2).sum(0) / ((loo_mean - Ys) ** 2).sum(0)
    mape = 100 * (np.abs(Ps - Ys) / np.abs(Ys).clip(1e-9)).mean(0)
    return {"MAE": mae, "MAPE": mape, "skill": 1 - mae / base, "Q2": q2}


def permutation_pvalue(make_model, ds, X, n_perm=200, seed=0, centre_tags=False):
    """P-value of the LOSO skill under random relabelling of whole samples (labels move with their sample)."""
    rng = np.random.default_rng(seed)
    _, Ps, Ys = cross_predict(make_model, ds, X, centre_tags=centre_tags)
    obs = metrics(Ps, Ys)["skill"]
    samples = np.unique(ds.sample)
    lab = {g: ds.Y[ds.sample == g][0] for g in samples}
    count = np.zeros(len(ELEMENTS))
    Y0 = ds.Y
    for _ in range(n_perm):
        perm = dict(zip(samples, rng.permutation(samples)))
        ds.Y = np.array([lab[perm[g]] for g in ds.sample])
        _, Pp, Yp = cross_predict(make_model, ds, X, centre_tags=centre_tags)
        count += metrics(Pp, Yp)["skill"] >= obs
    ds.Y = Y0
    return obs, (count + 1) / (n_perm + 1)
