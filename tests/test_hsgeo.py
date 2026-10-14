import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hsgeo import preprocess as pp  # noqa: E402
from hsgeo.data import ELEMENTS, Dataset, load_crop_means  # noqa: E402
from hsgeo.evaluate import cross_predict, metrics, tag_centre  # noqa: E402
from hsgeo.features import band_depth  # noqa: E402
from hsgeo.models import BestBand, MeanPredictor, PCARidge, PLS  # noqa: E402


@pytest.fixture(scope="module")
def ds():
    return load_crop_means()


def test_dataset_shape(ds):
    assert ds.X.shape == (106, len(ds.wl)) and ds.Y.shape == (106, len(ELEMENTS))
    assert len(np.unique(ds.sample)) == 28
    # one label per sample: every crop of a sample carries the same composition
    for s in np.unique(ds.sample):
        assert np.ptp(ds.Y[ds.sample == s], axis=0).max() == 0


def test_mean_predictor_has_zero_skill(ds):
    s, P, Y = cross_predict(MeanPredictor, ds, ds.X)
    m = metrics(P, Y)
    # rows are crops, so samples with four crops weigh a little more than in the per-sample LOO mean
    assert np.abs(m["skill"][:9]).max() < 0.03


def test_no_leakage_in_loso(ds):
    """A model that memorises its training rows must have no advantage under leave-one-sample-out."""
    class Lookup:
        def fit(self, X, Y, g):
            self.X, self.Y = X, Y; return self

        def predict(self, X):
            d = ((X[:, None] - self.X[None]) ** 2).sum(-1)
            return self.Y[d.argmin(1)]
    _, P, Y = cross_predict(Lookup, ds, ds.X, protocol="sample")
    _, Pc, _ = cross_predict(Lookup, ds, ds.X, protocol="crop")
    # lookup gains when another crop of the same sample is in the training set (only partly: fine and
    # coarse crops of one sample differ a lot), and has no skill on unseen samples
    assert metrics(Pc, Y)["skill"][:9].mean() > metrics(P, Y)["skill"][:9].mean() + 0.08
    assert abs(metrics(P, Y)["skill"][:9].mean()) < 0.05


def test_band_depth_of_synthetic_absorption():
    wl = np.linspace(1000, 2500, 301)
    R = 0.5 + 1e-4 * (wl - 1000) - 0.1 * np.exp(-0.5 * ((wl - 2200) / 15) ** 2)
    assert band_depth(R, wl, 2100, 2200, 2300) == pytest.approx(0.1 / (0.5 + 0.12), rel=0.05)


def test_preprocessing_shapes(ds):
    for name in ("raw", "snv", "d1", "cr"):
        Z = pp.apply(name, ds.X[:5].astype(float), ds.wl, 58)
        assert Z.shape == (5, len(ds.wl)) and np.isfinite(Z).all()
    cr = pp.apply("cr", ds.X[:5].astype(float), ds.wl, 58)
    assert cr.max() <= 1 + 1e-9


def test_tag_centre_uses_training_rows_only():
    X = np.arange(8.0)[:, None]
    tag = np.array(["fine", "coarse"] * 4)
    tr = np.array([True] * 6 + [False] * 2)
    Xc = tag_centre(X, tag, tr)
    assert Xc[0, 0] == 0 - np.mean([0, 2, 4]) and Xc[7, 0] == 7 - np.mean([1, 3, 5])


@pytest.mark.parametrize("M", [PLS, PCARidge, BestBand])
def test_models_recover_linear_signal(M):
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 3)
    Z = rng.normal(size=(20, 3))[g] + 0.05 * rng.normal(size=(60, 3))
    W = rng.normal(size=(3, 40))
    X = Z @ W + 0.05 * rng.normal(size=(60, 40))
    Y = Z[:, :2] * [1.0, 0.1] + [5.0, 0.3]
    d = Dataset(X, Y, g.astype(str), np.arange(60).astype(str), np.array(["fine"] * 60), np.arange(40.0))
    _, P, Ys = cross_predict(M, d, X)
    sk = metrics(P, Ys)["skill"]
    assert sk[0] > 0.3
