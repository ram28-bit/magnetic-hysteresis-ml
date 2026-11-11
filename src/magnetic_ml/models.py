import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import build_features
from .physics import fit_steinmetz, waveform_factors


def _boosting():
    return HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=24,
        l2_regularization=0.1,
        random_state=config.SEED,
    )


def _network():
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            alpha=1e-4,
            early_stopping=True,
            max_iter=800,
            random_state=config.SEED,
        ),
    )


def _loop_network():
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(256, 256),
            alpha=1e-5,
            early_stopping=True,
            max_iter=600,
            random_state=config.SEED,
        ),
    )


def _fit_parameters(frame):
    return fit_steinmetz(
        frame["frequency_hz"].to_numpy(float),
        frame["b_peak_t"].to_numpy(float),
        frame["core_loss_w_per_m3"].to_numpy(float),
        frame["waveform"].to_numpy(),
    )


class SteinmetzBaseline:
    def fit(self, frame):
        self.global_ = _fit_parameters(frame)
        self.material_ = {}
        self.band_ = {}
        for material, group in frame.groupby("material_id"):
            self.material_[material] = _fit_parameters(group)
            bands = np.digitize(group["temperature_c"], config.TEMPERATURE_EDGES)
            for band in np.unique(bands):
                subset = group[bands == band]
                if len(subset) >= 12:
                    self.band_[(material, int(band))] = _fit_parameters(subset)
        return self

    def _parameters(self, frame):
        bands = np.digitize(frame["temperature_c"], config.TEMPERATURE_EDGES)
        rows = []
        for material, band in zip(frame["material_id"], bands):
            rows.append(
                self.band_.get((material, int(band)))
                or self.material_.get(material)
                or self.global_
            )
        return np.array(rows, dtype=float)

    def predict(self, frame):
        k, alpha, beta = self._parameters(frame).T
        f = frame["frequency_hz"].to_numpy(float)
        b = frame["b_peak_t"].to_numpy(float)
        base = k * np.power(f, alpha) * np.power(b, beta)
        return base * waveform_factors(frame["waveform"].to_numpy(), alpha)


class TabularLossModel:
    def __init__(self, kind):
        self.kind = kind
        self.model = _boosting() if kind == "boosting" else _network()

    def fit(self, frame):
        y = np.log10(frame["core_loss_w_per_m3"].to_numpy(float))
        self.model.fit(build_features(frame).to_numpy(), y)
        return self

    def predict(self, frame):
        return np.power(10.0, self.model.predict(build_features(frame).to_numpy()))


class HybridLossModel:
    def __init__(self):
        self.baseline = SteinmetzBaseline()
        self.correction = _boosting()

    def fit(self, frame):
        self.baseline.fit(frame)
        residual = np.log(frame["core_loss_w_per_m3"].to_numpy(float) / self.baseline.predict(frame))
        self.correction.fit(build_features(frame).to_numpy(), residual)
        return self

    def predict(self, frame):
        delta = self.correction.predict(build_features(frame).to_numpy())
        return self.baseline.predict(frame) * np.exp(delta)


class LoopModel:
    def __init__(self):
        self.model = _loop_network()

    def fit(self, frame, flux):
        target = flux / frame["b_peak_t"].to_numpy(float)[:, None]
        self.model.fit(build_features(frame).to_numpy(), target)
        return self

    def predict(self, frame):
        normalised = self.model.predict(build_features(frame).to_numpy())
        return normalised * frame["b_peak_t"].to_numpy(float)[:, None]
