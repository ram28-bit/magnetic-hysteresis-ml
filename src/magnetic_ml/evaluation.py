import numpy as np

from .physics import closure_error, loop_area


def core_loss_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    relative = np.abs(error) / y_true
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": float(100.0 * np.mean(relative)),
        "r2": 1.0 - ss_res / ss_tot,
        "mae_log10": float(np.mean(np.abs(np.log10(y_pred) - np.log10(y_true)))),
        "within_10": float(100.0 * np.mean(relative <= 0.10)),
        "within_20": float(100.0 * np.mean(relative <= 0.20)),
        "within_30": float(100.0 * np.mean(relative <= 0.30)),
    }


def loop_metrics(h, b_true, b_pred):
    rmse = float(np.sqrt(np.mean((b_true - b_pred) ** 2)))
    area_true = np.array([loop_area(x, y) for x, y in zip(h, b_true)])
    area_pred = np.array([loop_area(x, y) for x, y in zip(h, b_pred)])
    closure = np.array([closure_error(y) for y in b_pred])
    return {
        "b_rmse_t": rmse,
        "area_mape": float(100.0 * np.mean(np.abs(area_pred - area_true) / area_true)),
        "peak_b_mae_t": float(np.mean(np.abs(np.max(np.abs(b_pred), axis=1) - np.max(np.abs(b_true), axis=1)))),
        "closure_error": float(np.mean(closure)),
    }


def make_splits(frame, seed=42, test_fraction=0.2):
    rng = np.random.default_rng(seed)
    index = np.arange(len(frame))
    shuffled = rng.permutation(index)
    random_test = shuffled[: int(test_fraction * len(frame))]

    def pair(mask):
        mask = np.asarray(mask)
        return index[~mask], index[mask]

    random_mask = np.zeros(len(frame), dtype=bool)
    random_mask[random_test] = True
    return {
        "random": pair(random_mask),
        "unseen_frequency": pair(frame["frequency_hz"].to_numpy() >= frame["frequency_hz"].quantile(0.85)),
        "unseen_temperature": pair(frame["temperature_c"].to_numpy() >= 105.0),
        "unseen_waveform": pair(frame["waveform"].to_numpy() == "pwm"),
        "unseen_material": pair(frame["material_id"].to_numpy() == "N49"),
    }
