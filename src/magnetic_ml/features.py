import numpy as np
import pandas as pd

from . import config
from .physics import slope_statistics


def build_features(frame):
    f = frame["frequency_hz"].to_numpy(float)
    t = frame["temperature_c"].to_numpy(float)
    b = frame["b_peak_t"].to_numpy(float)
    kinds = frame["waveform"].to_numpy()
    stats = {k: slope_statistics(k) for k in config.WAVEFORMS}
    rms = np.array([stats[k]["rms"] for k in kinds])
    peak = np.array([stats[k]["peak"] for k in kinds])
    crest = np.array([stats[k]["crest"] for k in kinds])
    log_f = np.log10(f)
    log_b = np.log10(b)
    columns = {
        "log_frequency": log_f,
        "log_b_peak": log_b,
        "temperature_c": t,
        "temperature_squared": t**2,
        "log_frequency_b_peak": log_f * log_b,
        "log_dbdt_rms": np.log10(f * b * rms),
        "log_dbdt_max": np.log10(f * b * peak),
        "crest_factor": crest,
    }
    for material in config.MATERIAL_IDS:
        columns[f"material_{material}"] = (frame["material_id"].to_numpy() == material).astype(float)
    for kind in config.WAVEFORMS:
        columns[f"waveform_{kind}"] = (kinds == kind).astype(float)
    return pd.DataFrame(columns, index=frame.index)
