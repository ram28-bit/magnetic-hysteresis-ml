import numpy as np
import pandas as pd

from . import config
from .physics import waveform, waveform_factors


def peak_field(material, b_peak, temperature):
    spec = config.MATERIALS[material]
    mu = config.VACUUM_PERMEABILITY * spec["mu_r"] * (1.0 - 0.0015 * (np.asarray(temperature) - 25.0))
    scale = spec["bs"] / mu
    return scale * np.arctanh(np.asarray(b_peak) / spec["bs"])


def _core_loss(spec, f, t, b, kinds, rng):
    lf = np.log10(f / 1e5)
    lb = np.log10(b / 0.1)
    alpha = spec["alpha"] + 0.12 * lf
    beta = spec["beta"] + 0.25 * lf
    log_p = np.log10(spec["p0"]) + alpha * lf + beta * lb + 0.10 * lf * lb
    thermal = 1.0 + spec["tc"] * ((t - spec["t0"]) / 50.0) ** 2
    loss = np.power(10.0, log_p) * thermal * waveform_factors(kinds, alpha)
    return loss * rng.lognormal(0.0, 0.03, size=loss.shape)


def _loops(material, f, t, b, kinds):
    spec = config.MATERIALS[material]
    n = config.LOOP_POINTS
    h_peak = peak_field(material, b, t)
    scale = spec["bs"] / (config.VACUUM_PERMEABILITY * spec["mu_r"] * (1.0 - 0.0015 * (t - 25.0)))
    h = np.empty((len(f), n))
    flux = np.empty((len(f), n))
    for i in range(len(f)):
        lag = 0.008 + 0.03 * (f[i] / config.FREQUENCY_RANGE[1]) + 0.0002 * (t[i] - 25.0) / 10.0
        drive = waveform(kinds[i], n)
        delayed = waveform(kinds[i], n, phase=lag)
        h[i] = h_peak[i] * drive
        flux[i] = spec["bs"] * np.tanh(h_peak[i] * delayed / scale[i])
    return h, flux


def generate_dataset(samples_per_material=config.SAMPLES_PER_MATERIAL, seed=config.SEED):
    rng = np.random.default_rng(seed)
    frames, h_all, b_all = [], [], []
    for material, spec in config.MATERIALS.items():
        n = samples_per_material
        f = np.power(10.0, rng.uniform(np.log10(config.FREQUENCY_RANGE[0]), np.log10(config.FREQUENCY_RANGE[1]), n))
        t = rng.uniform(*config.TEMPERATURE_RANGE, n)
        b = np.power(10.0, rng.uniform(np.log10(config.FLUX_RANGE[0]), np.log10(config.FLUX_RANGE[1]), n))
        kinds = rng.choice(config.WAVEFORMS, n)
        loss = _core_loss(spec, f, t, b, kinds, rng)
        h, flux = _loops(material, f, t, b, kinds)
        frames.append(
            pd.DataFrame(
                {
                    "material_id": material,
                    "material_family": spec["family"],
                    "frequency_hz": f,
                    "temperature_c": t,
                    "waveform": kinds,
                    "b_peak_t": b,
                    "b_peak_to_peak_t": 2.0 * b,
                    "h_peak_a_per_m": peak_field(material, b, t),
                    "core_loss_w_per_m3": loss,
                    "source": "synthetic",
                }
            )
        )
        h_all.append(h)
        b_all.append(flux)
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "sample_id", np.arange(len(frame)))
    return frame, np.vstack(h_all), np.vstack(b_all)


def save_dataset(frame, h, b):
    raw = config.DATA_DIR / "raw"
    processed = config.DATA_DIR / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw / "measurements.csv", index=False)
    np.savez_compressed(processed / "loop_sequences.npz", h=h, b=b, sample_id=frame["sample_id"].to_numpy())


def load_dataset():
    frame = pd.read_csv(config.DATA_DIR / "raw" / "measurements.csv")
    loops = np.load(config.DATA_DIR / "processed" / "loop_sequences.npz")
    return frame, loops["h"], loops["b"]
