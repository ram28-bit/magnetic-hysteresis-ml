from functools import lru_cache

import numpy as np

from .config import WAVEFORMS


def waveform(kind, n=256, phase=0.0):
    t = (np.arange(n) / n - phase) % 1.0
    if kind == "sine":
        return np.sin(2.0 * np.pi * t)
    triangle = 1.0 - 4.0 * np.abs(((t + 0.25) % 1.0) - 0.5)
    if kind == "triangle":
        return triangle
    if kind == "trapezoid":
        return np.clip(1.5 * triangle, -1.0, 1.0)
    if kind == "pwm":
        duty = 0.25
        return np.where(t < duty, -1.0 + 2.0 * t / duty, 1.0 - 2.0 * (t - duty) / (1.0 - duty))
    raise ValueError(f"unknown waveform: {kind}")


@lru_cache(maxsize=None)
def _slope(kind, n=2048):
    w = waveform(kind, n)
    return np.abs(n * (np.roll(w, -1) - w))


@lru_cache(maxsize=None)
def slope_statistics(kind):
    s = _slope(kind)
    rms = float(np.sqrt(np.mean(s**2)))
    peak = float(np.max(s))
    return {"mean": float(np.mean(s)), "rms": rms, "peak": peak, "crest": peak / rms}


def waveform_factor(kind, alpha):
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    s = _slope(kind)[None, :]
    r = _slope("sine")[None, :]
    a = alpha[:, None]
    return np.mean(s**a, axis=1) / np.mean(r**a, axis=1)


def waveform_factors(kinds, alpha):
    kinds = np.asarray(kinds)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), kinds.shape)
    out = np.ones(kinds.shape)
    for kind in WAVEFORMS:
        mask = kinds == kind
        if mask.any():
            out[mask] = waveform_factor(kind, alpha[mask])
    return out


def steinmetz(f, b, k, alpha, beta):
    return k * np.power(f, alpha) * np.power(b, beta)


def igse(f, b, kinds, k, alpha, beta):
    return steinmetz(f, b, k, alpha, beta) * waveform_factors(kinds, alpha)


def fit_steinmetz(f, b, p, kinds, iterations=4):
    log_f = np.log(np.asarray(f, dtype=float))
    log_b = np.log(np.asarray(b, dtype=float))
    p = np.asarray(p, dtype=float)
    design = np.column_stack([np.ones_like(log_f), log_f, log_b])
    alpha = 1.5
    coef = np.zeros(3)
    for _ in range(iterations):
        y = np.log(p / waveform_factors(kinds, alpha))
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        alpha = float(np.clip(coef[1], 0.5, 3.0))
    return float(np.exp(coef[0])), alpha, float(coef[2])


def loop_area(h, b):
    h = np.append(h, h[0])
    b = np.append(b, b[0])
    return float(abs(np.sum(0.5 * (h[1:] + h[:-1]) * np.diff(b))))


def loop_loss(h, b, frequency):
    return frequency * loop_area(h, b)


def closure_error(b):
    span = float(np.ptp(b))
    return abs(float(b[-1] - b[0])) / span if span else 0.0
