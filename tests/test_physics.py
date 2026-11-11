import numpy as np

from magnetic_ml.physics import fit_steinmetz, loop_area, steinmetz, waveform, waveform_factor


def test_sine_factor_is_unity():
    assert abs(waveform_factor("sine", 1.5)[0] - 1.0) < 1e-9


def test_triangle_factor_exceeds_sine_for_high_alpha():
    assert waveform_factor("triangle", 1.6)[0] > 0.8


def test_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    f = rng.uniform(5e4, 5e5, 400)
    b = rng.uniform(0.02, 0.3, 400)
    p = steinmetz(f, b, 3.2e-3, 1.5, 2.6)
    kinds = np.array(["sine"] * 400)
    k, alpha, beta = fit_steinmetz(f, b, p, kinds)
    assert abs(alpha - 1.5) < 1e-6
    assert abs(beta - 2.6) < 1e-6
    assert abs(k / 3.2e-3 - 1.0) < 1e-6


def test_ellipse_area():
    theta = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    h = 3.0 * np.cos(theta)
    b = 2.0 * np.sin(theta)
    assert abs(loop_area(h, b) - np.pi * 3.0 * 2.0) < 0.05


def test_waveforms_are_bounded():
    for kind in ("sine", "triangle", "trapezoid", "pwm"):
        w = waveform(kind, 256)
        assert w.max() <= 1.0 + 1e-9 and w.min() >= -1.0 - 1e-9
