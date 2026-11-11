from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

SEED = 42
LOOP_POINTS = 64
SAMPLES_PER_MATERIAL = 1400
TEMPERATURE_EDGES = (40.0, 70.0, 100.0)
VACUUM_PERMEABILITY = 4e-7 * 3.141592653589793

WAVEFORMS = ("sine", "triangle", "trapezoid", "pwm")

MATERIALS = {
    "N87": {"family": "ferrite", "p0": 4.8e4, "alpha": 1.45, "beta": 2.55, "bs": 0.49, "mu_r": 2200.0, "t0": 95.0, "tc": 0.35},
    "3C94": {"family": "ferrite", "p0": 6.2e4, "alpha": 1.38, "beta": 2.62, "bs": 0.47, "mu_r": 2300.0, "t0": 100.0, "tc": 0.30},
    "N49": {"family": "ferrite", "p0": 3.9e4, "alpha": 1.62, "beta": 2.48, "bs": 0.49, "mu_r": 1500.0, "t0": 80.0, "tc": 0.42},
    "NANO": {"family": "nanocrystalline", "p0": 2.6e4, "alpha": 1.33, "beta": 2.12, "bs": 1.20, "mu_r": 8000.0, "t0": 60.0, "tc": 0.12},
}

MATERIAL_IDS = tuple(MATERIALS)

FREQUENCY_RANGE = (5e4, 5e5)
FLUX_RANGE = (0.02, 0.30)
TEMPERATURE_RANGE = (25.0, 120.0)
