import joblib
import numpy as np
import pandas as pd

from . import config
from .data import peak_field
from .physics import loop_loss, waveform


class Predictor:
    def __init__(self, path=None):
        self.bundle = joblib.load(path or config.MODEL_DIR / "bundle.joblib")

    @property
    def materials(self):
        return list(self.bundle["materials"])

    @property
    def ranges(self):
        return self.bundle["ranges"]

    def _warnings(self, frequency_hz, temperature_c, b_peak_t):
        values = {"frequency_hz": frequency_hz, "temperature_c": temperature_c, "b_peak_t": b_peak_t}
        labels = {"frequency_hz": "Frequency", "temperature_c": "Temperature", "b_peak_t": "Peak flux density"}
        messages = []
        for key, value in values.items():
            low, high = self.ranges[key]
            if not low <= value <= high:
                messages.append(f"{labels[key]} is outside the training range")
        return messages

    def predict(self, material, frequency_hz, temperature_c, b_peak_t, waveform_kind):
        if material not in self.materials:
            raise ValueError(f"unknown material: {material}")
        if waveform_kind not in config.WAVEFORMS:
            raise ValueError(f"unknown waveform: {waveform_kind}")
        frame = pd.DataFrame(
            {
                "material_id": [material],
                "frequency_hz": [frequency_hz],
                "temperature_c": [temperature_c],
                "b_peak_t": [b_peak_t],
                "waveform": [waveform_kind],
            }
        )
        estimates = {
            "steinmetz": float(self.bundle["baseline"].predict(frame)[0]),
            "boosting": float(self.bundle["boosting"].predict(frame)[0]),
            "network": float(self.bundle["network"].predict(frame)[0]),
            "hybrid": float(self.bundle["hybrid"].predict(frame)[0]),
        }
        flux = self.bundle["loop"].predict(frame)[0]
        h_peak = float(peak_field(material, b_peak_t, temperature_c))
        field = h_peak * waveform(waveform_kind, config.LOOP_POINTS)
        return {
            "core_loss_w_per_m3": estimates,
            "loop": {"h_a_per_m": field.tolist(), "b_t": flux.tolist()},
            "loop_loss_w_per_m3": float(loop_loss(field, flux, frequency_hz)),
            "warnings": self._warnings(frequency_hz, temperature_c, b_peak_t),
        }
