from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import FLUX_RANGE, FREQUENCY_RANGE, TEMPERATURE_RANGE
from .service import Predictor

app = FastAPI(title="Magnetic Core Loss API", version=__version__)


class PredictionRequest(BaseModel):
    material: str
    frequency_hz: float = Field(gt=0, le=5e6)
    temperature_c: float = Field(ge=-40, le=250)
    b_peak_t: float = Field(gt=0, lt=2.0)
    waveform: Literal["sine", "triangle", "trapezoid", "pwm"] = "sine"


@lru_cache(maxsize=1)
def get_predictor():
    return Predictor()


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/materials")
def materials():
    return {
        "materials": get_predictor().materials,
        "waveforms": ["sine", "triangle", "trapezoid", "pwm"],
        "ranges": {
            "frequency_hz": FREQUENCY_RANGE,
            "temperature_c": TEMPERATURE_RANGE,
            "b_peak_t": FLUX_RANGE,
        },
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return get_predictor().predict(
            request.material,
            request.frequency_hz,
            request.temperature_c,
            request.b_peak_t,
            request.waveform,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
