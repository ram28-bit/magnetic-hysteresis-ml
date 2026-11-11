import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from magnetic_ml.config import FLUX_RANGE, FREQUENCY_RANGE, TEMPERATURE_RANGE, WAVEFORMS
from magnetic_ml.service import Predictor

st.set_page_config(page_title="Core Loss Studio", page_icon="🧲", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.5rem; max-width: 1180px;}
    h1 {font-weight: 600; letter-spacing: -0.02em;}
    div[data-testid="stMetric"] {background: #f5f5f7; border-radius: 16px; padding: 18px 20px;}
    div[data-testid="stMetricValue"] {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor():
    return Predictor()


def format_loss(value):
    return f"{value / 1e3:,.1f} kW/m³"


predictor = load_predictor()

st.title("Core Loss Studio")
st.caption("Physics-guided prediction of magnetic core loss and B–H behaviour")

with st.sidebar:
    st.header("Operating point")
    material = st.selectbox("Material", predictor.materials)
    waveform_kind = st.selectbox("Waveform", WAVEFORMS)
    frequency_khz = st.slider(
        "Frequency (kHz)", FREQUENCY_RANGE[0] / 1e3, FREQUENCY_RANGE[1] / 1e3, 100.0, step=5.0
    )
    temperature = st.slider("Temperature (°C)", TEMPERATURE_RANGE[0], TEMPERATURE_RANGE[1], 90.0, step=1.0)
    flux_mt = st.slider("Peak flux density (mT)", FLUX_RANGE[0] * 1e3, FLUX_RANGE[1] * 1e3, 100.0, step=1.0)

result = predictor.predict(material, frequency_khz * 1e3, temperature, flux_mt / 1e3, waveform_kind)
estimates = result["core_loss_w_per_m3"]

for message in result["warnings"]:
    st.warning(message)

columns = st.columns(4)
labels = {"hybrid": "Hybrid", "steinmetz": "Steinmetz / iGSE", "boosting": "Gradient boosting", "network": "Neural network"}
reference = estimates["steinmetz"]
for column, key in zip(columns, labels):
    delta = None if key == "steinmetz" else f"{100 * (estimates[key] / reference - 1):+.1f}% vs Steinmetz"
    column.metric(labels[key], format_loss(estimates[key]), delta, delta_color="off")

left, right = st.columns([3, 2])

with left:
    loop = result["loop"]
    figure = go.Figure(
        go.Scatter(
            x=loop["h_a_per_m"] + loop["h_a_per_m"][:1],
            y=loop["b_t"] + loop["b_t"][:1],
            mode="lines",
            line={"color": "#0a84ff", "width": 3},
            fill="toself",
            fillcolor="rgba(10,132,255,0.08)",
        )
    )
    figure.update_layout(
        title="Predicted B–H loop",
        xaxis_title="H (A/m)",
        yaxis_title="B (T)",
        height=440,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        template="simple_white",
    )
    st.plotly_chart(figure, use_container_width=True)

with right:
    table = pd.DataFrame(
        {
            "Model": [labels[key] for key in labels],
            "Loss (kW/m³)": [round(estimates[key] / 1e3, 2) for key in labels],
        }
    )
    st.subheader("Model comparison")
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.metric("Loop-area loss", format_loss(result["loop_loss_w_per_m3"]))
    st.caption("Loop-area loss is computed from the predicted loop as f · ∮H dB.")
