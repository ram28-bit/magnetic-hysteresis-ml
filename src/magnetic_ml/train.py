import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .data import generate_dataset, save_dataset
from .evaluation import core_loss_metrics, loop_metrics, make_splits
from .models import HybridLossModel, LoopModel, SteinmetzBaseline, TabularLossModel


def _factories():
    return {
        "steinmetz": SteinmetzBaseline,
        "boosting": lambda: TabularLossModel("boosting"),
        "network": lambda: TabularLossModel("network"),
        "hybrid": HybridLossModel,
    }


def evaluate_core_loss(frame):
    rows = []
    predictions = {}
    for split, (train, test) in make_splits(frame, config.SEED).items():
        train_frame, test_frame = frame.iloc[train], frame.iloc[test]
        for name, factory in _factories().items():
            predicted = factory().fit(train_frame).predict(test_frame)
            scores = core_loss_metrics(test_frame["core_loss_w_per_m3"], predicted)
            rows.append({"split": split, "model": name, "n_test": len(test), **scores})
            if split == "random":
                predictions[name] = (test_frame["core_loss_w_per_m3"].to_numpy(), predicted)
    return pd.DataFrame(rows), predictions


def evaluate_loops(frame, h, b):
    train, test = make_splits(frame, config.SEED)["random"]
    model = LoopModel().fit(frame.iloc[train], b[train])
    predicted = model.predict(frame.iloc[test])
    scores = loop_metrics(h[test], b[test], predicted)
    return scores, (frame.iloc[test], h[test], b[test], predicted)


def plot_parity(predictions, path):
    fig, axes = plt.subplots(1, len(predictions), figsize=(4 * len(predictions), 4), sharex=True, sharey=True)
    for ax, (name, (actual, predicted)) in zip(np.atleast_1d(axes), predictions.items()):
        ax.loglog(actual, predicted, ".", markersize=2, alpha=0.5, color="#0a84ff")
        low, high = actual.min(), actual.max()
        ax.plot([low, high], [low, high], color="#1d1d1f", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Measured loss (W/m³)")
    np.atleast_1d(axes)[0].set_ylabel("Predicted loss (W/m³)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_loops(sample, path, count=4):
    frame, h, b, predicted = sample
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4))
    for ax, i in zip(axes, np.linspace(0, len(frame) - 1, count).astype(int)):
        row = frame.iloc[i]
        ax.plot(h[i], b[i], color="#1d1d1f", label="measured")
        ax.plot(h[i], predicted[i], color="#0a84ff", linestyle="--", label="predicted")
        ax.set_title(f"{row.material_id} · {row.waveform} · {row.frequency_hz / 1e3:.0f} kHz")
        ax.set_xlabel("H (A/m)")
    axes[0].set_ylabel("B (T)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fit_final(frame, h, b):
    bundle = {
        "baseline": SteinmetzBaseline().fit(frame),
        "boosting": TabularLossModel("boosting").fit(frame),
        "network": TabularLossModel("network").fit(frame),
        "hybrid": HybridLossModel().fit(frame),
        "loop": LoopModel().fit(frame, b),
        "ranges": {
            "frequency_hz": [float(frame["frequency_hz"].min()), float(frame["frequency_hz"].max())],
            "temperature_c": [float(frame["temperature_c"].min()), float(frame["temperature_c"].max())],
            "b_peak_t": [float(frame["b_peak_t"].min()), float(frame["b_peak_t"].max())],
        },
        "materials": sorted(frame["material_id"].unique()),
    }
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_DIR / "bundle.joblib", compress=3)


def main():
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame, h, b = generate_dataset()
    save_dataset(frame, h, b)

    table, predictions = evaluate_core_loss(frame)
    table.to_csv(config.REPORT_DIR / "core_loss_metrics.csv", index=False)
    loop_scores, sample = evaluate_loops(frame, h, b)
    (config.REPORT_DIR / "loop_metrics.json").write_text(json.dumps(loop_scores, indent=2))

    plot_parity(predictions, config.FIGURE_DIR / "parity.png")
    plot_loops(sample, config.FIGURE_DIR / "loops.png")
    fit_final(frame, h, b)

    summary = table.pivot(index="split", columns="model", values="mape").round(2)
    print(summary.to_string())
    print(json.dumps(loop_scores, indent=2))


if __name__ == "__main__":
    main()
