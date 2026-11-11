from magnetic_ml.data import generate_dataset
from magnetic_ml.evaluation import core_loss_metrics, make_splits
from magnetic_ml.features import build_features
from magnetic_ml.models import HybridLossModel, SteinmetzBaseline


def test_hybrid_beats_baseline_on_random_split():
    frame, _, _ = generate_dataset(samples_per_material=300)
    train, test = make_splits(frame)["random"]
    baseline = SteinmetzBaseline().fit(frame.iloc[train]).predict(frame.iloc[test])
    hybrid = HybridLossModel().fit(frame.iloc[train]).predict(frame.iloc[test])
    truth = frame.iloc[test]["core_loss_w_per_m3"]
    assert core_loss_metrics(truth, hybrid)["mape"] < core_loss_metrics(truth, baseline)["mape"]


def test_feature_matrix_has_no_missing_values():
    frame, _, _ = generate_dataset(samples_per_material=50)
    assert not build_features(frame).isna().any().any()
