from __future__ import annotations

import numpy as np
import torch

from pv_pipeline.evaluation.metrics import build_prediction_rows, classification_metrics
from pv_pipeline.io_utils import ensure_finite_json
from pv_pipeline.models.mobilenet_v3 import (
    build_mobilenet,
    configure_stage_a,
    configure_stage_b,
    trainable_parameter_names,
)
from pv_pipeline.models.small_cnn import SmallCNN
from pv_pipeline.training.torch_engine import selection_is_better


def test_small_cnn_shape_and_parameter_count(context):
    torch.manual_seed(42177)
    model = SmallCNN(context.config["small_cnn"])
    output = model(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, 4)
    assert sum(parameter.numel() for parameter in model.parameters()) == 389924


def test_mobilenet_stage_contract(context):
    torch.manual_seed(42177)
    model, lock = build_mobilenet(context, pretrained=False)
    assert lock is None
    assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 4)
    configure_stage_a(model)
    assert trainable_parameter_names(model) == [
        "classifier.3.bias",
        "classifier.3.weight",
    ]
    configure_stage_b(model, 9)
    names = trainable_parameter_names(model)
    assert any(name.startswith("features.9.") for name in names)
    assert all(name.startswith("features.9.") or name.startswith("features.10.") or name.startswith("features.11.") or name.startswith("features.12.") or name.startswith("classifier.") for name in names)


def test_metrics_use_all_four_labels(context):
    metrics = classification_metrics([0, 1, 2, 3], [0, 1, 1, 3], context.classes)
    assert len(metrics["per_class"]) == 4
    assert np.asarray(metrics["confusion_matrix"]).shape == (4, 4)
    assert metrics["samples"] == 4


def test_prediction_contract_and_nonfinite_guard(context):
    manifest = [
        {"sample_id": "a" * 64, "class_id": 0},
        {"sample_id": "b" * 64, "class_id": 1},
    ]
    rows = build_prediction_rows(
        experiment_id="test",
        pipeline_id="small_cnn",
        split="val",
        manifest_rows=manifest,
        predicted=[0, 2],
        scores=np.asarray([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]]),
        score_type="probability",
        classes=context.classes,
    )
    assert rows[1]["predicted_class_name"] == "Tomato___Late_blight"
    try:
        ensure_finite_json({"score": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("NaN must be rejected")


def test_model_selection_tie_break():
    incumbent = {"val_macro_f1": 0.9, "val_loss": 0.5, "epoch": 3}
    assert selection_is_better(
        {"val_macro_f1": 0.91, "val_loss": 0.9, "epoch": 4}, incumbent, 1e-12
    )
    assert selection_is_better(
        {"val_macro_f1": 0.9, "val_loss": 0.4, "epoch": 4}, incumbent, 1e-12
    )
    assert selection_is_better(
        {"val_macro_f1": 0.9, "val_loss": 0.5, "epoch": 2}, incumbent, 1e-12
    )
