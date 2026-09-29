from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from .config import ProjectContext
from .data.transforms import decode_image, deep_eval_transform, hog_feature
from .determinism import configure, resolve_device
from .hashing import sha256_file
from .models.mobilenet_v3 import build_mobilenet
from .models.serialization import load_model
from .models.small_cnn import SmallCNN


def _resolve_model(path: Path, pipeline_id: str) -> Path:
    if path.is_file():
        return path
    suffix = ".joblib" if pipeline_id == "hog_svm" else ".safetensors"
    candidate = path / f"model{suffix}"
    if not candidate.is_file():
        raise FileNotFoundError(f"Could not locate model artifact: {candidate}")
    return candidate


def predict_image(
    context: ProjectContext,
    *,
    pipeline_id: str,
    image_path: Path,
    model_artifact: Path,
    device_name: str,
) -> dict[str, Any]:
    configure(int(context.config["seed"]))
    image = decode_image(image_path)
    model_path = _resolve_model(model_artifact, pipeline_id)
    if pipeline_id == "hog_svm":
        model = joblib.load(model_path)
        features = hog_feature(image, context.config["hog_svm"])[None, :]
        scores = np.asarray(model.decision_function(features)[0], dtype=np.float64)
        score_type = "decision_margin"
    else:
        device = resolve_device(device_name)
        if pipeline_id == "small_cnn":
            model = SmallCNN(context.config["small_cnn"])
        elif pipeline_id == "mobilenet_v3_small":
            model, _ = build_mobilenet(context, pretrained=False)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_id}")
        model = model.to(device)
        load_model(model, model_path, device=device)
        tensor = deep_eval_transform(image, context.config["common_deep_input"])
        model.eval()
        with torch.inference_mode():
            logits = model(tensor.unsqueeze(0).to(device))[0]
            scores = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float64)
        score_type = "probability"
    if scores.shape != (4,) or not np.isfinite(scores).all():
        raise ValueError("Inference produced invalid scores")
    predicted_id = int(scores.argmax())
    return {
        "schema_version": "1.0.0",
        "pipeline_id": pipeline_id,
        "model_sha256": sha256_file(model_path),
        "input_sha256": sha256_file(image_path),
        "predicted_class_id": predicted_id,
        "predicted_class_name": context.class_by_id[predicted_id].class_name,
        "score_type": score_type,
        "scores": [float(value) for value in scores],
    }
