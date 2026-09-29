from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from ..artifacts import experiment_layout, finalize_checksums
from ..config import ProjectContext
from ..data.dataset import load_manifest_rows
from ..data.transforms import decode_image, hog_feature
from ..determinism import configure, resolve_device
from ..hashing import sha256_file
from ..io_utils import load_json, write_bytes, write_csv, write_json, write_jsonl
from ..models.mobilenet_v3 import build_mobilenet
from ..models.serialization import load_model
from ..models.small_cnn import SmallCNN
from ..training.torch_engine import evaluate_torch
from .metrics import classification_metrics
from .runner import PIPELINE_IDS, verify_test_gate


def _condition_key(kind: str, severity: float) -> str:
    return f"{kind}_{format(float(severity), '.12g').replace('.', 'p')}"


def _noise(image: Image.Image, sample_id: str, sigma: float, seed: int) -> Image.Image:
    digest = hashlib.sha256(
        f"pv-a1-v1/robustness/noise|{seed}|{sample_id}|{format(sigma, '.12g')}".encode(
            "utf-8"
        )
    ).digest()
    seed64 = int.from_bytes(digest[:8], byteorder="big")
    rng = np.random.Generator(np.random.PCG64(seed64))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    corrupted = np.clip(pixels + rng.normal(0.0, sigma, pixels.shape), 0.0, 1.0)
    return Image.fromarray(np.rint(corrupted * 255.0).astype(np.uint8), mode="RGB")


def _corrupt(
    image: Image.Image,
    kind: str,
    severity: float,
    sample_id: str,
    seed: int,
) -> Image.Image:
    if kind == "gaussian_noise":
        return _noise(image, sample_id, severity, seed)
    if kind == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=severity))
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(severity)
    raise ValueError(f"Unknown corruption: {kind}")


def build_robustness_cache(
    context: ProjectContext,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Path]]]:
    config = context.config["robustness"]
    conditions = [
        *(('gaussian_noise', float(value)) for value in config["gaussian_noise_sigma"]),
        *(('gaussian_blur', float(value)) for value in config["gaussian_blur_radius"]),
        *(('brightness', float(value)) for value in config["brightness_factor"]),
    ]
    cache_root = context.root / "data" / "robustness_cache"
    manifest_rows = []
    lookup: dict[str, dict[str, Path]] = {}
    seed = int(context.config["seed"])
    for kind, severity in conditions:
        key = _condition_key(kind, severity)
        lookup[key] = {}
        condition_root = cache_root / kind / format(severity, ".12g")
        for row in rows:
            target = condition_root / f"{row['sample_id']}.png"
            if not target.is_file():
                source = decode_image(context.extracted_root / Path(row["relative_path"]))
                corrupted = _corrupt(source, kind, severity, row["sample_id"], seed)
                buffer = BytesIO()
                corrupted.save(buffer, format="PNG", compress_level=9)
                write_bytes(target, buffer.getvalue())
            lookup[key][row["sample_id"]] = target
            manifest_rows.append(
                {
                    "condition": key,
                    "corruption": kind,
                    "severity": severity,
                    "sample_id": row["sample_id"],
                    "relative_path": target.relative_to(context.root).as_posix(),
                    "bytes": target.stat().st_size,
                    "sha256": sha256_file(target),
                }
            )
    write_jsonl(cache_root / "manifest.jsonl", manifest_rows)
    write_json(
        cache_root / "manifest_lock.json",
        {
            "schema_version": "1.0.0",
            "source_manifest_sha256": sha256_file(context.manifest_path),
            "rows": len(manifest_rows),
            "manifest_sha256": sha256_file(cache_root / "manifest.jsonl"),
        },
    )
    return manifest_rows, lookup


def _drop_record(clean: dict[str, Any], corrupted: dict[str, Any]) -> dict[str, Any]:
    clean_f1 = float(clean["macro_f1"])
    corrupted_f1 = float(corrupted["macro_f1"])
    per_class_clean = {row["class_id"]: row["f1"] for row in clean["per_class"]}
    per_class_corrupted = {
        row["class_id"]: row["f1"] for row in corrupted["per_class"]
    }
    return {
        "macro_f1": corrupted_f1,
        "macro_f1_absolute_drop": clean_f1 - corrupted_f1,
        "macro_f1_relative_drop": (clean_f1 - corrupted_f1) / clean_f1 if clean_f1 else 0.0,
        "per_class_f1_drop": [
            {
                "class_id": class_id,
                "drop": float(per_class_clean[class_id] - per_class_corrupted[class_id]),
            }
            for class_id in range(4)
        ],
        "confusion_matrix": corrupted["confusion_matrix"],
    }


def run_robustness(
    context: ProjectContext,
    *,
    pipeline: str,
    device_name: str,
) -> Path:
    verify_test_gate(context)
    configure(int(context.config["seed"]))
    test_rows = load_manifest_rows(context, "test")
    _manifest, lookup = build_robustness_cache(context, test_rows)
    selected = PIPELINE_IDS if pipeline == "all" else (pipeline,)
    summary_rows = []
    for pipeline_id in selected:
        layout = experiment_layout(context, pipeline_id)
        clean = load_json(layout.metrics / "test.json")
        results = {}
        if pipeline_id == "hog_svm":
            model = joblib.load(layout.pipeline_root / "model.joblib")
            for condition, paths in lookup.items():
                features = np.stack(
                    [
                        hog_feature(
                            decode_image(paths[row["sample_id"]]),
                            context.config["hog_svm"],
                        )
                        for row in test_rows
                    ]
                )
                predicted = np.asarray(model.predict(features), dtype=np.int64)
                metrics = classification_metrics(
                    [row["class_id"] for row in test_rows], predicted, context.classes
                )
                results[condition] = _drop_record(clean, metrics)
        else:
            device = resolve_device(device_name)
            if pipeline_id == "small_cnn":
                model = SmallCNN(context.config["small_cnn"])
                batch_size = int(context.config["small_cnn"]["training"]["batch_size"])
            else:
                model, _ = build_mobilenet(context, pretrained=False)
                selection = load_json(layout.pipeline_root / "selection.json")
                stage_key = "stage_b" if selection["selected_stage"] == "stage_b" else "stage_a"
                batch_size = int(context.config["mobilenet_v3"][stage_key]["batch_size"])
            model = model.to(device)
            load_model(model, layout.pipeline_root / "model.safetensors", device=device)
            for condition, paths in lookup.items():
                evaluation = evaluate_torch(
                    context,
                    model,
                    test_rows,
                    device,
                    batch_size=batch_size,
                    image_loader=lambda row, condition_paths=paths: decode_image(
                        condition_paths[row["sample_id"]]
                    ),
                )
                results[condition] = _drop_record(clean, evaluation.metrics)
        write_json(layout.metrics / "robustness.json", results)
        for condition, values in results.items():
            summary_rows.append(
                {
                    "pipeline_id": pipeline_id,
                    "condition": condition,
                    "macro_f1": values["macro_f1"],
                    "absolute_drop": values["macro_f1_absolute_drop"],
                    "relative_drop": values["macro_f1_relative_drop"],
                }
            )
        finalize_checksums(layout)
    output = context.root / "artifacts" / "comparison" / "robustness_summary.csv"
    write_csv(
        output,
        summary_rows,
        ("pipeline_id", "condition", "macro_f1", "absolute_drop", "relative_drop"),
    )
    return output
