from __future__ import annotations

import copy
import hashlib

import numpy as np
from PIL import Image

from pv_pipeline.data.dataset import EpochHashSampler
from pv_pipeline.data.transforms import (
    deep_eval_transform,
    hog_feature,
    stateless_train_transform,
)
from pv_pipeline.training.hog_svm import _feature_config_hash, build_hog_classifier


def synthetic_image() -> Image.Image:
    y, x = np.mgrid[0:160, 0:192]
    array = np.stack(((x * 3) % 256, (y * 5) % 256, (x + y * 2) % 256), axis=-1)
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def test_hog_feature_length_and_finiteness(context):
    vector = hog_feature(synthetic_image(), context.config["hog_svm"])
    assert vector.shape == (8100,)
    assert np.isfinite(vector).all()


def test_hog_solver_matches_canonical_matrix_shape(context):
    classifier = context.config["hog_svm"]["classifier"]
    train_samples = 4 * context.config["dataset"]["selected_count_per_class"]["train"]
    feature_count = context.config["hog_svm"]["hog"]["expected_feature_length"]
    assert train_samples < feature_count
    assert classifier["dual"] is True


def test_hog_feature_cache_ignores_classifier_only_changes(context):
    changed_classifier = copy.deepcopy(context)
    changed_classifier.config["hog_svm"]["classifier"]["dual"] = False
    assert _feature_config_hash(changed_classifier) == _feature_config_hash(context)

    changed_feature = copy.deepcopy(context)
    changed_feature.config["hog_svm"]["hog"]["orientations"] = 8
    assert _feature_config_hash(changed_feature) != _feature_config_hash(context)


def test_hog_smoke_and_training_share_canonical_classifier(context):
    model = build_hog_classifier(context, 0.1)
    classifier = model.named_steps["classifier"]
    expected = context.config["hog_svm"]["classifier"]
    assert classifier.dual is expected["dual"]
    assert classifier.tol == expected["tol"]
    assert classifier.max_iter == expected["max_iter"]
    assert classifier.random_state == expected["random_state"]


def test_deep_transforms_are_stateless(context):
    sample = "a" * 64
    first = stateless_train_transform(
        synthetic_image(),
        sample,
        0,
        context.config["common_deep_input"],
        context.config["stateless_train_augmentation"],
        42177,
    )
    replay = stateless_train_transform(
        synthetic_image(),
        sample,
        0,
        context.config["common_deep_input"],
        context.config["stateless_train_augmentation"],
        42177,
    )
    next_epoch = stateless_train_transform(
        synthetic_image(),
        sample,
        1,
        context.config["common_deep_input"],
        context.config["stateless_train_augmentation"],
        42177,
    )
    assert tensor_hash(first) == tensor_hash(replay)
    assert tensor_hash(first) != tensor_hash(next_epoch)
    assert tuple(first.shape) == (3, 224, 224)


def test_eval_transform_shape(context):
    tensor = deep_eval_transform(synthetic_image(), context.config["common_deep_input"])
    assert tuple(tensor.shape) == (3, 224, 224)


def test_epoch_sampler_is_stable_and_epoch_specific():
    rows = [{"sample_id": format(index, "064x")} for index in range(20)]
    sampler = EpochHashSampler(rows, 42177)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != first
    assert sorted(first) == list(range(20))
