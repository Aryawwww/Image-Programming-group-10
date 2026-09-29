from __future__ import annotations

import copy
from dataclasses import replace

from pv_pipeline.data.manifest import (
    SourceRecord,
    image_identifier,
    resolve_leaf_id,
    select_records,
)


def test_image_identifier_matches_dataset_logic():
    assert image_identifier("uuid___Mary_HL 1_copy.JPG") == "Mary_HL 1_"
    assert image_identifier("uuid_final_masked.png") == "uuid"


def test_leaf_id_single_multiple_and_fallback():
    leaf_map = {
        "one": ["Tomato___healthy:::1.0"],
        "many": ["Potato___healthy:::2.0", "Tomato___Late_blight:::3.0"],
        "nomatch": ["Potato___healthy:::4.0", "Peach___healthy:::5.0"],
    }
    assert resolve_leaf_id("one.JPG", "Tomato___healthy", leaf_map) == "Tomato___healthy:::1.0"
    assert (
        resolve_leaf_id("many.JPG", "Tomato___Late_blight", leaf_map)
        == "Tomato___Late_blight:::3.0"
    )
    assert resolve_leaf_id("missing.JPG", "Tomato___healthy", leaf_map) == "fallback_missing"
    assert resolve_leaf_id("nomatch.JPG", "Tomato___healthy", leaf_map) == "fallback_nomatch"


def _record(class_id: int, class_name: str, split: str, group: int, index: int):
    return SourceRecord(
        relative_path=f"raw/color/{class_name}/{split}_{group}_{index}.JPG",
        source_split="official_train" if split == "train" else "official_test",
        class_id=class_id,
        class_name=class_name,
        leaf_id=f"leaf-{group}",
        group_id=f"{class_name}::leaf-{group}",
    )


def test_validation_groups_do_not_leak_into_training(context):
    config = copy.deepcopy(context.config)
    config["dataset"]["selected_count_per_class"] = {"train": 2, "val": 2, "test": 1}
    config["dataset"]["holdout_exclusions"] = []
    reduced = replace(context, config=config)
    train = []
    test = []
    for class_info in context.classes:
        for group in range(4):
            train.extend(
                _record(class_info.class_id, class_info.class_name, "train", group, index)
                for index in range(2)
            )
        test.append(_record(class_info.class_id, class_info.class_name, "test", 99, 0))
    selected = select_records(reduced, train, test)
    for class_info in context.classes:
        train_groups = {
            record.group_id
            for split, _rank, record in selected
            if split == "train" and record.class_id == class_info.class_id
        }
        val_groups = {
            record.group_id
            for split, _rank, record in selected
            if split == "val" and record.class_id == class_info.class_id
        }
        assert train_groups.isdisjoint(val_groups)


def test_holdout_exclusion_is_replaced_with_same_class_candidate(context):
    config = copy.deepcopy(context.config)
    config["dataset"]["selected_count_per_class"] = {"train": 2, "val": 2, "test": 1}
    train = []
    test = []
    for class_info in context.classes:
        for group in range(4):
            train.extend(
                _record(class_info.class_id, class_info.class_name, "train", group, index)
                for index in range(2)
            )
        test.extend(
            _record(class_info.class_id, class_info.class_name, "test", 99, index)
            for index in range(2)
        )
    excluded_path = test[0].relative_path
    config["dataset"]["holdout_exclusions"] = [
        {"relative_path": excluded_path, "reason": "unit_test"}
    ]
    reduced = replace(context, config=config)
    selected = select_records(reduced, train, test)
    selected_test = [record for split, _rank, record in selected if split == "test"]
    assert excluded_path not in {record.relative_path for record in selected_test}
    assert len(selected_test) == len(context.classes)
