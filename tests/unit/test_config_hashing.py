from __future__ import annotations

from pv_pipeline.hashing import namespaced_hash, sample_id, source_tree_hash


def test_class_mapping_is_explicit(context):
    assert [item.class_id for item in context.classes] == [0, 1, 2, 3]
    assert context.class_by_name["Tomato___healthy"].class_id == 0
    assert context.class_by_name["Tomato___Leaf_Mold"].class_id == 3


def test_namespaced_hash_is_stable():
    first = namespaced_hash("pv-a1-v1/sample", 42177, "train", "class", "path")
    second = namespaced_hash("pv-a1-v1/sample", 42177, "train", "class", "path")
    assert first == second
    assert len(first) == 64
    assert first != namespaced_hash("pv-a1-v1/sample", 42177, "val", "class", "path")


def test_sample_id_includes_file_hash():
    assert sample_id("raw/color/a.jpg", "0" * 64) != sample_id(
        "raw/color/a.jpg", "1" * 64
    )


def test_source_tree_ignores_bytecode(context):
    before = source_tree_hash(context.root)
    cache = context.root / "src" / "pv_pipeline" / "__pycache__" / "ignored.pyc"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"not-source")
    assert source_tree_hash(context.root) == before
    cache.unlink()
