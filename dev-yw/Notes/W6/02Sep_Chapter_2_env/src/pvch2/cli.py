from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from pvch2 import __version__
from pvch2.config import load_config
from pvch2.data.inventory import inventory_raw, list_class_images
from pvch2.paths import dataset_root, leaf_map_path, output_dir, raw_dir
from pvch2.preprocess.pipelines import available_pipelines, run_pipeline


def verify() -> None:
    cfg = load_config()
    print(f"pvch2 {__version__}")
    print(f"python {sys.version.split()[0]}")
    print(f"env    {cfg['_env_root']}")
    print(f"data   {dataset_root(cfg)}")
    missing = []
    for kind in ("color", "grayscale", "segmented"):
        p = raw_dir(kind, cfg)
        ok = p.is_dir()
        print(f"  raw/{kind:10} {'OK' if ok else 'MISSING'}  {p}")
        if not ok:
            missing.append(str(p))
    lm = leaf_map_path(cfg)
    print(f"  leaf-map     {'OK' if lm.is_file() else 'MISSING'}  {lm}")
    if missing:
        raise SystemExit("dataset paths missing")
    print("pipelines:", ", ".join(available_pipelines(cfg)))
    print("verify: ok")


def inventory() -> None:
    cfg = load_config()
    report = {kind: inventory_raw(kind) for kind in ("color", "grayscale", "segmented")}
    out = output_dir(cfg) / "inventory.json"
    slim = {
        k: {kk: vv for kk, vv in rec.items() if kk != "per_class"}
        | {"n_classes": rec["n_classes"]}
        for k, rec in report.items()
    }
    out.write_text(json.dumps({k: {kk: rec[kk] for kk in ("kind", "root", "n_classes", "n_images", "min_class", "max_class")} for k, rec in report.items()}, indent=2), encoding="utf-8")
    for k, rec in report.items():
        print(f"{k:10} classes={rec['n_classes']:2} images={rec['n_images']:6} min={rec['min_class']:4} max={rec['max_class']:5}")
    print(f"wrote {out}")


def preview() -> None:
    cfg = load_config()
    color_root = raw_dir("color", cfg)
    class_dir = color_root / "Apple___Apple_scab"
    images = list_class_images(class_dir)
    if not images:
        raise SystemExit(f"no images in {class_dir}")
    src = images[0]
    dest = output_dir(cfg) / "preview"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"source {src}")
    for name in available_pipelines(cfg):
        result = run_pipeline(name, src, cfg)
        out = dest / f"{name}.png"
        cv2.imwrite(str(out), result.bgr)
        extra = ""
        if result.mask is not None:
            cv2.imwrite(str(dest / f"{name}_mask.png"), result.mask)
            extra = f" mask={int(result.mask.mean())}"
        print(f"  {name:12} {result.bgr.shape} used_mask={result.meta['used_mask']} -> {out.name}{extra}")
    print(f"preview dir {dest}")


def check() -> None:
    from pvch2.checks.runner import run_checks

    raise SystemExit(run_checks(sys.argv[2:] or None))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    {"verify": verify, "inventory": inventory, "preview": preview, "check": check}[cmd]()
