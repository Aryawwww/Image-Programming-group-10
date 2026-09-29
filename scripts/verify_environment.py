from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
import sysconfig
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PYTHON = (3, 11, 9)
DETERMINISTIC_ENV = {
    "PYTHONHASHSEED": "42177",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
IMPORT_NAMES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "pillow": "PIL",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "huggingface-hub": "huggingface_hub",
    "safetensors": "safetensors",
    "joblib": "joblib",
    "jsonschema": "jsonschema",
    "pytest": "pytest",
}


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line or any(token in line for token in (";", "[", "]", " @ ")):
            raise ValueError(f"Unsupported requirement at {path}:{line_number}: {line}")
        name, version = line.split("==", 1)
        key = canonical_name(name.strip())
        if not key or not version.strip() or key in result:
            raise ValueError(f"Invalid or duplicate requirement at {path}:{line_number}: {line}")
        result[key] = version.strip()
    return result


def parse_hashed_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if "==" not in tokens[0] or len(tokens) < 2:
            raise ValueError(f"Invalid hashed requirement at {path}:{line_number}: {line}")
        name, version = tokens[0].split("==", 1)
        key = canonical_name(name.strip())
        hashes = [token.removeprefix("--hash=sha256:") for token in tokens[1:]]
        hashes_valid = all(
            token.startswith("--hash=sha256:")
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for token, value in zip(tokens[1:], hashes, strict=True)
        )
        if not key or not version.strip() or key in result or not hashes_valid:
            raise ValueError(f"Invalid hashed requirement at {path}:{line_number}: {line}")
        result[key] = version.strip()
    return result


def add_check(report: dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        report["errors"].append(f"{name}: {detail}")


def verify_static_files(project_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    required = {
        "project_config": project_root / "configs" / "project_config.json",
        "classes": project_root / "configs" / "classes.json",
        "reference_requirements": project_root / "requirements" / "reference.txt",
        "bootstrap_requirements": project_root / "requirements" / "bootstrap.txt",
        "pyproject": project_root / "pyproject.toml",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    add_check(report, "required_files", not missing, {"missing": missing})
    if missing:
        return {}

    config = load_json(required["project_config"])
    classes = load_json(required["classes"])
    class_rows = classes.get("classes", [])
    class_ids = [row.get("class_id") for row in class_rows]
    class_names = [row.get("class_name") for row in class_rows]
    counts = config.get("dataset", {}).get("selected_count_per_class", {})

    add_check(report, "spec_version", config.get("spec_version") == "1.0.0", config.get("spec_version"))
    add_check(report, "seed", config.get("seed") == 42177, config.get("seed"))
    add_check(report, "class_ids", class_ids == [0, 1, 2, 3], class_ids)
    add_check(report, "unique_class_names", len(class_names) == len(set(class_names)) == 4, class_names)
    add_check(
        report,
        "split_counts",
        counts == {"train": 630, "val": 135, "test": 135},
        counts,
    )
    add_check(
        report,
        "total_selected_images",
        len(class_rows) * sum(counts.values()) == 3600 if counts else False,
        len(class_rows) * sum(counts.values()) if counts else None,
    )
    add_check(
        report,
        "model_output_classes",
        config.get("small_cnn", {}).get("architecture", {}).get("output_classes") == 4
        and config.get("mobilenet_v3", {}).get("output_classes") == 4,
        {
            "small_cnn": config.get("small_cnn", {}).get("architecture", {}).get("output_classes"),
            "mobilenet_v3": config.get("mobilenet_v3", {}).get("output_classes"),
        },
    )

    reference = parse_requirements(required["reference_requirements"])
    bootstrap = parse_requirements(required["bootstrap_requirements"])
    add_check(report, "reference_requirement_count", len(reference) == 15, len(reference))
    add_check(report, "bootstrap_requirement_count", len(bootstrap) == 3, len(bootstrap))

    pyproject = tomllib.loads(required["pyproject"].read_text(encoding="utf-8"))
    pyproject_dependencies: dict[str, str] = {}
    dependency_errors = []
    for dependency in pyproject.get("project", {}).get("dependencies", []):
        if "==" not in dependency:
            dependency_errors.append(dependency)
            continue
        name, version = dependency.split("==", 1)
        pyproject_dependencies[canonical_name(name)] = version
    add_check(
        report,
        "pyproject_matches_reference",
        not dependency_errors and pyproject_dependencies == reference,
        {"invalid": dependency_errors, "dependencies": pyproject_dependencies},
    )

    lock_path = project_root / "deployment.lock.json"
    if lock_path.is_file():
        lock = load_json(lock_path)
        mismatches = []
        for relative_path, expected_hash in lock.get("locked_files", {}).items():
            target = project_root / relative_path
            actual_hash = sha256_file(target) if target.is_file() else None
            if actual_hash != expected_hash:
                mismatches.append(
                    {"path": relative_path, "expected": expected_hash, "actual": actual_hash}
                )
        add_check(report, "deployment_lock", not mismatches, {"mismatches": mismatches})
    else:
        add_check(report, "deployment_lock", False, "deployment.lock.json is missing")

    resolved_path = project_root / "requirements" / "resolved.txt"
    resolved: dict[str, str] = {}
    if resolved_path.is_file():
        resolved = parse_hashed_requirements(resolved_path)
        direct = {**bootstrap, **reference}
        direct_mismatches = [
            {"package": name, "expected": version, "resolved": resolved.get(name)}
            for name, version in sorted(direct.items())
            if resolved.get(name) != version
        ]
        add_check(
            report,
            "resolved_lock_contains_direct_requirements",
            not direct_mismatches,
            {"packages": len(resolved), "mismatches": direct_mismatches},
        )

    report["static_hashes"] = {
        name: sha256_file(path) for name, path in required.items()
    }
    if resolved_path.is_file():
        report["static_hashes"]["resolved_requirements"] = sha256_file(resolved_path)
    return {"reference": reference, "bootstrap": bootstrap, "resolved": resolved}


def installed_inventory() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name") or distribution.name
        packages[canonical_name(raw_name)] = distribution.version
    return [
        {"name": name, "version": packages[name]}
        for name in sorted(packages)
    ]


def deterministic_torch_test(seed: int) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    else:  # pragma: no cover - compatibility path for older PyTorch releases
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "conv"):
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    else:  # pragma: no cover - compatibility path for older PyTorch releases
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)

    def run_once() -> tuple[str, Any]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        layer = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        inputs = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8) / 255.0
        output = layer(inputs).detach().cpu().contiguous()
        return hashlib.sha256(output.numpy().tobytes()).hexdigest(), output

    digest_a, output_a = run_once()
    digest_b, output_b = run_once()
    return {
        "passed": bool(torch.equal(output_a, output_b) and digest_a == digest_b),
        "output_sha256": digest_a,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }


def verify_strict(
    project_root: Path,
    requirements: dict[str, dict[str, str]],
    report: dict[str, Any],
) -> None:
    python_version = tuple(sys.version_info[:3])
    add_check(report, "python_version", python_version == EXPECTED_PYTHON, list(python_version))

    environment_values = {key: os.environ.get(key) for key in DETERMINISTIC_ENV}
    add_check(
        report,
        "deterministic_environment",
        environment_values == DETERMINISTIC_ENV,
        environment_values,
    )

    expected = {**requirements.get("bootstrap", {}), **requirements.get("reference", {})}
    installed = {
        canonical_name(item["name"]): item["version"] for item in installed_inventory()
    }
    mismatches = []
    for name, version in sorted(expected.items()):
        actual = installed.get(name)
        if actual != version:
            mismatches.append({"package": name, "expected": version, "actual": actual})
    add_check(report, "direct_package_versions", not mismatches, {"mismatches": mismatches})

    resolved = dict(requirements.get("resolved", {}))
    if resolved:
        pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
        local_name = canonical_name(pyproject["project"]["name"])
        local_version = str(pyproject["project"]["version"])
        resolved[local_name] = local_version
        inventory_mismatches = [
            {"package": name, "expected": version, "actual": installed.get(name)}
            for name, version in sorted(resolved.items())
            if installed.get(name) != version
        ]
        unexpected = sorted(set(installed) - set(resolved))
        add_check(
            report,
            "resolved_package_inventory",
            not inventory_mismatches and not unexpected,
            {"mismatches": inventory_mismatches, "unexpected": unexpected},
        )
        try:
            local_module = importlib.import_module("pv_pipeline")
            local_detail = {
                "distribution": local_name,
                "version": getattr(local_module, "__version__", None),
            }
            add_check(
                report,
                "local_project_import",
                local_detail["version"] == local_version,
                local_detail,
            )
        except Exception as exc:
            add_check(report, "local_project_import", False, repr(exc))
    else:
        add_check(report, "resolved_package_inventory", False, "requirements/resolved.txt is missing")

    import_errors = []
    for package_name in sorted(requirements.get("reference", {})):
        module_name = IMPORT_NAMES.get(package_name)
        if module_name is None:
            import_errors.append({"package": package_name, "error": "missing import map"})
            continue
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised only on broken environments
            import_errors.append({"package": package_name, "error": repr(exc)})
    add_check(report, "package_imports", not import_errors, {"errors": import_errors})

    if not import_errors and not mismatches:
        try:
            torch_test = deterministic_torch_test(42177)
            add_check(report, "torch_determinism", torch_test["passed"], torch_test)
        except Exception as exc:  # pragma: no cover - exercised only on broken environments
            add_check(report, "torch_determinism", False, repr(exc))

    inventory = installed_inventory()
    executable_path = Path(sys.executable).resolve()
    fingerprint_payload = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_sha256": sha256_file(executable_path),
        "python_abi": sysconfig.get_config_var("SOABI"),
        "platform_tag": sysconfig.get_platform(),
        "packages": inventory,
        "deterministic_environment": DETERMINISTIC_ENV,
        "project_config_sha256": sha256_file(project_root / "configs" / "project_config.json"),
        "classes_sha256": sha256_file(project_root / "configs" / "classes.json"),
        "reference_requirements_sha256": sha256_file(
            project_root / "requirements" / "reference.txt"
        ),
        "resolved_requirements_sha256": sha256_file(
            project_root / "requirements" / "resolved.txt"
        ),
        "deployment_lock_sha256": sha256_file(project_root / "deployment.lock.json"),
    }
    report["environment_fingerprint_payload"] = fingerprint_payload
    report["environment_fingerprint_sha256"] = hashlib.sha256(
        canonical_json_bytes(fingerprint_payload)
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the canonical PlantVillage environment")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "strict"), default="strict")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "mode": args.mode,
        "status": "running",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": [],
        "errors": [],
    }
    requirements = verify_static_files(project_root, report)
    if args.mode == "strict" and not report["errors"]:
        verify_strict(project_root, requirements, report)
    report["status"] = "pass" if not report["errors"] else "fail"

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": report["status"], "report": str(output_path)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
