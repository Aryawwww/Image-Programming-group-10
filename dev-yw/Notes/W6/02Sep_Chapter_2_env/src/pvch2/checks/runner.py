from __future__ import annotations

import json
import sys

from pvch2.checks.sections import REGISTRY
from pvch2.config import load_config
from pvch2.paths import output_dir


def run_checks(sections: list[str] | None = None) -> int:
    cfg = load_config()
    wanted = sections or list(REGISTRY)
    unknown = [s for s in wanted if s not in REGISTRY]
    if unknown:
        print("unknown sections:", ", ".join(unknown))
        print("available:", ", ".join(REGISTRY))
        return 2

    results = []
    failed = 0
    for sid in wanted:
        res = REGISTRY[sid](cfg)
        mark = "PASS" if res.passed else "FAIL"
        if not res.passed:
            failed += 1
        print(f"[{mark}] {res.section}  {res.title}")
        for item in res.items:
            print(f"    {'ok' if item.ok else 'NO'}  {item.name}" + (f"  ({item.detail})" if item.detail else ""))
        results.append(
            {
                "section": res.section,
                "title": res.title,
                "passed": res.passed,
                "items": [{"name": i.name, "ok": i.ok, "detail": i.detail} for i in res.items],
                "artifacts": [str(p) for p in res.artifacts],
            }
        )
    out = output_dir(cfg) / "checks" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{len(wanted) - failed}/{len(wanted)} sections passed  -> {out}")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    code = run_checks(argv or None)
    raise SystemExit(code)
