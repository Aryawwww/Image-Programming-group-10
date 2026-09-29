# Environment Deployment Standard

## 1. Scope

This document defines the canonical Windows software environment for the three PlantVillage pipelines. It is normative for both human operators and AI agents. The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** describe required, prohibited, recommended, and optional behavior.

The canonical profile is deliberately hardware-neutral. GPU-specific PyTorch builds are separate execution profiles and MUST NOT replace the reference wheelhouse or its lock.

## 2. Canonical Invariants

An environment is canonical only when all conditions below hold:

1. Python is exactly `3.11.9` on `win_amd64`.
2. Python is located under `.runtime/python311`, and the virtual environment is `.venv`.
3. The system `PATH`, file associations, launcher, and other Python installations remain unchanged.
4. Every direct and transitive package is listed in `requirements/resolved.txt` with an exact version and at least one SHA-256 hash.
5. Third-party installation uses only `wheelhouse`, with `--no-index` and `--require-hashes`; the local project is then installed editable with `--no-index --no-deps --no-build-isolation`.
6. `deployment.lock.json`, `wheelhouse_manifest.json`, and `requirements/resolved.txt` all pass hash verification.
7. `pip check`, package imports, exact inventory comparison, and the deterministic PyTorch smoke test pass.
8. Training and evaluation enter Python through `scripts/run_deterministic.ps1`.

`environment_report.json` with `status: pass` is the machine-verifiable acceptance record.

## 3. State Machine

The deployment stages are ordered and idempotent:

```text
Preflight -> Runtime -> Wheelhouse -> Install -> Verify
```

| Stage | Inputs | Outputs | Network |
| --- | --- | --- | --- |
| `Preflight` | deployment lock, configs, direct pins, scripts | `preflight_report.json` | forbidden |
| `Runtime` | signed Python.org installer | `.runtime/python311`, `.venv`, installer audit record | first run only |
| `Wheelhouse` | direct pins or existing resolved lock | 54-wheel local repository, manifest, hashed transitive lock | first run or explicit rebuild |
| `Install` | verified wheelhouse and resolved lock | installed `.venv`, `pip_check.txt`, inventory snapshot | forbidden |
| `Verify` | installed `.venv`, immutable project inputs | `environment_report.json` and environment fingerprint | forbidden |

Later stages MUST NOT bypass their prerequisites. Existing valid outputs are verified and reused. Existing invalid or incomplete `.venv` and `wheelhouse` directories are preserved and cause a fail-closed stop; the deployer never deletes them implicitly.

## 4. Operator Procedures

### 4.1 First Canonical Deployment

From the project root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1 -Action All -AllowOnline
```

Network access is permitted only while acquiring the signed Python installer and wheels. The Python installer MUST have a valid Authenticode signature whose subject contains `Python Software Foundation`. Its SHA-256 is fixed in `deployment.lock.json` and recorded again in `artifacts/environment/python_installer_lock.json`.

### 4.2 Routine Health Check

```powershell
.\scripts\bootstrap.ps1 -Action Verify
```

The command exits nonzero on any version drift, unexpected package, import failure, lock mismatch, or nondeterministic smoke-test result.

### 4.3 Offline Reinstallation

With `.runtime`, `wheelhouse`, `requirements/resolved.txt`, and the manifest present:

```powershell
.\scripts\bootstrap.ps1 -Action Install
.\scripts\bootstrap.ps1 -Action Verify
```

These commands MUST complete without network access. A missing wheel causes installation to fail rather than contact an index.

### 4.4 Deterministic Project Commands

Run a Python module:

```powershell
.\scripts\run_deterministic.ps1 -Module pv_pipeline -Arguments @('doctor')
```

Run a repository script:

```powershell
.\scripts\run_deterministic.ps1 -ScriptPath scripts/example.py -Arguments @('--config', 'configs/project_config.json')
```

The wrapper fixes the Python hash seed, BLAS thread counts, cuBLAS workspace mode, and project root before Python starts.

## 5. Human and Agent Rules

- MUST run `Preflight` before changing or rebuilding an environment.
- MUST use `.venv/Scripts/python.exe`; activation is optional and is not proof of correctness.
- MUST NOT run an unpinned `pip install`, use the system Python, or overwrite canonical lock files during a training run.
- MUST NOT silently delete or repair an invalid environment. Preserve it, report the failed invariant, and create a reviewed replacement.
- MUST treat edits to configs, requirements, or deployment scripts as a specification change. Recompute the deployment lock and rerun all verification.
- SHOULD archive `deployment.lock.json`, `requirements/resolved.txt`, the wheelhouse, wheelhouse manifest, and environment report with the experiment.
- MAY use a separate GPU profile, but it needs its own directory, resolved lock, wheelhouse manifest, device report, and environment fingerprint.

## 6. Acceptance Checklist

Before data preparation or training begins, confirm:

```text
[ ] bootstrap.ps1 -Action Verify exits 0
[ ] environment_report.json has status "pass"
[ ] Python version is 3.11.9
[ ] package inventory has 55 distributions: 54 locked third-party distributions plus the local project
[ ] pip_check.txt says "No broken requirements found."
[ ] wheelhouse manifest reports 54 files with no hash errors
[ ] training command uses run_deterministic.ps1
```

The number 54 is fixed for this `win_amd64` lock. A reviewed dependency update may change it and MUST produce a new lock, manifest, and acceptance record.

## 7. Reproducibility Boundary

The environment fingerprint proves equivalence of the Python executable, package inventory, deterministic process settings, and canonical static inputs. It does not prove equality of the dataset, pretrained model bytes, source code revision, GPU driver, or model outputs. Those are frozen and checked by the later data, model, and experiment pipelines.
