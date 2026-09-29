# PlantVillage Three-Pipeline Project

本项目已经部署 HOG/SVM、Small CNN 与 MobileNetV3-Small 三条统一分类管道。真实 PlantVillage 缩减数据集、确定性 manifest、公共评估协议和 VS Code 工作流均位于本目录。

- [三技术管道实施手册](docs/PIPELINE_RUNBOOK.md)
- [VS Code 工作流](docs/VSCODE_WORKFLOW.md)
- [环境部署规范](docs/ENVIRONMENT_DEPLOYMENT.md)
- [智能体规则](AGENTS.md)

快速健康检查：

```powershell
.\scripts\pv.ps1 doctor --config configs/project_config.json
.\scripts\test.ps1
.\scripts\pv.ps1 smoke-test --config configs/project_config.json --device auto
```

下面是项目软件环境的建立与审计说明。

本目录是 PlantVillage 三主线项目的自动化环境入口。环境部署遵循：

- Python `3.11.9`；
- 规范中的固定直接依赖；
- 项目内 `.runtime` 与 `.venv`，不修改系统 PATH；
- 首次联网建立本地 wheelhouse，并生成包含全部传递依赖与 SHA-256 的锁；
- 后续使用 wheelhouse 离线重建；
- 固定 deterministic environment variables；
- 生成环境 fingerprint、完整 package inventory 和健康报告。

完整规范与人/智能体操作约束见 [Environment Deployment Standard](docs/ENVIRONMENT_DEPLOYMENT.md)。

## 一键部署

首次运行需要网络：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1 -Action All -AllowOnline
```

该命令依次执行：

```text
Preflight -> Runtime -> Wheelhouse -> Install -> Verify
```

如果 Python 3.11.9 尚不存在，部署器从 Python.org 下载 64-bit installer，验证 Authenticode 签名后安装到 `.runtime/python311`。它不会加入 PATH、创建快捷方式或删除其他 Python。

## 分阶段运行

```powershell
.\scripts\bootstrap.ps1 -Action Preflight
.\scripts\bootstrap.ps1 -Action Runtime -AllowOnline
.\scripts\bootstrap.ps1 -Action Wheelhouse -AllowOnline
.\scripts\bootstrap.ps1 -Action Install
.\scripts\bootstrap.ps1 -Action Verify
```

建立 wheelhouse 后，`Install` 和 `Verify` 不需要网络。`requirements/resolved.txt` 是当前 Windows/Python 平台的完整依赖锁；安装使用 `--require-hashes`，不会从 wheelhouse 外取包。若现有 `.venv` 版本错误，部署器会停止并保留它，避免隐式删除用户环境。

## 确定性运行入口

后续所有训练或评估命令通过：

```powershell
.\scripts\run_deterministic.ps1 -Module pv_pipeline -Arguments @("doctor", "--config", "configs/project_config.json")
```

脚本会设置 `PYTHONHASHSEED`、BLAS thread 数和项目根目录，再调用锁定虚拟环境中的 Python。

## 审计产物

部署成功后生成：

```text
artifacts/environment/
├── python_installer_lock.json
├── wheelhouse_manifest.json
├── requirements.resolved.txt
├── pip_freeze.txt
├── pip_check.txt
└── environment_report.json
```

`environment_report.json` 中的 `environment_fingerprint_sha256` 不包含时间戳、主机名或用户名，可用于判断两个环境是否属于同一 canonical software environment。

## 设备边界

当前自动化只冻结软件环境，不针对具体 GPU backend 调优。设备专用 PyTorch wheel、AMP 或编译模式必须作为单独 profile 加入，不能覆盖 canonical reference environment。
