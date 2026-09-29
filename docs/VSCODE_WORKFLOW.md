# VS Code 实验工作流

## 1. 打开项目

VS Code 工作区根目录必须是：

```text
C:\Users\32142\Documents\Codex\2026-08-19\cha\plantvillage_pipeline
```

项目已经提供 `.vscode/settings.json`、`tasks.json`、`launch.json` 和 `.env`。Python 扩展应自动选择 `.venv/Scripts/python.exe`，Pylance 从 `src` 解析 `pv_pipeline`。

当前部署状态与实测 hash 见 `docs/DEPLOYMENT_ACCEPTANCE.md`。

在 VS Code 状态栏确认解释器路径以 `.venv\Scripts\python.exe` 结尾。集成终端会自动得到固定 seed、单线程 BLAS、cuBLAS workspace 和 `PYTHONPATH`。

## 2. 常用任务

从 `Terminal > Run Task` 运行：

| VS Code 任务 | 用途 |
| --- | --- |
| `PV: Environment Verify` | 校验 Python、依赖、wheelhouse 与环境 fingerprint |
| `PV: Doctor` | 校验配置、数据锁、manifest、设备与确定性环境 |
| `PV: Tests` | 运行单元和集成测试 |
| `PV: Data Validate` | 重验 3,600 张图像、schema 与 split 泄漏 |
| `PV: Smoke Test` | 对固定 24 张图片运行三管道双重 replay |
| `PV: Train HOG-SVM` | 训练四个 C 候选并选择 baseline |
| `PV: Train Small CNN` | 完整训练 30 epochs |
| `PV: Train MobileNetV3` | 训练 Stage A 10 epochs 与 Stage B 15 epochs |
| `PV: Freeze Test Gate` | 三模型 validation 选择后冻结 test gate |
| `PV: Evaluate Frozen Test` | 对同一 540 张 test 图片执行一次统一评估 |
| `PV: Robustness All` | 运行 noise、blur 与 brightness 稳定性测试 |
| `PV: Compare Models` | 生成统一表格、bootstrap、混淆矩阵与错误样本 |
| `PV: Verify Complete Run` | 最终 checksum 与完整性审计 |

默认 Build Task 是 smoke test，默认 Test Task 是 pytest。

## 3. 调试

`Run and Debug` 中包含：

- `PV: Doctor`
- `PV: Smoke Test`
- `PV: Train Current Pipeline`

第三项默认指向 Small CNN。需要逐步调试其他模型时，只修改 `launch.json` 中该配置的 pipeline 参数，不修改 `project_config.json`。

调试运行不应打开 test split。完整 test 仍必须使用冻结 gate 的任务。

## 4. 推荐实验顺序

```text
Environment Verify
Doctor
Tests
Data Validate
Smoke Test
人工检查 data/manifests/montages
Train HOG-SVM
Train Small CNN
Train MobileNetV3
检查三份 validation selection
Freeze Test Gate
Evaluate Frozen Test
Robustness All
Compare Models
Verify Complete Run
```

`artifacts/run_state.json` 记录 P0 至 P9 状态。长时间训练或 Codex 会话中断后，先读取该文件，再执行尚未完成 phase 的精确下一条命令。

## 5. CPU 与 GPU

当前 canonical reference 环境是 `torch 2.9.0+cpu`，因此 `--device auto` 解析为 CPU。本机为 Ryzen 7 6800H 与集成 AMD Radeon，未检测到 NVIDIA/CUDA 设备。代码已支持 `cpu/cuda/auto`，但 CUDA PyTorch 必须部署在带 NVIDIA GPU 的主机并使用独立 profile、虚拟环境、wheelhouse、依赖锁和 environment fingerprint。

不要在 reference `.venv` 中直接用 CUDA wheel 或 DirectML package 覆盖 CPU torch。这样会使已有环境、smoke golden hash 和实验 ID 失效，也会破坏模型间的标准统一。

## 6. 故障定位

- `Deterministic process environment is not active`：通过 VS Code Task 或 `scripts/pv.ps1` 运行，不要直接调用 Python。
- `Test gate no longer matches`：代码或 artifact 已变化；保留旧实验并创建新实验，不能修改 gate JSON。
- `Stale HOG cache`：manifest 或 HOG 配置变化；保留旧 cache 作为证据，使用新实验目录重新生成。
- `CUDA was requested`：当前 profile 是 CPU；使用 `--device cpu`，或先建立独立 GPU profile。
- 磁盘空间不足：优先保留 source archive、manifest、selected models 和锁文件；不要删除仍被实验引用的 artifact。
