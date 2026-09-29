# PlantVillage Pipeline Agent Rules

本文件约束在本目录工作的 Codex 和其他自动化智能体。目标是让后续会话能够继续实验，而不改变已经冻结的数据与比较协议。

## Canonical Entry Points

- 解释器：`.venv/Scripts/python.exe`
- 项目命令：`scripts/pv.ps1 <command>`
- 测试：`scripts/test.ps1`
- 环境验证：`scripts/bootstrap.ps1 -Action Verify`
- 唯一实验配置：`configs/project_config.json`
- 类别真相：`configs/classes.json`
- 执行状态：`artifacts/run_state.json`

不要直接调用系统 Python，不要执行无版本约束的 `pip install`，不要通过临时环境变量改变 seed、线程数或后端确定性设置。

## Required Workflow

1. 开始工作时读取 `artifacts/run_state.json`、`docs/PIPELINE_RUNBOOK.md` 和相关 artifact 的 `selection.json`。
2. 修改代码后运行 `scripts/test.ps1` 和 `scripts/pv.ps1 smoke-test --config configs/project_config.json --device cpu`。
3. 训练三模型的固定顺序为 HOG/SVM、Small CNN、MobileNetV3-Small；模型之间共享同一 manifest 与公共指标实现。
4. validation 只能用于选择模型。三条主线全部完成后才能运行 `freeze-test-gate`。
5. test gate 冻结后，任何 source tree、config、manifest、selection 或 model hash 变化都要求建立新实验，不能覆盖旧 test 输出。
6. `verify-run` 通过前，不得声明完整实验完成。

## Immutable Contracts

- manifest SHA-256：从 `data/manifests/manifest_lock.json` 读取，不要写死在代码中。
- 每类 train/val/test 数量：`630/135/135`。
- 类别 ID 固定为 `0..3`，不得使用目录字母排序重新编号。
- HOG 长度固定为 `8100`；Canonical `LinearSVC` 固定 `dual=true`，因为 2,520 个训练样本少于 8,100 个特征。
- Deep eval 变换与 MobileNetV3 `IMAGENET1K_V1` 保持一致。
- 主选择指标为 validation macro F1，test macro F1 只用于冻结后的最终比较。
- SVM 的 decision margin 不是 probability。

## Data Storage

`data.zip` 保留为来源快照。实现验证归档内全部官方 split 路径，但只原字节抽取 canonical manifest 中的 3,600 张图像。不要全量解压归档，也不要重新编码 JPEG；当前磁盘空间不足以安全保留重复全量副本。

## Resume Handoff

暂停或交接时在工作说明中提供：

```text
Current phase:
Completed artifacts:
Verified hashes:
Commands already run:
Last successful check:
Blocking error:
Unapproved deviations:
Exact next command:
```

额度更新后的会话应从 `artifacts/run_state.json` 和最后一个成功 artifact 继续，不应重新发明 pipeline 或清除现有环境。
