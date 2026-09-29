# 三技术管道实施手册

## 1. 已部署输入

- 数据源：`mohanty/PlantVillage`
- revision：`afb5ce38128c7bac085b31d499d28b631ef8fc80`
- 数据类别：番茄 healthy、Early blight、Late blight、Leaf Mold
- canonical manifest：3,600 张，train/val/test 为 2,520/540/540
- 全局 seed：`42177`
- 公共主指标：macro F1

源文件、manifest、预训练权重和每个实验 artifact 都有 SHA-256。当前状态以 `artifacts/run_state.json` 为准。

## 2. 公共输入到输出

```text
fixed Hugging Face revision
  -> source file lock and ZIP member audit
  -> leaf-group-aware deterministic selection
  -> canonical manifest and selected raw JPEG extraction
  -> shared class IDs and metrics
  -> pipeline-specific preprocessing and training
  -> validation-only model selection
  -> frozen test gate
  -> one-time test, robustness and paired comparison
  -> checksums and report assets
```

所有命令均从项目根目录通过 `scripts/pv.ps1` 运行。

## 3. Phase P0-P3

```powershell
.\scripts\pv.ps1 data acquire --config configs/project_config.json
.\scripts\pv.ps1 data build-manifest --config configs/project_config.json
.\scripts\pv.ps1 data validate --config configs/project_config.json
.\scripts\pv.ps1 smoke-test --config configs/project_config.json --device auto
```

验收：

- 四个来源文件与 `dataset_lock.json` 一致；
- 54,305 条官方 color split 路径全部存在于归档；
- manifest 为 3,600 行，每类 630/135/135；
- leaf group 与文件哈希无跨 split 泄漏；
- `data/manifests/holdout_exclusions.json` 与配置一致，任何冻结前已暴露的官方 test 样本都不在 manifest；
- smoke 的两次输出 hash 完全相同；
- 人工查看 `data/manifests/montages/class_0.jpg` 至 `class_3.jpg`。

P7 前不得解码、显示或单图推理 canonical test 图片。若发生诊断性暴露，必须将路径和原因写入 `dataset.holdout_exclusions`，重新运行 build-manifest 与 validate，由同类下一候选自动补足；禁止直接手改 JSONL。

## 4. HOG/SVM Baseline

```powershell
.\scripts\pv.ps1 train hog_svm --config configs/project_config.json
```

流程：RGB 解码，128x128 bilinear resize，灰度化，9 orientation HOG，8100 维特征，StandardScaler，LinearSVC。分别训练 `C=0.01/0.1/1/10`，按 validation macro F1 选择，容差内选择更小 C。

规范配置固定 `dual=true`、`tol=1e-3`、`max_iter=20000`。本项目的训练矩阵为 2,520 个样本、8,100 个特征，属于 `n_samples < n_features`；对偶求解更适合此矩阵形状。实测 `dual=false, tol=1e-5` 在单 CPU 线程运行 30 分钟仍未完成完整候选网格；`C=0.1, dual=true, tol=1e-4` 需要 8,722 次迭代和 281.4 秒，而 `C=1/10, dual=true, tol=1e-3` 分别只需 116/115 次迭代和 9.48/9.18 秒，且均正常收敛。因此严格检查收敛的 `1e-3` 是默认可实现配置，任何 `ConvergenceWarning` 仍会使训练失败。HOG 特征会按 manifest hash 和特征配置 hash 缓存；首次生成约需 160 MB，后续重跑会验证样本顺序后复用。

主要输出：

```text
artifacts/<experiment_id>/hog_svm/
  model.joblib
  feature_config.json
  selection.json
  history.csv
  predictions/val.jsonl
  metrics/val.json
  checksums.sha256
```

## 5. Small CNN

```powershell
.\scripts\pv.ps1 train small_cnn --config configs/project_config.json --device auto
```

四个 Conv-BN-ReLU-MaxPool block 从随机初始化训练 30 epochs。样本顺序与增强由 `sample_id + epoch + seed` 决定，禁止 DataLoader shuffle。每个 epoch 结束后计算完整 validation；按更高 macro F1、更低 loss、更早 epoch 选择。

最终权重使用 safetensors，optimizer 状态不进入推理 artifact。

## 6. MobileNetV3-Small

```powershell
.\scripts\pv.ps1 train mobilenet_v3_small --config configs/project_config.json --device auto
```

Stage A 只训练新 `classifier.3` 共 10 epochs。Stage B 从 best Stage A 恢复，只解冻 `features[9:]` 与 classifier，共 15 epochs。BatchNorm running stats 在两阶段均冻结。最终候选只允许 best Stage A 与 best Stage B。

预训练权重保存在 `data/pretrained` 并由独立 lock 固定完整 SHA-256。

## 7. Test Gate 与统一评估

三条管道 validation artifact 完整且人工复核后：

```powershell
.\scripts\pv.ps1 freeze-test-gate --config configs/project_config.json
.\scripts\pv.ps1 evaluate --pipeline all --split test --config configs/project_config.json --device auto
```

gate 锁定 manifest、config、source tree、三模型与三份 selection。任何不一致都会拒绝 test。三个模型都对同一顺序的 540 个 sample ID 输出预测。

## 8. Robustness 与比较

```powershell
.\scripts\pv.ps1 robustness --pipeline all --config configs/project_config.json --device auto
.\scripts\pv.ps1 compare --config configs/project_config.json
.\scripts\pv.ps1 report build-assets --config configs/project_config.json
.\scripts\pv.ps1 verify-run --config configs/project_config.json
```

比较采用 accuracy、macro precision/recall/F1、每类指标与 4x4 confusion matrix。10,000 次配对分层 bootstrap 使用相同重采样 indices。只有点差至少 0.005 且 95% interval 不含 0 时，才报告某模型表现更好。

## 9. 恢复规则

网络中断可继续固定 revision 下载。环境、manifest、模型或 gate hash 不一致时必须停止，不得静默修复。额度或会话恢复后执行：

```powershell
Get-Content -Raw artifacts\run_state.json
.\scripts\pv.ps1 doctor --config configs/project_config.json
```

然后根据 run state 运行第一个未完成 phase。不要删除旧实验目录，不要在 test 结果出现后返回训练阶段覆盖原结果。

Small CNN 与 MobileNet 每个 epoch 都原子更新 `checkpoints/<stage>.pt`，其中包含 model、optimizer、scheduler、当前最佳候选和 Python/NumPy/PyTorch RNG 状态。相同 source tree、config 与 sample IDs 下重新执行同一训练命令会从下一 epoch 继续；identity 不一致时拒绝恢复。
