# 技术管道部署验收记录

验收日期：2026-08-28  
工作区：`C:\Users\32142\Documents\Codex\2026-08-19\cha\plantvillage_pipeline`

## 1. 验收结论

三条技术管道已完成代码、环境、数据、CLI、VS Code 任务、测试、统一产物协议和恢复机制部署，可直接进入后续正式实验。当前已完整训练并重载验证 HOG/SVM；Small CNN 与 MobileNetV3-Small 已通过包含前向、反向、优化器更新和验证推理的双次确定性 smoke。深度模型的完整 30 epochs 和 10+15 epochs 训练属于后续实验，不属于部署验收结果。

当前 canonical 源树 hash 为 `7113cb17e3fa4b466f31ec99c04e6a34871b4d955ae04519c395fdb1668c95a2`。当前 HOG/SVM 实验 ID 为 `44e011204f972179`。旧实验目录是部署调试证据，不得作为当前报告结果引用。

## 2. 环境与设备

- Python：3.11.9，项目环境为 `.venv`。
- PyTorch：2.9.0+cpu。
- 依赖：54 个锁定第三方 wheel，加本地 editable 项目，共 55 个 distribution。
- 设备：AMD Ryzen 7 6800H，8 核 16 线程，29,734,412,288 bytes 物理内存，集成 AMD Radeon Graphics。
- CUDA：不可用；系统未检测到 NVIDIA 设备或 `nvidia-smi`。
- 结论：本机 `--device auto` 必须解析为 CPU。不要把 DirectML 与 canonical PyTorch 结果混合。未来迁移到 NVIDIA 主机时，另建 CUDA profile、锁文件、wheelhouse 和 environment fingerprint。

验收命令：

```powershell
.\scripts\bootstrap.ps1 -Action Verify
.\scripts\pv.ps1 doctor --config configs/project_config.json --device auto
.\scripts\test.ps1
```

结果：environment verify 通过，doctor 8/8 通过，pytest 22/22 通过，全部 Python 源文件通过 `compileall`。

## 3. 数据验收

- 上游 revision：`afb5ce38128c7bac085b31d499d28b631ef8fc80`。
- `data.zip` SHA-256：`fba30c6a7965e49be94b47a62f8aff6cfb1c35c27f475f22092b56db41745e84`。
- canonical manifest：3,600 行，SHA-256 为 `bf94c05f60d4520e94a5d7d17f23514d3e62d7070e95c0c3ca692417de9f7960`。
- 四类各自 train/val/test 为 630/135/135；总计 2,520/540/540。
- schema、图像解码、文件 hash、leaf group 跨 split 泄漏和文件重复检查均通过。
- 已目检 `data/manifests/montages/class_0.jpg` 至 `class_3.jpg`：四组均为番茄叶片，健康、Early blight、Late blight 和 Leaf Mold 的组内外观与病斑差异合理，未见明显损坏文件或异类图片。
- 一张曾用于冻结前模型重载诊断的官方 test 图片已在 `configs/project_config.json` 的 `holdout_exclusions` 中登记，不再进入 manifest；同类下一候选已确定性补入，已暴露文件保存在 `data/quarantine/pre_gate_exposed`。该处理保留 540 张未打开的 blind test 集。

P2 仍保留为人工门禁。正式深度训练前，人类操作者必须再次打开四张 montage 并记录批准；AI 目检不得冒充人工批准。

## 4. 三管道执行验收

固定 24 张真实图像的 smoke 连续运行两次，三管道输出 hash 均为：

```text
db3ab1345a29736f672ab33e2c9943034454e910427fc295665ac2fc35588537
```

两次完全一致，prediction schema 无错误。Smoke 覆盖 HOG 特征与 SVM 拟合、Small CNN 单次优化、预训练 MobileNetV3 Stage A 单次优化及三个模型的验证推理。

HOG/SVM 完整验收：

- 特征：128x128 灰度 HOG，8,100 维；train cache 约 132 MB，val cache 约 28 MB。
- 求解器：`LinearSVC(dual=true, tol=1e-3, max_iter=20000)`。
- 四候选 macro F1：C=0.01 为 0.67118977，C=0.1 为 0.67312444，C=1 和 C=10 均为 0.67719047。
- Tie-break 选择较小的 `C=1`；validation accuracy 为 0.68148148。
- 最终模型 SHA-256：`ede98f8e3a5f2a6a723c747ee2efc3c6360a206bb5c5c5b3ef053f414d6b5a12`。
- 同一模型 hash 在三个不同代码状态的独立构建中保持一致；当前产物的 9 个 checksum 全部通过。
- 一张真实 `Tomato___healthy` validation 图经独立模型重载后预测为 class 0，输入与模型 hash 均写入 JSON 输出。

未冻结 test gate，也未生成正式 test 指标。这是有意的：test 只能在三个模型完成 validation 选择后统一打开一次。

## 5. VS Code 启动顺序

在 VS Code 中打开项目根目录，确认解释器为 `.venv\Scripts\python.exe`，然后从 `Terminal > Run Task` 顺序运行：

```text
PV: Environment Verify
PV: Doctor
PV: Tests
PV: Data Validate
PV: Smoke Test
人工批准四张 montage
PV: Train HOG-SVM
PV: Train Small CNN
PV: Train MobileNetV3
人工检查三份 validation selection
PV: Freeze Test Gate
PV: Evaluate Frozen Test
PV: Robustness All
PV: Compare Models
PV: Verify Complete Run
```

当前 `PV: Train HOG-SVM` 已完成，可直接读取 `artifacts/44e011204f972179/hog_svm`。正式实验若未改变 config、代码、数据或环境，可复用该结果；任何一项改变都应产生新实验 ID。

## 6. 中断与恢复

- HOG/SVM：特征缓存由 manifest hash 与特征配置 hash 定位。重新运行同一任务会验证 sample order 后复用缓存。
- Small CNN：每 epoch 原子写入 `checkpoints/small_cnn.pt`。
- MobileNetV3：Stage A 与 Stage B 分别原子写入 `checkpoints/stage_a.pt` 和 `checkpoints/stage_b.pt`。
- 深度 checkpoint 包含模型、优化器、scheduler、history、最佳权重及 Python/NumPy/Torch/CUDA RNG 状态。重新运行完全相同的 VS Code Task 会从下一 epoch 恢复。
- 恢复前先读 `artifacts/run_state.json`。不要删除 checkpoint，不要在同一实验中修改 config 或源代码，不要重复开启第二个训练进程。

## 7. 当前阶段状态

```text
P0 source acquisition       complete
P1 manifest validation      complete
P2 human montage approval   pending by design
P3 deterministic smoke      complete
P4 HOG/SVM                  complete
P5 Small CNN full training  pending experiment
P6 MobileNet full training  pending experiment
P7 test gate                pending
P8 comparison               pending
P9 final run verification   pending
```

部署已经完成；P2、P5 至 P9 是正式实验工作，不是环境或代码部署缺失。
