# 原生浮点分叉审计：不训练，只定位失败起点

上一轮整数锚点诊断未复现 003240 第 93 帧、008333 第 61 帧的原生输出，不能据此归因。此次直接调用原生 OSTrack.initialize/track，保存真实浮点状态及实际输入张量，不读取预测 TXT，不改正常测试结果，不更新任何权重。

## 服务器操作

更新到 `codex/vdrm-native-float-branch-audit` 后，在 OSTrack 项目根目录、已激活的 OSTrack 环境运行。需要原来的 Tclean、Rfreeze、Rdisc 三个 epoch-300 权重和 GOT-10k train 数据；无需重跑 152 条序列或恢复旧预测文件。三个配置名及权重路径与上一轮完全相同。

测试与诊断（单卡，无种子命令行参数）：

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/diagnose_vdrm_native_float.py \
  --save_dir ./output \
  --device cuda
```

成功必须出现 `NATIVE FLOAT AUDIT PASSED: 516 native steps, 102 same-crop records, 102 fork records`。任何原生回放或控制分支不一致均中止，不生成新的归因包。

独立检查（不需要数据、权重或 GPU）：

```bash
python tracking/diagnose_vdrm_native_float.py \
  --check_bundle output/analysis/frozen_vdrm/native_float_audit.tar.gz
```

成功出现 `BUNDLE CHECK PASSED: 102 same-crop records, 102 fork records`。

只下载新文件：`output/analysis/frozen_vdrm/native_float_audit.tar.gz`。此前本地下载的文件保留即可。

已存在的包不会被覆盖。确需复跑时另取文件名：

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/diagnose_vdrm_native_float.py \
  --save_dir ./output \
  --device cuda \
  --output output/analysis/frozen_vdrm/native_float_audit_repeat.tar.gz
```

## 固定设计与安全检查

- 003240 从第 1 帧原生跟踪至第 100 帧，在第 88 帧前的 Rdisc 状态分叉，审计 88–100 帧。
- 008333 从第 1 帧原生跟踪至第 74 帧，在第 54 帧前的 Rdisc 状态分叉，审计 54–74 帧。
- 每个原生跟踪帧，都用捕获的实际张量再次前向，五项视觉输出、路由 logits、CE 索引、浮点框必须精确一致。推理输入不是从整数框重建。
- 每个审计帧，对三个原生浮点裁剪分别执行 Tclean、Rfreeze、Rdisc、Rdisc_route0、Rdisc_alpha0。alpha=0 必须精确等于同裁剪的 Tclean。
- 三条分叉轨迹从同一个真实 Rdisc 浮点状态开始，各自继续原生跟踪：control（不干预）、route0（仅新路由修正为零）、alpha0（整个残差关闭）。control 必须逐帧等于未分叉的 Rdisc。分叉后的裁剪由各自上一帧预测决定，不用 GT 重置。
- 原生裁剪、预处理、Hann、CENTER 框头、坐标映射及状态更新均未修改。只替换构造阶段的重复权重加载，使用已严格验证来源的网络。
- 当前 GT 只在前向完成后计算诊断；仅第 1 帧 GT 用于标准初始化。没有优化器、反传、训练；全部参数及缓冲区前后哈希相同；异常时恢复临时 alpha、hook 和 forward 捕获器。
- 输出为独立压缩包，包含浮点历史、分叉轨迹、原图、裁剪、实际输入 NPZ 与哈希、响应/CE/路由/注入图、模型来源及环境信息。独立检查复核所有文件哈希、每条轨迹的浮点前后帧连续性和控制分支一致性，不解压任意文件。

这只是两个已知开发集案例的诊断，不是新测试成绩，也不能自动认定身份切换。关闭部件后的轨迹改善需结合原图人工确认；本次原生轨迹若未复现历史失败，先检查 checkpoint/环境差异，不能宣布修复。即使负例改善，后续设计也必须复核此前正收益案例和固定开发集门槛；可靠度训练与联合模型仍不在本步骤范围内。

分叉只检验指定窗口内的后续作用。未改善不能排除更早帧已累积的影响；alpha0 分支从 Rdisc 的历史状态开始，也不是从第 1 帧运行的 Tclean。二者不能混称。

## 本地验证记录

145 项完整回归测试全部通过，包括文档中可复制命令的解析、三臂构造器与原版原生 tracker 等价、浮点初始化、GT 输入限制、异常恢复、缺失历史图像预检、禁止覆盖、干预别名保护、输入哈希、轨迹连续性，以及大压缩包按存储顺序一次读取。原生 ViT-B/CE/CENTER、原生 OSTrack tracker/Preprocessor 的 RTX 5070 CUDA 合成测试完成 516 个主轨迹跟踪步骤、102 个同裁剪记录和 102 个分叉记录；全部精确复现检查及完整状态哈希检查通过，独立包复检通过。没有训练步骤，也没有服务器真实数据或实际训练权重的本地复现。
