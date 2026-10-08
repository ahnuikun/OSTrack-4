# 固定身份头：原生 Tclean 轨迹上的只读候选验证

## 决策（2026-10-08，运行新实验前固定）

离线 v1 包已独立复算：4032 个训练配对，1193 个开发配对，152 个开发序列；基础响应错误只有 11 对、涉及 8 序列。mean_rank 修复 5/误改 2，mean_id 修复 4/误改 6，spatial_id 修复 5/误改 3。三个头相对基础响应的整体正确率配对 95% 区间均跨 0。

跨模板辨别增益不等于同场景纠错增益；不继续训练，不降低原 v1 门槛，不据此删除/选定某个头。原 v1 的所有未通过标志保持不变。

## 本次只改变验证协议，不改模型

- 使用同一个原始结果包内的三个 ep020 权重，无初始化、优化器或新训练。
- 从显式 `data/got10k/train` 的固定 152 开发 ID 读取全部帧；不经过 local.py 的 val/test 路径选择。检查 list 的编号顺序、标注及逐帧文件。
- Tclean 调用现有原生 initialize/track。只有第一帧 GT 用于初始化，此后搜索位置始终来自原生浮点预测，无 GT 重置、无整数 TXT 回放。
- 同一次视觉前向的输出同时交给三个只读身份头。头的选择不进入 tracker.state；这是 **Tclean 原生闭环轨迹上的单步反事实**，不是三个新跟踪器的闭环结果。
- 全部帧都记录，包括不可见、无正确候选、支持不足的帧，不按头的表现筛选。

## GT 无关候选与控制

主分析：基础 Hann 响应最高的 5 个候选，按原 ROI 支持规则（至少 3 个共同位置）过滤后，以身份余弦排序。没有可用候选则回退原生框；同分优先原生候选，否则固定较小 grid index。GT 不参与候选生成、过滤或选择，不强行加入正确框。

辅助诊断：全部 256 个预测候选使用相同规则。它只检查 top-5 候选池是否遗漏可选框，不用辅助结果自动挑部署策略。保留中间 IoU 的框，不再只比较一个 GT 正例与四个极低 IoU 负例。

每帧验证全部候选中的基础响应峰值，经过相同映射/裁剪后逐值等于原生 tracker 返回框。每序列第二帧额外验证相同输入重放逐张量一致。骨干和三个头的前后状态哈希相同、参数梯度为空，0 优化步骤。输出单独写入 analysis，不接触旧 tracking_results。

## 预注册报告与解释

- 整体可见帧、正常帧（cover≥7，沿用离线 v1）、严重部分遮挡（cover=1..3）、重新出现后 10 帧的序列宏平均 IoU 与配对区间。
- 正确阈值 IoU≥0.5；基础错误→头正确为修复，基础正确→头错误为误改。只称候选选择错误，不自动标为身份切换。
- 报告全体可见帧、原生错误帧、全 256 候选有好框、支持有效候选有好框、top-5 有好框的覆盖，分别区分搜索/定位无解、特征支持不足、候选池遗漏和排序错误。
- 至少 30 个主候选池可修复的困难帧、分布在至少 10 个序列，才称困难证据数量足够；多帧不能当成独立序列。数量不足不追加自适应挖掘、不改门槛。
- 保留原始/逐序列修复与误改计数、序列 bootstrap（内部固定 42，5000 次）区间。只有同裁剪整体改善区间下界 >0、正常帧平均 IoU 下降≤0.5 pp、严重遮挡/重现未下降且覆盖足够，才讨论后续小规模原生闭环验证。此处不自动授权整合或训练。
- 重新出现指标仍为单步同裁剪指标，不计算或声称新的闭环恢复率、恢复时延；这需要下一阶段独立状态轨迹。

## 操作

分支 `codex/vdrm-identity-native-probe`。本次不训练，一张 GPU 足够；一个 Tclean 前向服务所有头。不需要旧预测 TXT、特征缓存或三个独立 pth，仅需原结果包和 Tclean ep300。

```bash
git fetch origin
git switch codex/vdrm-identity-native-probe
git pull --ff-only origin codex/vdrm-identity-native-probe
```

测试：

```lua
CUDA_VISIBLE_DEVICES=0 python tracking/probe_vdrm_identity_native.py \
  --screen_bundle output/analysis/identity_offline_v1/screen/identity_offline_results.tar.gz \
  --data_root ./data/got10k/train \
  --save_dir ./output \
  --output_dir output/analysis/identity_native_v1
```

检查：

```bash
test -f output/analysis/identity_native_v1/identity_native_results.tar.gz \
  && echo "Identity native probe bundle OK"
```

分析（CPU，无原始图片/权重依赖）：

```css
python tracking/analyze_vdrm_identity_native.py \
  --bundle output/analysis/identity_native_v1/identity_native_results.tar.gz
```

应完成 152/152 序列。只下载 `output/analysis/identity_native_v1/identity_native_results.tar.gz`。目录禁止覆盖；中断时使用新的输出目录，不删除现有文件。包的校验通过仅代表完整性和从保存分数/框复算一致，不代表新头通过闭环门槛。CPU 无法只凭该包重算原始图像主干前向。

## 本地工程检查

已完成真实服务器离线包、缓存和小头权重复核（详见 `VDRM_IDENTITY_SCREEN_REVIEW_20261008.md`），未修改用户文件。13 项新增测试通过，覆盖 152 个真实固定 ID 的合成数据目录、304 次原生步骤、全帧记录、无 GT 候选生成、输入标注变化不影响推理、浮点解码逐值一致、来源错配/轨迹篡改拒绝及实际文档命令解析。16 项旧离线身份及 12 项原生浮点回归通过，共 41 项。

原生 ViT-B / RTX 5070 合成测试跑完整 152 序列，每序列核对重放、状态与 CPU 指标复算。合成测试只验证工程路径，不是模型效果；本地未运行真实服务器 Tclean 的新原生轨迹。这里的模型固定与身份分数复算已经验证，真实开发轨迹的覆盖/效果仍需服务器结果。
