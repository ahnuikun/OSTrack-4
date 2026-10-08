# Rqual 下一步：固定候选的头部输入归因，不再训练

## 目标与边界

Rqual ep020 冻结审计通过，但原定五案例门槛未通过，见 `VDRM_RQUAL_REVIEW_20261008.md`。先查新头错误排序依赖哪些输入，再决定删什么、重做什么。本轮不延长训练、不跑 152/UAV、不接可靠度/M2，不修改已有模型、配置、结果 TXT 或分析缓存。

仅回放现有 `rqual_early_probe.tar.gz` 中 298 个 Rqual 原生归一化模型输入。不重新加载数据集，不重跑四臂闭环；需要现有 Rqual ep020 和源 Tclean ep300 权重，不需要 Rfreeze/Rdisc 权重。检查 checkpoint SHA256、完整模型 state、每帧原输出、原生 Hann 峰、原生预测框/IoU；不匹配直接停止。

## 固定什么，改变什么

所有干预共用同一裁剪、冻结视觉特征、256 个实际候选框、有效性 mask、原生 Hann 和最终能量公式。当前 GT 仅在输出算完后评估候选 IoU，不进入模型或供体选择。

| 干预 | 改变的质量头输入 | 不改变 |
| --- | --- | --- |
| native | 不改变 | 原始 Rqual |
| identity_zero | 乘积、绝对差、余弦三组置零，共 65 维 | 其余 6 维 |
| response_zero | 原 CENTER logit 置零，1 维 | 原 CENTER 候选和最终基础响应均保留 |
| geometry_zero | 宽高/offset 置零，4 维 | 实际候选框本身不改 |
| support_zero | ROI 支持度特征置零，1 维 | 有效候选 mask 仍按原始支持度 |
| 四种 template_swap | 只把质量头匹配向量换成其他四序列供体，重算前 65 维 | 搜索嵌入和非身份特征不改 |

每个供体取该序列第一张已保存帧的模板 ROI 投影向量，不看 GT 选择；每帧穷举另外四个序列。这里没有替换主干的模板图像，因此测量的是**质量头对已提取身份线索的依赖**，不是整个模型对模板的依赖。模板特征来自联合特征编码，也不能视作完全独立的身份表示。

每帧 9 组，共 298 个固定输入、1,192 次相关模板替换。置零/换模板是分布外干预，不是可直接部署的新模型；不能据单步 IoU 宣称闭环收益。全部干预都报告，不选择最佳供体充当模型结果。

## 服务器操作

分支：`codex/vdrm-rqual-cue-audit`。

```bash
git fetch origin
git switch codex/vdrm-rqual-cue-audit
git pull --ff-only origin codex/vdrm-rqual-cue-audit
```

在项目根目录、OSTrack 环境执行。

训练：**不训练，复用 Rqual ep020。** 不增加种子命令行参数。

检查：

```bash
test -f output/analysis/frozen_quality/rqual_early_probe.tar.gz \
  && test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020/OSTrack_ep0020.pth.tar \
  && test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300/OSTrack_ep0300.pth.tar \
  && echo "Rqual cue audit inputs OK"
```

若旧诊断包只在本地，上传已有 `frozen_quality/rqual_early_probe.tar.gz` 回这个路径即可，不重跑测试。脚本本身也会严格检查输入、权重和哈希，不依赖上面的 shell 检查兜底。

测试（单卡，固定输入回放与干预导出）：

```lua
CUDA_VISIBLE_DEVICES=0 python tracking/probe_vdrm_candidate_cues.py \
  --bundle output/analysis/frozen_quality/rqual_early_probe.tar.gz \
  --save_dir ./output \
  --output output/analysis/frozen_quality/rqual_cue_audit.tar.gz
```

预期 `Frozen input replays: 298/298`，最终 `RQUAL CUE AUDIT PASSED (integrity only)`。不是新模型效果通过。脚本拒绝覆盖已有输出；确需重跑请为 `--output` 换新文件名，后续检查使用同一新名字。

分析（CPU 独立复算，不需要 GPU/数据集/权重）：

```css
python tracking/probe_vdrm_candidate_cues.py \
  --check_bundle output/analysis/frozen_quality/rqual_cue_audit.tar.gz
```

预期 `RQUAL CUE BUNDLE CHECK PASSED: 298 records`。导出时已自动执行一次独立复算，此命令可在复制/下载后再次验证。

只需下载一个新增文件：

`output/analysis/frozen_quality/rqual_cue_audit.tar.gz`

下载到原来的本地 `frozen_quality` 目录即可，旧日志、权重审计和诊断包不重复下载。包内含报告、逐帧指标、完整特征/候选框、全部 9 组输出和小 MLP 权重；本地可继续做头部干预，无需再下载完整跟踪权重。

## 决策方法，而非自动宣布新结构有效

1. 先确认 native 与原诊断逐帧一致，0 优化步骤、前后模型哈希一致。任何一致性失败都先修诊断，不解释模型。
2. 查看 003240:93、001664:74/75 的正确/错误候选排序，检查 007878 的已修复早期帧和再失效段，同时保护 008311 的恢复前驱帧及 008333 正例。报告全部干预的变化，不能只报有利帧。
3. 若原响应/几何屏蔽能纠正多个直接误选，且模板替换敏感性弱：支持下一版删除这些输入、使用外观身份配对排序；仍须与分布外效应区分，不能直接把测试时置零部署。
4. 若模板/身份干预明显改变排序，但本目标模板仍把错误实例排第一：优先重做 ROI 空间/身份表征，而不是继续拟合标量 IoU。
5. 若固定候选确实无好框或目标在裁剪外：该帧不能靠候选重排直接解决，不把恢复问题硬塞给质量头。
6. 下一训练候选须先在训练集内、排除固定开发序列的困难实例配对上做离线身份排序验证，使用短预算与明确保护门槛。现阶段不擅自选择新结构、不用这五例制作训练标签、也不进入 Qonly/M2 联合阶段。

## 实现与验证范围

新增独立诊断入口、导出包 CPU 校验器、合成回归测试与原生 ViT-B/CUDA smoke；不改模型训练代码或既有配置。测试覆盖输入分组、不修改输入、hook 异常清理、拒绝覆盖、错误 checkpoint、重算校验、供体规则、伪造/损坏数据拒绝和 CPU-only 检查。真实服务器权重不在本地，服务器首次回放的严格校验仍然必需。

2026-10-08 本地验证：新增 10 项测试全部通过；相关 Rqual 原有 14 项回归通过，覆盖正式 tracker、嵌套结果 writer、配置哈希、20 轮保存/续训、四卡入口预检与冻结隔离。文档中的两个 Python 命令已通过参数解析测试，不带种子参数。

原生 RTX 5070 / PyTorch 2.7.0+cu128 合成验证：原流程生成 1,996 跟踪步骤、298 帧源包，再回放 298 帧，回放最大误差 0；全部 9 组干预通过 CPU 小 MLP 独立复算，前后模型 state 和源包哈希不变，全程 0 优化步骤。导出包约 34.7 MB（合成数据大小，不保证真实数据相同）。报告：`output/analysis/vdrm_candidate_cues_gpu_smoke.json`。

另用真实下载包的 298 帧逐一验证 native 几何校验：最大 CPU/GPU 框解码差 0.0001221 像素，IoU 差小于 0.000001；沿用旧分析器 0.002 像素容差，峰值仍须严格相同。该项不需要也没有加载用户权重，不能代替真实权重的 GPU 回放。
