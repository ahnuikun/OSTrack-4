# 身份表征离线筛选：先验证，再接跟踪器

## 决策与预注册

Rqual 输入归因已独立复算 298 帧、每帧 9 组，与源包及 checkpoint/state 哈希一致，回放误差 0。原响应置零修复 003240:93（0→0.834），但损坏 001664:74（0.860→0）；身份特征置零修复 001664:75（0→0.921）。几何/支持度置零没有超过 5 pp 的改善。四供体模板替换共 1,192 次，49 次改峰、0 次超过 5 pp 改善、2 次超过 5 pp 退化。不同目标的五个模板投影余弦为 0.939–0.989；这支持检查身份表征，但不能单凭五例宣称全局塌缩或所有视觉特征无用。

因此不发布旧权重“置零某输入”的补丁，不续训旧 Rqual。下一个实验是**离线身份匹配能力筛选**，不是 Ronly/Qonly、不是新跟踪模型；不训练可靠度、不接 M2/J、不访问 UAV 或 GOT 官方 val/test。

## 对照与最小结构

三个同参数量的小头均从头初始化，不继承 Rqual 投影/MLP；均使用同一 Tclean ep300 冻结特征缓存：

- `mean_rank`：3×3 ROI 有效特征均值，LayerNorm（无可学习仿射）→32 维无 bias 线性投影→归一化余弦。仅候选排序监督。
- `mean_id`：相同均值结构和候选排序，增加跨模板身份监督。
- `spatial_id`：相同两种监督，保留 9 个相对位置，逐位置使用同一 LayerNorm/投影，平均有效对应位置余弦。不增加可靠度门控、路由或可学习部件权重。

`mean_id−mean_rank` 检查跨模板监督的作用；`spatial_id−mean_id` 检查空间保留的作用。这个 rank-only 对照避免把新监督的收益误归因于新结构。

每头在正式 ViT-B 上只有 24,576 参数，初始化权重、样本顺序、预算相同。身份头不接收 CENTER 响应、宽高/offset、IoU 或 GT；支持 mask 仅用于排除无特征位置。响应用于离线选困难负候选、作为比较基准，不作为头输入。

全部头使用当前模板下，好候选优于四个低 IoU 高响应候选的监督。两个 `_id` 头另加：固定好候选下，本模板优于批内其他序列模板；`mean_rank` 不加入该项。跨序列负配对假定目标不同，仍可能有重复实例；同序列样本不互作负例。低 IoU 候选可能是背景，不能自动称为相似身份干扰物。

## 数据和预算

配置：`experiments/vdrm_identity/offline_v1.json`，内部 seed=42，命令不提供种子选项。训练仅从 GOT VOT train ID 中固定抽 512 序列；评估仅用原固定 152 开发 ID；严格检查不重叠。每序列最多 8 个当前/前一帧均可见的帧，初始化模板来自第一帧，搜索裁剪锚点用前帧 GT 加固定抖动，不用当前帧 GT 选择预测。

这是 **GT 辅助裁剪下的离线配对任务**，不是原生闭环跟踪。当前 GT 只在冻结前向后标注候选、构造训练/评估配对；好候选不足或有效部件不足时记跳过原因，不伪造正样本。含四个低 IoU 高响应负候选的配对并不保证四个不同实例。

正例取有效好候选中基础 Hann 响应最高的框，不只取 IoU 最高框，避免把基准本来能选对的好框排除、虚构困难样本。负例取有效低 IoU 候选中基础 Hann 响应最高的四个。双方在同一五候选集合上比较；这仍不是 Tclean 的完整候选/原生跟踪指标。

先单次提取冻结特征，再各训练 20 个**缓存 epoch**，而非跑 20/300 轮图像主干。最多约 4,096 个训练配对；batch=64。只使用最后一轮，不根据开发结果挑 checkpoint 或调超参。暂不集成跟踪器，因此没有新的标准 tracking_results TXT。

## 通过条件

先要求训练有效配对 ≥256；开发有效配对 ≥100、有效序列 ≥30；基准排错的困难配对 ≥30。计数不足标为“证据不足”，不判通过。

三个头分别与同候选 Tclean 基础响应比较：序列宏平均候选 top-1 正确率不低于基准超过 0.5 pp；正常帧不低于基准超过 0.5 pp；困难配对增益 ≥5 pp；本模板对四个固定异序列供体的 top-1 正确率 ≥80%（有效身份查询至少 100 对、30 序列）。同分按失败计，防止常数分数利用正例位置。

跨模板监督只有在 `mean_id` 通过独立门槛、对 `mean_rank` 的模板检索配对差值 95% 序列 bootstrap 下界 >0，且候选排序/正常帧不下降超过 0.5 pp 时才有保留证据。空间结构只有在 `spatial_id` 通过独立门槛、对 `mean_id` 的候选排序配对差值 95% 区间下界 >0，且正常帧下降不超过 0.5 pp 时才有增加复杂度的证据。没有增益证据的结构不默认保留。

离线通过也不等于残差/可靠度闭环通过。之后仍需原五案例原生轨迹保护、固定开发集完整门槛。失败时不自动追加 epoch，不把这些开发样本拿去训练。

## 操作说明

分支：`codex/vdrm-identity-offline-screen`。在服务器项目根目录、OSTrack 环境执行。用户权重、真实数据和原诊断包不上传 GitHub。

```bash
git fetch origin
git switch codex/vdrm-identity-offline-screen
git pull --ff-only origin codex/vdrm-identity-offline-screen
```

输入检查：只需要 Tclean ep300 和 GOT-10k train。不需要 Rqual/Rresp/Rfreeze/Rdisc 权重，也不需要旧测试 TXT。

```bash
test -f data/got10k/train/list.txt \
  && test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300/OSTrack_ep0300.pth.tar \
  && echo "Offline identity inputs OK"
```

特征准备（单卡、只读主干）：

```lua
CUDA_VISIBLE_DEVICES=0 python tracking/cache_vdrm_identity.py \
  --config experiments/vdrm_identity/offline_v1.json \
  --data_root ./data/got10k/train \
  --save_dir ./output \
  --output_dir output/analysis/identity_offline_v1/cache
```

正常计划为 **512 train + 152 dev = 664 序列**；每条显示有效配对数，跳过原因保存到 `cache/index.json`。有效配对数不是固定值，不能用“非 152/152”误判此离线采样失败。身份来自 train/list.txt 和固定 ID 划分；支持 train 路径为符号链接。主干前后 state 哈希必须一致，0 优化步骤。

训练（同一命令依次训练 mean_rank / mean_id / spatial_id 三头，单卡足够）：

```lua
CUDA_VISIBLE_DEVICES=0 python tracking/train_vdrm_identity.py \
  --cache_dir output/analysis/identity_offline_v1/cache \
  --output_dir output/analysis/identity_offline_v1/screen \
  --device cuda
```

每头仅 24,576 参数；20 个缓存 epoch，数据最多 4,096 个训练配对，每头最多 1,280 次小矩阵优化步骤。主干不在训练进程中加载，无法被优化器更新。固定内部 seed=42，三个头初始化与样本排列一致。先用实际缓存 batch 审计输入无梯度、头有梯度及跨序列身份负配对存在，失败则停止。无需占用四张卡做 DDP。

检查：

```bash
test -f output/analysis/identity_offline_v1/screen/Identity_mean_rank_ep0020.pth \
  && test -f output/analysis/identity_offline_v1/screen/Identity_mean_id_ep0020.pth \
  && test -f output/analysis/identity_offline_v1/screen/Identity_spatial_id_ep0020.pth \
  && test -f output/analysis/identity_offline_v1/screen/identity_offline_results.tar.gz \
  && echo "Offline identity heads and results OK"
```

测试：上述训练命令在三个头训练完后，自动只对开发缓存进行一次离线测试，不运行 tracker，也不访问 UAV。所有开发分数/描述子都会导出，不挑最佳 epoch。

分析（CPU，不需要原缓存/数据集/权重）：

```css
python tracking/analyze_vdrm_identity.py \
  --bundle output/analysis/identity_offline_v1/screen/identity_offline_results.tar.gz
```

应出现 `IDENTITY OFFLINE BUNDLE CHECK PASSED`，它只表示包完整、分数和指标复算一致，不代表性能门槛通过。是否通过看 `offline_gate_passed`；身份监督的增益证据看 `identity_supervision_supported`，空间结构的证据看 `spatial_complexity_supported`。

只下载新增文件 `output/analysis/identity_offline_v1/screen/identity_offline_results.tar.gz`。可以放回本地原 `frozen_quality` 目录，旧文件不再下载。包中包含三个小头权重、梯度审计、20 轮训练统计、逐样本证据、投影后描述子和配对置信区间。

所有输出目录拒绝覆盖；未完成/中断运行如需重跑，请使用新的输出目录，并保持后续命令路径对应。不自动删除旧目录，不自动续训。若缓存有效配对不足，先提供 `cache/index.json` 和报错，不改门槛或增加轮数。

## 已知边界

- 这三个头不对旧 Rqual 的“某一组置零”做部署承诺，是新的待验证假设；均值及 rank-only 提供更简单的对照，额外监督与空间结构都需证明增益才保留。
- ROI 来自冻结联合模板/搜索编码，仍可能包含条件化或背景信息。跨模板分辨能力不等价于独立图像 ReID；高召回/误报率可靠度校准另行验证。
- 当前配对含 GT 选择的好候选，性能是可排序条件下的离线能力，不是完整候选选择/闭环 AO。统计所有无好候选与低支持跳过，不能删掉失败再宣称跟踪有效。
- 只筛选 512 个训练序列，不是用新数据覆盖原训练配方；若失败，只说明该结构/监督/短预算组合未通过。

## 本地验证

2026-10-08：16 项新增测试通过，包括真实固定 ID 划分、154 序列合成数据目录的缓存入口全链路、错误数据集/权重预检、无支持候选跳过、冻结状态不变、头输入梯度隔离、同序列负例排除、排序-only 目标对照、同分失败、三臂训练/保存/复算、训练数据访问范围、结果篡改拒绝、两次训练权重逐张量一致，以及本文命令的实际参数解析。相关旧 Rqual/归因回归 24 项通过；未修改旧模型、配置或结果 writer。

原生 RTX 5070 / PyTorch 2.7.0+cu128 合成验证通过：ViT-B 输出 768 维冻结 ROI 特征，三头各 24,576 参数、各训练 20 个合成缓存 epoch、保存结果，并通过 CPU 分数/指标独立复算。主干 state 不变，缓存阶段 0 优化步骤。合成标签/特征扰动只用于工程验证，不是效果证据；没有在本地训练用户模型，也未在本地运行真实服务器 checkpoint。报告：`output/analysis/vdrm_identity_offline_gpu_smoke.json`。

结果包的 CPU 分析复算投影后描述子的分数、供体配对、宏平均和置信区间，不声称无原缓存就能重新计算整个主干或投影前特征。真实数据的有效配对覆盖与性能结论仍需服务器运行。
