# Rresp 早期失效审计

上一轮五案例没有通过：003240 出现身份切换；007878 在旧保存窗口开始前已漂移；008311 的旧窗口中冻结候选没有合格的小目标框。001664 的响应修正收益需要保留。

本轮复用 Tclean、Rfreeze、Rdisc、Rresp 四个 epoch-300 权重。无训练、无参数更新。新增 `early_failure` 保存方案与 CPU 分析器，定位早期首次损害时是否仍有好框可选。

## 更新代码

新分支：`codex/vdrm-rresp-early-failure-audit`。项目根目录、OSTrack 环境执行：

```bash
git fetch origin
git switch --track origin/codex/vdrm-rresp-early-failure-audit
```

已在该分支时无需重复创建本地分支。旧配置和训练权重继续使用，不需要新的配置文件。

## 检查

```bash
for cfg in \
  vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
  vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300 \
  vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300
do
  test -f "output/checkpoints/train/ostrack/$cfg/OSTrack_ep0300.pth.tar" \
    || { echo "Missing checkpoint: $cfg"; exit 1; }
done
echo "Four checkpoints OK"
```

训练数据使用原固定 GOT-10k 152 条训练开发集入口。运行前会检查五案例完整图像前缀、标签、checkpoint 身份与来源；不读预测 TXT，不使用 UAV/VisDrone 验证集。

## 测试

```lua
CUDA_VISIBLE_DEVICES=0 python tracking/diagnose_vdrm_response.py \
  --save_dir ./output \
  --profile early_failure
```

该诊断单卡串行维护四条独立轨迹，四卡服务器只需占用一张卡。全部从第 1 帧初始化，后续由各自浮点预测驱动。当前帧 GT 仅用于前向结束后的评价，不用于裁剪/选峰/状态更新。

| 序列 | 保存帧（1-based） | 用途 |
| --- | --- | --- |
| 003240 | 88–100 | 复核相似实例切换 |
| 008333 | 54–74 | 复核 Hann 排序与重现 |
| 001664 | 65–100 | 保留真实正例 |
| 007878 | 2–130 | 补齐最早分歧、漂移及后续恢复窗口 |
| 008311 | 2–100 | 补齐小目标尺度失效之前的候选图 |

终点与旧五案例完全一致，只前移两个案例的保存起点；共 **1,996 个原生跟踪步骤、298 个保存帧**。应输出：

```text
RRESP PROBE CHECK PASSED: 1996 native steps, 298 records; NOT a performance pass
```

保存到 `output/analysis/frozen_response_early/rresp_early_probe.tar.gz`。包内含原图、四臂裁剪与响应/回归图、原生浮点轨迹、Rresp 实际输入、同裁剪 Tclean 回放、真实最终 CE 保留掩码、padding 掩码和基准 logits。旧 `focus_summary.json` 仍按原五窗口计算；新增 `expanded_summary.json` 单独记录扩展窗口，不替换旧对照口径。

## 分析

```css
python tracking/analyze_vdrm_response.py \
  --bundle output/analysis/frozen_response_early/rresp_early_probe.tar.gz \
  --output output/analysis/frozen_response_early/rresp_early_analysis.tar.gz
```

分析仅需 CPU 和诊断包。先验证包完整性，再独立重算所有轨迹 IoU、冻结候选框、同裁剪选择、GT 搜索支持和首次分歧。成功输出：

```text
RRESP ANALYSIS COMPLETE: 298 records; no performance/identity verdict
```

分析包中包含 `summary.json`、`per_frame.json`、`report.md` 和来源哈希。其“全部候选最优 IoU”使用 GT 做离线可行性检查，不能拿来充当真实跟踪性能。公共 logit 提升、去均值、最终 log-response 修正仅为保存地图上的单步数值对照，不会产生新的闭环预测轨迹，也不构成新模块通过。

## 下载

只下载两个本轮新文件，旧本地包继续保留：

1. `output/analysis/frozen_response_early/rresp_early_probe.tar.gz`
2. `output/analysis/frozen_response_early/rresp_early_analysis.tar.gz`

完整性检查已包含在测试和分析中，不需要额外运行旧 `analyze_uav_suite.py`。这两个新文件不会落到标准 `tracking_results` 或 `result_plots`。已有同名输出会拒绝覆盖；若要重复运行，为两条命令分别使用新的 `--output` 路径，并把分析命令的 `--bundle` 指向新的测试包。

## 决策门槛

先检查首次损害附近的原图及同裁剪候选。若好框仍在但选错，优先修身份/候选质量排序与公共偏移；若好框已全部失效，检查更早的尺度状态和定位能力；若目标已出裁剪，定位导致丢失搜索支持的更早事件。正常帧保护、001664 正例及身份切换必须同时复查。

五案例通过后才进入原固定 152 条开发集筛选。本轮不启动新的完整训练，也不进入可靠度/M2/J。

## 上传前验证

2026-10-07，本次相关 23 项测试通过（216.988 秒）。覆盖默认旧命令兼容、新命令解析与输出路径、完整早期帧集合、四臂全部原生轨迹在扩展保存前后逐项一致、旧五窗口 AO 一致、冻结权重哈希不变、GT 不参与单步对照选峰、缺失帧/伪造 IoU/非法 CE 修正拒绝、已有输出拒绝覆盖，以及 CPU 分析不访问数据集/权重/CUDA。

RTX 5070 上的原生 ViT-B 合成验证完成 1,996 个跟踪步骤和 298 帧打包/独立分析；007878、008311 两条轨迹与正式 tracker 输出精确一致，诊断优化器步骤为 0。另用已下载的服务器真实旧 Rresp 包完成 179 帧分析，重现此前五案例 AO。GPU 合成验证用于检查运行正确性，服务器真实早期失败帧仍需按上方命令生成。
