# Rresp：最终候选响应专用残差，独立单种子训练

## 为什么进入这一臂

服务器 Rscore 包的五案例审阅没有通过：它在 008333 遮挡段保留响应收益，但重现段仍退化；001664、007878 的轨迹与 Tclean 相同，没有保住此前其他臂的正例。003240 的改善主要来自去掉 Rdisc 回归反馈，而不是超过 Tclean。因此停止扩大 Rscore，不接可靠度/M2，不组合 J。

Rresp 不是已证明有效的新模型，而是一次明确的设计检验：**冻结 Tclean 的全部权重和缓冲区，只训练最终响应图的修正头。** 保留原生裁剪、Hann、CENTER 解码和状态更新；不加入扩大搜索、运动重置或新回归分支。

## 架构与监督

- 从既有 Tclean epoch-300 权重逐项严格复制全部参数，包括旧 VDRM 的标量与精确为零的 alpha。旧 VDRM 仅为 checkpoint 兼容保留，不训练旧部件路由、可靠度标量或 alpha。
- 一次冻结的视觉前向后，用最终搜索特征、初始化模板目标框内的全局目标原型、原始 CENTER logits，计算模板匹配乘积/差值/余弦特征。小卷积头输出 logit 修正，最后一层零初始化，修正范围固定为 ±2。
- 新头输入 detach，整个主干和原 CENTER head 明确 no_grad、始终 eval。原 size_map/offset_map 不变；**选择了不同候选后最终框当然可以改变**，不声称整个预测框恒定。
- CE 最终删除的格点修正为零。训练正样本为 GT 框中实际存活、未 padding、未被人工遮挡的格点；无正样本或无负样本时跳过，不把最近搜索格点伪造为 GT。模板初始化小目标的最近模板格点回退只用初始化框，不使用当前帧 GT。
- 两个损失：平衡 sigmoid focal 分类；最终 `log(score_map × Hann)` 的困难负候选 top-4 排序。排序与测试 Hann 后 argmax 一致，margin=0.2（对数单位），GT 周围忽略 1 格。权重分别 1.0/0.5；GIoU、L1、旧 VDRM 辅助损失全部为零。
- 结构化遮挡和同类别干扰物粘贴沿用 Rdisc 的 0.5/0.3 训练配方。既有 GOT 训练数据入口排除固定 152 条开发序列；训练跳过验证 loader。只做 seed=42，启动器默认值已校验，下面命令不需要种子参数。

新字段仅在 Rresp YAML 中启用，不更改旧配置默认内容和旧 checkpoint 配置哈希。训练/续训/保存/测试均校验独立 `frozen_response_contract`，保护全部基准参数与 BN 缓冲区；优化器必须只包含 `response_residual.*`。

## 服务器：现在只运行训练与五案例检验

分支：`codex/vdrm-rresp-frozen-response`。在项目根目录、OSTrack 环境执行。不要删除旧模型权重或本地分析包。

```bash
git fetch origin
git switch --track origin/codex/vdrm-rresp-frozen-response
```

如果已经切到该分支，无需再次创建本地分支。训练源只需要 Tclean 原来的 epoch-300 权重和原训练数据；后面的五案例诊断另需保留 Rfreeze/Rdisc 的两个 epoch-300 权重作对照。第一行仅检查源权重，不训练 B0/Tclean。

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300/OSTrack_ep0300.pth.tar \
  && echo "Tclean source checkpoint OK"
```

### Rresp

训练：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

训练开始前自动用实际训练 minibatch 审计，不做优化步骤。检查失败则直接退出，不允许继续训练；日志 `output/logs/vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300-frozen-preflight.json`。

检查：

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300/OSTrack_ep0300.pth.tar \
  && echo "Rresp checkpoint OK"
```

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/check_vdrm_response.py \
  --config vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300 \
  --save_dir ./output \
  --checkpoint output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300/OSTrack_ep0300.pth.tar
```

应出现 `RRESP CHECKPOINT CHECK PASSED`。这里是合成输入的接线审计，不是性能评估。它检查关闭修正精确还原 Tclean、同输入回归图不变、两个损失只对新头有梯度。

测试（先五案例，附带分析与打包）：

```lua
CUDA_VISIBLE_DEVICES=0 python tracking/diagnose_vdrm_response.py \
  --save_dir ./output
```

分析/独立完整性检查：

```css
python tracking/diagnose_vdrm_response.py \
  --check_bundle output/analysis/frozen_response/rresp_probe.tar.gz
```

成功输出 `RRESP PROBE CHECK PASSED: 1996 native steps, 179 records`，只表示数值与完整性检查通过。五窗口与此前固定窗口相同：003240 88–100、008333 54–74、001664 65–100、007878 68–130、008311 55–100。四臂各从第 1 帧开始按自身浮点预测跟踪；不读旧 TXT，不用当前 GT 重置或选候选。GT 只在前向完成后计算 IoU/阶段。

只下载两个新增文件：

1. `output/analysis/frozen_response/rresp_probe.tar.gz`：包括四臂浮点轨迹、179 帧原图/各自 crop/响应图、Rresp 真实输入、同裁剪 Tclean/关闭修正回放、`focus_summary.json`、权重来源哈希；足以检查正例和失败段。
2. `output/analysis/frozen_response/vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300/checkpoint_audit.json`。

旧本地分析包不需再次下载。已有同名诊断包会拒绝覆盖，重跑时用 `--output` 指定新文件名，不删除旧包。五案例脚本不会写入/覆盖标准 tracking TXT 或旧 result_plots。

先检查 003240 身份、008333 的完整遮挡/重现窗口，以及 001664/007878/008311 既有正例是否保留。**修正回归图不变不保证不发生裁剪反馈失败**：不同峰仍会选择不同框。若两类问题未解决或明显丢失正例，停止扩大测试，返回该包分析，不调阈值掩盖问题。五案例属于已使用开发案例，不能算无偏验证集。

## 五案例审阅通过后，才运行固定 152 条开发集

下面两条标准测试均可重复执行：默认跳过已有结果文件，补齐此前删除的文件；文件是否完整由随后的分析预检判定，测试的跳过逻辑本身只检查文件存在。都使用规范 `<配置>/got10k/<序列>.txt`，不是 GOT 官方测试集，也不是 UAV 验证集。

测试 Tclean 对照：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  --dataset got10k_vdrm_dev \
  --num_gpus 4 \
  --threads 4
```

测试 Rresp：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300 \
  --dataset got10k_vdrm_dev \
  --num_gpus 4 \
  --threads 4
```

分析：

```css
python tracking/analyze_vdrm_module1_dev.py \
  --tracker_params \
    vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
    vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300 \
  --reference vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  --report_name vdrm_rresp_dev \
  --visibility_stages \
  --per_sequence
```

需显示 `152/152`；缺失、不完整预测或 flat/nested 冲突都会拒绝分析。新结果在 `output/test/tracking_results/ostrack/vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300/got10k/`；分析 cache 与阶段报告在 `output/test/result_plots/vdrm_rresp_dev/eval_data.pkl` 和 `visibility_stages.json`。

本轮预注册筛选：`Rresp−Tclean` 的部分遮挡/严重部分遮挡/重新出现 AO 为正；正常帧 AO 下降不超过 **0.5 pp**；不增加人工确认的差异身份切换；配对恢复率不得下降、双方成功的恢复时延不得变长。报告序列配对 bootstrap 区间，不将单种子或五案例结果宣称为最终显著收益。若无足够恢复事件，注明未证明而不是当作通过。通过后才讨论多种子，可靠度与搜索恢复策略仍为独立阶段。

## 本地验证

2026-10-07：完整 173 项回归测试通过（206.878 秒）；包含旧配置哈希不变、零初始化/关闭修正的精确 Tclean 输出、两项损失的梯度隔离、全部基准参数与缓冲区保护、CE/裁剪/遮挡/padding 的缺失正样本跳过、Hann 后排序与梯度方向、优化器参数集合、源权重缺失和非四卡启动在数据加载前失败、不构建验证 loader、正式保存与续训、真实命令解析与无种子参数、诊断包篡改/覆盖拒绝、完整浮点闭环、正式 tracker 和规范结果 writer。

RTX 5070 上另完成原生 ViT-B/CE/CENTER 的合成验证：1,996 个跟踪步骤、179 个诊断帧，全部精确回放、关闭修正精确 Tclean、同输入回归图不变、同峰 bbox 精确一致；正式 tracker 100 帧与诊断轨迹一致。只在临时合成模型/图像上做了 **1 次优化器更新**，确认 101,217 个新头参数可训练、全部基准参数和缓冲区未改变；诊断本身仍为 0 次优化。报告 `output/analysis/vdrm_response_gpu_smoke.json`。

上述合成验证不等于服务器真实权重/数据上的性能通过。本地没有四卡训练环境，未运行四卡正式训练；服务器入口会额外审计真实训练 minibatch 并在失败时退出。Rresp YAML 配置 SHA256：`88c2e3e7d0d20fb14a033e23083c3c33e227dc384d16571f62d2e255319b9c65`。
