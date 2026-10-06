# Rscore：先验证响应/回归解耦，不训练

## 设计与边界

已有服务器原生浮点审计已复现历史失败。003240 的路由干预先产生小回归差异，再经裁剪反馈影响身份选择；008333 直接关路由会损失部分遮挡收益。保存图的离线组合在 008333/61 与 001664/74 上保留了主要响应收益，但尚未证明闭环改善。因此下一步只有一个新臂：

`Rscore = 同一实际输入上的 Rdisc score_map + Tclean size_map/offset_map`。

保持原生裁剪、CE、Hann、CENTER 解码、坐标映射及状态更新；每条轨迹从第 1 帧独立开始，此后只用自身上一帧浮点预测。当前 GT 只用于前向后的分析。使用原有 epoch-300 权重，不做反传、优化器、可靠度/M2 或训练。两次完整前向是归因探针，不是最终低开销架构。

独立 tracker 为 `ostrack_rscore`，测试参数名为 `vitb_256_mae_ce_vdrm_rscore_probe_s42_32x4_ep300`。参数模块显式映射到原 Tclean/Rdisc 配置，**不是新训练 YAML**；原 YAML 及 checkpoint 配置哈希不变。不支持用 `tracking/train.py` 训练这个名字，也不支持 UAV/保留验证集测试。

## 第一关：五案例闭环（现在运行）

在新分支 `codex/vdrm-rscore-closed-loop-probe` 的项目根目录、OSTrack 环境运行。需要原来的 Tclean、Rfreeze、Rdisc 三个 300 轮权重以及 GOT-10k train；不需要旧预测 TXT 或旧包。合计五案例：003240 88–100、008333 54–74、001664 65–100、007878 68–130、008311 55–100。这些连续窗口覆盖上一轮五案例的全部关键帧，包含正例后续退化段，不只挑收益帧；窗口在运行新结果前固定。

测试及分析打包：

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/diagnose_vdrm_rscore.py \
  --save_dir ./output
```

成功应出现 `RSCORE PROBE CHECK PASSED: 1996 native steps, 179 records`。这表示数值/完整性检查通过，**不是性能通过**。它包含四臂真实浮点轨迹、原图、各自 crop/响应/尺寸/偏移图、Rscore 实际输入及哈希、同裁剪拆分、阶段汇总、权重来源和前后完整状态哈希。

独立检查：

```bash
python tracking/diagnose_vdrm_rscore.py \
  --check_bundle output/analysis/frozen_vdrm/rscore_probe.tar.gz
```

只下载新增的 `output/analysis/frozen_vdrm/rscore_probe.tar.gz`。已有本地包不需要再次下载。已有同名包会拒绝覆盖；必要时通过 `--output` 指定新的 `.tar.gz` 名称，不删除旧文件。

第一关判断：人工检查 003240 是否仍发生差异身份切换；检查 008333 的 59/61/62 遮挡帧和 64–73 重现窗口，不能只挑第 64 帧；保护 001664/007878/008311 的既有正例。对照 TC/RF/RD 同次原生运行。案例 AO、单帧 IoU 或“数值检查成功”均不自动替代身份人工确认；单帧不可见不计为正式配对恢复事件。

如果两个失败案例没有改善或明显丢失正例，停止扩大测试，先分析新包，不训练、不扫阈值。下面的第二关仅在第一关审阅后运行。

## 第二关：固定 152 条训练开发集（第一关后）

四卡仅测试新臂，正常结果目录与三条旧臂完全隔离：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_name ostrack_rscore \
  --tracker_param vitb_256_mae_ce_vdrm_rscore_probe_s42_32x4_ep300 \
  --dataset got10k_vdrm_dev \
  --num_gpus 4 \
  --threads 4
```

唯一写入目录（标准 save_dir=./output）：

`output/test/tracking_results/ostrack_rscore/vitb_256_mae_ce_vdrm_rscore_probe_s42_32x4_ep300/got10k/<序列>.txt`

先确认原三臂的固定开发集结果仍在服务器（不要求重新下载）。若此前被删，需用原三臂各自的标准测试命令补齐 **got10k_vdrm_dev**，不能用 UAV 结果代替。分析会在计算前检查四臂是否都有 152 条完整、有限值、正宽高的预测，拒绝缺失或短序列，不允许悄悄补齐成绩；旧 flat 路径仅兼容读取，有冲突则报错。

配对分析（可混用两个 tracker，但参数名各不相同）：

```bash
python tracking/analyze_vdrm_module1_dev.py \
  --tracker_names ostrack ostrack ostrack ostrack_rscore \
  --tracker_params \
    vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
    vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
    vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300 \
    vitb_256_mae_ce_vdrm_rscore_probe_s42_32x4_ep300 \
  --reference vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  --report_name vdrm_rscore_dev \
  --visibility_stages \
  --per_sequence
```

输出在 `output/test/result_plots/vdrm_rscore_dev/`（标准 local.py 路径）。新增下载 `eval_data.pkl` 和 `visibility_stages.json` 即可。必须显示 **152/152**；不要用 `--dataset all`，此次不动四个 UAV 数据集/LaSOT。

第二关预登记筛选条件：相对 Tclean，部分遮挡与重现的配对平均 IoU 差均为正；正常帧 AO 下降不超过 **0.2 pp**（本轮新指定的小容差，结果生成前固定，不冒充既有阈值）；严重部分遮挡单独报告。配对恢复率不能下降，双方均恢复事件的恢复时延不能增加；零事件或置信区间不足以判断时标注证据不足。人工确认不得新增相似实例身份切换。AO/AUC、阶段帧数、序列级配对 bootstrap CI 同时报出，单种子开发筛选不声称五种子显著性。

如果通过，再设计一次前向的“响应专用残差”训练臂：稳定回归、仅响应路径学习残差；具体单流结构和梯度隔离另做审计。若仍错身份，下一设计才考虑最终候选实例排序监督。只有残差门槛通过，才另开 q_vis/q_id 离线校准闭环，Qonly−M 通过后才考虑 RM/Jnew。

## 自动安全检查

严格核对 checkpoint 来源、配置哈希、视觉权重与 Tclean 一致；只有源配置参与冻结合同校验，禁止参数覆盖。alpha=0 时五项视觉输出及解码框必须逐元素等于同裁剪 Tclean；Hann 峰不变时框必须等于 Tclean；逐帧实际输入自回放必须精确一致；探针分析不得推进 tracker 状态。全部权重和缓冲区前后哈希不变，梯度为空、eval 模式、零优化步骤。完整前缀图像在加载模型前预检，异常恢复 alpha/hook/forward 捕获器。独立包检查按存储顺序一次读取、校验每个成员、输入哈希及轨迹闭环，不解压任意路径。

## 本地验证记录

2026-10-07：157 项完整回归测试通过，包括文档四条可复制命令解析、无种子命令行参数、冻结来源、两模型加载无需 Rfreeze、禁止训练、缺失历史图像预检、拒绝覆盖、异常恢复、组件来源、raw pred_boxes 一致性、α=0、同峰框相等、包成员单次读取、输入哈希、浮点闭环、实际正式 tracker/结果 writer，以及四臂混合 tracker 的完整 152 条合成结果读取、分析 cache 身份和阶段报告。

RTX 5070 上的原生 ViT-B/CE/CENTER 与原生 Preprocessor 合成验证完成 1,996 个跟踪步骤、179 个诊断帧；全部精确回放、同峰框相等、α=0 视觉输出/框相等及完整参数/缓冲区哈希检查通过，独立包复检通过。额外使用正式 `ostrack_rscore` 参数与 tracker 入口加载两份权重，100 帧轨迹与审计入口逐帧精确一致。没有优化步骤。上述为合成模型/数据验证，不是服务器实际训练权重上的新成绩；真实效果仍须运行第一关后审阅。
