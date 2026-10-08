# Rqual：冻结实际候选框质量排序，先做 20 轮筛选

## 早期失效审阅结论

已独立复算服务器 early_failure 包的 298 帧，全部分析值与服务器一致；四臂完整轨迹、原五窗口指标、权重哈希与此前包一致，新增 119 帧，没有训练或改动用户权重。

- 008311 第 38 帧：四臂同时失去 GT 搜索覆盖，目标在相邻两帧横向移动约 279 像素，超过当时约 188 像素搜索边长。后续大框/候选 IoU 上限低是失效后的状态，不能作为“最初回归头坏了”的证据。候选排序不能恢复搜索区域以外的目标。
- 007878 第 15 帧：在 Rresp 相同裁剪上，Tclean 候选 IoU 0.8193 被新头选成 0.4694。后来曾恢复，不能把该帧直接说成不可逆漂移；第 64–66 帧目标不可见、四臂均失效，随后搜索/状态反馈不同。此前把该序列 GT 说成整个人不准确，GT 是小目标（球），分析按 GT 框而非整个人。
- 003240 第 93 帧：相同裁剪存在 IoU 0.9205 的正确候选，Rresp 仍选择 IoU 0 的错误实例；改为最终能量注入或只去均值都未修复。这不是只调 alpha/幅度就能解决的问题。
- 001664 的正例确实保留，但第 75 帧直接把旧修正改为能量注入会由 IoU 0.9656 变为 0。因此不发布“旧权重换公式”的补丁，重新训练独立小头。

这些是已使用开发案例的机制证据，不是无偏泛化证据，也没有自动确认身份切换或宣称统计显著。

## 本轮改什么，不改什么

**新训练臂只有 Rqual，20 epochs；不是 Qonly 可靠度模型，也不接 M2/J。**

1. 源模型仍为已有 Tclean ep300，严格复制并冻结主干、CENTER、旧 VDRM 所有权重与缓冲区；alpha 精确固定为零。旧部件仅留作权重兼容，不训练旧路由/可见度/残差分支，不继续训练 Rresp。
2. 用冻结 CENTER 的 size/offset 解码全部 256 个实际候选框。每个框在冻结最终搜索特征上做 3×3 ROI 采样，初始化模板框也做 ROI 采样，再学习模板—候选匹配的小 MLP。输入含匹配乘积、差值、余弦、原响应、框尺度/offset、保留特征支持度；新头 **34,977 参数**。
3. ROI 采样无反向，学习投影在采样后，避免 CUDA grid_sample 反向引入视觉梯度。一个中心格点被 CE 删除不等于整个候选框无特征：有效性由框内 ROI 支持决定，而不是中心格点是否存活。
4. 预测质量 logit `q`，在有效候选中去均值后生成 `delta = 2*tanh((q-mean(q))/2)`；无支持候选 delta=0。最终 `score = base_score * exp(delta-max(delta))`，随后使用原生 Hann 和 CENTER 解码。公共缩放不改变候选排序；不再经过“base logit 加修正再 sigmoid”。零修正/关闭头精确还原 Tclean，size/offset/视觉特征不变。
5. 标签是实际候选与搜索 GT 的 **归一化裁剪坐标 IoU**，不是格点是否落入 GT；未使用源图 clip 后 IoU 当训练标签。高/中/低 IoU 分组平衡的 soft-IoU BCE；好候选 ≥0.5 对坏候选 ≤0.1 的 Hann 后 top-4 排序；基准已选好框时保护它不被差 0.05 以上的候选替代。无好候选时只保留质量监督和小幅度正则，不伪造正样本。
6. GT 无有效像素支持（越界、padding 或人工完全遮挡）时跳过该样本监督。自然遮挡程度没有通过该规则自动识别；不能声称本轮训练已解决不可见性标签问题。GT 只进入 loss/离线分析，不进入当前帧模型前向。
7. 损失权重质量/排序/保护/无好框正则为 1/0.5/0.5/0.1；学习率 4e-4，第 16 轮乘 0.1；FP32、每卡 32、4 卡。训练数据/增强沿用原方案，GOT 训练入口排除固定 152 开发序列，不构建验证 loader。

质量 sigmoid 是训练用排序信号，**尚未校准，不称为 q_vis/q_id**。统一缩放后的最大响应也不是可靠度，不能直接用于 M2 阈值。

## 先跑以下命令

在服务器项目根目录、OSTrack 环境执行。分支为 `codex/vdrm-rqual-candidate-ranking`。

```bash
git fetch origin
git switch codex/vdrm-rqual-candidate-ranking
git pull --ff-only origin codex/vdrm-rqual-candidate-ranking
```

不删除/覆盖旧模型或下载包。训练需要 Tclean ep300；后续四臂五案例诊断还需要 Rfreeze/Rdisc ep300。不需要重新训练这三个旧模型。

```bash
for cfg in \
  vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
  vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300
do
  test -f "output/checkpoints/train/ostrack/$cfg/OSTrack_ep0300.pth.tar" \
    || { echo "Missing checkpoint: $cfg"; exit 1; }
done
test -f data/got10k/train/list.txt || { echo "GOT training data missing"; exit 1; }
```

### Rqual

训练：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

启动器默认 seed=42，配置和实际种子会校验，命令不加种子参数。每次开始/续训前，用实际训练 minibatch 审计梯度边界，失败直接退出。保护/无好框损失可能合法为零；若该批没有好坏配对，排序项记为不活跃而不是伪造样本。报告明确记录各类样本数；独立合成测试覆盖活跃排序梯度方向。训练日志保留这些支持统计以检查训练信号是否真的出现。

检查：

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020/OSTrack_ep0020.pth.tar \
  && echo "Rqual checkpoint OK"
```

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/check_vdrm_response.py \
  --config vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020 \
  --save_dir ./output \
  --checkpoint output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020/OSTrack_ep0020.pth.tar
```

应出现 `RQUAL CHECKPOINT CHECK PASSED`。这是无优化步骤的合成接线检查，不是性能通过。

测试（先四臂五案例，保存早期失效段）：

```lua
CUDA_VISIBLE_DEVICES=0 python tracking/diagnose_vdrm_response.py \
  --save_dir ./output \
  --candidate Rqual \
  --profile early_failure \
  --output output/analysis/frozen_quality/rqual_early_probe.tar.gz
```

分析：

```css
python tracking/analyze_vdrm_response.py \
  --bundle output/analysis/frozen_quality/rqual_early_probe.tar.gz \
  --output output/analysis/frozen_quality/rqual_early_analysis.tar.gz
```

测试应为 **1,996 native steps、298 records**；分析应为 **298 records**。四臂是 Tclean/Rfreeze/Rdisc/Rqual，各自从第一帧按原生浮点状态闭环运行。当前 GT 只在前向完成后用于分析。两个脚本都拒绝覆盖现有包；需重跑时为 `--output` 换新名字。诊断不写标准 TXT，不覆盖旧 result_plots。

本轮只下载以下新增文件，旧本地包不用重复下载：

- `output/analysis/frozen_quality/rqual_early_probe.tar.gz`
- `output/analysis/frozen_quality/rqual_early_analysis.tar.gz`
- `output/analysis/frozen_response/vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020/checkpoint_audit.json`（检查入口沿用 frozen_response 目录）
- `output/logs/vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020-frozen-preflight.json`
- `output/logs/ostrack-vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020.log`

## 先验筛选门槛，不在结果出来后更换

已使用案例只作失败筛选：003240 原窗口 AO 不低于 Tclean 超过 0.5 pp，且人工核实第 93 帧不选错误实例；001664 部分遮挡窗口 AO 至少 60%，保住原有约 65% 正例的大部分；008333 部分遮挡和重现窗口均不低于 Tclean 超过 0.5 pp；007878 第 15 帧同裁剪下降不超过 5 pp、正常帧窗口不低于 Tclean 超过 0.5 pp，原重现事件恢复时延不超过 Tclean 的 8 帧。008311 不能要求纯排序找回裁剪外目标，但要报告早期失效是否提前以及后续恢复，不能把共同失效记作新头收益。

任一明显失败，停止扩大，不自动续到 300 轮、不接 M2、不调阈值掩盖问题。若 20 轮尚未学到有效排序，只能判“本配方筛选未通过”，不能据此宣称整个候选质量思路已被否定。

### 五案例审阅通过后，才跑固定 152 开发序列

现在不运行下面三条。Tclean 已有完整结果则复用；被删除才补测，绝不换成 VisDrone train 或 UAV 验证集。标准结果统一为 `<配置>/got10k/<序列>.txt`；分析会检查每臂 152 条、帧数和嵌套/平铺冲突。

测试 Tclean（需要补齐时）：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  --dataset got10k_vdrm_dev \
  --num_gpus 4 \
  --threads 4
```

测试 Rqual：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020 \
  --dataset got10k_vdrm_dev \
  --num_gpus 4 \
  --threads 4
```

分析：

```css
python tracking/analyze_vdrm_module1_dev.py \
  --tracker_params \
    vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
    vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020 \
  --reference vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  --report_name vdrm_rqual_dev \
  --visibility_stages \
  --per_sequence
```

输出 `152/152`。Rqual 标准结果在 `output/test/tracking_results/ostrack/vitb_256_mae_ce_vdrm_rqual_s42_32x4_ep020/got10k/`；配对结果在 `output/test/result_plots/vdrm_rqual_dev/eval_data.pkl` 和 `visibility_stages.json`。

完整开发门槛沿用：部分遮挡、严重部分遮挡、重新出现 AO 增量为正；正常帧 AO 下降 ≤0.5 pp；不增加人工确认的差异身份切换；配对恢复率不下降、共同恢复事件的时延不变长。报告序列配对 bootstrap 区间；样本不足写未证明。单种子筛选后才考虑五种子确认和独立可靠度阶段。

## 本地验证记录

完整 `test_vdrm*.py` 回归 **191 项通过（445.008 秒）**，覆盖新头数学/标签/梯度隔离、旧配置哈希、20 轮正式保存/续训、四卡与源权重预检、命令解析/默认种子、正式 tracker、规范结果 writer、诊断与独立分析、旧开发流程；日志 `output/analysis/vdrm_rqual_regression.log`。

2026-10-08，RTX 5070、PyTorch 2.7.0+cu128：原生 ViT-B/CE/CENTER，**每卡 batch=32** 的合成前向/损失/反向与 1 次合成优化更新通过，全部 Tclean 参数和缓冲区哈希不变，新头有更新。该单卡合成过程峰值 allocated 显存约 1.66 GiB，不代表实际四卡训练总占用。

审计中质量/排序/保护损失的新头梯度 L2 分别为 0.25459/0.01717/0.03212，视觉基准与 alpha 梯度张量均为 0；32 个样本可监督、32 个可排序、24 个满足基准保护条件。无好框正则此批为 0，符合预期。

合成闭环完成 1,996 原生跟踪步骤、298 诊断记录；独立候选解码最大误差 0.0000153 像素。007878/008311 两条完整前缀与正式 tracker 浮点框精确一致，规范 `<配置>/got10k/` writer 通过。诊断本身 0 次优化；唯一 1 次更新发生在临时随机模型和合成图像上，没有训练/更新用户模型。报告：`output/analysis/vdrm_candidate_quality_gpu_smoke.json`。

真实 early_failure 包的 298 帧在共用分析代码改动后再次独立复算，与旧服务器分析所有值一致，旧完整轨迹/窗口/权重哈希仍一致。报告：`output/analysis/vdrm_rresp_early_review_20261007/checks.json`。

Rqual 配置 SHA256：`9b3c44335100b798fdbe7b2b724772e08d0da232545d47c73f016a3c974890cc`。Tclean/Rfreeze/Rdisc/Rresp 配置哈希兼容性纳入回归测试。

合成测试不等于服务器真实数据性能通过。尚未进行四卡正式训练，也未在本地复现服务器 PyTorch 2.3.1 环境；服务器训练入口仍会审计真实 minibatch，失败即停止。
