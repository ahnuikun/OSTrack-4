# VDRM 模块一：冻结视觉与判别路由开发闭环

本轮只研究模块一残差，不训练新的完整 J，不加入 M2，不把空间路由判别器称为独立可靠度头。

## 设计依据与边界

已有七臂开发结果与关键帧说明：关闭全部残差会损失有效恢复能力；直接抹平部件可靠度、抹平路由或强制 20% 残差裁剪并不稳定。因此保留四部件、部件差异和 V8 有界残差，不直接删除整个残差模块。

本次增加两个严格配对的训练臂，二者都加载同一个 Tclean epoch-300 checkpoint：

| 臂 | 视觉 backbone / box head | 残差 | 新判别路由 | 监督 |
| --- | --- | --- | --- | --- |
| Tclean | 已完成训练的原权重 | alpha=0 | 无 | 本轮不重训 |
| Rfreeze | 精确复制 Tclean，参数和运行状态冻结 | V8，重新初始化，仅 5 个标量可训练 | 无 | 跟踪损失 |
| Rdisc | 与 Rfreeze 相同 | 同 Rfreeze | 29,121 个新增参数 | 跟踪损失 + 0.5 路由损失 |

- `Rfreeze - Tclean`：不改变视觉权重时的纯残差收益。
- `Rdisc - Rfreeze`：加入判别路由和对应监督这一完整处理的收益，不能进一步区分结构与监督各自的贡献。
- `Rdisc - Tclean`：候选模块一的净收益。

不是只设置 `requires_grad=False`：主干和预测头保持 eval，冻结 BN buffers、dropout 和 DropPath；CE 使用 Tclean 推理时的固定比例。残差之后仍保留计算图，不能用 `no_grad` 截断跟踪损失对残差的反传。

两个训练臂采用相同样本配置、增强、四卡每卡 batch=32、300 epochs 和学习率。新判别头初始化不改变主随机流，零初始化输出使初始路由与原 V8 相同。启动器已有默认种子 42，配置同时校验它；以下操作命令不需要也不包含种子参数。

## 判别路由怎么工作

对每个模板部件和每个保留的搜索 token，读取 detach 后的视觉特征。共享投影到 32 维，输入投影差异、乘积、原余弦相似度、搜索 token 坐标及部件坐标，经过 64 维 MLP 输出路由修正。

修正加入 V8 原路由 logits；仍保留原部件可靠度和残差系数。推理没有 GT 候选位置输入，没有模板在线更新策略。

路由辅助监督使用 `original_logits.detach() + correction`，只更新新判别头，不更新视觉主干、原可靠度/路由标量或 alpha。跟踪损失仍可通过实际残差更新这些残差参数。

训练标签严格按 CE 前原搜索网格生成：

- 只有真实保留下来、中心位于对应 GT 部件内的 token 才是正例。
- GT 其他部件、边界半个 token 的邻域不作为该部件负例。
- padding、人工遮挡覆盖的正例 token 被忽略；缺少正例的部件不监督，不用最近 token 伪造正例。
- 优先监督真实保留的同类粘贴干扰物；没有可用粘贴 token 时，从真实背景挖掘 top-4 高响应负例，不虚构粘贴标签。
- 正负各半；负例中难负例与普通背景各半（两组都存在时）。

## 自动防错

正式训练前，每个进程在一个真实训练 minibatch 上运行无更新审计：

1. 校验源 checkpoint 的配置身份、epoch=300、网络类型、alpha=0，严格加载全部键。
2. alpha 临时置零时，检查预测框、响应、size、offset、backbone features 与 Tclean 逐元素完全一致。
3. 跟踪损失必须能更新 alpha；路由监督不得对视觉、原残差标量或 alpha 产生梯度。
4. 新判别头必须获得有效路由梯度；审计不调用优化器，恢复 RNG 和 alpha。

失败直接中止，训练启动器也返回非零状态，不再吞掉子进程错误。rank 0 保存审计 JSON。

每次保存/恢复检查冻结视觉权重和 BN buffers 的 SHA256。checkpoint 同时保存源文件 SHA256、配置 SHA256 和训练臂身份；测试严格校验配置及实际 epoch=300，不允许把其他臂、其他 epoch 文件改名后冒充。

本轮训练不建立验证 loader，`GOT10K_votval` 不被读取。YAML 保留原 VAL 字段只是为了兼容配置结构，不执行验证。

## 数据与判定协议

训练沿用 Tclean 的训练集组合，其中 GOT-10k 使用 `GOT10K_vottrain`。开发只使用固定 `got10k_vdrm_dev` 的 152 个训练目录来源序列；这些 ID 与 GOT-10k VOT train/val 清单均不重叠。它不是官方 GOT-10k test，也不使用 UAV 四数据集或 LaSOT 来挑方案。

正常帧容差此前未给出具体数值，本轮预注册为 **0.2 pp 的序列等权 AO**；这是开发容差，不是统计等价性的证明。若要改，必须在查看新训练结果前确定。

残差候选进入后续阶段需要同时满足：

- 全开发集 AUC / AO 相对 Tclean 不下降，并报告配对序列 bootstrap 区间。
- 部分遮挡和重新出现阶段 AO 相对 Tclean 为正；严重部分遮挡单独报告。
- 正常阶段 AO 相对 Tclean 不低于 -0.2 pp。
- 配对恢复率和时延没有明显恶化，人工核实的差异身份切换不增加。
- 单种子筛选不能作为五种子确认；bootstrap 序列区间也不能替代种子方差。

如果 Rfreeze 无净收益，不能靠把训练视觉重新放开来宣称通过。Rdisc 也必须单独通过净收益门槛。

离线路由检查使用相同的 Rfreeze 结果锚定搜索 crop，均匀选择每序列最多 4 个 `cover>=7, absence=0` 非初始化帧，两模型输入完全相同。GT 只在推理后生成评分标签，不进入模型。报告“GT 对应部件峰值超过背景峰值”的比例及 logit gap、有效部件/序列覆盖率、配对区间。

离线路由门槛：`Rdisc - Rfreeze` 排序正确率为正；报告其 95% 配对区间，若区间跨零只算趋势，不能称为显著改善。缺少正例的 CE 部件被剔除，必须同时查看覆盖率。这是 GT 区域与背景区分的代理指标，**不是 q_id 校准、相似目标身份准确率或身份切换计数**。

阶段分析按 GT cover 定义，初始化帧排除：正常=8、部分遮挡=1..7、严重部分遮挡=1..3（均要求 absence=0）。重新出现窗口为不可见结束后最多 10 个连续可见帧。恢复事件要求不可见至少 3 帧，随后至少可评价 5 帧；恢复为 30 帧内首次连续 5 帧 IoU>=0.5。未恢复保留在恢复率分母，恢复时延差只在双方都恢复的事件计算，有幸存者选择限制。cover 是粗粒度等级，不等同精确遮挡面积百分比。

不能自动从框 IoU 断言身份切换。差异身份仍需回看原图，重点复核已发现的 008669、004083、002126、001201、007878、007764 等序列。GT 区域低响应还可能是 crop 不含目标或 CE 已删目标，不能一概归因于路由。

## 服务器操作

先更新代码，保留已有 output 和数据目录，不重新训练 Tclean。

```bash
git fetch origin
git switch codex/vdrm-frozen-discriminative-route
git pull --ff-only
conda activate OSTrack
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300/OSTrack_ep0300.pth.tar \
  && echo "Tclean source checkpoint OK"
```

训练前提仍是原 Tclean 使用的 LASOT / GOT10K train / COCO17 / TRACKINGNET 数据可读。正式入口强制四进程。按下面顺序运行，前一步失败时不要继续下一步。两组不要同时使用同一组 GPU。

### Rfreeze

训练：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

检查：

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300/OSTrack_ep0300.pth.tar \
  && echo "Rfreeze checkpoint OK"

CUDA_VISIBLE_DEVICES=0 python tracking/check_vdrm_frozen.py \
  --config vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
  --save_dir ./output \
  --checkpoint output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300/OSTrack_ep0300.pth.tar
```

独立检查使用固定合成输入验证训练后连接和等价性，不能代替训练入口的真实 minibatch 审计。

测试：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
  --dataset got10k_vdrm_dev \
  --num_gpus 4 \
  --threads 4
```

分析（复用此前已完成的 Tclean 152 条开发结果）：

```css
python tracking/analyze_vdrm_module1_dev.py \
  --tracker_params \
    vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
    vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
  --reference vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  --report_name vdrm_frozen_dev_rfreeze \
  --visibility_stages \
  --per_sequence
```

### Rdisc

训练：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/train.py \
  --script ostrack \
  --config vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300 \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_lmdb 0 \
  --use_wandb 0
```

检查：

```bash
test -f output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300/OSTrack_ep0300.pth.tar \
  && echo "Rdisc checkpoint OK"

CUDA_VISIBLE_DEVICES=0 python tracking/check_vdrm_frozen.py \
  --config vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300 \
  --save_dir ./output \
  --checkpoint output/checkpoints/train/ostrack/vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300/OSTrack_ep0300.pth.tar
```

先离线检查路由（读取两组权重和 Rfreeze 已完成的开发结果，不训练）：

```css
CUDA_VISIBLE_DEVICES=0 python tracking/evaluate_vdrm_frozen_routes.py \
  --tracker_params \
    vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
    vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300 \
  --save_dir ./output
```

测试：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300 \
  --dataset got10k_vdrm_dev \
  --num_gpus 4 \
  --threads 4
```

分析（净收益与路由增量分别保存，均重新计算，不读过期 cache）：

```css
python tracking/analyze_vdrm_module1_dev.py \
  --tracker_params \
    vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
    vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
    vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300 \
  --reference vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  --report_name vdrm_frozen_dev \
  --visibility_stages \
  --per_sequence

python tracking/analyze_vdrm_module1_dev.py \
  --tracker_params \
    vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
    vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300 \
  --reference vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
  --report_name vdrm_frozen_dev_pair \
  --visibility_stages \
  --per_sequence
```

分析需要每一臂完整 152/152，缺失立即报错。若 Tclean 开发结果缺失，只补测 Tclean，不重训：

```lua
CUDA_VISIBLE_DEVICES=0,1,2,3 python tracking/test_uav_suite.py \
  --tracker_param vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  --dataset got10k_vdrm_dev \
  --num_gpus 4 \
  --threads 4
```

### 输出与回传

以下路径以服务器仓库根目录及默认环境配置为基准。测试读取服务器已有 `lib/test/evaluation/local.py`，不会覆写机器相关数据路径。

| 内容 | 默认路径 |
| --- | --- |
| 权重 | `output/checkpoints/train/ostrack/<配置名>/OSTrack_ep0300.pth.tar` |
| 实际 minibatch 审计 | `output/logs/<配置名>-frozen-preflight.json` |
| 训练后合成输入检查 | `output/analysis/frozen_vdrm/<配置名>/checkpoint_audit.json` |
| 新逐帧预测 | `output/test/tracking_results/ostrack/<配置名>/got10k/<序列>.txt` |
| 离线路由比较 | `output/analysis/frozen_vdrm/route_offline.json` |
| 三臂评估缓存 | `output/test/result_plots/vdrm_frozen_dev/eval_data.pkl` |
| 三臂遮挡/恢复 | `output/test/result_plots/vdrm_frozen_dev/visibility_stages.json` |
| 路由增量评估/阶段报告 | `output/test/result_plots/vdrm_frozen_dev_pair/` |

旧 flat GOT-10k 文件仅作读取兼容，新的 writer 只写 `got10k/`；如果同名 flat/nested 结果不一致，分析拒绝选择其中一份。旧 `vdrm_module1_dev` 报告保留。

完成后下载：两臂的 preflight / checkpoint_audit JSON、route_offline.json、两个最终报告目录、两臂逐帧预测目录。终端分析输出也请保存。暂时不用下载两份大权重。

## 实施检查记录

- 123 项单元/集成检查通过，覆盖旧 VDRM、原结果 writer/reader、严格源加载、参数/模式/BN 冻结、alpha=0 等价、梯度隔离、无伪造标签、checkpoint 保存/恢复及离线/阶段分析；Python 编译和 diff whitespace 检查通过。
- 本地单张 CUDA GPU 用实际 ViT-B、batch=32、合成数据验证了两臂反传、优化器、保存、严格重新加载和训练后 alpha=0 等价；不属于真实数据训练或性能结果。
- 本地四进程 CPU/Gloo 验证了两臂 DDP 优化及参数同步。没有本地四 GPU，不能声称四卡 NCCL 已实跑；服务器会先审计真实 batch 再开始更新。
- 命令参数与配置名称另有自动校验；机器数据可读性和真实源权重仍由服务器入口校验。

模块一通过后才开展独立 detach 的 `q_vis/q_id` 可靠度训练与校准；可靠度通过才比较 `Qonly - M`；最后才考虑 RM/Jnew。
