# VDRM 模块一：固定开发集与同权重因果筛查

本阶段**不训练**，三份新配置都严格读取已有 `vitb_256_mae_ce_vdrm_v8_ronly_s42_32x4_ep300` 的 epoch-300 checkpoint。训练入口会拒绝这些 `inference_only` 配置。

## 已有证据与边界

- Ronly 的有效 α 约为 −0.494；六条诊断序列的平均残差更新范数约为搜索 token 范数的 0.30–0.36。
- 在六条预先挑选的困难序列里，原 Ronly 的 866/866 个失败帧的视觉可靠度均高于 0.7；四个部件可靠度的平均最大差值仅约 0.03。
- 将 α 置零后，六条里五条的平均 IoU 下降；`Surfing04` 从 0.428 降到 0.171。因此不能直接删除完整残差。
- 这些序列是按失败现象选择的，不能用它们单独选模型，也不能由此证明残差相对于 Tclean 有净收益。

## 固定开发集

`got10k_vdrm_dev` 是 GOT-10k 官方 train 目录中的 152 条固定序列，ID 为 `got10k_val_split.txt − got10k_vot_val_split.txt`。代码启动时检查它与 `got10k_vot_train_split.txt`、`got10k_vot_val_split.txt` 均不相交。这不是当前 Ronly 训练使用的 GOT-10k 子集，也不是它的 VOT-val 子集。四个 UAV benchmark 仅作为已发生问题的事后诊断；本轮结构选择依靠该固定开发集。

## 同权重实验臂

| 臂 | 相对 Ronly 唯一的推理改动 | 检验问题 |
| --- | --- | --- |
| Ronly | 无 | 基准 |
| α0 | 残差更新置零 | 整体残差在这组权重下是否提供即时收益 |
| qflat | 每帧把有效部件的 `q_i` 换成它们的均值 | 部件间可靠度差异是否有用；保留每帧平均强度 |
| routeflat | 每个部件的空间 route 换成其搜索 token 均值 | 空间选择性是否有用；保留每部件平均 route mass |
| clip20 | 残差更新范数上限为搜索 token 范数的 20% | 当前约 30% 的更新是否过大 |

三项消融都是**已训练模型的局部扰动**，不能直接当作重新训练后的性能结论。B0 和 Tclean 同时在开发集运行，才能判断候选是否真正超过干净训练基线。

## 判定规则（运行前固定）

1. 先确认七个模型在同一 152/152 开发集全部完成；比较官方 AUC、配对每序列 AO 差与 bootstrap 区间，并查看失败序列而非仅看总分。
2. `qflat` 不劣于 Ronly、且困难序列无新的持续失败时，下一训练版移除**部件可靠度参与残差加权**；`q_vis/q_id` 放到独立、detach 的可靠度任务。`qflat` 只检验部件间加权，不证明独立可靠度本身无用。
3. `routeflat` 明显变差时保留空间路由；若不变或变好，则当前 route 没有显示正贡献，下一训练版需用 GT 前景/同类干扰负样本监督独立路由头，监督特征先 detach，且不得继续把现有 route 称为有效部件。
4. `clip20` 同时改善开发集平均与尾部失败、并超过 Tclean 时，下一训练版从训练开始使用有界残差；不可将事后推理裁剪当成最终模型。
5. 如果所有候选仍低于 Tclean，不再组合 M2/可靠度，也不继续调 α；转向“可见部件的目标-干扰判别路由 + 有界残差”新结构。先验证路由对真实目标与同类干扰物的区分，再做单种子残差训练。

## 服务器顺序

先测试 B0、Tclean、Ronly、α0、qflat、routeflat、clip20 的 `got10k_vdrm_dev`；全部测试命令只调用已有 checkpoint，不调用 `tracking/train.py`。然后一次运行 `tracking/analyze_vdrm_module1_dev.py`，它会从七臂结果 TXT 计算 152/152 的对照表、逐序列 AO 和配对 bootstrap 区间。缓存写入 `output/test/result_plots/vdrm_module1_dev/eval_data.pkl`，可用 `tee` 另存终端报告。

测试前确认 GOT-10k train 目录在服务器的 `data/got10k/train`，以及 B0、Tclean、Ronly 的 epoch-300 checkpoint 均存在。
