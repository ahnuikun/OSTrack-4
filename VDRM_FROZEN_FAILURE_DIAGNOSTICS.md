# 冻结残差失败案例诊断：先定位，不重新训练

当前固定 GOT-10k **训练集开发划分**的 Rdisc−Tclean：全集 AUC +0.500 pp，严重部分遮挡 AO +3.292 pp，但重新出现 AO −0.334 pp，恢复事件仍为 3/9。离线空间路由排序变好不等于残差闭环已经通过，更不是 q_vis/q_id 的可靠度校准通过。本步骤不访问正式验证/测试集，不启动 Rnext、RM 或 Jnew。

## 服务器操作

先更新到 `codex/vdrm-frozen-failure-diagnostics`。三个 epoch-300 权重、三组固定开发集预测框、GOT 训练数据仍放原处，不需要移动。默认读取测试环境配置的 `results_path/ostrack/<配置名>/got10k/<序列>.txt`；旧平铺结果兼容读取，若两份内容冲突直接终止。**不写任何 tracking_results。**

诊断（单卡足够，无种子命令行参数）：

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/diagnose_vdrm_frozen_failures.py \
  --save_dir ./output \
  --device cuda
```

程序先检查三组权重文件、五条完整预测/GT/可见度标签及所选图像，再加载模型并校验源权重哈希、视觉参数哈希、配置哈希、网络身份和 epoch。成功末尾出现 `DIAGNOSTIC BUNDLE CHECK PASSED`。

独立检查（只读压缩包，不需要图像、权重或 GPU）：

```bash
python tracking/diagnose_vdrm_frozen_failures.py \
  --check_bundle output/analysis/frozen_vdrm/failure_diagnostics.tar.gz
```

下载 **一个文件**：

```text
output/analysis/frozen_vdrm/failure_diagnostics.tar.gz
```

需要重复运行时用新文件名，已有包不会被覆盖：

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/diagnose_vdrm_frozen_failures.py \
  --save_dir ./output \
  --device cuda \
  --output output/analysis/frozen_vdrm/failure_diagnostics_repeat.tar.gz
```

只有确实自定义过结果存储根目录才加 `--results_root <各配置目录的父目录>`；它不是数据集目录，也不是某个配置的 got10k 子目录。

## 诊断设计与解释边界

固定五条序列：008333（重新出现损失）、003240（相对 Tclean 的损失）、001664（Rfreeze 收益在 Rdisc 中消失）、007878 / 008311（正收益对照）。窗口自动由完整预测轨迹与 GT 的差异选出，保留完整轨迹，不删除失败序列，不改官方分数。仅选取初始化之后的帧。通常各序列约 11–30 帧，实际取决于遮挡/恢复窗口。

每帧依次使用 Tclean、Rfreeze、Rdisc **各自保存的上一帧预测框**作为搜索锚点。在每个锚点上，三个模型读取完全相同的模板/搜索张量，再对 Rdisc 做两个只读反事实：

- `Rdisc_route0`：仅将新空间路由头的输出修正置零，保留 Rdisc 已学习的原残差标量，不等同于 Rfreeze。
- `Rdisc_alpha0`：临时令 alpha=0。每个搜索裁剪上，五项原生输出必须逐元素等于 Tclean，否则终止。

临时 hook 和 alpha 在异常时也恢复；全程无优化器、反传或参数更新。开始/结束的完整模型状态哈希必须相同。解码复用原生 Hann 窗、CENTER 框头、坐标映射和 clip_box。当前帧 GT 只生成事后标签/诊断，不输入模型、不给搜索裁剪定位。初始化帧 GT 是标准单目标跟踪协议输入。

预测 TXT 以整数保存，而原跟踪内部使用浮点状态，故这是**开放环整数锚点诊断**，不是原闭环的逐位复现。包中记录 replay 对原预测框的差异；不能把这些反事实 IoU 当作新的测试成绩。

包中有：完整五条 GT/cover/absence 与三臂预测/IoU、自动窗口、原图与初始化图、三个锚点裁剪、原生响应/框、四部件 q_i、门控饱和、路由 GT/背景排名、原始网格 / 残差插入处 / 最后一次 CE 后的真实 GT 单元数、残差注入范数/能量图、两种反事实、模型来源及各成员 SHA256。q_i 是现有门控信号，不是经校准的新可靠度。

## 收到诊断后如何改模块一

| 同一裁剪上的证据 | 后续实现方向（尚未授权训练新臂） |
| --- | --- |
| GT 大部分在裁剪外，换 Tclean 锚点恢复 | 首先处理搜索状态漂移；不要把它误诊为当前帧路由分类差 |
| GT 在裁剪内，但原网格有 GT 单元、CE 保留后为零 | 检查残差插入位置 / CE 保护；任何保护不能用推理时 GT |
| GT 单元保留，关闭新路由修正后变好，路由峰在背景/门控饱和 | 缩减或约束路由修正，检查位置特征与负样本覆盖 |
| 关闭路由修正无效，alpha=0 恢复，注入能量偏大或集中错区 | 优先约束残差幅度/支持区域，删除造成错误注入的部件 |
| 图像显示相似实例且新路由在错误实例更强 | 人工确认身份事件，再设计实例判别监督；仅低 IoU 不算身份切换 |

改动必须同时复核 007878 / 008311 的正收益，不能以修复负例为由消掉已有效的恢复能力；设计冻结后再训练一个可归因的新残差臂。可靠度闭环仍独立进行，两个闭环未分别过门槛前不组合 Jnew。

## 本地验证

新增单元 / 端到端测试覆盖文档命令解析（不含 --seed）、整数初始化、裁剪边缘视觉支持、窗口索引、缺失图像、结果路径兼容/冲突、伪造源权重、异常恢复、完整打包与校验损坏。原生 ViT-B/CE/CENTER 的 CUDA 合成 fixture 已完成 61 个选定帧、183 个三锚点裁剪、915 次只读前向；每个裁剪 alpha=0 五项输出完全等于 Tclean，三个模型完整状态哈希不变，最终包和独立复检均通过。该验证不代表真实服务器数据已运行，也没有训练步骤。
