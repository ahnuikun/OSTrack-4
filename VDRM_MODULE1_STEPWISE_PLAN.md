# 模块一逐步验证计划

日期：2026-10-09。原则：每次只执行当前一步；收到结果后，再确定下一步代码并发布独立新分支。后续分支名为预定名称，尚未创建，不能提前切换。每步都保留失败结果，不通过不直接进入下一轮完整训练。

## 计划表

| 步骤 | 要回答的问题 | 代码/实验内容 | 判定与下一步 | 服务器资源 | 分支与状态 |
| --- | --- | --- | --- | --- | --- |
| 01 原模块证据采集 | 实际是哪类失败？原型、q、route和残差是什么状态？ | 既有Tclean/Rfreeze/Rdisc；原生浮点前缀、自回放、alpha0同输入检查；保存部件/匹配/CE/输入/输出；CPU候选解码和失败分型 | 工程校验通过只表示证据可信；人工核验身份、定位首次分叉，给02确定样本与标签 | 单卡，无训练；1497原生步骤，499保存帧 | `codex/vdrm-module1-step01-evidence`：本轮实现 |
| 02 部件表示验证 | 四格原型能否区分目标与同类实例？ | 优先复用01；补人工实例框；原型相似度、token支持、目标/干扰间隔；整体/四格/token集合对照；明确哪些对照需额外前向 | 原型身份分离无能力则先修表示；有能力则进入03检查使用方式 | CPU为主；仅缺少的输入对照用单卡，无完整训练 | `codex/vdrm-module1-step02-parts`：待01审查后实现 |
| 03 q与route联合验证 | q、空间路由各有什么作用，是否相互补偿？ | 原q/均值q × 原route/空间均值route；alpha0；量/范数对照；同输入与原生闭环分别记录；Ronly另立模型合同，不混解释 | 分清部件信息、全局强度与交互；不能用推理干预代替训练消融 | 单卡，无训练；帧数由01确定 | `codex/vdrm-module1-step03-gates`：待02后实现 |
| 04 理想标签能力上限 | 即使q和route正确，原结构是否能改善？ | 真实/人工可见性oracle、目标/同类实例oracle route及联合；只作诊断；测候选间隔、定位、CE和残差方向 | oracle有效而学习头无效→修监督；联合oracle仍无效→修表示/残差；检查分布外干预限制 | 单卡，无参数更新 | `codex/vdrm-module1-step04-oracle`：待03后实现 |
| 05 针对断点的小预算训练 | 一个具体改动能否解决已定位问题？ | 只选一个主要断点；同训练域自然遮挡/同类负例；小预算训练；梯度、参数、数据合同；基线/A/B/A+B仅在检验两组件时安排 | 必须同时保正例与修负例；失败停止该配置，不能靠延长300轮宣布机制成立 | 先小预算，完整训练预算另行登记 | `codex/vdrm-module1-step05-<方向>`：待04后确定 |
| 06 独立闭环与正式消融 | 新模块是否有净收益且组件可归因？ | 序列隔离拟合/校准/审计；训练时组件消融；多个种子；原生闭环；遮挡、身份、重现、定位、尾部与成本 | 通过后再做正式评估；原152/guard192及已查看序列不再冒充独立审计 | 按05结果制定 | `codex/vdrm-module1-step06-validation`：待05后实现 |

每一步交付固定包括：设计文档、代码、必要测试、新GitHub分支和提交号、真实可解析的服务器命令、预期输出、下载清单、失败处置。未实现的步骤不预先提供虚构命令。后续以用户返回的新包为输入推进，不重复索取已下载旧包。

## 第01步的范围

只读取三个既有epoch300权重和固定GOT训练开发集五案例。没有优化器、反传、训练、当前帧GT驱动裁剪或选峰。初始化GT按标准SOT协议使用；其后GT只评价已经生成的输出。

三个模型各自从第一帧运行，使用各自真实浮点预测。每步精确输入自回放；保存窗口内Rdisc裁剪的alpha0视觉输出/CE必须逐元素等于同输入Tclean。不是重用整数TXT。

| 序列 | 保存帧，1-based | 用途 |
| --- | --- | --- |
| 003240 | 2–100 | 相似动物身份错误及更早分歧 |
| 008333 | 2–74 | 遮挡收益与重现损失的完整前缀 |
| 001664 | 2–100 | 残差正例、球员区分及更早分歧 |
| 007878 | 2–130 | 早期分歧、遮挡与恢复；GT是球 |
| 008311 | 2–100 | 小目标、尺度与搜索支持 |

保存499个时点、1497份模型内部证据、1497个原生跟踪步骤。每步有一次自回放，另有499×2个Rdisc同裁剪旁路检查，总计3992次网络前向。五案例沿用已使用开发案例，仅用于机制诊断。所有案例均从第2帧保存，防止首次损害发生在旧选择窗口之前再次缺少内部图。

保存：模板/搜索插入处token、四格mask与原型、余弦匹配、q及其标定参数、route/logit、真实注入delta、保留索引、最终CE掩码、实际标准化输入、padding、原生响应/size/offset/Hann、Rdisc同裁剪Tclean的响应/size/offset、原图、完整前缀框/GT/cover/absence、模型来源哈希。保存输入和token为float32，包可能达数百MB至数GB，建议预留至少10GB磁盘及16GB空闲主存；实际大小随压缩率变化。

CPU分析重新验证哈希、浮点状态连接、原型/余弦/q/route/残差公式、原生候选解码和完整前缀IoU，报告首次框分歧与Rdisc同裁剪直接修复/损害；分型为不可见/无效、裁剪外、无好候选、有好候选未选、选到好候选。最后两类不能自动解释成身份事件。

## 服务器操作（Linux/Bash，项目根目录，已有OSTrack环境）

### 1. 更新分支

先确认服务器工作区没有未提交改动；若有，保留它们，不执行reset/clean。

```bash
git status --short
git fetch origin
git switch codex/vdrm-module1-step01-evidence
git pull --ff-only origin codex/vdrm-module1-step01-evidence
git log -1 --oneline
```

首次切换时git switch会跟踪同名origin分支。已经在该分支时仍可执行。不要切到计划表中尚未发布的后续分支。

### 2. 权重检查

```bash
for cfg in \
  vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300 \
  vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300 \
  vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300
do
  test -f "output/checkpoints/train/ostrack/$cfg/OSTrack_ep0300.pth.tar" \
    || { echo "Missing checkpoint: $cfg"; exit 1; }
done
echo "Three checkpoints OK"
```

沿用既有服务器GOT数据与local.py配置，不改数据路径。脚本加载GPU模型前会检查152开发集身份、五案例全部所需图像前缀和标签。来源合同不匹配时直接报错，不绕过校验。

### 3. 运行诊断

```bash
CUDA_VISIBLE_DEVICES=0 python tracking/diagnose_vdrm_module1_evidence.py \
  --save_dir ./output
```

预期预检与完成信息：

```text
PRECHECK PASSED: 5 cases, 1497 native steps, 499 saved frames
STEP01 EVIDENCE CHECK PASSED: 1497 native steps, 499 records; NOT a performance pass
```

过程每25帧打印进度。该代码单卡串行运行，不需要四卡DDP，不接受--seed，不运行tracking/train.py。不会写标准tracking_results/result_plots。诊断包存在时拒绝覆盖。

### 4. CPU独立分析

```bash
python tracking/analyze_vdrm_module1_evidence.py \
  --bundle output/analysis/module1_step01/module1_evidence.tar.gz \
  --output output/analysis/module1_step01/module1_analysis.tar.gz
```

完成标记：

```text
STEP01 ANALYSIS COMPLETE; no automatic identity/performance verdict
```

该入口不加载数据集/权重，不要求CUDA，读取大数组时逐成员校验；产物有summary.json、per_frame.json、report.md。它是独立CPU复算，不是只读取保存汇总。

### 5. 可选独立复检

```bash
python tracking/diagnose_vdrm_module1_evidence.py \
  --check_bundle output/analysis/module1_step01/module1_evidence.tar.gz
```

### 6. 下载与下一步

只下载两个新文件：

1. `output/analysis/module1_step01/module1_evidence.tar.gz`
2. `output/analysis/module1_step01/module1_analysis.tar.gz`

将两包提供给当前聊天。若失败，提供完整报错与失败前进度；没有完成标记的包不用于设计决策。不要因此重训旧模型。

下一轮先核对模型来源、轨迹、数组重算和原图，再输出各案例首次分歧与状态分型，确定02所需的人工实例框/部件证据及最小补采。第01步没有性能通过门槛；其目的在于让后续判断可复核。

若确需重复，诊断用--output指定新包，分析--bundle指向新包、--output也使用新名称；不覆盖旧包。没有自动删除或搬移操作。

## 本地验证记录

发布前对应CPU协议/完整性/候选解码及既有native_float回归共32项测试通过。CPU完整打包fixture覆盖499时点、1497份模型记录。RTX5070上的原生ViT-B/CE/CENTER CUDA短前缀验证通过：两组标签共12个原生步骤、12份内部记录，非零Rdisc路由修正，采集轨迹与无钩子原生轨迹精确一致，初始化后的GT/可见性标签变化不改变预测，模型状态哈希不变，独立解码最大误差0。该GPU验证是合成短前缀，不是完整五案例真实服务器验证。

设计依据：VDRM_MODULE1_DESIGN_REASSESSMENT_20261009.md。使用workflow-engineer技能制定里程碑和验证顺序，使用pr-writer技能整理提交说明。
