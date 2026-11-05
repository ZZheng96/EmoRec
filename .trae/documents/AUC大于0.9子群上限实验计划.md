# Context

用户希望先把历史指标高于 0.9 的个体单独成组，先优化该组，观察这条路线的真实上限。现有高分结果主要来自被试内留一 trial 评估，不能直接等价为跨被试泛化；其中实验095 adaptive 及166–177存在训练类别先验/反向输出退化，不能作为有效高分基线。因此本计划将“历史高分子群”与“新实验评估”严格解耦，并同时测量无标签基础能力与有标签适配上限。

# 推荐方案

## 1. 冻结历史子群名单

- 首先以实验095的固定 `linear_svc_raw_auc > 0.90` 生成并保存冻结名单，不使用095 adaptive结果。
- 名单来源固定为 `test/experiments/095_within_subject_model_direction/per_subject.csv`，同时记录筛选阈值、人数、HC/DEP构成和来源文件哈希。
- 另将“固定SVC窗口与F1均>0.9的稳定HC五人组（HC05、HC13、HC21、HC22、HC30）”作为更严格的敏感性子群；不把退化高分纳入名单。
- 新实验不得根据新OOF结果重新筛选或修改名单，避免选择偏差。

## 2. 新建独立实验目录

在 `test/experiments/204_high_auc_subgroup_upper_bound/` 中新增实验脚本、配置、指标、预测、输入元数据和中文实验报告，不修改历史实验目录。

## 3. 评估协议

复用实验032/057的 subject-wise 分层外层折与指标汇总：

- 训练/验证按被试隔离，HC/DEP只用于分层和报告，不作为模型输入。
- 使用已有 `plus_pe`、频谱/时序、graph 等缓存或预测，先建立 `k=0` 无标签基础条件。
- 同时报告全体、冻结高AUC子群、非高AUC子群、HC/DEP及逐被试结果。
- 指标固定报告 pooled AUC、F1、balanced accuracy、subject-macro AUC/F1、trial-level AUC/F1和每被试明细。
- 模型、阈值、融合方式先预注册；外层验证标签只用于最终评分。

## 4. 上限分层

至少运行以下条件：

1. `fixed_k0`：复用当前跨被试基础模型，不使用目标被试标签。
2. `subgroup_train_k0`：训练仅使用历史高AUC子群的训练折被试，验证仍只在该子群验证被试上评分；同时保留全体训练/子群验证对照，区分“专门训练”与“验证组容易”。
3. `adaptive_k0`：在不读取验证标签的前提下，比较预先固定的模型分支/融合，不复用095的退化adaptive结果。
4. `support_k1/k2/k4`：复用实验198的完整物理trial support/query隔离，测量有标签适配上限；明确标记为研究上限，不能作为公开无标签提交方案。

如第一阶段结果稳定，再进行固定融合（概率、Platt、logit）和轻量模型比较；不在同一OOF上反复挑选后宣称无偏提升。

## 5. 重点审计

- 逐折检查模型是否退化为训练类别比例或其反向输出。
- support/query必须按完整物理trial划分，同一trial的5个窗口不得跨集合。
- 训练集40窗口与公开集8窗口的窗口语义分开报告。
- 子群样本量小，必须同时给出每被试结果和重复间波动，不只看pooled AUC。
- 不把095的0.995 adaptive或166–177的完美分数作为有效上限。

# 关键文件与复用模块

- 历史筛选：[095 per_subject.csv](D:/BCIC2026/EmoRec/test/experiments/095_within_subject_model_direction/per_subject.csv)
- 跨被试折分与汇总：[032 run_experiment.py](D:/BCIC2026/EmoRec/test/experiments/032_repeated_subject_folds/run_experiment.py)
- 特征与主协议：[057 run_experiment.py](D:/BCIC2026/EmoRec/test/experiments/057_multiscale_permutation_entropy/run_experiment.py)
- support/query适配：[198 run_experiment.py](D:/BCIC2026/EmoRec/test/experiments/198_protocol_k_adaptation/run_experiment.py)
- 融合参考：[203 run_experiment.py](D:/BCIC2026/EmoRec/test/experiments/203_logit_fusion/run_experiment.py)
- 退化审计参考：[195 实验报告](D:/BCIC2026/EmoRec/test/experiments/195_HC31_inner_model_selection_audited/实验报告.md)

# 验证与交付

- 运行新实验时在终端打印阶段、fold、repeat和当前指标进度。
- 检查预测文件行数、subject/trial键、有限值和support/query无交集。
- 生成 `metrics.json`、`summary.csv`、逐被试结果、预测文件和 `实验报告.md`；报告明确当前AUC和F1，并区分跨被试结果与有标签适配上限。
- 只有当结果相对现有基线有稳定、协议合法的增益时，才把对应条件用于下一轮公开候选；否则保留为上限分析，不替换主候选。
