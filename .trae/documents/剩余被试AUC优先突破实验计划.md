# 剩余被试 AUC 优先突破实验计划

## Summary

针对实验204冻结高AUC子群之外的被试，先以严格被试内物理trial隔离的OOF为开发评估，优先提升每个被试的AUC，F1同步记录但不作为首轮停止条件。首轮采用小批次验证：

- 有CSP或多特征提升证据：`DEP07、DEP09、DEP16、HC10、HC29`
- 稳定反向/困难对照：`DEP03、DEP17`

首轮结果只解释为被试内个体化上限或诊断结果，不宣称公开测试集无标签泛化成绩；quota=4只作辅助指标，不作为主要达标依据。

## Current State Analysis

1. 实验204的057 plus-PE与006 graph固定融合在非冻结子群仅达到窗口AUC `0.759703`、F1 `0.690289`，说明剩余个体是主要瓶颈。
2. 实验095显示大量个体存在方向差异，但adaptive结果因模型/方向选择及先验退化不可信；本计划不直接复用adaptive高分。
3. 实验060显示plus-PE对个体存在异质性：DEP07、DEP16附近有继续验证价值，DEP09存在退化风险；因此首轮需要保留基线对照，不能假设PE必然有效。
4. 实验122/123/141已经提供严格的pair-OoF协议：每次留出一个正类和一个负类物理trial，模型/方向只能由训练trial内OOF选择，外层query标签不得参与选择。
5. 每名被试只有8个物理trial、40个窗口，窗口级指标和trial级指标均需报告；同一物理trial的5个窗口不得跨训练/验证集合。
6. 数据集说明确认训练被试每人4个中性和4个积极视频，训练窗口由每个视频切成5个10秒段；因此trial级隔离和`subject + original_trial_id + y_true`对齐检查是必要的。

## Proposed Changes

### 1. 创建独立实验目录

新建：

`D:\BCIC2026\EmoRec\test\experiments\205_remaining_subject_auc_breakthrough\`

目录包含：

- `run_experiment.py`
- `config.json`
- `实验报告.md`
- `metrics.json`
- `summary.csv`
- `predictions/`
- `input_metadata/`
- `selections/`

不覆盖实验122、123、141或057的既有产物。

### 2. 参数化严格trial-level pair-OoF脚本

以[122实验脚本](file:///D:/BCIC2026/EmoRec/test/experiments/122_DEP02_window_regularized_csp_pairs/run_experiment.py)的正则化FilterBank CSP为主模板，以[141实验脚本](file:///D:/BCIC2026/EmoRec/src/141_HC16_csp_transfer_pairs/run_experiment.py)的缓存对齐、基线和provenance为补充，改为按配置循环处理首轮7名被试。

每个被试执行：

1. 从`数据/训练集`加载对应MAT文件，验证30通道、40个窗口、8个物理trial。
2. 计算5个频带协方差特征。
3. 使用正负物理trial pair作为外层留出组合。
4. 在外层训练trial内部，对候选模型、正则化、组件数、频带块及raw/reverse方向进行OOF选择。
5. 外层query只用于生成预测，不参与任何选择。
6. 生成每个被试独立窗口/ trial预测、候选选择审计和来源哈希。

### 3. 首轮模型矩阵

为控制首轮范围，固定以下候选，不在外层标签上搜索：

- CSP shrinkage：`lambda ∈ {0.0, 0.1, 0.3}`
- CSP组件：`{2, 4}`
- 频带块：`all、band_2、band_3、band_4`
- 模型：`LinearSVC、LogisticRegression`
- `C ∈ {0.1, 1, 10}`
- 方向：`raw、reverse`

同时计算：

- 057 plus-PE固定OOF基线（若能按被试严格对齐）
- CSP单模型
- CSP与plus-PE的固定小网格融合：CSP权重`{0.25, 0.5, 0.75}`，其余为plus-PE；融合权重只能由训练trial内OOF确定
- DEP03、DEP17额外保留显式raw/reverse对照，不能根据外层query标签事后选方向

如057 OOF与个体窗口无法严格对齐，则只报告CSP与已有汇总基线，不强行拼接。

### 4. 评估和报告

`metrics.json`和`summary.csv`按被试、方法、重复/外层组合记录：

- 窗口AUC
- 窗口固定阈值F1（阈值0.5）
- trial均值AUC
- trial固定阈值F1
- trial quota=4 F1（辅助，不作为主结果）
- pair mean AUC/F1
- 每个被试的基线、最佳方法及AUC增益
- CSP/PE融合是否改善窗口与trial层级
- 选择次数、raw/reverse方向比例
- 最差pair表现和预测有限性

报告明确区分：

- AUC优先开发指标：目标AUC≥0.95；
- F1是同步观察指标，首轮不以F1未达标阻止继续；
- 被试内OOF结果是诊断/上限，不是公开集成绩；
- quota不等同固定阈值F1；
- 不能把095 adaptive或单被试CSP结果转化为公开集无标签方案。

### 5. 首轮扩展规则

首轮完成后按预先规则决定是否扩展：

- 若某方法在至少3名被试上相对该被试基线AUC提升，并且没有明显破坏回归保护个体，则将该方法扩展到全部非冻结被试。
- 若CSP仅对个别被试有效，则保留为个体诊断路线，不作为全体默认模型。
- 若DEP03/DEP17的reverse只在单次或少数pair出现，禁止推广为稳定方向；若在重复/多数pair中稳定，单独纳入困难DEP方向规则研究。
- 若首轮出现高AUC但固定阈值F1严重下降，记录为排序提升而非完整达标，继续以AUC优先但不夸大结果。

## Assumptions & Decisions

1. 本轮用户已选择“AUC优先”和“先跑小批次”。因此不直接运行全部剩余被试。
2. 首轮被试固定为`DEP07、DEP09、DEP16、HC10、HC29、DEP03、DEP17`，不得依据本轮结果临时增删。
3. 允许被试内有标签OOF用于开发和上限分析，但不允许使用公开测试集标签；公开集仍保持无标签协议。
4. 所有新方向产物只写入205目录；不修改历史实验结果。
5. GPU优先原则适用于可GPU化部分；当前CSP/协方差特征主要依赖SciPy/CPU，使用线程限制控制资源并在终端输出被试和pair进度，避免无进度长时间运行。
6. 实验完成后需维护总进度文件，并在用户明确要求提交时执行Git提交和远程同步；本计划阶段不自动提交代码。

## Verification Steps

1. 运行前检查7名被试MAT文件唯一存在，缓存/标签/物理trial顺序合法。
2. 检查每个被试40窗口、8物理trial、每trial5窗口、4正4负。
3. 检查每个外层pair的训练/验证物理trial交集为空。
4. 检查`outer_validation_labels_used_for_selection=False`、`public_labels_used=False`及来源文件SHA256。
5. 检查所有预测概率有限，窗口和trial输出行数正确，且无重复或缺失被试。
6. 检查基线、CSP、CSP+PE每个被试均有AUC/F1；若某层级无法计算，明确记录原因而不是填充伪值。
7. 检查报告中明确AUC和F1，并列出首轮每个被试的最佳方法、基线、增益和是否达到AUC 0.95。
8. 只有在首轮结果完成审计后，才依据上述扩展规则规划下一轮全部剩余被试实验。
