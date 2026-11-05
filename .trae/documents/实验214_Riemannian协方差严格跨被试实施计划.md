# 实验214：Riemannian协方差严格跨被试实施计划

## Context
当前冻结基线 `057 plus_pe` 窗口级 AUC 约为 0.800，STFT 残差方案约为 0.805，但 DEP AUC 仍约 0.688。继续调简单 DE、CNN 或 HC/DEP 硬专家难以接近 0.9。本实验新增独立协方差/Riemannian 分支，重点处理被试间 EEG 协方差域偏移，并与057基线进行严格、可审计的固定融合。

## 推荐方案

1. 在 `test/experiments/214_subjectwise_riemannian_covariance/` 创建独立实验目录，新增 `config.json`、`run_experiment.py`、`test_protocol.py`、`实验报告.md`，运行产物限制在该目录。
2. 使用 `src/emo_rec/data.py` 读取原始训练窗口；排除 HC43，保持69名被试、2760窗口、552物理trial协议。显式记录 `subject_id`、`health_group`、`physical_trial_index`、`window_index`、`y_true`、`fold`，不使用047 STFT缓存。
3. 复用 `src/emo_rec/covariance.py` 的收缩协方差和切空间数学；在214入口中实现GPU优先的 `torch.float64` 批量协方差、训练折参考矩阵白化、矩阵对数和465维上三角特征。正式运行前用CPU公共实现做数值一致性抽样校验。
4. 固定复用002/057的subject-wise外层折。每折只用训练窗口拟合协方差参考、标准化器和分类器；验证被试只允许用于无标签自身统计（如预声明的subject z-score），不使用验证标签调参。保留 raw 与 subject-zscore 两个预声明条件，分类器先固定为 `StandardScaler + LogisticRegression(C=1.0)`。
5. 读取冻结的057 OOF预测，双方先按规范键排序、检查无重复、数量、标签和fold完全一致，再按真实键合并；禁止按数组位置融合。预声明主融合为 `0.8 * 057 + 0.2 * covariance`，057单独作为控制基线。
6. 运行固定折及预声明重复折，输出每个分支和融合的窗口AUC/F1、HC/DEP AUC/F1、trial AUC/F1、subject macro、最差折、有限值覆盖率和Δ指标。终端持续打印device、condition、repeat、fold、GPU batch进度。
7. 只有在协方差融合重复平均AUC稳定超过057且DEP不恶化、协议测试和键审计全部通过后，才生成公开候选；公开标签不读取，公开AUC/F1记录为未知。

## 关键文件

- `test/experiments/214_subjectwise_riemannian_covariance/run_experiment.py`
- `test/experiments/214_subjectwise_riemannian_covariance/config.json`
- `test/experiments/214_subjectwise_riemannian_covariance/test_protocol.py`
- `src/emo_rec/covariance.py`
- `test/experiments/057_multiscale_permutation_entropy/predictions/plus_pe_fixed002_oof.csv`
- `test/experiments/032_repeated_subject_folds/run_experiment.py`

## 验证

1. 运行协议测试：形状、正定性、GPU/CPU数值误差、465维、subject/trial隔离、OOF覆盖和057键集合。
2. 用当前Python环境执行214入口，确认CUDA优先且每折有进度输出。
3. 检查 `metrics.json`、`summary.csv`、`实验报告.md` 和 `input_metadata/provenance.json`，确认AUC/F1及HC/DEP结果完整。
4. 仅当验收条件满足时再生成公开预测；随后检查80行、每个公开被试8条trial、概率有限且位于[0,1]、无公开标签读取。
