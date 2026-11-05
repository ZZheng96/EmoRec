# 实验212：STFT残差重复种子稳定性验证计划

## Context

实验210在严格跨被试、物理trial隔离的固定5折OOF中，将057 plus_pe与STFT-CNN按预先固定权重0.2融合，窗口AUC达到0.805161，较057基线0.799903提升0.005258；但STFT-CNN单模较弱，且F1没有同步提升。实验211已证明HC/DEP硬专家路由无效。剩余时间优先验证实验210的AUC增益是否稳定，而不是继续搜索已失败的专家模型。

## 推荐实现

1. 新建独立目录 `test/experiments/212_stft_residual_seed_stability`，不覆盖210/211。
2. 复用实验210的STFT缓存、057 plus_pe OOF、固定subject-wise折和协议；保持排除HC43、69名被试、2760窗口、552物理trial。
3. 对STFT-CNN残差模型运行预先声明的3个随机种子；每个折只用训练被试拟合STFT均值/std并训练模型，验证被试只做推理。权重固定为0.2，不根据验证标签调整。
4. 所有OOF结果先构造并校验规范键：`subject_id + health_group + physical_trial_index + window_index + y_true + fold`；逐种子确认键集合、数量、标签和fold完全一致后再按键平均概率，禁止按数组位置拼接。
5. 比较单种子、重复种子平均、057冻结基线和实验210结果；报告窗口AUC/F1、HC/DEP AUC/F1、trial AUC/F1、subject macro、最差折、每种子结果和平均结果。固定阈值为0.5，quota仅作辅助（如已有实现）。
6. 记录缓存与源码SHA256、随机种子、设备、训练折统计、模型参数、键校验和公开标签未使用证明，输出`metrics.json`、`summary.csv`、`实验报告.md`、`predictions/`和`input_metadata/`审计文件。公开集不读取标签；只有重复种子平均结果不低于实验210且协议审计通过，才允许生成公开候选，否则保留057/210候选。

## 关键复用文件

- `test/experiments/210_stft_cnn_baseline_fusion/run_experiment.py`
- `test/experiments/210_stft_cnn_baseline_fusion/config.json`
- `test/experiments/047_time_frequency_cnn/cache/stft_windows.npz`
- `test/experiments/057_multiscale_permutation_entropy/predictions/plus_pe_fixed002_oof.csv`
- `数据/数据集说明文档.md`
- `Review Competition Requirements and Progress.md`

## Verification

- 终端打印每个seed、fold、epoch进度，并确认使用CUDA（若可用）。
- 运行后检查所有OOF概率有限、覆盖2760行、每被试40窗口、每物理trial5窗口、fold与subject完全隔离。
- 检查重复种子键校验通过，报告同时写明AUC和固定阈值F1；公开AUC/F1保持未知。
- 若结果超过实验210且不是偶然单种子增益，再由用户决定是否生成公开提交文件；完成后更新总进度并提交Git同步远程。
