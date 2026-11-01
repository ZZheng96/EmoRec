# 实验068：训练上下文匹配

冻结协议见test_protocol.txt。69被试2760训练行全部保留；双路60 fits；验证统一a8。
五重复汇总不包含fixed002；Δ为训练a8减训练a40。

|位置|控制AUC|匹配AUC|ΔAUC|控制F1|匹配F1|ΔF1|HC ΔAUC|DEP ΔAUC|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|all_rows|0.802382|0.810227|+0.007845|0.723856|0.734082|+0.010226|+0.007701|+0.009502|
|0|0.787938|0.800176|+0.012237|0.718444|0.723883|+0.005439|+0.011603|+0.016400|
|1|0.805393|0.815703|+0.010310|0.727533|0.736656|+0.009123|+0.011706|+0.004500|
|2|0.815850|0.819555|+0.003705|0.742747|0.748485|+0.005738|-0.000019|+0.016860|
|3|0.791593|0.803043|+0.011450|0.704395|0.723326|+0.018931|+0.011215|+0.015700|
|4|0.809832|0.812755|+0.002922|0.725813|0.737908|+0.012096|+0.004901|-0.000520|

## 预注册接受判断
{"accepted": true, "checks": {"mean_delta_auc": true, "mean_delta_f1": true, "auc_nonnegative": true, "f1_nonnegative": true, "hc_no_excess_drop": true, "dep_no_excess_drop": true}}
接受线：五重复all_rows平均ΔAUC和ΔF1均>=0.005；两指标各至少4/5 split非负；HC/DEP平均ΔAUC均>=-0.005。
结果：达到预注册接受线。不能据此宣称公开提升。

## 逐重复all_rows配对差值
|split|ΔAUC|ΔF1|HC ΔAUC|DEP ΔAUC|
|---|---:|---:|---:|---:|
|fixed002|+0.006127|+0.004266|+0.005622|+0.008784|
|repeat2027|+0.006792|+0.003112|+0.009580|-0.000484|
|repeat2028|+0.010814|+0.014259|+0.007286|+0.019912|
|repeat2029|+0.001484|+0.005231|+0.003045|-0.000692|
|repeat2030|+0.003366|+0.007470|+0.004653|+0.005844|
|repeat2031|+0.016771|+0.021059|+0.013940|+0.022928|

耗时212.15秒；控制六份OOF对067 p_context8最大差1.11e-16。
sklearn HGB无GPU实现，CPU例外n_jobs=10/thread=1；未访问公开数组。所有输入和复用源码hash见input_metadata/provenance.json。
分组使用已知训练视频结构，仅属诊断，不代表公开刺激/时间匹配；五位置不是独立样本，五重复复用同一队列；不宣称独立队列、显著性或公开提升。

## 方法动机及相反机制
- [Batch Renormalization](https://arxiv.org/abs/1702.03275)：上游核读原文摘要；small/non-iid batch及训练/推理激活差异提供方法动机。
- [Ghost Noise for Regularizing Deep Neural Networks, AAAI2024](https://arxiv.org/abs/2305.17205)：上游核读原文摘要；小batch可能带来正则收益，同时引入train-test discrepancy，并非越大越好。
两篇均为方法动机，而非EEG zscore+HGB效果证据；不外推成绩。
含自身的ddof=0 n窗zscore（非退化特征）满足|z_i|<=sqrt(n-1)：其余n-1个中心化值和为-z_i，由Cauchy不等式得到该界。因此8窗/40窗不仅估计方差不同，支持范围也不同（sqrt(7) vs sqrt(39)）。统计量匹配有可检验机制，但不保证分类改善。
完整逐折/逐被试指标、五重复样本SD和配对values见metrics.json；每位置逐split差值见paired_delta.csv。

## 测试与实际文件清单
- 训练前：5 passed, 1 deselected（10.97秒，排除尚未生成的OOF重算）。
- 训练后：6 passed（38.07秒），含真实映射、手工归一化、禁止公开数组、全部30折训练上下文独立重算、两路训练保留所有行、六套控制复现、全部OOF指标/配对汇总/接受判断及输入源码hash重算。
- 五重复all_rows AUC、F1差值均为5/5非负。
- 根目录：config.json、test_protocol.txt、run_experiment.py、test_068.py、metrics.json、summary.csv、paired_delta.csv、summary.md、run_output.txt、pytest_before.txt、pytest_after.txt。
- predictions/：fixed002_oof.csv、repeat2027_oof.csv、repeat2028_oof.csv、repeat2029_oof.csv、repeat2030_oof.csv、repeat2031_oof.csv。每份含两路预测，共6份而非合并跨重复预测。
- input_metadata/：provenance.json、folds_fixed002.csv、folds_repeat2027.csv、folds_repeat2028.csv、folds_repeat2029.csv、folds_repeat2030.csv、folds_repeat2031.csv。
共24个文件；仅本实验目录有写入，未更新进度，未执行git。
