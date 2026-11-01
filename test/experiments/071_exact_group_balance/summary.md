# 实验071：严格HC/DEP等权ERM

冻结协议见test_protocol.txt。控制与严格等权组训练均为a8→a8；处理权重 w_i=N_train/(2*N_group_train)，组总权重各N_train/2、均值1、不裁剪。

|split|ΔAUC|ΔF1|HC ΔAUC|DEP ΔAUC|
|---|---:|---:|---:|---:|
|fixed002|-0.006194|-0.003093|-0.007997|-0.002072|
|repeat2042|-0.005651|-0.007572|-0.006751|-0.001692|
|repeat2043|-0.005370|-0.003901|-0.007078|+0.000676|
|repeat2044|-0.007632|-0.004977|-0.008667|-0.003488|
|repeat2045|-0.001467|+0.002237|-0.003419|+0.005052|
|repeat2046|-0.006007|-0.001768|-0.006282|-0.005064|

五重复均值：ΔAUC=-0.005225，ΔF1=-0.003196，HC ΔAUC=-0.006440，DEP ΔAUC=-0.000903。
接受判断：未通过。
方法背景（不作为EEG证据）：Sagawa et al. https://arxiv.org/abs/1911.08731；HGB sample_weight文档 https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html。普通平均可能掩盖弱组，naive groupDRO也可能泛化失败；本实验是固定等权ERM，不是DRO。训练44 HC/25 DEP与公开5/5仅支持动机，不能解决组内条件偏移。
耗时：1678.80秒；控制fixed002逐行复现070 p_train8最大误差≤1e-12；未访问公开数组。
