# 实验062：跨通道正交包络相关

本轮使用abs幅度，不是原Hipp log power方法的完整复现。Sensor orthogonalization不等于源空间消除混叠，也不能证明脑连接，不能消除所有ghost/间接混叠；10秒短窗可靠性未知。参考Hipp https://pmc.ncbi.nlm.nih.gov/articles/PMC3861400/ 、Nolte https://pmc.ncbi.nlm.nih.gov/articles/PMC7683718/ 、Palva2018 https://doi.org/10.1016/j.neuroimage.2018.02.032 。公开缓存仅复用057既有特征核对alignment，不计算公开新特征、不进行公开全训练及预测。

|方法|五重复AUC均值±SD|F1均值|DEP AUC均值|
|---|---:|---:|---:|
|control_pe|0.794851±0.003865|0.724749|0.685235|
|plus_oAEC|0.796941±0.005322|0.728151|0.690310|

|方法|ΔAUC|ΔF1|ΔDEP AUC|fixed002 ΔAUC|达线|
|---|---:|---:|---:|---:|---|
|plus_oAEC|+0.002090|+0.003401|+0.005074|+0.002286|False|

真实耗时：362.36秒；不调参。
