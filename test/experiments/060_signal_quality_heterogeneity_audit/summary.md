# 060 信号质量异质性审计

本轮获得用户授权后运行；固定 seed=2060。仅使用已生成的058 split OOF与057 repeat OOF，不训练、不选择阈值、不删分析对象、不使用公开预测。质量代理是粗粒度信号尺度/极值相对标准差指标，不是确诊 artifact，也不报告伪迹率。

主分析：69名被试；HC43仅作为mat审计参照记录，不进入相关分析。12个相关（2代理×2 endpoint×3群组）在本检验family内做BH校正；Spearman，2000次分组bootstrap CI，4999次组内置换双侧p。all组置换在HC/DEP内进行，bootstrap按组抽样。057 repeat仅作方向稳定性，不是独立验证。

## 主分析结果

 quality_proxy            endpoint group       rho    ci_low   ci_high  p_perm_two_sided     q_bh
     std_proxy plus_pe_auc_primary   all  0.288663  0.067983  0.502762            0.2660 0.354667
     std_proxy plus_pe_auc_primary    HC  0.248705 -0.078704  0.526213            0.1104 0.354667
     std_proxy plus_pe_auc_primary   DEP -0.258511 -0.608371  0.203107            0.2176 0.354667
     std_proxy        gain_primary   all -0.160582 -0.405701  0.099584            0.1568 0.354667
     std_proxy        gain_primary    HC -0.232575 -0.531561  0.098473            0.1368 0.354667
     std_proxy        gain_primary   DEP -0.077323 -0.541664  0.388311            0.7112 0.775855
range_sd_proxy plus_pe_auc_primary   all  0.090923 -0.149089  0.323931            0.2368 0.354667
range_sd_proxy plus_pe_auc_primary    HC -0.206984 -0.510062  0.132587            0.1836 0.354667
range_sd_proxy plus_pe_auc_primary   DEP -0.034237 -0.439262  0.388107            0.8732 0.873200
range_sd_proxy        gain_primary   all -0.083561 -0.315960  0.167343            0.4816 0.577920
range_sd_proxy        gain_primary    HC -0.344633 -0.598201 -0.035133            0.0272 0.326400
range_sd_proxy        gain_primary   DEP  0.395076 -0.059338  0.753074            0.0584 0.350400

## 057 repeat方向稳定性（非独立验证）

 quality_proxy           endpoint group       rho    ci_low  ci_high  p_perm_two_sided     q_bh
     std_proxy plus_pe_auc_repeat   all  0.299356  0.065638 0.505349            0.2544 0.482057
     std_proxy plus_pe_auc_repeat    HC  0.236749 -0.103249 0.509007            0.1384 0.482057
     std_proxy plus_pe_auc_repeat   DEP -0.228462 -0.585967 0.208912            0.2624 0.482057
     std_proxy        gain_repeat   all -0.079526 -0.337278 0.172636            0.4716 0.628800
     std_proxy        gain_repeat    HC -0.250705 -0.548875 0.070972            0.1024 0.482057
     std_proxy        gain_repeat   DEP  0.229660 -0.242796 0.637436            0.2812 0.482057
range_sd_proxy plus_pe_auc_repeat   all  0.090724 -0.133935 0.331425            0.2028 0.482057
range_sd_proxy plus_pe_auc_repeat    HC -0.208063 -0.498492 0.130122            0.1872 0.482057
range_sd_proxy plus_pe_auc_repeat   DEP -0.040769 -0.429801 0.380194            0.8528 0.930327
range_sd_proxy        gain_repeat   all -0.001169 -0.257717 0.255915            0.9900 0.990000
range_sd_proxy        gain_repeat    HC -0.117987 -0.412690 0.205862            0.4476 0.628800
range_sd_proxy        gain_repeat   DEP  0.119254 -0.341380 0.554488            0.5856 0.702720

## 方法动机与边界

PREP原论文讨论多维坏通道检测与鲁棒参考：https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2015.00016/full；Autoreject原论文采用逐传感器峰峰值阈值并交叉验证：https://arxiv.org/html/1612.08194v3。本文整段极值/std代理不能替代逐窗逐通道检测，弱关联不能排除伪迹；没有0.95的新性能证据，也不构成伪迹率估计。因果与上限解释边界沿用053理论，不修改053。

结果与源文件hash见 `metrics.json`；逐被试合并表见 `tables/subject_quality_vs_oof.csv`。
