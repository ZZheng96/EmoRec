# 实验064：单窗谱形状动态

冻结协议见protocol.md；不调参，不公开训练或预测。

|方法|五重复AUC均值±SD|F1均值|DEP AUC均值|
|---|---:|---:|---:|
|control_pe|0.794851±0.003865|0.724749|0.685235|
|plus_spectral_js|0.792721±0.004118|0.721240|0.682816|

五重复ΔAUC=-0.002130，ΔF1=-0.003509，ΔDEP AUC=-0.002419；fixed002 ΔAUC=-0.002778。接受：False。未达线即停止。
真实运行耗时：195.36秒；控制对057最大差：1.11e-16。12份OOF见predictions；完整逐折/逐被试指标见metrics.json。

## 运行与验证记录
使用`.venv/Scripts/python.exe -B`；首次训练前保存config.json/protocol.md。训练前专项测试3 passed、1 skipped（结果尚未生成）；随后一次真实运行完成70人原始特征提取、排除HC43后的69人2760行、30对折任务共60模型；运行后4 passed（8.61秒，无跳过）。记录见tests_before.txt、run_output.txt、tests_after.txt。直接JS公式与SciPy平方距离一致，缩放/符号不变、零输入、batch独立、真实行/折隔离、12份OOF全指标与汇总重算、057控制复现均通过。输入哈希及缓存逐行对齐审计见input_metadata。
五次重复的配对ΔAUC全部为负（-0.002523、-0.001073、-0.004907、-0.001713、-0.000436），本轮否决，不调参。无执行阻塞或依赖困难；效果未达预设接受条件。未修改全局进度，未执行git操作。

Shen/Lin2019（https://pmc.ncbi.nlm.nih.gov/articles/PMC6831623/ ）跨日非平稳仅作动机，不证明秒级JS有效。SciPy jensenshannon为sqrt(divergence)，本轮用base2 divergence。1秒窗频率分辨率和随机谱估计限制；50%重叠引入相邻相关；非平稳不等于情绪机制；mean/std丢失变化方向。
