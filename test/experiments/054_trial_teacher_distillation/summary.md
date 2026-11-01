# 实验054：trial-teacher 蒸馏（检验053理论预测P3）

数据与协议同052：001频谱150维+009时序450维，排除HC43，逐被试/user无标签z-score；fixed002五折+seeds 2027..2031重复，阈值0.5，threadpool_limits(1)。
trial键=(subject, original_trial_id+4*y)，每键5窗（已断言）。trial id仅在外层训练被试内部用于平均内层OOF概率构造训练目标，从不作为预测输入，也不触及外层验证标签。

方法（预注册，无调参）：
- control：与052 control相同HGB（同种子2052+fold）。
- soft_target_student：外层训练被试内5折被试划分（seed 2054）得内层OOF窗口概率，t=所属trial的OOF均值；软目标0.5*y+0.5*t；同HGB在复制行上训练（标签1权重soft，标签0权重1-soft，等价软标签logloss）。
- teacher_confidence_weighting：硬标签，w=0.25+0.75*(y==1?t:1-t)，归一化为均值1。

teacher质量（30个外层折均值，训练被试内）：窗口OOF AUC 0.7819，trial均值AUC 0.8516。

|方法|分割|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|最差被试AUC|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|control|fixed002|0.7886|0.7085|0.7069|0.7722|0.8410|0.6762|0.4075|
|soft_target_student|fixed002|0.7912|0.7124|0.7101|0.7720|0.8458|0.6737|0.4025|
|teacher_confidence_weighting|fixed002|0.7897|0.7105|0.7080|0.7720|0.8425|0.6779|0.4300|
|control|repeat2027|0.7885|0.7215|0.7225|0.7739|0.8388|0.6819|0.4625|
|soft_target_student|repeat2027|0.7905|0.7190|0.7188|0.7745|0.8404|0.6830|0.4550|
|teacher_confidence_weighting|repeat2027|0.7914|0.7241|0.7250|0.7757|0.8398|0.6876|0.4625|
|control|repeat2028|0.7889|0.7185|0.7178|0.7728|0.8399|0.6784|0.3600|
|soft_target_student|repeat2028|0.7875|0.7225|0.7214|0.7692|0.8409|0.6698|0.3500|
|teacher_confidence_weighting|repeat2028|0.7868|0.7214|0.7221|0.7709|0.8405|0.6686|0.3375|
|control|repeat2029|0.7863|0.7206|0.7185|0.7713|0.8333|0.6832|0.4075|
|soft_target_student|repeat2029|0.7900|0.7180|0.7163|0.7741|0.8412|0.6778|0.4000|
|teacher_confidence_weighting|repeat2029|0.7909|0.7198|0.7185|0.7770|0.8388|0.6868|0.4200|
|control|repeat2030|0.7910|0.7219|0.7228|0.7782|0.8423|0.6829|0.3525|
|soft_target_student|repeat2030|0.7900|0.7232|0.7232|0.7749|0.8413|0.6783|0.3975|
|teacher_confidence_weighting|repeat2030|0.7953|0.7248|0.7246|0.7799|0.8454|0.6871|0.3950|
|control|repeat2031|0.7925|0.7177|0.7170|0.7750|0.8427|0.6846|0.4225|
|soft_target_student|repeat2031|0.7957|0.7167|0.7152|0.7793|0.8462|0.6873|0.4250|
|teacher_confidence_weighting|repeat2031|0.7944|0.7253|0.7250|0.7785|0.8443|0.6872|0.4300|
|control|repeat_mean|0.7894|0.7200|0.7197|0.7743|0.8394|0.6822|0.4010|
|soft_target_student|repeat_mean|0.7907|0.7199|0.7190|0.7744|0.8420|0.6792|0.4055|
|teacher_confidence_weighting|repeat_mean|0.7918|0.7231|0.7230|0.7764|0.8418|0.6835|0.4090|

## 相对control（五重复均值差）

|方法|ΔAUC|ΔF1|ΔDEP AUC|ΔHC AUC|Δ最差被试|fixed002 ΔAUC|逐重复ΔAUC|接受|
|---|---:|---:|---:|---:|---:|---:|---|---|
|soft_target_student|+0.0013|-0.0002|-0.0029|+0.0026|+0.0045|+0.0027|+0.0021, -0.0014, +0.0037, -0.0010, +0.0032|否|
|teacher_confidence_weighting|+0.0023|+0.0030|+0.0013|+0.0024|+0.0080|+0.0011|+0.0029, -0.0021, +0.0046, +0.0043, +0.0020|否|

control复现检查（与052 control OOF最大绝对差）：fixed002=1.11e-16, repeat2027=1.11e-16, repeat2028=1.11e-16, repeat2029=1.11e-16, repeat2030=1.11e-16, repeat2031=1.11e-16

## P3检验：被试AUC增益 vs 053 (trial AUC − window AUC)

|方法|Spearman ρ (p)|Pearson r (p)|高gap半数平均增益|低gap半数平均增益|全体平均增益|
|---|---|---|---:|---:|---:|
|soft_target_student|+0.221 (0.068)|+0.190 (0.118)|+0.0036|-0.0034|+0.0001|
|teacher_confidence_weighting|+0.344 (0.004)|+0.311 (0.009)|+0.0055|-0.0012|+0.0022|

注：053的trial/window AUC来自同一control家族OOF，与本实验增益非独立，相关仅作描述性证据；逐被试明细见per_subject_gain_vs_gap.csv。

## 决策（预注册）

接受条件：五重复均值ΔAUC ≥ +0.005 且 ΔF1 ≥ 0 且 ΔDEP AUC ≥ 0。
结论：soft_target_student未达标；teacher_confidence_weighting未达标。两种蒸馏均未达标，按协议停止，不调参。

真实运行耗时：954.7秒。
