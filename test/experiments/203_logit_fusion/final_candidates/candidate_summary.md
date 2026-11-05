# 实验203公开候选说明

## 核验范围

本说明仅补充候选记录，不修改203及其他旧实验。已核验 `203_logit_fusion/predictions` 中四份公开CSV：

| 文件 | 数据行数 | 列名与顺序 | 结论 |
|---|---:|---|---|
| `logit_zscore_public.csv` | 80 | `user_id, trial_id, prob. of positive emotion` | 通过 |
| `platt_probability_public.csv` | 80 | `user_id, trial_id, prob. of positive emotion` | 通过 |
| `raw_logit_public.csv` | 80 | `user_id, trial_id, prob. of positive emotion` | 通过 |
| `raw_probability_public.csv` | 80 | `user_id, trial_id, prob. of positive emotion` | 通过 |

四份文件均包含表头以下80条记录，列名、列顺序与公开提交格式一致。测试集为10名被试×8段视频，真实标签不公开；数据集说明明确指出公开测试集标签未知，因此不能从公开CSV计算真实AUC/F1。

## OOF候选与选择

| 候选 | OOF AUC | OOF F1 | 定位 | 选择说明 |
|---|---:|---:|---|---|
| `logit_zscore` | **0.795511** | 0.723266 | AUC优先候选 | 203四种预设融合中OOF AUC最高；F1比031 Platt低0.000523，不能同时宣称AUC和F1均提升。 |
| `031 Platt` / `platt_probability` | 0.793526 | **0.723789** | F1/稳健候选 | F1高于logit_zscore 0.000523；与031 Platt基线一致，校准路径和表现更保守，适合作为F1/稳健备选。 |
| `0.85 * 057 + 0.15 * 006` | **0.803836** | **0.726616** | **当前主候选** | 057与006固定权重融合的既有结果，OOF AUC/F1均高于203单独候选；当前仍保留为正式主候选。 |

补充基线：031 Platt为 AUC=0.793526、F1=0.723789；203 `logit_zscore`相对其 ΔAUC=+0.001985、ΔF1=-0.000523。057+006 0.85融合相对057 `plus_pe`的增益为 ΔAUC=+0.003933、ΔF1=+0.001706。

## 最终排序与使用建议

1. **主候选：057 + 006，权重0.85/0.15**，同时优先兼顾AUC与F1。
2. **AUC优先候选：203 `logit_zscore`**，仅在目标明确偏向排序指标时考虑。
3. **F1/稳健候选：031 Platt（对应203 `platt_probability`路径）**，用于偏好阈值0.5下F1或更保守校准时的备选。

## 公开标签与选择风险

- 公开测试标签未知，以上公开文件只能作为提交预测，公开AUC/F1均记为 **unknown**，不得用公开结果反向调权或调阈值。
- OOF指标来自训练集固定的subject-wise外层划分；在多个实验和候选之间比较会产生探索性选择偏差，OOF领先不等于公开集必然领先。
- 203复用了031的外层预测与inner OOF；`logit_zscore`的标准化统计量按协议只来自相应训练侧inner OOF，但候选选择仍应视为有限验证证据。
- 057+006的0.85权重来自固定扫描结果，仍可能受单一划分和有限被试数量影响；其“主候选”是当前工作决策，不是无偏泛化结论。
