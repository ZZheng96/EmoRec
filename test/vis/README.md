# EEG基础统计与可视化审计

运行（项目根目录）：

```powershell
.\.venv\Scripts\python.exe test\vis\run_visual_audit.py
```

脚本只读取 `数据/训练集/`，不会读取 `数据/公开测试集/`，也不会修改旧实验目录。训练集按 `src/emo_rec/data.py` 的约定，将每个情绪数组 `(30, 50000)` 切成20个不重叠的10秒窗；每名被试共40窗。HC43明确排除，因此主分析为69人（HC44、DEP25），总计2760窗。neu/pos是视频情绪标签，HC/DEP是被试组别，二者严格分开。

通道布局图使用脚本中的固定二维近似坐标，仅用于标注和可读性，不声称为实测电极几何位置。输出包括：

- `channel_layout.png`：30通道布局；
- `representative_waveform_envelope.png`：HC01（若存在）的neu 20窗稳健10–90%包络；
- `hc_dep_effect_heatmap.png`：DEP−HC平均log-bandpower热图；
- `subject_channel_emotion_summary.csv`：被试×通道×情绪的RMS、绝对均值和五频带log功率；
- `hc_dep_effects.csv`：组均值、DEP−HC差异、Cohen's d及描述性方向比例；
- `subject_heterogeneity.csv`：被试级幅值基线和窗口/通道异质性摘要；
- `audit_metadata.json`、`summary.md`：审计元数据与边界说明。

本目录是描述性数据诊断，不是可泛化模型证据：不训练、不调参、不访问公开测试集、不生成公开预测。AUC/F1在此审计中不适用，报告为N/A。
