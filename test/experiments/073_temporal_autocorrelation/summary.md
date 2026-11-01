# 实验073：连续原始波形多滞后自相关

接受：False；联合非负 1/5。

- 五新split平均Δauc: -0.001897921
- 五新split平均Δf1: -0.001166770
- 五新split平均Δhc_auc: -0.002144112
- 五新split平均Δdep_auc: -0.001343200

60 fits；总耗时 224.98 秒；特征/加载/哈希 6.28 秒。
CUDA float64自相关；冻结HGB无GPU实现，CPU例外10 jobs/1 thread。
控制720维、处理780维，均a8→a8；六套控制逐行复现070，最大误差1.11e-16。
未访问公开数组。未通过则停止，不调参、不挑位置。

命令（项目根目录）：
```powershell
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider test/experiments/073_temporal_autocorrelation/test_073.py -q
.venv/Scripts/python.exe -B test/experiments/073_temporal_autocorrelation/run_experiment.py
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider test/experiments/073_temporal_autocorrelation/test_073.py -q
```
运行前pytest：6 passed、1 skipped（尚无真实OOF），6.23秒；运行后最终pytest：7 passed，33.47秒。
首次运行后测试发现缓存字符串元数据保存成object dtype，已改为Unicode并重新生成缓存，未改变候选、配置、特征公式或OOF，未额外拟合。初次失败日志保存在pytest_after_initial_failure.txt；修复时间和原始/修复后源码hash记录于provenance。
前后测试输出见pytest_before.txt和pytest_after.txt。输入SHA256及运行环境见input_metadata/provenance.json。

产物：config.json、test_protocol.txt、run_experiment.py、test_073.py、cache/autocorrelation.npz、predictions/{fixed002,repeat2042..2046}_oof.csv、input_metadata/folds_*.csv、input_metadata/provenance.json、metrics.json、paired_delta.csv、summary.csv、summary.md、run_output.txt及三个pytest日志。
