# 064 冻结实验协议

首次训练前保存；单一配置、不调参、不进行公开集全训练或预测，仅写064目录。

每个独立10秒、30通道、250Hz窗：scipy.signal.stft Hann，nperseg=250，noverlap=125，nfft=250，boundary=None，padded=False，detrend=False。保留1≤f≤45Hz；abs(z)^2按每帧频率和归一化，零功率帧使用45频点均匀分布。19帧的18个相邻对计算base2 Jensen–Shannon divergence（非平方根），按通道求mean/std(ddof=0)，排列mean30再std30，共60维。逐被试无标签zscore，拼接057 source.load_data的720维控制得到780维。不调用062 load_data，不引入oAEC。

仅control_pe和plus_spectral_js。复用062配对训练/保存骨架。fixed002及seed2027..2031五次被试级分层5折；排除HC43，69人2760行；HGB200、lr=.03、leaf15、minleaf20、l2=1、early_stopping=False，外折seed=2052+fold，n_jobs=10，阈值.5，训练折StandardScaler沿用骨架。

接受条件同时满足：五重复均值ΔAUC≥.005、ΔF1≥0、ΔDEP AUC≥0，且fixed002 ΔAUC>0。未达线停止，不调参。

保存原始输入哈希、配置/代码/缓存哈希、缓存alignment、逐行配对及被试隔离审计、六套折、12份OOF、全指标、摘要及运行记录。测试覆盖直接JS参考、scale/sign invariance、零输入有限、batch独立、真实行及折隔离、重算全部指标、控制与057 plus_pe OOF差≤1e-12。先测试再运行再测试；运行前结果相关测试允许skip，运行后必须全部通过。

## 新颖性与局限
009仅5个2秒bandpower均值/std/斜率；040为包络调制；047为STFT CNN；061为ordinal JS，均不是本轮谱形状相邻帧JS。
主线提供的Shen/Lin2019研究（https://pmc.ncbi.nlm.nih.gov/articles/PMC6831623/ ，10人80sessions跨日情绪EEG时空谱非平稳）仅作动机，不证明秒级JS有效。SciPy官方说明jensenshannon返回sqrt(divergence)：https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html 。本轮使用divergence。
1秒窗限制频率分辨率且有随机谱估计误差；50%重叠导致相邻相关；非平稳不等于情绪机制；mean/std丢失变化方向。逐被试无标签zscore利用该被试整批窗，但不读取标签、不混合不同被试。
