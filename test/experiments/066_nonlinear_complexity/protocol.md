# 实验066：单窗非线性复杂度增量（预注册冻结协议）

目标是检验单个原始10秒、30通道、2500点窗口的非线性复杂度特征，是否相对057的720维PE基线带来增量。仅新增本目录文件；不修改旧实验、不提交git、不使用公开标签或公开指标决策。

## 特征冻结

每个独立窗口逐通道计算，不跨窗口、不使用标签、身份、trial、行序或邻窗。SampEn固定m=2、tau=1、r=0.2×该通道窗口标准差；模板按Chebyshev距离匹配，排除self-match。B是m维模板的匹配对计数，A是m+1维模板的匹配对计数，二者均为有序模板对中每个i<j的一次计数，SampEn=-log(A/B)。std=0、B=0或A=0时返回0，结果必须有限。匹配按行分块，避免构造O(n²)巨大距离矩阵；参数不因耗时改变。

LZC固定对每通道窗口以窗口中位数二值化（等于中位数取1），统计LZ76增量短语数c；归一化为c×log2(n)/n，n=2500。常数序列仍返回有限值。特征顺序严格为SampEn30再LZC30，共60维。

复用057的720维PE基线（001的150维log power + 009的450维temporal + 057的120维PE），训练和公开数据各自逐被试/user无标签z-score后拼接。方法仅有control_pe（720维）与plus_complexity（780维），其余HGB、StandardScaler、阈值、折分、seed、排除HC43与057一致。

## 验证与接受线

先运行专项pytest，确认SampEn直接小序列参考、LZ76直接字符串参考、SampEn scale invariant、zero/finite、batch independent、真实缓存行/subject/fold对齐；再运行完整实验，再运行专项pytest。固定002及2027..2031五次被试级分层5折，训练fold seed=2052+fold，公开全训seed=2052。接受线相对control_pe：五重复均值ΔAUC≥0.005、ΔF1≥0、ΔDEP AUC≥0，且fixed002 ΔAUC>0；不达线停止，不调参。

SampEn和LZC是复杂度描述量，不是机制证据。短窗、匹配阈值、离散化阈值及幅值归一化均可能敏感；结果不声称神经机制或跨协议普适性。

保存complexity cache、predictions、input_metadata、metrics/summary/run_output，以及相关原始输入、缓存、配置和脚本SHA256。公开集只输出预测，不计算指标。
