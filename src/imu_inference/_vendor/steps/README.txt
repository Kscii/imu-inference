GP2 通用计步估计附加包

用途
输入25Hz三轴加速度（m/s²），输出已分析观测时间内的估计步数和连续行进片段步频。不改变原四类动作标签、时长和时间线。支持按输入显式排除无效、缺失或不应分析的区间；没有分析结果返回null，真实分析后没有候选步才返回0；不按覆盖比例补齐全天。

实现与训练
成熟mobgap1.2.0 Shin范数算法提供候选步；一个轻量HistGradientBoosting分类器判断适合计步的时段。模型以31项旋转不变加速度特征处理4秒上下文，每秒输出一次。24名OxWalk公共开发人员用于3折按人交叉验证及最终拟合；15名人员仅作冻结后的留出评价。没有真实设备数据参与拟合、阈值选择、归一化或标定。

实测与边界
留出测试15人、14.94小时、27,631标注步，髋部25Hz数据。
整录制总步数平均绝对百分比误差6.14%；现成步行筛选器对照30.99%。这是总数误差，不是事件识别正确率。
防止多计/漏计抵消的10秒分段计数误差23.40%；零步区间50.80误计步/小时。
1–5步片段归一化绝对误差67.31%；>10步片段3.50%。短步行、停顿误计仍是主要限制。
参考合格片段上的步频MAE5.69步/分钟，按人平均输出覆盖77.43%。这些结果没有达到全部原拟议严格门槛。
±150ms事件F1按数值边界修正后的评分为56.29%（旧报告55.87%；物理容差仍为150ms，计数与步频结果不变），因此不把候选时刻当可靠触地事件，不输出经过验证的步间变异、空间步长、左右不对称或疾病结论。
本包可接入累计步数估计和连续行进概览，不应宣传为全天胸前计步全面达标。公共髋部成绩不是本机胸前实测准确率；不要求补标看不到脚的设备视频。只在明确的适用范围说明下使用估计值。

安装
Python3.12，LinuxCPU。
python -m pip install -r requirements.txt
完整离线锁定依赖见requirements-offline.txt和配套wheels归档（若随交付提供）。建议在独立环境安装，避免mobgap的PyWavelets版本约束影响其他算法。

输入
NPZ包含accel_mps2（N×3浮点、m/s²）；可选valid、excluded为N长度bool；可选time_s为严格递增、约40ms间隔的秒时间戳。输入已按25Hz规则网格准备；50Hz原始数据先复用主包的预处理。遇到真实时间缺口应拆成独立连续输入，不能跨缺口补步。少于4秒的连续数据保留为无法分析时间，不伪造零步。

命令
python step_count.py --input example.npz --output steps.json
python step_count.py --input example.npz --activity-json activity.json --output combined.json
组合输出保持activity对象原样，新增steps对象；客户端读取steps.estimated_step_count。原主包step_count字段的null不会被静默改写。

Python
from step_count import StepCounter
counter = StepCounter()  # 每个进程复用实例
result = counter.predict(accel_mps2, valid=valid, excluded=excluded)

主要字段
estimated_step_count：已分析时间内估计步数；estimated_cadence_spm：有效连续候选片段的时间加权步频估计；analyzed_s、unresolved_valid_s、invalid_s、excluded_s分别解释覆盖。coverage_of_observed表示算法处理了多少有效信号，不表示步数正确率。
候选时刻仅在include_candidates=True的工程诊断中返回，不能作为经过验证的生理触地时间。步频不跨预测入口间断、缺失或>2秒候选事件间隔连接。

计算
CPU即可；本包没有新的神经网络训练，也不需要GPU。具体吞吐、内存、15条保存结果精确回放以及边界检查见RUNTIME_CHECKS.json。

证据
VALIDATION.json：完整主指标与未通过门槛。
CAPABILITIES.json：支持字段和限制。
DATA_CITATION.txt及THIRD_PARTY_LICENSES：数据来源、算法来源与许可证。
SHA256SUMS：本地交付文件完整性。

本地工程包；未自动发送客户或外部发布。原主分类包及旧统计包保留。
