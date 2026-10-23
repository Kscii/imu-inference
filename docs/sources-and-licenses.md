# 数据来源、规模与授权核对

核对日期：2026-10-23。本文列出冻结交付模型实际涉及的来源；作者数据集总规模与本项目训练用量分开报告。具体训练人员、分支用途和保留人数以 [模型手册](model.md) 及冻结审计为准，不把来源总人数相加当作最终训练人数。

## 来源和实际角色

| 来源与作者入口 | 作者发布规模 | 本交付中的角色 |
|---|---|---|
| [HAPT](https://archive.ics.uci.edu/dataset/341/smartphone-based-recognition-of-human-activities-and-postural-transitions)；Reyes-Ortiz 等 | 30 人；腰部手机六轴，50 Hz，6 种基本动作及 6 种姿态转换 | SB1 运动头及 E7 分支；官方原始 70/30 人员划分不能替代本项目各分支清单 |
| [HARTH](https://archive.ics.uci.edu/dataset/779/harth)；Logacjov、Bach 等 | 原始 UCI 版 22 人；腰背与大腿三轴、50 Hz，每人约 2 小时 | SB1 姿态及运动头、E7；项目使用的视图不等于作者所有部位 |
| [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth-dataset)；Banos 等 | 10 人、12 种活动，50 Hz | SB1/E7；胸部加速度可用，胸部文件没有可直接等同使用的陀螺仪三轴 |
| [MotionSense](https://github.com/mmalekzadeh/motion-sense)；Malekzadeh 等 | 24 人、6 种活动，手机传感器 | SB1 运动二分类补充，未将其全部作为胸前姿态训练 |
| [Opportunity](https://archive.ics.uci.edu/dataset/226/opportunity-activity-recognition)；Roggen 等 | 4 人，多传感器日常动作 | SB1/E7；按冻结通道、标签及人员使用 |
| [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2-physical-activity-monitoring)；Reiss、Stricker | 9 人，18 种活动；胸、腕、踝 IMU，100 Hz | SB1/E7 采用胸部视图 |
| [WISDM 2019](https://archive.ics.uci.edu/dataset/507/wisdm-smartphone-and-smartwatch-activity-and-biometrics-dataset)；Weiss | 51 人、18 种活动，每种约 3 分钟；手机及手表 | SB1 运动头使用手机子集；不能把两个设备算作两名参与者 |
| [FORTH-TRACE](https://zenodo.org/records/841301)；Karagiannaki、Panousopoulou、Tsakalides | 15 人、5 个部位、16 种活动；51.2 Hz 九轴 | R15/E1/E5 中使用时钟合格的躯干加速度范围；未把未经确认的角速度单位作为 SI 输入 |
| [UNIPD-BPE](https://doi.org/10.17605/OSF.IO/YJ9Q4)；Guidolin、Menegatti、Reggiani | 15 名参与者；同步 RGB-D 与全身惯性系统，多人场景不增加独立人数 | R15/E1 使用实测 T8 加速度；作者运动学辅助生成弱标签，非自采设备信号、非合成 IMU |
| [ElderHAR](https://zenodo.org/records/17191997)；作者记录 17191997 | 54 位老年参与者，腰背六轴、50 Hz；作者报告约 15 小时、260 万以上样本 | E1/E5 训练 43 人、保留 11 人；不能据此宣称完成本设备老年人验证 |
| [Mendeley ADL v2](https://data.mendeley.com/datasets/wjpbtgdyzm/2)；Ruzzon、Carfi、Ishikawa、Mastrogiovanni、Murakami | 10 人、9 种日常动作；原始采样周期 30 ms | E5 使用 8 人；SitDown/StandUp 正例及易混淆负例，按冻结适配器重采样 |
| [OxWalk](https://ora.ox.ac.uk/objects/uuid:19d3cb34-e2b3-4177-91b6-1bad0e0163e7)；Small 等 | 39 人，腕与髋部 25/100 Hz 加速度、视频步事件参考 | GP2：24 人开发及拟合，15 人保留评价；不是活动分类的 9 人设备参考 |
| [RealWorld HAR](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/)；Sztyler、Stuckenschmidt | 15 人、7 个佩戴位置 | E7 公共保留／方案选择参考，本交付相关分支没有以其拟合权重 |
| [UniMiB SHAR](https://arxiv.org/abs/1611.07688)；Micucci、Mobilio、Napoletano | 30 人，11,771 个动作／跌倒短片段 | E7 公共保留／方向诊断参考；其切片不能冒充全天连续测试 |

表内人数和总时长只描述相应作者版本；无完整采纳时长审计的来源不填“训练总小时数”。SB1 最终训练人员并集为 108，E7 各分支又有交叠，详见 [SB1 审计](../provenance/sb1-training-audit.json) 和 [E7 审计](../provenance/e7-training-sources.json)。研究工作区的其他新数据源、新实验不自动进入本次冻结模型。

## 授权证据与范围

- HAPT、HARTH、MHEALTH、Opportunity、PAMAP2、WISDM 的 UCI 当前来源页标示 CC BY 4.0；保留作者、数据集入口和引用。原始数据不随包重新分发。
- MotionSense 作者仓库采用 MIT；其原始许可证随 `notices/activity/THIRD_PARTY_LICENSES/MotionSense-source-MIT.txt` 保留。
- UNIPD-BPE 的[作者数据说明](https://doi.org/10.3390/data7060079)明确写出数据许可 CC0，区别于论文自身 CC BY。作者 OSF 节点是 `yj9q4`，不是任意其他 UniPD 数据。
- ElderHAR 作者 Zenodo API 的许可为 CC BY 4.0；Mendeley ADL v2 作者页为 CC BY 4.0；OxWalk 的 Oxford 记录为 CC BY。GP2 保留其作者引用。
- RealWorld、UniMiB 在本交付的相应分支作为研究评价来源；没有再分发其原始信号，也不以论文的开放访问许可代替数据授权。此处不作未来商用训练准入承诺。
- 训练／评价数据的许可与代码依赖许可证不同；MIT 自有代码不取消来源条件。模型文件的最终适用范围见 [MODEL_LICENSES.md](../MODEL_LICENSES.md)。

## FORTH 标记的追溯

冻结 R15 元数据保存 `research_only=true`；历史 `AF1_FS1` 准入协议还明确保存 `commercial_weight_release_allowed=false`。这些是当时研究工程的准入限制，保留用于追溯，不修改旧文件。

本次重新读取作者 [Zenodo API](https://zenodo.org/api/records/841301)，得到 `license.id=cc-by-sa-4.0`。作者 GitHub README 的 LICENCE 段要求引用论文，未在该段另列非商业限制。**CC BY-SA 4.0 允许商用，但要求对受其约束的改编材料保留相同许可及署名**；不能把它写成 CC BY-NC，也不能仅凭允许商用就将其改成 MIT。[许可说明](https://creativecommons.org/licenses/by-sa/4.0/)

R15、E1、E5 都在训练来源清单中使用 FORTH；因此授权核对覆盖三个权重，不能只处理名称上带 research_only 的 R15。关于训练权重是否构成该数据的法律衍生作品，本文不作确定判断。候选包的具体发布授权仍以 MODEL_LICENSES.md 及发布核对记录为准。

本说明是来源证据及交付范围记录，不替代作者授权，也不代表来源作者认可本产品。日期、地址和响应摘要见 [来源核对记录](../provenance/source-license-review.json)。
