# 模型文件授权

适用版本：imu-inference 1.0.1。发布许可决定日期：2026-10-23；权利人：Kscii。

## 文件与许可范围

以下三个训练权重由 Kscii 按 **Creative Commons Attribution-ShareAlike 4.0 International（CC BY-SA 4.0）** 授权，限于 Kscii 有权授予的版权及类似权利。根目录 MIT 不适用于这三个文件。

| 模型 | 仓库路径 | SHA256 |
|---|---|---|
| R15 | `src/imu_inference/assets/e7/r15.ubj` | `b8d8a183f3f2b960771b540e1437fde6a8438f1ececb50ac00fb73a63c294201` |
| E1 | `src/imu_inference/assets/e7/head.ubj` | `b9596e53ce786ee1f9630ce23af461e66524abd26095cc5c385756c4b598ab95` |
| E5 | `src/imu_inference/assets/e7/event.ubj` | `af66d3f248e16caf49bad472bd05cc1df5348005564d96cb917011bc232ffd3f` |

wheel 安装后对应 `imu_inference/assets/e7/` 下的同名文件。完整许可：[LICENSES/CC-BY-SA-4.0.txt](LICENSES/CC-BY-SA-4.0.txt)；[官方许可](https://creativecommons.org/licenses/by-sa/4.0/)。完整文本和本说明随 wheel 的 `imu_inference-1.0.1.dist-info/licenses/` 以及服务镜像保留。

其余可授权的自有 SB1、R3、GP2 模型材料及自有代码、文档采用根目录 [MIT](LICENSE)。第三方代码、模型材料与原版权声明仍遵守各自条款，参见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。此说明不将原始数据或整个依赖环境统一改为 MIT 或 CC BY-SA。

## 来源署名与转换说明

- **训练模型发布者：Kscii。** R15、E1、E5 均使用 FORTH-TRACE 的合格躯干加速度范围，也使用其他来源；各分支来源、人员划分及训练方法见 [模型手册](docs/model.md)、[训练与算法](docs/training-and-algorithms.md) 和 `provenance/e7-training-sources.json`。
- **FORTH-TRACE 数据作者：Katerina Karagiannaki、Athanasia Panousopoulou、Panagiotis Tsakalides。** 数据标题：*The FORTH-TRACE dataset for human activity recognition of simple activities and postural transitions using a Body Area Network*，v1，2016。[作者发布记录及引用](https://doi.org/10.5281/zenodo.841301)，[数据许可 CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。
- **作者 README 要求的论文引用：** Katerina Karagiannaki, Athanasia Panousopoulou, Panagiotis Tsakalides. *A Benchmark Study on Feature Selection for Human Activity Recognition*. ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp), ACM, 2016.
- **转换内容：** 本项目对合格信号范围执行通道选择、时钟检查、重采样、单位／坐标规范化、标签映射、特征提取并训练 XGBoost 分支，交付学习到的权重。原始数据不随包分发。本次 1.0.1 只明确许可与文档，三个权重的字节与原冻结模型相同。
- 数据作者没有参与本产品授权决定，不代表其认可本产品、指标或商业用途；保留数据及模型相应的免责声明。

## 使用与再分发

CC BY-SA 4.0 允许商业使用、复制、修改和再分发。分享这些权重时，按完整许可保留署名、来源、许可及免责声明，标示修改；分享受许可约束的改编材料时，遵守 ShareAlike 条件，不对所授权权利添加额外限制。许可本身包含无担保及责任限制条款。许可只及于权利人有权许可的范围，不替代其他依法适用的权利。

三个权重与 MIT 服务代码共同打包，各自的许可范围仍按上表及第三方声明区分。这不是向服务接口输入的客户数据或所有业务代码自动适用 CC BY-SA 的声明。

## 历史记录与本次决定

旧 R15 元数据的 `research_only=true` 和旧准入协议的 `commercial_weight_release_allowed=false` 原样保留，用于追溯当时的研究准入状态。本次重新核对作者来源后，权利人确认对上述三个权重主动采用 CC BY-SA 4.0；当前发布许可由本文件及 [新发布记录](provenance/source-license-review.json) 明确。

这是一项主动的分项许可决定，不是对所有利用 CC BY-SA 数据训练的权重都依法构成衍生作品的判断。也不追溯撤回任何此前已经有效授予的权利，不修改旧标签、历史包或原始审计记录。
