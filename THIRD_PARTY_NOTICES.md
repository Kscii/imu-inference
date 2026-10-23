# 第三方声明与授权范围

根目录 LICENSE 为 Kscii 自有代码和配套文档的 MIT 授权。模型文件的具体授权范围见 [MODEL_LICENSES.md](MODEL_LICENSES.md)，不能将根许可证解释为覆盖所有上游数据和组件。

- 继承组件的 `IMU Fall Benchmark contributors` 版权声明原样保留。
- `src/imu_inference/notices/activity/THIRD_PARTY_LICENSES/` 保存 NumPy、SciPy、PyTorch、XGBoost 等原始声明和清单。
- `src/imu_inference/_vendor/steps/THIRD_PARTY_LICENSES/` 保存 mobgap 的 Apache 2.0 声明；GP2 来源见该目录上一级的 DATA_CITATION.txt。
- MotionSense 作者的 MIT 声明随 activity notices 保留。
- FORTH-TRACE 作者 Katerina Karagiannaki、Athanasia Panousopoulou、Panagiotis Tsakalides 的数据记录为 https://doi.org/10.5281/zenodo.841301，许可 CC BY-SA 4.0。使用该来源训练的 R15、E1、E5 权重由 Kscii 按 CC BY-SA 4.0 发布；转换说明、引用和文件摘要见 MODEL_LICENSES.md，完整许可见 LICENSES/CC-BY-SA-4.0.txt。
- 公共数据及其作者引用、授权核对见 [数据来源](docs/sources-and-licenses.md)。本交付不再分发原始公共或自采传感信号。
- 依赖的原版权、许可和免责声明保留其效力；数据来源或算法作者不因被引用而认可本产品或其准确率。
