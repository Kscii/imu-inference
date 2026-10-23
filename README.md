# IMU 推理库

**1.0.1 · Linux x86_64 · Python 3.12 · CPU**

包含 SB1 分块分类、E7 整段分类和 GP2 实验计步与统计。**录制结束后分析推荐 E7 整段处理**；需持续输出时使用 SB1 分块处理。当前同一研究参考的 ES1 区间 accuracy 分别为 75.49% / 72.67%，主指标 macro recall 为 74.09% / 67.72%；改善有适用范围，详见[模型与验证](docs/model.md)。权重随 wheel 安装，运行时不下载模型。

文档修订日期：2026-10-24。只更新解释与建议，发布的 v1.0.1 wheel、标签、模型和接口行为保持冻结。模式描述仍保留 `whole_record.experimental=true`，不因文档推荐而自动改变该字段。

## 安装与使用

```bash
uv sync --frozen --extra runtime
uv run --frozen --extra runtime pytest -q
uv build --wheel --sdist
```

服务使用固定 wheel 与 SHA256。独立安装还需 uv.lock 或 requirements-runtime.lock 及 CPU Torch 来源，不能只固定 wheel；离线运行优先使用交付的 Docker 镜像。

```python
import numpy as np
from imu_inference import InferenceEngine, PipelineConfig, InputBatch

engine = InferenceEngine(PipelineConfig(mode="incremental", steps_enabled=True), threads=2)
# 仅为合成接口示例；实际需上传测得的六轴。
acc = np.tile([0.0, 0.0, 9.80665], (3000, 1))
gyro = np.zeros_like(acc)
result = engine.predict(acc, angular_velocity=gyro)
print(result["statistics"])

stream = engine.stream()
events = stream.append(InputBatch.create(acc[:1500], angular_velocity=gyro[:1500]))
checkpoint = stream.checkpoint()
stream = engine.restore(checkpoint)
events += stream.append(InputBatch.create(acc[1500:], angular_velocity=gyro[1500:], start=1500))
events += stream.finish()
```

一个进程加载一次 engine，每条记录独立 stream；同一 stream 不允许并发。库只收有序、不重复的位置，网络乱序、幂等、持久化由服务负责。保存 checkpoint 后恢复的模型与配置须匹配。

输入加速度单位 m/s² 且保留重力，角速度 rad/s，25 Hz。valid=false 标记缺失，excluded=true 标记排除，保留时间位置。输出事件范围为整数采样位置 `[start,end)`；classification 与 steps 有独立进度，结束用 finish 完成尾段。

whole_record 用 PipelineConfig(mode="whole_record") 和完整 predict 输入，最长 25 小时，不支持流式拼接等价。纵向符号适配默认关闭。无法分析的计步保持 null，不等于零步。模型置信分数不是经过客户校准的正确概率。

当前四类标签合并走路和跑步，未接入独立走跑二分类。可按需另行提供研究候选；公司设备跑步样本不足且现有验证未通过，不作为即用功能，见[走跑候选及验证](docs/walk-run.md)。

## 交付与更新

src/ 包含实际模型、算法、第三方实现及声明；tests/ 是自动测试；scripts/ 是研究复算与性能工具；provenance/ 保存来源与证据。源码来源见 [清单](provenance/delivery-origin.json)。

本版本固定权重推理可复现，不包含完整训练工程或原始数据。发布通过 GitHub Release，部署禁止原地改权重；更新形成新 wheel、摘要、回归记录和固定版本，由服务仓库显式接入。

1.0.0 的历史构建与测试记录保持原样。1.0.1 的模型等价证据见 [版本回归](provenance/1.0.1/patch-equivalence.json)，新制品构建与完整测试记录随对应 Release 提供。

自有代码署名 Kscii，采用 [MIT](LICENSE)；模型授权范围单列于 [MODEL_LICENSES.md](MODEL_LICENSES.md)，第三方条款见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。数据规模、官方入口及许可核对见 [来源说明](docs/sources-and-licenses.md)。
