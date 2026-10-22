"""FP32 neural export probe only. Does not change the shipped inference backend."""

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from imu_inference.engine import FrozenBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    backend = FrozenBackend(threads=2, steps_enabled=False)
    results = []
    for index, (net, arm) in enumerate(backend.activity.model["neural"]):
        path = args.output / f"neural-{index:02}-{arm}.onnx"
        example = torch.zeros((1, 8, 633 if arm == "G" else 636), dtype=torch.float32)
        torch.onnx.export(
            net,
            example,
            str(path),
            opset_version=17,
            dynamo=False,
            input_names=["signal"],
            output_names=["logits"],
            dynamic_axes={"signal": {0: "batch", 2: "time"}, "logits": {0: "batch", 2: "time"}},
        )
        onnx.checker.check_model(str(path))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        errors = []
        times = []
        mismatches = 0
        for batch, length in [(1, 633), (3, 401), (16, 636)]:
            x = (
                np.random.default_rng(1000 + index + length)
                .normal(0, 0.4, (batch, 8, length))
                .astype(np.float32)
            )
            with torch.inference_mode():
                expected = net(torch.from_numpy(x)).numpy()
            actual = session.run(None, {"signal": x})[0]
            errors.append(float(np.max(np.abs(expected - actual))))
            mismatches += int(np.count_nonzero(expected.argmax(1) != actual.argmax(1)))
            # Warm both backends, then alternate to reduce order effects.
            native = []
            converted = []
            for _ in range(3):
                t = time.perf_counter()
                with torch.inference_mode():
                    net(torch.from_numpy(x))
                native.append(time.perf_counter() - t)
                t = time.perf_counter()
                session.run(None, {"signal": x})
                converted.append(time.perf_counter() - t)
            times.append(
                dict(
                    batch=batch,
                    length=length,
                    native_s=float(np.median(native)),
                    onnx_s=float(np.median(converted)),
                )
            )
        results.append(
            dict(
                member=index,
                arm=arm,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size,
                max_abs_error=max(errors),
                argmax_mismatch=mismatches,
                timing=times,
            )
        )
        report = dict(
            schema="imu.onnx.probe.v1",
            scope="FP32 neural members on synthetic representations only",
            promoted=False,
            missing_gates=["full-pipeline ES1", "per-class", "frame", "duration", "GP2"],
            versions={"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
            members=results,
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(index, arm, max(errors), mismatches, flush=True)


if __name__ == "__main__":
    main()
