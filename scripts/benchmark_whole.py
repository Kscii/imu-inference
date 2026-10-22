"""Capacity only: deterministic synthetic six-axis data, not accuracy evidence."""

import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
from imu_inference import InferenceEngine, PipelineConfig

p = argparse.ArgumentParser()
p.add_argument("--hours", type=float, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--steps", action="store_true")
a = p.parse_args()
n = round(a.hours * 3600 * 25)
t = np.arange(n, dtype=np.float64) / 25
x = np.column_stack([0.15 * np.sin(t * 9), 9.80665 + 0.6 * np.sin(t * 9), 0.1 * np.cos(t * 9)]).astype(
    np.float32
)
g = np.column_stack([0.02 * np.sin(t * 9), 0.01 * np.cos(t * 9), 0.015 * np.sin(t * 9)]).astype(np.float32)
del t
start = time.monotonic()
engine = InferenceEngine(PipelineConfig(mode="whole_record", steps_enabled=a.steps), threads=a.threads)
loaded = time.monotonic()
r = engine.predict(x, angular_velocity=g)
end = time.monotonic()
out = dict(
    schema="imu.whole.capacity.v1",
    input="synthetic_continuous_six_axis_NOT_ACCURACY",
    hours=a.hours,
    samples=n,
    threads=a.threads,
    steps=a.steps,
    load_seconds=loaded - start,
    inference_seconds=end - loaded,
    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    provenance=r["provenance"],
    statistics=r["statistics"],
    event_count=len(r["events"]),
)
a.output.write_text(json.dumps(out, indent=2))
print(json.dumps({k: v for k, v in out.items() if k not in ("provenance", "statistics")}), flush=True)
