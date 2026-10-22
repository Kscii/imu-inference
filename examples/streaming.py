"""Compare a synthetic batch with checkpointed streaming; no network access."""
import json
import numpy as np
from imu_inference import InferenceEngine, InputBatch

engine = InferenceEngine(threads=2)
t = np.arange(3001)/25
x = np.column_stack((.3*np.sin(t*2*np.pi*1.8), .15*np.cos(t*2*np.pi*1.8),
                     9.80665+1.7*np.sin(t*2*np.pi*1.8))).astype(np.float32)
expected = engine.predict(x, angular_velocity=np.zeros((len(x), 3), np.float32))
stream = engine.stream()
events = []
for start in range(0, len(x), 137):
    events.extend(stream.append(InputBatch.create(x[start:start+137], start=start, angular_velocity=np.zeros((len(x[start:start+137]), 3), np.float32))))
    # In production, save this atomically with the emitted events before acknowledging progress.
    checkpoint = json.loads(json.dumps(stream.checkpoint()))
    stream = engine.restore(checkpoint)
events.extend(stream.finish())
assert stream.statistics.result() == expected["statistics"]
print(json.dumps(stream.statistics.result(), indent=2))
