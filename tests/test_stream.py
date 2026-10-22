import json
import numpy as np
import pytest
from imu_inference import InferenceEngine, InputBatch, PipelineConfig
from imu_inference.steps import StepStream


def normalized(events):
    return sorted(events, key=lambda e: (e["kind"], e.get("start", e.get("first")), e.get("end", 0)))


@pytest.mark.parametrize("size", [1, 17, 25, 137, 2249, 4000])
def test_partition_and_restart_are_invariant(backend, signal, size):
    x, v, e = signal
    engine = InferenceEngine(backend=backend)
    expected = engine.predict(x, valid=v, excluded=e, angular_velocity=np.zeros((len(x), 3), np.float32))
    stream = engine.stream()
    events = []
    for a in range(0, len(x), size):
        b = min(len(x), a+size)
        events.extend(stream.append(InputBatch.create(x[a:b], start=a, valid=v[a:b], excluded=e[a:b], angular_velocity=np.zeros((len(x[a:b]), 3), np.float32))))
        if a < 4500 <= b:
            checkpoint = json.loads(json.dumps(stream.checkpoint(), allow_nan=False))
            stream = engine.restore(checkpoint)
        assert len(stream.activity.x) <= 3000
        assert len(stream.steps.x) <= 1100
    events.extend(stream.finish())
    assert normalized(events) == normalized(expected["events"])
    assert stream.statistics.result() == expected["statistics"]
    assert stream.finish() == []
    assert len(stream.activity.x) == len(stream.steps.x) == 0


def test_gate_region_coverage_and_step_ownership(backend, signal):
    x, valid, excluded = signal
    stream = StepStream(backend)
    events = []
    for a in range(0, len(x), 250):
        b = min(len(x), a+250)
        events.extend(stream.feed(InputBatch.create(x[a:b], start=a,
                                                   valid=valid[a:b], excluded=excluded[a:b], angular_velocity=np.zeros((len(x[a:b]), 3), np.float32))))
    events.extend(stream.finish())
    coverage = np.zeros(len(x), int)
    found = []
    for row in events:
        if row["kind"] == "step_segment":
            coverage[row["start"]:row["end"]] += 1
            found.extend(row["event_samples"])
    np.testing.assert_array_equal(coverage, valid & ~excluded)
    assert found == sorted(set(found))
    assert all(valid[i] and not excluded[i] for i in found)
    assert sum(r["estimated_steps"] for r in events if r["kind"] == "step_bout") == len(found)


@pytest.mark.parametrize("n", [1, 24, 25, 99, 100, 1499, 1500, 1501, 2249, 2250, 3000])
def test_short_tails_and_first_readiness(backend, n):
    engine = InferenceEngine(backend=backend)
    stream = engine.stream()
    x = np.column_stack((np.arange(n), np.zeros(n), np.ones(n)*9.8))
    events = stream.append(InputBatch.create(x, angular_velocity=np.zeros((len(x), 3), np.float32)))
    assert any(e["kind"] == "activity_core" for e in events) == (n >= 2250)
    stream.finish()
    s = stream.statistics.result()
    assert s["through_sample"] == n
    assert sum(s["duration_s"].values()) == n/25
    assert s["steps"]["estimated_step_count"] is None if n < 100 else True


def test_checkpoint_rejects_mutation_and_versions(backend):
    engine = InferenceEngine(backend=backend)
    stream = engine.stream()
    stream.append(InputBatch.create([[0, 0, 9.8]]*500, angular_velocity=np.zeros((len([[0, 0, 9.8]]*500), 3), np.float32)))
    checkpoint = stream.checkpoint()
    wrong = InferenceEngine(PipelineConfig(steps_enabled=False), backend=backend)
    with pytest.raises(ValueError, match="version"):
        wrong.restore(checkpoint)
    checkpoint["state"]["received"] += 1
    with pytest.raises(ValueError, match="checksum"):
        engine.restore(checkpoint)


def test_explicit_input_and_disabled_steps(backend):
    engine = InferenceEngine(PipelineConfig(steps_enabled=False), backend=backend)
    stream = engine.stream()
    with pytest.raises(ValueError, match="contiguous"):
        stream.append(InputBatch.create([[0, 0, 9.8]], start=1, angular_velocity=np.zeros((len([[0, 0, 9.8]]), 3), np.float32)))
    with pytest.raises(ValueError, match="finite"):
        InputBatch.create([[float("nan"), 0, 0]], angular_velocity=np.zeros((len([[float("nan"), 0, 0]]), 3), np.float32))
    with pytest.raises(ValueError, match="boolean"):
        InputBatch.create([[0, 0, 0]], valid=[1], angular_velocity=np.zeros((len([[0, 0, 0]]), 3), np.float32))
    result = engine.predict([[0, 0, 9.8]]*100, angular_velocity=np.zeros((len([[0, 0, 9.8]]*100), 3), np.float32))
    assert result["statistics"]["steps"]["enabled"] is False
    assert result["statistics"]["steps"]["estimated_step_count"] is None


@pytest.mark.long
def test_hour_state_size_is_bounded(backend):
    stream = InferenceEngine(backend=backend).stream()
    largest = 0
    for a in range(0, 25*3600, 1500):
        x = np.column_stack((np.arange(a, a+1500), np.zeros(1500), np.ones(1500)*9.8))
        stream.append(InputBatch.create(x, start=a, angular_velocity=np.zeros((len(x), 3), np.float32)))
        largest = max(largest, len(json.dumps(stream.checkpoint())))
    assert largest < 160_000
    stream.finish()
    assert stream.statistics.samples == 25*3600
