import numpy as np
import pytest
from imu_inference import InferenceEngine, InputBatch
from imu_inference.steps import StepStream

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def engine():
    return InferenceEngine(threads=2)


def fixture(n=9007):
    t = np.arange(n)/25
    x = np.column_stack((.3*np.sin(t*2*np.pi*1.8), .15*np.cos(t*2*np.pi*1.8),
                         9.80665+1.7*np.sin(t*2*np.pi*1.8)))
    x[(t % 100) > 75] = [0, 0, 9.80665]
    return x.astype(np.float32)


def test_gp2_matches_frozen_full_api_including_masks(engine):
    x = fixture()
    valid = np.ones(len(x), bool)
    excluded = np.zeros(len(x), bool)
    valid[1500:1550] = False
    valid[1600:1630] = False
    excluded[2450:2520] = True
    expected = engine.backend.counter.predict(x, valid=valid, excluded=excluded, include_candidates=True)
    stream = StepStream(engine.backend)
    events = []
    for a in range(0, len(x), 137):
        b = min(len(x), a+137)
        events.extend(stream.feed(InputBatch.create(x[a:b], start=a,
                                                    valid=valid[a:b], excluded=excluded[a:b], angular_velocity=np.zeros((len(x[a:b]), 3), np.float32))))
    events.extend(stream.finish())
    indices = [i for e in events if e["kind"] == "step_segment" for i in e["event_samples"]]
    assert np.array_equal(np.array(indices)/25, expected["candidate_event_times_s"])
    assert sum(e["end"]-e["start"] for e in events
               if e["kind"] == "step_segment" and e["analyzed"]) == round(expected["analyzed_s"]*25)
    bouts = [
        dict(start_s=e["first"]/25, end_s=e["last"]/25, estimated_steps=e["estimated_steps"],
             estimated_cadence_spm=e["estimated_cadence_spm"])
        for e in events if e["kind"] == "step_bout"
    ]
    assert len(bouts) == len(expected["bouts"])
    for got, want in zip(bouts, expected["bouts"], strict=True):
        assert got["estimated_steps"] == want["estimated_steps"]
        assert got["start_s"] == want["start_s"] and got["end_s"] == want["end_s"]
        if want["estimated_cadence_spm"] is not None:
            assert got["estimated_cadence_spm"] == pytest.approx(want["estimated_cadence_spm"], abs=1e-10)


def test_activity_matches_independent_frozen_core_calls(engine):
    x = fixture(3001)
    result = engine.predict(x, angular_velocity=np.zeros((len(x), 3), np.float32))
    reconstructed = []
    for a in range(0, len(x), 1500):
        b = min(len(x), a+1500)
        lo, hi = max(0, a-750), min(len(x), b+750)
        values = np.column_stack((x[lo:hi], np.zeros((hi-lo, 3), np.float32)))
        legacy = engine.backend.activity.predict(values)
        tag = np.empty(hi-lo, dtype=object)
        for row in legacy["timeline"]:
            tag[round(row["start_s"]*25):round(row["end_s"]*25)] = row["label"]
        reconstructed.extend(tag[a-lo:b-lo].tolist())
    got = [row["label"] for e in result["events"] if e["kind"] == "activity_core"
           for row in e["timeline"] for _ in range(row["end"]-row["start"])]
    assert got == reconstructed
