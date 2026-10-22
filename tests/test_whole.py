import numpy as np
import pytest
from imu_inference import InferenceEngine, InputBatch, PipelineConfig, capabilities
from imu_inference.contracts import MAX_RECORD_SAMPLES


def test_six_axis_is_required_and_modes_are_explicit(backend):
    with pytest.raises(TypeError):
        InputBatch.create([[0, 9.8, 0]])
    with pytest.raises(ValueError):
        InputBatch.create([[0, 9.8, 0]], angular_velocity=None)
    with pytest.raises(ValueError):
        PipelineConfig(sign_adaptation=True)
    engine = InferenceEngine(PipelineConfig(mode="whole_record", steps_enabled=False), backend=backend)
    with pytest.raises(ValueError, match="complete recording"):
        engine.stream()
    x = np.tile([0, 9.8, 0], (1601, 1))
    gyro = np.zeros_like(x)
    valid = np.ones(len(x), bool)
    valid[125:150] = False
    excluded = np.zeros(len(x), bool)
    excluded[150:200] = True
    result = engine.predict(x, angular_velocity=gyro, valid=valid, excluded=excluded)
    assert result["statistics"]["invalid_s"] == 1
    assert result["statistics"]["excluded_s"] == 2
    assert result["statistics"]["finished"]
    assert result["provenance"]["config"]["mode"] == "whole_record"
    assert MAX_RECORD_SAMPLES == 2250000
    assert capabilities()["schema"] == "imu.capabilities.v2"


@pytest.mark.model
def test_real_e7_empty_observation_and_sign_fallback():
    from imu_inference.e7 import E7Model

    model = E7Model(1)
    x = np.tile([0, 9.8, 0, 0, 0, 0], (100, 1)).astype(np.float32)
    valid = np.zeros(len(x), bool)
    p = np.full((len(x), 4), 0.25, np.float32)
    tags, diag = model.predict(x, valid, p, sign_adaptation=True)
    assert np.all(tags == -1) and not diag["accepted"] and not diag["applied"]
    valid[:] = True
    tags, diag = model.predict(x, valid, p, sign_adaptation=True)
    assert np.isin(tags, [0, 1, 2, 3]).all() and diag["sign"] == 1 and not diag["accepted"]
