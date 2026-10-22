import numpy as np
import pytest


class FakeBackend:
    """Coordinate-aware deterministic backend for scheduling and recovery tests."""
    def whole_record(self, values, valid, baseline, sign_adaptation):
        return baseline.argmax(1).astype(np.int8), {"enabled": sign_adaptation, "applied": False}

    def activity_probabilities(self, values, valid):
        scores = np.column_stack((np.abs(values[:, 1])+1, np.abs(values[:, 2])+2,
                                  np.ones(len(values))*3, np.ones(len(values))*4))
        return (scores/scores.sum(1)[:, None]).astype(np.float32)

    def step_gate(self, values, center):
        coordinate = int(values[min(center, len(values)-1), 0])
        return .8 if (coordinate//25) % 100 < 80 else .2

    def step_contacts(self, values):
        return np.flatnonzero(values[:, 0].astype(int) % 17 == 0), "OK"


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def signal():
    x = np.column_stack((np.arange(9007), np.sin(np.arange(9007)/12),
                         np.ones(9007)*9.80665)).astype(np.float32)
    valid = np.ones(len(x), bool)
    excluded = np.zeros(len(x), bool)
    valid[875:940] = False
    valid[960:985] = False
    valid[1100:1110] = False
    excluded[2300:2400] = True
    return x, valid, excluded
