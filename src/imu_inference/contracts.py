"""Small versioned input contract shared by batch and streaming inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import base64
import hashlib
import json
import numpy as np

RATE_HZ = 25
LABELS = ("lying", "sitting", "standing", "walking_running")
PROFILE = "sb1-fc1-l60-h30-gp2-v1"
WHOLE_PROFILE = "e7-hard-positive-smoothed-2026102531-v1"
MAX_RECORD_SAMPLES = 25 * 3600 * RATE_HZ
PROFILES = {"incremental": PROFILE, "whole_record": WHOLE_PROFILE}


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pack_array(array):
    return base64.b64encode(array.tobytes()).decode("ascii")


def unpack_array(value, dtype, *, max_elements):
    if not isinstance(value, str) or len(value) > max_elements*np.dtype(dtype).itemsize*2+8:
        raise ValueError("checkpoint array exceeds bounded size")
    raw = base64.b64decode(value, validate=True)
    result = np.frombuffer(raw, dtype=dtype).copy()
    if len(result) > max_elements:
        raise ValueError("checkpoint array exceeds bounded size")
    return result


@dataclass(frozen=True)
class PipelineConfig:
    """Only validated settings are accepted; threads are execution settings."""
    steps_enabled: bool = True
    mode: str = "incremental"
    sign_adaptation: bool = False
    profile: str | None = None

    def __post_init__(self):
        if (type(self.steps_enabled) is not bool or type(self.sign_adaptation) is not bool
                or self.mode not in PROFILES
                or self.profile not in (None, PROFILES[self.mode])):
            raise ValueError("unsupported pipeline configuration")
        if self.sign_adaptation and self.mode != "whole_record":
            raise ValueError("experimental sign adaptation requires whole_record mode")
        object.__setattr__(self, "profile", PROFILES[self.mode])

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self))).hexdigest()


@dataclass(frozen=True)
class InputBatch:
    """Consecutive positions on a recording's immutable 25 Hz sample grid."""
    start: int
    acceleration: np.ndarray
    valid: np.ndarray
    excluded: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self):
        if type(self.start) is not int or self.start < 0:
            raise ValueError("start must be a nonnegative integer sample index")
        x = np.array(self.acceleration, dtype="<f4", copy=True)
        if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
            raise ValueError("finite nonempty N x 3 acceleration in m/s2 required")
        object.__setattr__(self, "acceleration", x)
        for key in ("valid", "excluded"):
            m = np.array(getattr(self, key), copy=True)
            if m.dtype != bool or m.shape != (len(x),):
                raise ValueError(f"{key} must be a boolean array of length N")
            m.flags.writeable = False
            object.__setattr__(self, key, m)
        g = np.array(self.angular_velocity, dtype="<f4", copy=True)
        if g.shape != x.shape or not np.isfinite(g).all():
            raise ValueError("angular_velocity must be finite N x 3 in rad/s; six axes are required")
        g.flags.writeable = False
        object.__setattr__(self, "angular_velocity", g)
        x.flags.writeable = False

    @classmethod
    def create(cls, acceleration, *, angular_velocity, start=0, valid=None, excluded=None):
        n = len(acceleration)
        return cls(start, acceleration, np.ones(n, bool) if valid is None else valid,
                   np.zeros(n, bool) if excluded is None else excluded, angular_velocity)

    @property
    def end(self):
        return self.start + len(self.acceleration)

    @property
    def values(self):
        return np.column_stack((self.acceleration, self.angular_velocity))

    def slice(self, start, end):
        if not self.start <= start < end <= self.end:
            raise ValueError("slice outside batch")
        sl = slice(start-self.start, end-self.start)
        return InputBatch(start, self.acceleration[sl], self.valid[sl], self.excluded[sl],
                          self.angular_velocity[sl])

    @property
    def digest(self):
        h = hashlib.sha256(canonical_json({"schema": 2, "start": self.start,
                                         "count": len(self.acceleration),
                                         "gyro_present": self.angular_velocity is not None}))
        for a in (self.acceleration, self.valid, self.excluded):
            h.update(a.tobytes())
        if self.angular_velocity is not None:
            h.update(self.angular_velocity.tobytes())
        return h.hexdigest()


def capabilities():
    from . import __version__
    return {
        "schema": "imu.capabilities.v2", "library_version": __version__, "profile": PROFILE,
        "default_mode": "incremental",
        "profiles": {
            "incremental": {"profile": PROFILE, "experimental": False, "streaming": True},
            "whole_record": {"profile": WHOLE_PROFILE, "experimental": True, "streaming": False,
                             "max_samples": MAX_RECORD_SAMPLES, "max_duration_hours": 25,
                             "sign_adaptation": {"experimental": True, "default_enabled": False},
                             "acceptance": "EXPERIMENTAL_ES1_80_GATE_NOT_PASSED"},
        },
        "input": {"sample_rate_hz": RATE_HZ, "acceleration_unit": "m/s2",
                  "gravity": "retained", "angular_velocity": "required_rad/s", "axes": 6,
                  "schema": "imu.input.v2",
                  "placement": "fixed_chest", "missing": "explicit_masks"},
        "activity": {"model": "stable-bags-sb1-v1/FINAL", "labels": list(LABELS),
                     "core_samples": 1500, "left_context_samples": 750,
                     "right_context_samples": 750, "stride_samples": 1500,
                     "confidence_calibrated": False,
                     "acceptance": "FROZEN_LIMITED_RELEASE_REPEAT_GATE_NOT_PASSED"},
        "steps": {"algorithm": "GP2-G3-B2-frozen", "experimental": True, "default_enabled": True,
                  "validation_scope": "OxWalk hip; chest accuracy unmeasured",
                  "gate_samples": 25, "detector_core_samples": 750,
                  "detector_context_samples": 50, "gate_future_samples": 100},
        "state_schema": "imu.stream.state.v1",
    }
