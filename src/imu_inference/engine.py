"""Public CPU engine and checkpointable, bounded streams."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib
import json

from . import __version__
from .activity import ActivityStream
from .contracts import InputBatch, PipelineConfig, canonical_json, capabilities, MAX_RECORD_SAMPLES, LABELS
from .statistics import Statistics
from .steps import StepStream


class FrozenBackend:
    def __init__(self, threads=2, steps_enabled=True):
        if type(threads) is not int or not 1 <= threads <= 32:
            raise ValueError("threads must be between 1 and 32")
        from ._vendor.activity.runtime import ActivityModel
        from ._vendor.activity import engine
        import torch
        torch.use_deterministic_algorithms(True)
        root = Path(__file__).parent
        self.activity = ActivityModel(root/"assets/activity", threads=threads)
        self.activity_engine = engine
        self.threads = threads
        if steps_enabled:
            from ._vendor.steps.step_count import StepCounter
            self.counter = StepCounter(threads=threads)
        else:
            self.counter = None

    def activity_probabilities(self, values, valid):
        return self.activity_engine.predict(self.activity.model, values, valid, self.activity.metadata)

    def step_gate(self, values, center):
        from ._vendor.steps.gp2_context import features
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=self.threads):
            return float(self.counter.model.predict_proba(features(values, centers=[center]))[0, 1])

    def step_contacts(self, values):
        from ._vendor.steps.gp2_algorithms import detect_core
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=self.threads):
            return detect_core(values, 25, "B2_shin")


def artifact_provenance(config=None):
    config = config or PipelineConfig()
    root = Path(__file__).parent
    model = json.loads((root/"assets/activity/model.json").read_text())
    steps = json.loads((root/"_vendor/steps/VALIDATION.json").read_text())
    code = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        code.update(path.relative_to(root).as_posix().encode()+b"\0"+path.read_bytes())
    provenance = {
        "library_version": __version__, "profile": config.profile,
        "code_sha256": code.hexdigest(),
        "config": asdict(config), "config_sha256": config.digest,
        "activity_metadata_sha256": hashlib.sha256((root/"assets/activity/model.json").read_bytes()).hexdigest(),
        "activity_weights_sha256": model["weights_sha256"],
        "steps_weights_sha256": steps["recipe"]["model_sha256"],
    }
    if config.mode == "whole_record":
        path = root/"assets/e7/model.json"
        provenance["whole_record_metadata_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance["whole_record_weights"] = json.loads(path.read_text())["models"]
    return provenance


class InferenceEngine:
    def __init__(self, config=None, *, threads=2, backend=None):
        self.config = config or PipelineConfig()
        self.backend = backend if backend is not None else FrozenBackend(threads, self.config.steps_enabled)
        if self.config.mode == "whole_record" and isinstance(self.backend, FrozenBackend):
            from .e7 import E7Model
            if not hasattr(self.backend, "whole_model"):
                self.backend.whole_model = E7Model(threads)
            self.whole_model = self.backend.whole_model
        else:
            self.whole_model = None
        self.provenance = artifact_provenance(self.config)
        self.fingerprint = hashlib.sha256(canonical_json(self.provenance)).hexdigest()

    def capabilities(self):
        return capabilities()

    def stream(self):
        if self.config.mode != "incremental":
            raise ValueError("whole_record mode requires a complete recording; stream is unavailable")
        return Stream(self)

    def restore(self, checkpoint):
        if self.config.mode != "incremental":
            raise ValueError("whole_record jobs restart from their durable complete input")
        return Stream.restore(self, checkpoint)

    def predict(self, acceleration, *, angular_velocity, valid=None, excluded=None):
        batch = InputBatch.create(acceleration, valid=valid, excluded=excluded,
                                  angular_velocity=angular_velocity)
        if self.config.mode == "whole_record":
            return self._predict_whole(batch)
        stream = self.stream()
        events = []
        for a in range(0, batch.end, 1500):
            events.extend(stream.append(batch.slice(a, min(batch.end, a+1500))))
        events.extend(stream.finish())
        return {"schema": "imu.result.v2", "provenance": deepcopy(self.provenance),
                "events": events, "statistics": stream.statistics.result(self.config.steps_enabled)}

    def _predict_whole(self, batch):
        import numpy as np
        from .steps import StepStream
        if batch.end > MAX_RECORD_SAMPLES:
            raise ValueError("complete recordings must not exceed 25 hours")
        values, observed = batch.values, batch.valid & ~batch.excluded
        baseline = self.backend.activity_probabilities(values, observed)
        if baseline.shape != (batch.end, 4) or not np.isfinite(baseline).all():
            raise RuntimeError("invalid whole-record baseline probabilities")
        if self.whole_model is not None:
            tags, diagnostic = self.whole_model.predict(
                values, observed, baseline, sign_adaptation=self.config.sign_adaptation)
        else:
            # Explicit injectable backends are for independent scheduling/recovery tests.
            tags, diagnostic = self.backend.whole_record(values, observed, baseline,
                                                         self.config.sign_adaptation)
        if tags.shape != (batch.end,) or not np.isin(tags[observed], [0, 1, 2, 3]).all():
            raise RuntimeError("invalid complete-record labels")
        status = np.where(batch.excluded, 2, np.where(batch.valid, 0, 1))
        tags = tags.copy()
        tags[~observed] = -1
        events = []
        # Output pagination is independent of whole-context model computation.
        for a in range(0, batch.end, 1500):
            b = min(a+1500, batch.end)
            t, s = tags[a:b], status[a:b]
            cuts = np.r_[0, np.flatnonzero((t[1:] != t[:-1]) | (s[1:] != s[:-1]))+1, b-a]
            timeline = [dict(start=a+int(lo), end=a+int(hi),
                             label=LABELS[int(t[lo])] if s[lo] == 0 else None,
                             status=("observed", "invalid", "excluded")[int(s[lo])],
                             mean_model_confidence=None)
                        for lo, hi in zip(cuts[:-1], cuts[1:], strict=True)]
            events.append(dict(kind="activity_core", start=a, end=b, timeline=timeline,
                               context_start=0, context_end=batch.end,
                               confidence_calibrated=False))
        if self.config.steps_enabled:
            steps = StepStream(self.backend)
            for a in range(0, batch.end, 250):
                events.extend(steps.feed(batch.slice(a, min(a+250, batch.end))))
            events.extend(steps.finish())
        events.sort(key=lambda e: int(e.get("start", e.get("first", 0))) // 1500)
        statistics = Statistics()
        statistics.consume(events)
        statistics.finish()
        return dict(schema="imu.result.v2", provenance=deepcopy(self.provenance), events=events,
                    statistics=statistics.result(self.config.steps_enabled),
                    diagnostics={"sign_adaptation": diagnostic}, input_sha256=batch.digest)


class Stream:
    """Ordered input only. Durable receivers own reordering and duplicate detection."""
    def __init__(self, engine):
        self.engine = engine
        self.activity = ActivityStream(engine.backend)
        self.steps = StepStream(engine.backend) if engine.config.steps_enabled else None
        self.statistics = Statistics()
        self.received = 0
        self.finished = False

    def append(self, batch):
        if self.finished or batch.start != self.received:
            raise ValueError("stream closed or non-contiguous input; reorder/deduplicate before append")
        events = []
        # Bound algorithm working buffers independently of caller upload length.
        for a in range(batch.start, batch.end, 250):
            part = batch.slice(a, min(batch.end, a+250))
            events.extend(self.activity.feed(part))
            if self.steps:
                events.extend(self.steps.feed(part))
        self.received = batch.end
        self.statistics.consume(events)
        return events

    def finish(self):
        if self.finished:
            return []
        events = self.activity.finish()
        if self.steps:
            events.extend(self.steps.finish())
        self.statistics.consume(events)
        self.statistics.finish()
        self.finished = True
        return events

    def checkpoint(self):
        state = {"schema": "imu.stream.state.v1", "fingerprint": self.engine.fingerprint,
                 "received": self.received, "finished": self.finished,
                 "activity": self.activity.state(),
                 "steps": self.steps.state() if self.steps else None,
                 "statistics": self.statistics.state()}
        return {"state": state, "sha256": hashlib.sha256(canonical_json(state)).hexdigest()}

    @classmethod
    def restore(cls, engine, checkpoint):
        state = deepcopy(checkpoint["state"])
        if hashlib.sha256(canonical_json(state)).hexdigest() != checkpoint["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        if state["schema"] != "imu.stream.state.v1" or state["fingerprint"] != engine.fingerprint:
            raise ValueError("checkpoint pipeline/version mismatch")
        obj = cls(engine)
        obj.activity = ActivityStream.restore(engine.backend, state["activity"])
        obj.steps = StepStream.restore(engine.backend, state["steps"]) if engine.config.steps_enabled else None
        obj.statistics = Statistics.restore(state["statistics"])
        obj.received, obj.finished = state["received"], state["finished"]
        if obj.received != obj.activity.end or (obj.steps and obj.received != obj.steps.end):
            raise ValueError("checkpoint component position mismatch")
        return obj
