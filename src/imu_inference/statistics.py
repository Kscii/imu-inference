"""Incremental integer accounting. No torch, model loading, or database dependency."""
from __future__ import annotations
from copy import deepcopy
from .contracts import LABELS


class Statistics:
    def __init__(self):
        self.samples = 0
        self.counts = dict.fromkeys(LABELS, 0)
        self.invalid = self.excluded = 0
        self.step_valid = self.step_analyzed = self.steps = 0
        self.step_through = 0
        self.cadence_intervals = self.cadence_span = 0
        self.closed_runs = dict.fromkeys(LABELS, 0)
        self.open_run = None
        self.finished = False

    def consume(self, events):
        for event in events:
            kind = event["kind"]
            if kind == "activity_core":
                if event["start"] != self.samples:
                    raise ValueError("non-contiguous or repeated activity core")
                for row in event["timeline"]:
                    n = row["end"]-row["start"]
                    if row["status"] == "observed":
                        self.counts[row["label"]] += n
                    elif row["status"] == "excluded":
                        self.excluded += n
                    else:
                        self.invalid += n
                    if (self.open_run is not None and
                            (row["label"] != self.open_run["label"] or
                             row["status"] != self.open_run["status"] or
                             row["start"] != self.open_run["end"])):
                        self._close_run()
                    if self.open_run is None:
                        self.open_run = {k: row[k] for k in ("start", "end", "label", "status")}
                    else:
                        self.open_run["end"] = row["end"]
                self.samples = event["end"]
            elif kind == "step_segment":
                n = event["end"]-event["start"]
                self.step_through = max(self.step_through, event["end"])
                self.step_valid += n
                if event["analyzed"]:
                    self.step_analyzed += n
                self.steps += len(event["event_samples"])
            elif kind == "step_bout" and event["estimated_cadence_spm"] is not None:
                self.cadence_intervals += event["estimated_steps"]-1
                self.cadence_span += event["last"]-event["first"]

    def _close_run(self):
        if self.open_run and self.open_run["status"] == "observed":
            self.closed_runs[self.open_run["label"]] += 1
        self.open_run = None

    def finish(self):
        self._close_run()
        self.step_through = self.samples
        self.finished = True

    def result(self, steps_enabled=True):
        observed = sum(self.counts.values())
        if observed+self.invalid+self.excluded != self.samples:
            raise RuntimeError("activity accounting invariant violated")
        return {
            "schema": "imu.statistics.v1", "through_sample": self.samples,
            "finished": self.finished, "observed_s": observed/25,
            "invalid_s": self.invalid/25, "excluded_s": self.excluded/25,
            "duration_s": {k: n/25 for k, n in self.counts.items()},
            "fraction_of_observed": {k: n/observed if observed else None for k, n in self.counts.items()},
            "closed_label_runs": dict(self.closed_runs),
            "open_label_run": deepcopy(self.open_run),
            "steps": {
                "enabled": steps_enabled, "experimental": True,
                "through_sample": self.step_through if steps_enabled else None,
                "estimated_step_count": self.steps if steps_enabled and self.step_analyzed else None,
                "analyzed_s": self.step_analyzed/25 if steps_enabled else None,
                "unresolved_valid_s": (self.step_valid-self.step_analyzed)/25 if steps_enabled else None,
                "coverage_of_processed_valid": self.step_analyzed/self.step_valid
                if steps_enabled and self.step_valid else None,
                "estimated_cadence_spm": 1500*self.cadence_intervals/self.cadence_span
                if steps_enabled and self.cadence_span else None,
                "cadence_scope": "closed_eligible_bouts",
                "validation_scope": "OxWalk hip; chest accuracy unmeasured",
            },
        }

    def state(self):
        return deepcopy(vars(self))

    @classmethod
    def restore(cls, state):
        obj = cls()
        if set(state) != set(vars(obj)):
            raise ValueError("invalid statistics checkpoint")
        obj.__dict__.update(deepcopy(state))
        obj.result()
        return obj
