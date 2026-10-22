"""Bounded equivalent of frozen GP2, including valid-span and gate-region origins."""
from __future__ import annotations
from collections import deque
import numpy as np
from .contracts import pack_array, unpack_array


class StepStream:
    def __init__(self, backend):
        self.backend = backend
        self.end = 0
        self.origin = None
        self.base = 0
        self.x = np.empty((0, 3), np.float64)
        self.gate_next = 0
        self.block_start = None
        self.jobs = deque()
        self.bout = None

    def feed(self, batch):
        if batch.start != self.end:
            raise ValueError("step stream requires contiguous explicit sample positions")
        out = []
        mask = batch.valid & ~batch.excluded
        edges = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1])+1, len(mask)]
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            start, end = batch.start+int(a), batch.start+int(b)
            if mask[a]:
                if self.origin is None:
                    self.origin = self.base = self.gate_next = start
                    self.x = np.empty((0, 3), np.float64)
                self.x = np.concatenate((self.x, batch.acceleration[a:b].astype(np.float64)))
                self.end = end
                out.extend(self._advance(False))
            else:
                out.extend(self._close_span())
                self.end = end
        return out

    def finish(self):
        return self._close_span()

    def _close_span(self):
        if self.origin is None:
            return []
        out = self._advance(True)
        self.origin = None
        self.x = np.empty((0, 3), np.float64)
        self.base = self.end
        return out

    def _advance(self, final):
        out = []
        if self.end-self.origin < 100:
            if final:
                out.append(self._segment(self.origin, self.end, [], False, "UNDER_FOUR_SECONDS"))
            return out
        while self.gate_next < self.end:
            g = self.gate_next
            if not final and g+100 > self.end:
                break
            lo, hi = max(self.origin, g-100), min(self.end, g+100)
            signal = self.x[lo-self.base:hi-self.base]
            positive = self.backend.step_gate(signal, g+12-lo) >= .5
            stop = min(self.end, g+25)
            if positive:
                if self.block_start is None:
                    self.block_start = g
                while stop-self.block_start >= 750:
                    a = self.block_start
                    self.jobs.append(("block", a, a+750))
                    self.block_start += 750
            else:
                self._end_region(g)
                self.jobs.append(("negative", g, stop))
            self.gate_next += 25
        if final:
            self._end_region(self.end)
        while self.jobs:
            kind, a, b = self.jobs[0]
            if kind == "block" and not final and b+50 > self.end:
                break
            self.jobs.popleft()
            if kind == "region_end":
                out.extend(self._close_bout())
            elif kind == "negative":
                out.append(self._segment(a, b, [], True, "OK"))
            else:
                lo, hi = max(self.origin, a-50), min(self.end, b+50)
                try:
                    indices, status = self.backend.step_contacts(self.x[lo-self.base:hi-self.base])
                    indices = np.asarray(indices, int)+lo
                    selected = indices[(indices >= a) & (indices < b)].tolist()
                    analyzed = status != "TOO_SHORT"
                except (ValueError, IndexError):
                    selected, analyzed, status = [], False, "DETECTOR_FAILED"
                out.append(self._segment(a, b, selected, analyzed, status))
                for index in selected:
                    if self.bout is not None and index-self.bout["last"] > 50:
                        out.extend(self._close_bout())
                    if self.bout is None:
                        self.bout = {"first": index, "last": index, "count": 1}
                    else:
                        self.bout["last"] = index
                        self.bout["count"] += 1
                if self.bout is not None and b-self.bout["last"] > 50:
                    out.extend(self._close_bout())
        if final:
            out.extend(self._close_bout())
            if self.jobs:
                raise RuntimeError("unflushed GP2 jobs")
        else:
            needed = [max(self.origin, self.gate_next-100)]
            if self.block_start is not None:
                needed.append(max(self.origin, self.block_start-50))
            needed.extend(max(self.origin, a-50) for k, a, b in self.jobs if k == "block")
            before = min(needed)
            self.x = self.x[before-self.base:].copy()
            self.base = before
        return out

    def _end_region(self, end):
        if self.block_start is not None:
            if end > self.block_start:
                self.jobs.append(("block", self.block_start, end))
            self.jobs.append(("region_end", end, end))
            self.block_start = None

    @staticmethod
    def _segment(start, end, indices, analyzed, status):
        return dict(kind="step_segment", start=start, end=end, event_samples=indices,
                    analyzed=analyzed, status=status, experimental=True)

    def _close_bout(self):
        if self.bout is None:
            return []
        b = self.bout
        self.bout = None
        eligible = b["count"] >= 6 and b["last"]-b["first"] >= 125
        return [dict(kind="step_bout", first=b["first"], last=b["last"],
                     estimated_steps=b["count"],
                     estimated_cadence_spm=1500*(b["count"]-1)/(b["last"]-b["first"])
                     if eligible else None, experimental=True)]

    def state(self):
        return dict(end=self.end, origin=self.origin, base=self.base, x=pack_array(self.x.astype("<f8")),
                    gate_next=self.gate_next, block_start=self.block_start,
                    jobs=list(self.jobs), bout=self.bout)

    @classmethod
    def restore(cls, backend, state):
        obj = cls(backend)
        for k in ("end", "origin", "base", "gate_next", "block_start", "bout"):
            setattr(obj, k, state[k])
        obj.x = unpack_array(state["x"], "<f8", max_elements=1500*3).reshape(-1, 3)
        obj.jobs = deque(tuple(j) for j in state["jobs"])
        if len(obj.x) > 1500 or not np.isfinite(obj.x).all() or len(obj.jobs) > 40:
            raise ValueError("invalid bounded GP2 checkpoint")
        if obj.origin is not None and (obj.base < obj.origin or len(obj.x) != obj.end-obj.base):
            raise ValueError("inconsistent GP2 buffer coordinates")
        return obj
