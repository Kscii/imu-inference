"""Bounded SB1/FC1 activity processing, anchored to the recording sample grid."""
from __future__ import annotations

import hashlib
import numpy as np
from .contracts import LABELS, pack_array, unpack_array


class ActivityStream:
    CORE = 1500
    HALO = 750

    def __init__(self, backend):
        self.backend = backend
        self.base = self.end = self.next = 0
        self.x = np.empty((0, 6), np.float32)
        self.valid = np.empty(0, bool)
        self.excluded = np.empty(0, bool)

    def feed(self, batch):
        if batch.start != self.end:
            raise ValueError("activity stream requires contiguous explicit sample positions")
        self.x = np.concatenate((self.x, batch.values))
        self.valid = np.r_[self.valid, batch.valid]
        self.excluded = np.r_[self.excluded, batch.excluded]
        self.end = batch.end
        return self._drain(False)

    def finish(self):
        return self._drain(True)

    def _drain(self, final):
        out = []
        while self.next < self.end:
            a = self.next
            b = min(a+self.CORE, self.end)
            if not final and (b-a < self.CORE or b+self.HALO > self.end):
                break
            lo, hi = max(0, a-self.HALO), min(self.end, b+self.HALO)
            sl = slice(lo-self.base, hi-self.base)
            x, valid, excluded = self.x[sl], self.valid[sl], self.excluded[sl]
            p = self.backend.activity_probabilities(x, valid & ~excluded)[a-lo:b-lo]
            if p.shape != (b-a, 4) or not np.isfinite(p).all():
                raise RuntimeError("invalid activity probability output")
            v, e = valid[a-lo:b-lo], excluded[a-lo:b-lo]
            observed = v & ~e
            if observed.any() and not np.allclose(p[observed].sum(1), 1, atol=1e-5):
                raise RuntimeError("activity probabilities do not sum to one")
            tags = p.argmax(1)
            tags[~observed] = -1
            status = np.where(e, 2, np.where(v, 0, 1))
            edges = np.r_[0, np.flatnonzero((tags[1:] != tags[:-1]) |
                                           (status[1:] != status[:-1]))+1, len(tags)]
            timeline = []
            confidence = p.max(1)
            for start, stop in zip(edges[:-1], edges[1:], strict=True):
                state = ("observed", "invalid", "excluded")[int(status[start])]
                timeline.append({
                    "start": a+int(start), "end": a+int(stop),
                    "label": LABELS[int(tags[start])] if state == "observed" else None,
                    "status": state,
                    "mean_model_confidence": float(np.mean(confidence[start:stop], dtype=np.float64))
                    if state == "observed" else None,
                })
            digest = hashlib.sha256()
            for value in (x.astype("<f4"), valid, excluded):
                digest.update(value.tobytes())
            out.append({
                "kind": "activity_core", "start": a, "end": b, "timeline": timeline,
                "context_start": lo, "context_end": hi, "input_sha256": digest.hexdigest(),
                "confidence_calibrated": False,
            })
            self.next = b
            self._discard(max(0, self.next-self.HALO))
        if final:
            self._discard(self.end)
        return out

    def _discard(self, before):
        drop = before-self.base
        self.x = self.x[drop:].copy()
        self.valid = self.valid[drop:].copy()
        self.excluded = self.excluded[drop:].copy()
        self.base = before

    def state(self):
        return dict(base=self.base, end=self.end, next=self.next, x=pack_array(self.x.astype("<f4")),
                    valid=pack_array(self.valid), excluded=pack_array(self.excluded))

    @classmethod
    def restore(cls, backend, state):
        obj = cls(backend)
        obj.base, obj.end, obj.next = (int(state[k]) for k in ("base", "end", "next"))
        obj.x = unpack_array(state["x"], "<f4", max_elements=3500*6).reshape(-1, 6)
        obj.valid = unpack_array(state["valid"], bool, max_elements=3500)
        obj.excluded = unpack_array(state["excluded"], bool, max_elements=3500)
        if (not 0 <= obj.base <= obj.next <= obj.end or len(obj.x) != obj.end-obj.base
                or obj.valid.shape != (len(obj.x),) or obj.excluded.shape != obj.valid.shape
                or len(obj.x) > 3500 or not np.isfinite(obj.x).all()):
            raise ValueError("invalid bounded activity checkpoint")
        return obj
