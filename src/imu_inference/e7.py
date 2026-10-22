"""Complete-record E7 inference from signal, with no research-cache dependency."""
from pathlib import Path
import hashlib
import json
import numpy as np


class E7Model:
    def __init__(self, threads=2):
        import xgboost as xgb
        root = Path(__file__).parent / "assets/e7"
        self.metadata = json.loads((root / "model.json").read_text())
        self.models = {}
        for key, entry in self.metadata["models"].items():
            path = root / entry["file"]
            if path.parent != root or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("E7 weight checksum mismatch")
            model = xgb.Booster()
            model.load_model(path)
            model.set_param({"device": "cpu", "nthread": threads})
            self.models[key] = model

    def predict(self, values, valid, baseline, *, sign_adaptation=False):
        from ._vendor.e7 import features as f, decode as d, polarity
        tags = baseline.argmax(1).astype(np.int8)
        tags[~valid] = -1
        diagnostic = {"enabled": sign_adaptation, "experimental": True,
                      "accepted": False, "sign": 1, "applied": False,
                      "reason": "disabled", "scope": "longitudinal_features_only"}
        if sign_adaptation:
            estimate, _ = polarity.estimate(values, valid)
            diagnostic.update(estimate, applied=estimate["accepted"] and estimate["sign"] == -1,
                              reason="accepted" if estimate["accepted"] else "insufficient_evidence")
        if not valid.any():
            return tags, diagnostic
        # The research grid starts at the same offset in each physical valid span.
        pos = np.concatenate([np.arange(a + min(2, b-a-1), b, 5) for a, b in f.spans(valid)])
        ix, reset = d.index_map(valid, pos)
        common = np.column_stack([f.invariant_features(values, valid, pos),
                                  f.integration(values, valid, pos)]).astype(np.float32)
        direction = self.models["r15"].inplace_predict(common)
        event = self.models["event"].inplace_predict(common)
        shape = f.shape_features(values, valid, pos)
        head_features = np.column_stack([common, shape[:, 37::3]]).astype(np.float32)
        signed_stats = np.array([0, 1, 3, 5, 6, 7, 8, 9, 10, 11])
        head_features[:, 285 + signed_stats] *= diagnostic["sign"]
        observation = self.models["head"].inplace_predict(head_features)
        del common, shape, head_features
        parent_features = f.six_axis_features(values, valid, pos)[:, f.GYRO_COLUMNS].copy()
        parent_features[:, 91 + 11*np.arange(7)] *= diagnostic["sign"]
        parent_probability = self.models["r3"].inplace_predict(parent_features)
        del parent_features
        correction = d.parent_decode(baseline[pos], np.column_stack([parent_probability, direction]),
                                     pos, valid, dict(threshold=.5, hold=5, back=0))
        changed = correction != baseline[pos].argmax(1)
        expanded = d.expand(pos, np.eye(4, dtype=np.float32)[correction]*changed[:, None], valid)
        use = valid & (expanded.sum(1) > .5)
        tags[use] = expanded[use].argmax(1)
        del expanded
        proposed = observation[ix[valid]]
        winner = proposed.argmax(1)
        peak = proposed.max(1)
        eligible = (np.isin(tags[valid], [1, 2]) & np.isin(winner, [1, 2])
                    & (peak >= proposed.dtype.type(.7))
                    & ((proposed == peak[:, None]).sum(1) == 1))
        tags[np.flatnonzero(valid)[eligible]] = winner[eligible]
        base = np.linalg.matrix_power(np.array(self.metadata["transition"]), 5)
        recipe = self.metadata["state_recipe"]
        posterior, _ = d.probabilities(observation, event, reset, base,
                                       recipe["weight"], recipe["threshold"])
        tags = d.state_overlay(tags, baseline, valid, ix, posterior["smoothed"], recipe["confidence"])
        return tags, diagnostic
