"""Compare the frozen E7 deployment chain with on-host research predictions.

Raw signals and sample-level predictions remain on the validation host. The output
contains anonymous row indices, counts, errors, hashes and timing only.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from imu_inference import InferenceEngine, PipelineConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol = (
        args.study_root / "autonomous-optimization-ao1-v1/research/light-transition-lt1-v1/protocol.json"
    )
    e7root = args.study_root / "full-absorption-af1-v1/autonomous-goal-ag2-v1/observation-state-e7-v1"
    reference = json.loads(protocol.read_text())["device"]
    engine = InferenceEngine(PipelineConfig(mode="whole_record", steps_enabled=False), threads=2)
    expected_hash = engine.whole_model.metadata["source_report_sha256"]
    if hashlib.sha256((e7root / "report.json").read_bytes()).hexdigest() != expected_hash:
        raise ValueError("research report differs from frozen E7 manifest")
    result = dict(
        schema="imu.e7.packaging-parity.v1",
        provenance=engine.provenance,
        reference_report_sha256=expected_hash,
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        records=[],
        raw_data_exported=False,
    )
    for i, item in enumerate(reference):
        started = time.monotonic()
        with np.load(item["signal_path"], allow_pickle=False) as z:
            x = z["x"]
        with np.load(item["mask_path"], allow_pickle=False) as z:
            valid = z["task_valid"].astype(bool)
        baseline = engine.backend.activity_probabilities(x, valid)
        with np.load(item["baseline_path"], allow_pickle=False) as z:
            old = z["p"][:, :4]
        row = dict(
            index=i,
            samples=len(x),
            baseline_max_abs_error=float(np.max(np.abs(baseline[valid] - old[valid]))),
            modes={},
        )
        with np.load(e7root / "predictions" / f"{i:02}.npz", allow_pickle=False) as cached:
            for enabled, name in [(False, "ORIGINAL"), (True, "AUTOMATIC_SIGN")]:
                tags, diagnostics = engine.whole_model.predict(x, valid, baseline, sign_adaptation=enabled)
                expected = cached[f"tag-HARD_POSITIVE-2026102531-{name}-smoothed"]
                row["modes"][name] = dict(
                    mismatched_samples=int(np.count_nonzero(tags[valid] != expected[valid])),
                    valid_samples=int(valid.sum()),
                    diagnostics=diagnostics,
                )
        row["seconds"] = time.monotonic() - started
        result["records"].append(row)
        args.output.write_text(json.dumps(result, indent=2))
    result["passed"] = len(reference) == 23 and all(
        m["mismatched_samples"] == 0 for r in result["records"] for m in r["modes"].values()
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
