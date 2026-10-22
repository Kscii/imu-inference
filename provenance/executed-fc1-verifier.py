"""Run where the existing FC1 data already reside. Export aggregate results only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np

ENGINE = None


def initialize():
    global ENGINE
    from imu_inference import InferenceEngine, PipelineConfig
    ENGINE = InferenceEngine(PipelineConfig(steps_enabled=False), threads=2)


def evaluate(item):
    record, reference = item
    for r in (record, reference):
        if hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest() != r["sha256"]:
            raise RuntimeError("reference input hash mismatch")
    with np.load(record["path"], allow_pickle=False) as z:
        x, valid = z["x"].copy(), z["valid"].copy()
    with np.load(reference["path"], allow_pickle=False) as z:
        expected, tags = z["p"].copy(), z["tag"].copy()
    p = np.empty_like(expected)
    started = time.monotonic()
    for a in range(0, len(x), 1500):
        b = min(len(x), a+1500)
        lo, hi = max(0, a-750), min(len(x), b+750)
        p[a:b] = ENGINE.backend.activity_probabilities(x[lo:hi], valid[lo:hi])[a-lo:b-lo]
    actual = p.argmax(1).astype(np.int16)
    actual[~valid] = -1
    return {"cohort": record["cohort"], "samples": len(x),
            "label_mismatches": int(np.count_nonzero(actual != tags)),
            "max_probability_difference": float(np.max(np.abs(p-expected))),
            "wall_s": time.monotonic()-started}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    records = json.loads((args.study/"records.json").read_text())
    references = {}
    for cohort in ("public", "device"):
        report = json.loads((args.study/f"reports/activity/cpu/{cohort}/pipeline-L60-H30.json").read_text())
        references.update({r["key"]: r for r in report["predictions"]})
    items = [(r, references[r["key"]]) for r in records]
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize) as pool:
        futures = [pool.submit(evaluate, item) for item in items]
        for future in as_completed(futures):
            rows.append(future.result())
            print(f"completed {len(rows)}/{len(items)}", flush=True)
    from imu_inference.engine import artifact_provenance
    cohorts = {}
    for name in ("public", "device"):
        selected = [r for r in rows if r["cohort"] == name]
        cohorts[name] = {"records": len(selected), "samples": sum(r["samples"] for r in selected),
                         "label_mismatches": sum(r["label_mismatches"] for r in selected),
                         "max_probability_difference": max(r["max_probability_difference"] for r in selected)}
    report = {"status": "PASS" if all(r["label_mismatches"] == 0 and
                                      r["max_probability_difference"] <= 1e-6 for r in rows) else "FAIL",
              "provenance": artifact_provenance(), "cohorts": cohorts,
              "scope": "Fixed-weight packaging regression, not a new accuracy or blind-customer test",
              "data_movement": "Original signals and per-record predictions remain on the source host"}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
