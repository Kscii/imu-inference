"""Score frozen predictions on Ubuntu; export aggregate durations, never raw signals.

No inference, training or threshold selection. Original participant IDs are kept
for the explicitly requested customer meeting appendix.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

CLASSES = ["lying", "sitting", "standing", "walking_running"]


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def duration_rows(person, matrices, recording_count):
    """Aggregate counts BEFORE division; unknown predictions retain denominators."""
    cm = np.sum(matrices, axis=0)
    total = int(cm.sum())
    if not total:
        raise ValueError("participant has no scored samples")
    truth, predicted = cm.sum(axis=1), cm[:, :4].sum(axis=0)
    rows = []
    for k, label in enumerate(CLASSES):
        t, p = int(truth[k]), int(predicted[k])
        relative = 100 * (p - t) / t if t else None
        rows.append(dict(
            participant_id=person, recording_count=recording_count,
            scored_samples=total, scored_minutes=total / 1500,
            class_id=k, label=label, truth_samples=t, predicted_samples=p,
            truth_minutes=t / 1500, predicted_minutes=p / 1500,
            signed_difference_minutes=(p - t) / 1500,
            signed_relative_error_percent=relative,
            absolute_relative_error_percent=abs(relative) if relative is not None else None,
            signed_share_difference_pp=100 * (p - t) / total,
            false_positive_minutes_if_truth_zero=p / 1500 if t == 0 else None,
            unknown_prediction_samples=int(cm[:, 4].sum()),
        ))
    return rows


def summarize(rows):
    out = []
    for k, label in enumerate(CLASSES):
        selected = [r for r in rows if r["class_id"] == k]
        present = [r for r in selected if r["truth_samples"] > 0]
        errors = [r["absolute_relative_error_percent"] for r in present]
        out.append(dict(
            label=label, eligible_participants=len(present), total_participants=len(selected),
            mean_absolute_relative_error_percent=float(np.mean(errors)) if errors else None,
            median_absolute_relative_error_percent=float(np.median(errors)) if errors else None,
            maximum_absolute_relative_error_percent=max(errors) if errors else None,
            mean_signed_relative_error_percent=float(np.mean([
                r["signed_relative_error_percent"] for r in present])) if present else None,
            mean_absolute_share_difference_pp=float(np.mean([
                abs(r["signed_share_difference_pp"]) for r in selected])),
            truth_minutes=sum(r["truth_minutes"] for r in selected),
            predicted_minutes=sum(r["predicted_minutes"] for r in selected),
            zero_truth_participants=[r["participant_id"] for r in selected if not r["truth_samples"]],
            zero_truth_false_positive_minutes=sum(
                r["false_positive_minutes_if_truth_zero"] or 0 for r in selected),
        ))
    return dict(per_class=out, person_duration_mae_pp=float(np.mean([
        abs(r["signed_share_difference_pp"]) for r in rows])))


def self_test():
    # Two recordings with unequal duration: sum before division (10%, not 50%).
    a, b = np.zeros((4, 5), int), np.zeros((4, 5), int)
    a[0, 0], a[1, 0], b[0, 0] = 10, 10, 90
    r = duration_rows("original-id", [a, b], 2)
    assert r[0]["signed_relative_error_percent"] == 10
    assert r[1]["signed_relative_error_percent"] == -100
    assert r[2]["signed_relative_error_percent"] is None
    # No truth for class 2: retain false positive time, never epsilon-divide.
    a[1, 2] = 25
    r = duration_rows("original-id", [a, b], 2)
    assert r[2]["false_positive_minutes_if_truth_zero"] == 25 / 1500
    assert summarize(r)["per_class"][2]["eligible_participants"] == 0
    # Unknown predictions count as missing predicted time, not missing truth.
    a[0, 4] = 100
    r = duration_rows("original-id", [a, b], 2)
    assert r[0]["truth_samples"] == 200 and r[0]["predicted_samples"] == 110
    assert r[0]["signed_relative_error_percent"] == -45
    assert r[0]["unknown_prediction_samples"] == 100
    # Opposite errors can cancel across records: intentional cumulative metric.
    c, d = np.zeros((4, 5), int), np.zeros((4, 5), int)
    c[0, 1], d[1, 0] = 25, 25
    r = duration_rows("original-id", [c, d], 2)
    assert r[0]["signed_relative_error_percent"] == r[1]["signed_relative_error_percent"] == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--es1", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--participant-reviews", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Duration metric edge-case checks passed")
        return
    if any(v is None for v in (args.es1, args.study, args.expected, args.output, args.participant_reviews)):
        parser.error("--es1, --study, --expected, --participant-reviews and --output are required")
    spec = importlib.util.spec_from_file_location("frozen_es1", args.es1 / "es1.py")
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    reference = scorer.verify_reference(args.es1)
    identities, identity_sources = {}, []
    for record in reference["records"]:
        rid = record["recording_id"].split(":", 1)[1]
        review_path = args.participant_reviews / rid / "review.json"
        review = read(review_path)
        assignment = review["participant_assignment"]
        assert review["recording_id"] == rid and assignment["status"] == "confirmed"
        original = assignment["participant_id"]
        previous = identities.setdefault(record["person"], original)
        assert previous == original, "inconsistent original participant assignment"
        identity_sources.append(dict(recording_id=record["recording_id"],
            reference_person=record["person"], participant_id=original,
            review_sha256=sha(review_path), assignment_status="confirmed"))
    assert len(identities) == len(set(identities.values())) == 9
    expected = read(args.expected)
    fc = args.study / "filter-context-fc1-v1"
    fc_records = read(fc / "records.json")
    by_key = {r["key"]: r["metadata"] for r in fc_records if r["cohort"] == "device"}
    fc_report = read(fc / "reports/activity/cpu/device/pipeline-L60-H30.json")
    fc_items = [dict(p, recording_id=by_key[p["key"]]["recording_id"],
                     person=by_key[p["key"]]["participant"], key="tag")
                for p in fc_report["predictions"]]
    e7 = args.study / "full-absorption-af1-v1/autonomous-goal-ag2-v1/observation-state-e7-v1"
    sub_path = e7 / "scores/submission-HARD_POSITIVE-2026102531-ORIGINAL-smoothed.json"
    sub = read(sub_path)
    assert sub["protocol_sha256"] == sha(args.es1 / "protocol.json")
    assert sub["reference_sha256"] == sha(args.es1 / "reference.json")
    output = dict(schema="imu.participant-duration.v1", sample_rate_hz=25,
                  reference_people=9, reference_recordings=23,
                  participant_identifiers="original IDs retained by explicit request",
                  scope="Repeatedly observed research reference; not customer blind or full-day validation",
                  aggregation="All recordings per participant before division; same ES1 valid labelled positions",
                  provenance=dict(
                      evaluator_sha256=sha(__file__), es1_sha256=sha(args.es1 / "es1.py"),
                      reference_sha256=sha(args.es1 / "reference.json"),
                      protocol_sha256=sha(args.es1 / "protocol.json"),
                      freeze_sha256=sha(args.es1 / "freeze.json"),
                      fc1_index_sha256=sha(fc / "records.json"),
                      fc1_report_sha256=sha(fc / "reports/activity/cpu/device/pipeline-L60-H30.json"),
                      e7_submission_sha256=sha(sub_path), expected_metrics_sha256=sha(args.expected)),
                  identity_mapping=identities, identity_sources=identity_sources, modes={})
    all_rows = []
    ids = {r["recording_id"] for r in reference["records"]}
    for mode, items, expected_key in [("incremental", fc_items, "fc1"),
                                      ("whole_record", sub["predictions"], "e7_default")]:
        predictions, sources = {}, []
        for item in items:
            ident = item["recording_id"]
            assert ident not in predictions
            assert sha(item["path"]) == item["sha256"], "prediction hash mismatch"
            with np.load(item["path"], allow_pickle=False) as z:
                predictions[ident] = z[item["key"]].copy()
            sources.append({k: item[k] for k in ("recording_id", "person", "sha256", "key")})
        assert set(predictions) == ids
        scored = scorer.score(reference, predictions, details=False)
        # Select by explicit mode key; refuse silently different scoring evidence.
        want = expected[expected_key] if expected_key == "fc1" else next(
            row for row in expected["e7"]
            if row["name"] == "HARD_POSITIVE-2026102531-ORIGINAL-smoothed")
        assert scored["strict"]["n"] == 816 and scored["frame"]["samples"] == 184383
        assert scored["strict"]["correct"] == want["strict"]["correct"]
        for field in ("accuracy", "macro_recall"):
            assert abs(scored["strict"][field] - want["strict"][field]) < 1e-12
        assert abs(scored["frame"]["accuracy"] - want["frame"]["accuracy"]) < 1e-12
        people, recording_counts = {}, {}
        for r in reference["records"]:
            truth = np.full(r["samples"], -1, np.int8)
            for event in r["events"]:
                truth[event["start"]:event["stop"]] = event["truth"]
            known = scorer.mask_for(r) & (truth >= 0)
            tags = scorer.clean_tags(predictions[r["recording_id"]], r["samples"])
            pred = np.where(tags >= 0, tags, 4)
            cm = np.bincount(5 * truth[known] + pred[known], minlength=20).reshape(4, 5)
            people.setdefault(r["person"], []).append(cm)
            recording_counts[r["person"]] = recording_counts.get(r["person"], 0) + 1
        assert len(people) == 9
        rows = [dict(mode=mode, reference_person=person, **row)
                for person in sorted(people, key=lambda p: identities[p])
                for row in duration_rows(identities[person], people[person], recording_counts[person])]
        summary = summarize(rows)
        assert abs(summary["person_duration_mae_pp"] - scored["frame"]["person_duration_mae_pp"]) < 1e-12
        assert abs(summary["person_duration_mae_pp"] - want["frame"]["person_duration_mae_pp"]) < 1e-12
        output["modes"][mode] = dict(summary=summary, rows=rows, prediction_sources=sources,
            existing_metrics_verified=True,
            es1=dict(strict=scored["strict"], frame=scored["frame"],
                     valid_output_frames=scored["valid_output_frames"],
                     excluded_from_scoring_frames=scored["excluded_from_scoring_frames"]))
        all_rows.extend(rows)
    output["passed"] = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "participant-duration.json").write_text(json.dumps(
        output, indent=2, ensure_ascii=False, allow_nan=False,
        default=lambda value: value.tolist()) + "\n")
    with (args.output / "participant-duration.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({k: v["summary"] for k, v in output["modes"].items()}, indent=2))


if __name__ == "__main__":
    main()
