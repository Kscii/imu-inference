"""Exact accounting and unsmoothed label-run statistics, not physical event counts."""
from __future__ import annotations
import datetime as dt
import math
import statistics as stats
from zoneinfo import ZoneInfo
from . import _calendar as base

VERSION = "gp1-statistics-v1"
US = 1_000_000


def _observed(r):
    return r["status"] == "observed" and r["label"] != "unclassified"


def _runs(rows):
    runs = []
    for r in rows:
        if not _observed(r):
            continue
        if (runs and runs[-1]["end_us"] == r["start_us"]
                and runs[-1]["recording_id"] == r["recording_id"]
                and runs[-1]["label"] == r["label"]):
            runs[-1]["end_us"] = r["end_us"]
        else:
            runs.append({k: r[k] for k in ("start_us", "end_us", "label", "recording_id")})
    # These flags describe observed boundaries, not proven onset/completion.
    before = {(r["recording_id"], r["end_us"]): r for r in rows}
    after = {(r["recording_id"], r["start_us"]): r for r in rows}
    for r in runs:
        left = before.get((r["recording_id"], r["start_us"]))
        right = after.get((r["recording_id"], r["end_us"]))
        r["left_censored"] = left is None or not _observed(left)
        r["right_censored"] = right is None or not _observed(right)
        r["duration_s"] = (r["end_us"] - r["start_us"]) / US
    return runs


def _periods(rows, max_pause_s):
    """Group locomotion with internal standing pauses; retain every pause."""
    if not math.isfinite(max_pause_s) or max_pause_s < 0:
        raise ValueError("max_standing_pause_s must be finite and nonnegative")
    out = []
    i = 0
    while i < len(rows):
        a = rows[i]
        if a["status"] != "observed" or a["label"] != "walking_running":
            i += 1
            continue
        active = a["end_us"] - a["start_us"]
        pause = 0
        last = i
        j = i + 1
        pending = 0
        while j < len(rows):
            r, prev = rows[j], rows[j-1]
            if (r["start_us"] != prev["end_us"] or r["recording_id"] != a["recording_id"]
                    or r["status"] != "observed"):
                break
            n = r["end_us"] - r["start_us"]
            if r["label"] == "standing":
                pending += n
                if pending > max_pause_s * US:
                    break
            elif r["label"] == "walking_running":
                active += n
                pause += pending
                pending = 0
                last = j
            else:
                break
            j += 1
        end = rows[last]["end_us"]
        assert active + pause == end - a["start_us"]
        out.append(dict(start_us=a["start_us"], end_us=end, recording_id=a["recording_id"],
                        elapsed_s=(end-a["start_us"])/US, locomotion_s=active/US,
                        standing_pause_s=pause/US, unresolved_s=0.0))
        i = last + 1
    return out


def _descriptive(rows, labels, max_pause_s):
    runs = _runs(rows)
    total_observed = sum(r["end_us"]-r["start_us"] for r in rows if r["status"] == "observed")
    per_class = {}
    for label in labels:
        if label == "unclassified":
            continue
        selected = [r for r in runs if r["label"] == label]
        lengths = [r["duration_s"] for r in selected]
        total = sum(lengths)
        per_class[label] = dict(
            label_run_count=len(lengths) if total_observed else None,
            observed_label_duration_s=total if total_observed else None,
            longest_label_run_s=max(lengths) if lengths else (0.0 if total_observed else None),
            median_label_run_s=stats.median(lengths) if lengths else None,
            duration_fraction_in_runs_under_3s=sum(v for v in lengths if v < 3)/total if total else None,
            duration_fraction_in_runs_under_8s=sum(v for v in lengths if v < 8)/total if total else None,
            censored_run_count=sum(r["left_censored"] or r["right_censored"] for r in selected),
        )
    return dict(
        algorithm_version=VERSION,
        interpretation="Model label runs, not verified physical bouts or completed events. No smoothing or label edits.",
        grouping_rule=dict(max_standing_pause_s=max_pause_s,
                           meaning="Reporting grouping only; internal standing time remains standing."),
        by_activity=per_class, strict_label_runs=runs,
        locomotion_periods=_periods(rows, max_pause_s),
        step_count=None, completed_transition_count=None,
    )


def summarize_recording(result, *, max_standing_pause_s=30.0):
    """Consume the frozen ActivityModel.predict JSON, without a guessed date."""
    if result.get("schema") != "imu.activity.recording.v1":
        raise ValueError("expected imu.activity.recording.v1")
    rid = result.get("recording_id")
    n = result.get("samples")
    fs = result.get("sample_rate_hz")
    if not isinstance(n, int) or n <= 0 or not isinstance(fs, (int, float)) or not math.isfinite(fs) or fs <= 0:
        raise ValueError("positive integer samples and finite sample_rate_hz required")
    duration = n / fs
    if not math.isfinite(result.get("recording_duration_s", float("nan"))) or abs(result["recording_duration_s"]-duration)>1e-7:
        raise ValueError("recording duration and sample clock disagree")
    payload = dict(schema="imu.activity.segments.v1", participant_id="recording-only",
                   model_version=result.get("model_version"), active_labels=result.get("active_labels"), segments=[])
    end = 0.0
    for i, r in enumerate(result["timeline"]):
        a, b = r["start_s"], r["end_s"]
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in (a,b)):
            raise ValueError("finite timeline times required")
        if abs(a-end)>1e-7 or b<=a or b>duration+1e-7:
            raise ValueError("recording timeline must cover the sample clock exactly without gaps")
        end = b
        payload["segments"].append(dict(recording_id=rid, segment_id=f"{rid}:{i}",
                                        start=base.iso(round(a*US)), end=base.iso(round(b*US)),
                                        label=r["label"], status=r["status"]))
    if abs(end-duration)>1e-7:
        raise ValueError("timeline does not reach recording end")
    rows, _ = base.canonical_segments(payload)
    labels = base.active_labels(payload)
    values = dict.fromkeys(labels, 0)
    invalid = excluded = 0
    for r in rows:
        delta = r["end_us"]-r["start_us"]
        if r["status"]=="observed":values[r["label"]]+=delta
        elif r["status"]=="invalid":invalid+=delta
        else:excluded+=delta
    measured = base.measurement(values, invalid, excluded, round(duration*US))
    if abs(measured["observed_s"]-result["observed_s"])>1e-6:
        raise ValueError("observed total disagrees with timeline")
    for key,value in values.items():
        if key not in result["duration_s"] or abs(value/US-result["duration_s"][key])>1e-6:
            raise ValueError("class durations disagree with timeline")
    for key in ("invalid_s","excluded_s"):
        if abs(measured[key]-result[key])>1e-6:raise ValueError(f"{key} disagrees with timeline")
    details = _descriptive(rows, labels, max_standing_pause_s)
    for key in ("strict_label_runs", "locomotion_periods"):
        for r in details[key]:
            r["start_s"] = r.pop("start_us")/US
            r["end_s"] = r.pop("end_us")/US
    return dict(schema="imu.product.recording-statistics.v1", recording_id=rid,
                model_version=result["model_version"], clock="recording-relative; no calendar inferred",
                measurement=measured, **details)


def _hour_edges(day, zone):
    """UTC boundaries for local clock hours, folds and non-hour DST jumps."""
    tomorrow = day + dt.timedelta(days=1)
    start = base.timestamp_us(dt.datetime.combine(day, dt.time(), zone).isoformat())
    stop = base.timestamp_us(dt.datetime.combine(tomorrow, dt.time(), zone).isoformat())
    edges = {start,stop}
    for hour in range(24):
        naive = dt.datetime.combine(day,dt.time(hour))
        for fold in (0,1):
            candidate = naive.replace(tzinfo=zone,fold=fold)
            utc = candidate.astimezone(base.UTC)
            if utc.astimezone(zone).replace(tzinfo=None)==naive:
                t=base.timestamp_us(utc.isoformat())
                if start<=t<=stop:edges.add(t)
    def offset(second):
        return (base.EPOCH+dt.timedelta(seconds=second)).astimezone(zone).utcoffset()
    a=start//US
    while a<stop//US:
        b=min(a+3600,stop//US)
        if offset(a)!=offset(b):
            lo,hi=a,b
            while hi-lo>1:
                mid=(lo+hi)//2
                if offset(mid)==offset(a):lo=mid
                else:hi=mid
            edges.add(hi*US)
        a=b
    return sorted(edges)


def summarize(payload, timezone, date_from, date_to, *, max_standing_pause_s=30.0):
    """Calendar statistics with explicit timezone and coverage, including DST."""
    original = base.summarize(payload,timezone,date_from,date_to)
    rows,duplicates=base.canonical_segments(payload)
    labels=base.active_labels(payload)
    zone=ZoneInfo(timezone)
    hourly=[]
    for day in original["daily"]:
        edges=_hour_edges(dt.date.fromisoformat(day["date"]),zone)
        for a,b in zip(edges,edges[1:]):
            values=dict.fromkeys(labels,0);invalid=excluded=0
            for r in rows:
                lo,hi=max(a,r["start_us"]),min(b,r["end_us"])
                if hi<=lo:continue
                if r["status"]=="observed":values[r["label"]]+=hi-lo
                elif r["status"]=="invalid":invalid+=hi-lo
                else:excluded+=hi-lo
            hourly.append(dict(start=base.iso(a),end=base.iso(b),
                               local_start=(base.EPOCH+dt.timedelta(microseconds=a)).astimezone(zone).isoformat(),
                               **base.measurement(values,invalid,excluded,b-a)))
    # Use actual timezone-aware midnights.
    range_start=_hour_edges(dt.date.fromisoformat(date_from),zone)[0]
    range_stop=_hour_edges(dt.date.fromisoformat(date_to),zone)[-1]
    clipped=[]
    for r in rows:
        a,b=max(r["start_us"],range_start),min(r["end_us"],range_stop)
        if b>a:clipped.append({**r,"start_us":a,"end_us":b})
    details=_descriptive(clipped,labels,max_standing_pause_s)
    for key in ("strict_label_runs","locomotion_periods"):
        for r in details[key]:
            r["start"]=base.iso(r.pop("start_us"));r["end"]=base.iso(r.pop("end_us"))
    return dict(schema="imu.product.calendar-statistics.v1",participant_id=payload["participant_id"],
                model_version=payload["model_version"],timezone=timezone,
                date_from=date_from,date_to=date_to,duplicates_removed=duplicates,
                daily=original["daily"],weekly=original["weekly"],hourly=hourly,**details)
