"""Public pure statistics API. No model, GPU, database or HTTP imports."""
from datetime import datetime, timedelta, timezone
from .contracts import LABELS
from ._vendor.statistics import summarize as _summarize


def events_to_segments(events, *, start_time, recording_id):
    """Convert immutable activity cores to timestamped half-open intervals."""
    start = datetime.fromisoformat(start_time.replace("Z", "+00:00")) if isinstance(start_time, str) else start_time
    if start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("explicit start_time timezone required")
    start = start.astimezone(timezone.utc)
    output = []
    for event in events:
        if event["kind"] != "activity_core":
            continue
        for row in event["timeline"]:
            output.append({
                "recording_id": recording_id,
                "segment_id": f"{recording_id}:{row['start']}:{row['end']}",
                "start": (start+timedelta(microseconds=row["start"]*40000)).isoformat(),
                "end": (start+timedelta(microseconds=row["end"]*40000)).isoformat(),
                "label": row["label"] if row["status"] == "observed" else None,
                "status": row["status"],
            })
    return output


def calendar_statistics(segments, *, participant_id, model_version, timezone, date_from, date_to,
                        max_standing_pause_s=30.0):
    """Hourly/daily/weekly coverage and descriptive label runs, including DST.

    date_from/date_to are inclusive ISO dates. The caller must resolve multiple
    inference revisions first; overlapping nonidentical segments are rejected.
    Label-run counts and locomotion periods are not physical movement counts.
    """
    return _summarize(
        dict(schema="imu.activity.segments.v1", participant_id=participant_id, model_version=model_version,
             active_labels=list(LABELS), segments=segments),
        timezone, date_from, date_to, max_standing_pause_s=max_standing_pause_s,
    )
