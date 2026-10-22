"""Model-independent activity statistics. Half-open intervals, UTC arithmetic.

No model loading, threshold tuning, interpolation, non-wear inference, or step
counting. The caller supplies final labels and an explicit local calendar.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

LABELS = ('lying', 'sitting', 'standing', 'walking_running',
          'posture_transition', 'other_activity', 'unclassified')
UTC = dt.timezone.utc
EPOCH = dt.datetime(1970, 1, 1, tzinfo=UTC)


def timestamp_us(value):
    if not isinstance(value, str):
        raise ValueError('timestamps must be ISO 8601 strings with offsets')
    value = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('timezone offset required; do not guess recording time')
    delta = value.astimezone(UTC) - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds


def iso(value):
    return (EPOCH + dt.timedelta(microseconds=value)).isoformat()


def seconds(value):
    return value / 1000000


def active_labels(payload):
    labels=payload.get('active_labels')
    if labels is None:return LABELS
    allowed=set(LABELS)-{'unclassified'}|{'sitting_standing'}
    if not isinstance(labels,list) or not labels or any(not isinstance(v,str) or v not in allowed for v in labels) or len(set(labels))!=len(labels):
        raise ValueError('active_labels must contain unique supported model classes')
    if 'sitting_standing' in labels and ('sitting' in labels or 'standing' in labels):
        raise ValueError('coarse and separate sit/stand schemas cannot be combined')
    return tuple(labels)+('unclassified',)


def canonical_segments(payload):
    if payload.get('schema') != 'imu.activity.segments.v1':
        raise ValueError('unsupported input schema')
    if not payload.get('participant_id') or not payload.get('model_version'):
        raise ValueError('participant_id and model_version are required')
    labels=active_labels(payload)
    seen = set()
    ids = {}
    rows = []
    duplicates = 0
    for item in payload['segments']:
        if not isinstance(item.get('recording_id'), str) or not item['recording_id']:
            raise ValueError('recording_id required for provenance and deduplication')
        a, b = timestamp_us(item['start']), timestamp_us(item['end'])
        if b <= a:
            raise ValueError('interval end must be strictly after start')
        status = item.get('status', 'observed')
        label = item.get('label')
        if status not in ('observed', 'invalid', 'excluded'):
            raise ValueError('status must be observed, invalid or excluded')
        if status == 'observed' and label not in labels:
            raise ValueError('observed interval requires a known label or unclassified')
        if status in ('invalid','excluded') and label is not None:
            raise ValueError('invalid signal must not carry an activity label')
        key = (item['recording_id'], a, b, status, label)
        identity = item.get('segment_id')
        if identity is not None:
            if not isinstance(identity, str) or not identity:
                raise ValueError('segment_id must be a nonempty string')
            if identity in ids and ids[identity] != key:
                raise ValueError('same segment_id has conflicting content')
            ids[identity] = key
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        rows.append(dict(start_us=a, end_us=b, status=status, label=label,
                         recording_id=item['recording_id'], segment_id=identity))
    rows.sort(key=lambda x: (x['start_us'], x['end_us']))
    for left, right in zip(rows, rows[1:]):
        if right['start_us'] < left['end_us']:
            raise ValueError('overlapping nonidentical intervals require upstream reconciliation')
    return rows, duplicates


def measurement(values, invalid, excluded, calendar):
    observed = sum(values.values())
    if observed + invalid + excluded > calendar:
        raise ValueError('observation accounting exceeds calendar duration')
    return dict(
        state='OBSERVED' if observed else 'NO_OBSERVATIONS',
        observed_s=seconds(observed),
        invalid_s=seconds(invalid),
        excluded_s=seconds(excluded),
        unobserved_s=seconds(calendar - observed - invalid - excluded),
        calendar_s=seconds(calendar),
        observation_fraction=observed / calendar if calendar else None,
        duration_s={k: seconds(v) if observed else None for k, v in values.items()},
        fraction_of_observed={k: v / observed if observed else None for k, v in values.items()},
        classified_fraction=(observed - values['unclassified']) / observed if observed else None,
    )


def summarize(payload, timezone, date_from, date_to):
    """Summarize an inclusive date range; missing dates remain missing, not zero."""
    zone = ZoneInfo(timezone)
    first, last = dt.date.fromisoformat(date_from), dt.date.fromisoformat(date_to)
    if last < first or (last - first).days > 3659:
        raise ValueError('date range must be ordered and at most ten years')
    segments, duplicates = canonical_segments(payload)
    labels=active_labels(payload)
    daily, internal = [], []
    day = first
    cursor = 0
    while day <= last:
        next_day = day + dt.timedelta(days=1)
        a = timestamp_us(dt.datetime.combine(day, dt.time(), zone).isoformat())
        b = timestamp_us(dt.datetime.combine(next_day, dt.time(), zone).isoformat())
        values = dict.fromkeys(labels, 0)
        invalid = 0
        excluded = 0
        timeline = []
        while cursor < len(segments) and segments[cursor]['end_us'] <= a:
            cursor += 1
        index = cursor
        while index < len(segments) and segments[index]['start_us'] < b:
            row = segments[index]
            lo, hi = max(a, row['start_us']), min(b, row['end_us'])
            if hi > lo:
                if row['status'] == 'observed':
                    values[row['label']] += hi - lo
                elif row['status']=='excluded':
                    excluded += hi-lo
                else:
                    invalid += hi - lo
                timeline.append(dict(start=iso(lo), end=iso(hi), status=row['status'],
                                     label=row['label'], recording_id=row['recording_id']))
            index += 1
        daily.append(dict(date=day.isoformat(), **measurement(values, invalid, excluded, b-a), timeline=timeline))
        internal.append((day, values, invalid, excluded, b-a))
        day = next_day
    weeks = {}
    for day, values, invalid, excluded, calendar in internal:
        y, week, _ = day.isocalendar()
        key = f'{y}-W{week:02d}'
        row = weeks.setdefault(key, dict(values=dict.fromkeys(labels, 0), invalid=0, excluded=0,
                                         calendar=0, observed_days=0, dates=[], missing_dates=[]))
        for label in labels:
            row['values'][label] += values[label]
        row['invalid'] += invalid
        row['excluded'] += excluded
        row['calendar'] += calendar
        row['dates'].append(day.isoformat())
        if sum(values.values()):
            row['observed_days'] += 1
        else:
            row['missing_dates'].append(day.isoformat())
    weekly = []
    for key, row in weeks.items():
        n = row['observed_days']
        weekly.append(dict(week=key, included_dates=row['dates'], observed_days=n,
                           missing_dates=row['missing_dates'],
                           **measurement(row['values'], row['invalid'], row['excluded'], row['calendar']),
                           mean_duration_per_observed_day_s={k: seconds(v)/n if n else None
                                                            for k, v in row['values'].items()}))
    return dict(schema='imu.activity.statistics.v1', participant_id=payload['participant_id'],
                model_version=payload['model_version'], active_labels=list(labels[:-1]), timezone=timezone,
                date_from=date_from, date_to=date_to, duplicates_removed=duplicates,
                denominator='observed time including unclassified; invalid, excluded segments and unobserved excluded',
                interpretation='Observation coverage is not verified wearing compliance. No full-day extrapolation.',
                step_count=None, completed_transition_count=None,
                daily=daily, weekly=weekly)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--timezone', required=True)
    parser.add_argument('--date-from', required=True)
    parser.add_argument('--date-to', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; choose a new versioned output')
    result = summarize(json.loads(args.input.read_text()), args.timezone, args.date_from, args.date_to)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    with args.output.open('x') as output:
        output.write(payload)


if __name__ == '__main__':
    main()
