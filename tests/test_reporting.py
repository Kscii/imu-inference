import pytest
from imu_inference.reporting import calendar_statistics, events_to_segments


def analyze(segments, day="2026-10-04", zone="Australia/Sydney"):
    return calendar_statistics(segments, participant_id="synthetic", model_version="test",
                               timezone=zone, date_from=day, date_to=day)


def test_dst_coverage_duplicate_and_conflicting_revision():
    events = [{"kind": "activity_core", "timeline": [
        {"start": 0, "end": 1500, "label": "walking_running", "status": "observed"},
        {"start": 1500, "end": 1750, "label": "standing", "status": "observed"},
        {"start": 1750, "end": 2000, "label": None, "status": "invalid"},
    ]}]
    segments = events_to_segments(events, start_time="2026-10-04T00:00:00+10:00", recording_id="r")
    got = analyze(segments+segments)
    assert len(got["hourly"]) == 23
    day = got["daily"][0]
    assert day["calendar_s"] == 23*3600 and day["observed_s"] == 70 and day["invalid_s"] == 10
    assert day["unobserved_s"] == 23*3600-80
    assert got["duplicates_removed"] == 3
    with pytest.raises(ValueError):
        analyze(segments+[{**segments[0], "label": "sitting", "segment_id": "other-version"}])


def test_unknown_day_is_missing_not_zero_activity():
    day = analyze([])["daily"][0]
    assert day["state"] == "NO_OBSERVATIONS"
    assert day["duration_s"]["walking_running"] is None
