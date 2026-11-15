"""The plain-English summaries on the Overview and 7-day forecast pages: deterministic templates
over the saved forecast, explanations and backtest."""
import numpy as np
import pandas as pd

from src.insights import driver_sentences, recent_accuracy, week_headline, week_insights

DAYS = pd.date_range("2026-11-09", periods=7)


def _fc(evening=(200, 195, 207, 210, 211, 211, 199), solar=(210, 207, 214, 218, 220, 222, 212), p90_extra=5):
    rows = []
    for t, vals in (("evening_peak", evening), ("solar_peak", solar), ("energy", [4400] * 7)):
        for d, v in zip(DAYS, vals):
            scale = 1 if t == "energy" else 1000
            rows.append({"model": "ensemble", "target": t, "date": d, "p10": (v - p90_extra) * scale,
                         "p50": v * scale, "p90": (v + p90_extra) * scale})
    return pd.DataFrame(rows)


def _ex():
    contrib = [{"Diwali week": -10.0, "recent trend": -3.0, "Monday": 1.0},
               *[{"Diwali week": -1.0, "Heat in North India": 0.5}] * 5,
               {"Sunday": -4.0, "Diwali week": -0.5}]
    return pd.DataFrame({"date": DAYS, "target": "evening_peak", "contrib": contrib, "ref": 205.0})


def test_headline_states_range_and_record_risk():
    fc = _fc()
    assert week_headline(fc, "ensemble", {"all_time": 270_000}) == \
        "Peak demand expected between *207 and 222 GW* this week. No record risk."
    assert "possible" in week_headline(fc, "ensemble", {"all_time": 225_000})   # P90 222+5 reaches it
    assert "likely" in week_headline(fc, "ensemble", {"all_time": 215_000})
    assert week_headline(fc, "ensemble", None).endswith("this week.")


def test_week_insights_name_driver_day_and_calendar_effect():
    out = dict(week_insights(_fc(), _ex(), "ensemble", "evening_peak"))
    assert list(out) == ["Biggest driver", "Day to watch", "Calendar effect"]
    assert out["Biggest driver"].startswith("Diwali week lowers the evening peak")
    assert "recent trend" not in " ".join(out.values())                 # generic inputs are not drivers
    assert out["Day to watch"].startswith("Tue 10 Nov: 195.0 GW")
    assert out["Calendar effect"] == "Sun 15 Nov: The weekend lowers the evening peak by 4.0 GW."  # not Diwali again


def test_week_insights_without_explanations_still_reports_the_day():
    out = week_insights(_fc(), None, "ensemble", "evening_peak")
    assert [k for k, _ in out] == ["Day to watch"]


def test_driver_sentences_are_plain_english_and_skip_small_effects():
    rows = driver_sentences({"Diwali week": -10.9, "Monday": 1.3, "recent trend": -5, "Rain (all India)": 0.1},
                            "energy")
    assert rows == [("Diwali week lowers energy demand", -10.9),
                    ("Day of the week (Monday) raises energy demand", 1.3)]


def test_recent_accuracy_compares_two_windows():
    dates = pd.date_range("2026-09-01", "2026-10-30")
    rows = []
    for d in dates:
        late = d > pd.Timestamp("2026-09-30")
        rows.append({"model": "ensemble", "target": "energy", "date": d, "h": 1, "actual": 100.0,
                     "p50": 101.0 if late else 102.0})
        rows.append({"model": "grid_india", "target": "energy", "date": d, "h": 1, "actual": 100.0,
                     "p50": 101.5 if late else 101.0})
    acc = recent_accuracy(pd.DataFrame(rows), "ensemble", days=30)
    assert np.isclose(acc["mape"], 1.0) and np.isclose(acc["mape_prev"], 2.0)
    assert acc["win"] == 100 and acc["win_prev"] == 0
    assert acc["n_days"] == 30
