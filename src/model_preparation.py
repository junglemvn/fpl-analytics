"""Chronological partitions and cutoff-aware FPL benchmarks."""

import numpy as np
import pandas as pd

from .data_preparation import MATCH_KEY, PLAYER_KEY, require_keys


SEASON_SPLITS = {
    "2022_2023": "train", "2023_2024": "train", "2024_2025": "validation",
    "2025_2026": "test", "2026_2027": "monitor",
}
FEATURES = ["recent_points_5", "recent_minutes_5", "prior_records"]
PREDICTIONS = ["zero", "train_mean", "train_position_mean", "recent_five"]
RULE_VERSION = "fixture_points_v1_season_split_cutoff_24h"


def prediction_calendar(fixtures, events, historical_buffer_hours=24):
    """Use saved current deadlines and a labelled historical cutoff proxy."""
    if historical_buffer_hours < 24:
        raise ValueError("Historical cutoff buffer must be at least 24 hours")
    require_keys(fixtures, ["season", "fpl_fixture_id"], "modelling fixtures")
    complete = fixtures.loc[fixtures.finished].copy()
    if complete[["event", "kickoff_time"]].isna().any().any():
        raise ValueError("Completed fixtures need gameweeks and kickoff times")
    calendar = complete.groupby(["season", "event"]).kickoff_time.min().reset_index()
    calendar = calendar.rename(columns={"event": "GW", "kickoff_time": "first_kickoff"})
    calendar["prediction_cutoff"] = calendar.first_kickoff - pd.Timedelta(
        hours=historical_buffer_hours,
    )
    calendar["cutoff_basis"] = "first_kickoff_minus_24h_proxy"
    if historical_buffer_hours != 24:
        calendar["cutoff_basis"] = f"first_kickoff_minus_{historical_buffer_hours}h_proxy"
    deadlines = events[["season", "id", "deadline_time"]].rename(columns={"id": "GW"})
    require_keys(deadlines, ["season", "GW"], "saved event deadlines")
    deadlines["deadline_time"] = pd.to_datetime(deadlines.deadline_time, utc=True)
    calendar = calendar.merge(deadlines, on=["season", "GW"], how="left",
                              validate="one_to_one")
    known = calendar.deadline_time.notna()
    calendar.loc[known, "prediction_cutoff"] = calendar.loc[known, "deadline_time"]
    calendar.loc[known, "cutoff_basis"] = "saved_fpl_deadline"
    if not calendar.prediction_cutoff.lt(calendar.first_kickoff).all():
        raise ValueError("Prediction cutoffs must precede their first fixture")
    return calendar.drop(columns="deadline_time").sort_values("prediction_cutoff")


def modelling_rows(matches, fixtures, calendar):
    """Preserve observations and targets, then assign seasons before deriving features."""
    require_keys(matches, MATCH_KEY, "modelling player records")
    require_keys(calendar, ["season", "GW"], "prediction calendar")
    if not matches.season.isin(SEASON_SPLITS).all():
        raise ValueError("Unconfigured season in modelling records")
    reference = fixtures[["season", "fpl_fixture_id", "event", "kickoff_time", "finished"]]
    linked = matches.merge(reference, on=["season", "fpl_fixture_id"], how="left",
                            suffixes=("", "_fixture"), validate="many_to_one")
    if (not linked.finished.all() or not linked.GW.eq(linked.event).all()
            or not linked.kickoff_time.eq(linked.kickoff_time_fixture).all()):
        raise ValueError("Player records disagree with completed fixture references")
    columns = MATCH_KEY + ["GW", "name", "position", "fpl_team_id", "kickoff_time", "minutes"]
    result = linked[columns].copy()
    current = result.season.eq("2026_2027")
    result["points"] = linked.P_adj.where(~current, linked.P_off).astype(float)
    result["target_basis"] = np.where(current, "P_off", "P_adj")
    if not np.isfinite(result.points).all():
        raise ValueError("Every observed modelling target must be finite")
    result["split"] = result.season.map(SEASON_SPLITS)
    result = result.merge(calendar, on=["season", "GW"], how="left", validate="many_to_one")
    if result.prediction_cutoff.isna().any():
        raise ValueError("Player records lack a prediction cutoff")
    return result.sort_values(["prediction_cutoff", *MATCH_KEY]).reset_index(drop=True)


def split_checks(rows, result_delay_hours=24):
    """Audit temporal order and whole-fixture/gameweek partitioning."""
    if result_delay_hours < 3:
        raise ValueError("Result delay must allow fixtures to finish")
    require_keys(rows, MATCH_KEY, "partitioned records")
    expected = rows.season.map(SEASON_SPLITS)
    checks = {
        "configured_season_assignment": rows.split.eq(expected).all(),
        "one_split_per_fixture": rows.groupby(
            ["season", "fpl_fixture_id"],
        ).split.nunique().eq(1).all(),
        "one_split_per_gameweek": rows.groupby(["season", "GW"]).split.nunique().eq(1).all(),
        "one_cutoff_per_gameweek": rows.groupby(
            ["season", "GW"],
        ).prediction_cutoff.nunique().eq(1).all(),
        "cutoff_precedes_target": rows.prediction_cutoff.lt(rows.kickoff_time).all(),
        "finite_targets": np.isfinite(rows.points).all(),
    }
    boundaries = zip(["train", "validation", "test"], ["validation", "test", "monitor"])
    for earlier, later in boundaries:
        past, future = rows.loc[rows.split.eq(earlier)], rows.loc[rows.split.eq(later)]
        if past.empty or future.empty:
            continue
        last_available = past.kickoff_time.max() + pd.Timedelta(hours=result_delay_hours)
        checks[f"{earlier}_results_before_{later}"] = (
            last_available < future.prediction_cutoff.min()
        )
    result = pd.DataFrame([{"check": key, "passed": bool(value)} for key, value in checks.items()])
    if not result.passed.all():
        raise ValueError("Chronological split validation failed")
    return result


def historical_features(rows, window=5, result_delay_hours=24):
    """Join each cutoff to the latest available same-season player history."""
    if window != 5:
        raise ValueError("This initial feature schema uses a fixed five-record window")
    if result_delay_hours < 3:
        raise ValueError("Result delay must allow fixtures to finish")
    require_keys(rows, MATCH_KEY, "feature records")
    source = rows.sort_values([*PLAYER_KEY, "kickoff_time", "fpl_fixture_id"]).copy()
    if source.duplicated(PLAYER_KEY + ["kickoff_time"]).any():
        raise ValueError("A player cannot have two fixtures at the same kickoff")
    grouped = source.groupby(PLAYER_KEY, sort=False)
    source["prior_records"] = grouped.cumcount() + 1
    for field, label in [("points", "recent_points_5"), ("minutes", "recent_minutes_5")]:
        source[label] = grouped[field].transform(
            lambda values: values.rolling(window, min_periods=1).mean(),
        )
    source["history_available_at"] = source.kickoff_time + pd.Timedelta(hours=result_delay_hours)
    source = source.rename(columns={"fpl_fixture_id": "history_fixture_id", "GW": "history_GW"})
    columns = PLAYER_KEY + FEATURES + [
        "history_available_at", "history_fixture_id", "history_GW",
    ]
    result = pd.merge_asof(
        rows.sort_values("prediction_cutoff"), source[columns].sort_values("history_available_at"),
        by=PLAYER_KEY, left_on="prediction_cutoff", right_on="history_available_at",
        direction="backward", allow_exact_matches=False,
    )
    result["prior_records"] = result.prior_records.fillna(0).astype("Int64")
    valid = result.history_available_at.notna()
    earlier = result.loc[valid, "history_available_at"].lt(
        result.loc[valid, "prediction_cutoff"],
    )
    if not earlier.all():
        raise ValueError("A feature uses results unavailable at its cutoff")
    if result.loc[valid, "history_fixture_id"].eq(result.loc[valid, "fpl_fixture_id"]).any():
        raise ValueError("A target fixture entered its own features")
    return result.sort_values(["prediction_cutoff", *MATCH_KEY]).reset_index(drop=True)


def fit_benchmarks(train):
    """Fit constant references exclusively on the training partition."""
    if train.empty or not train.split.eq("train").all():
        raise ValueError("Benchmark fitting accepts training records only")
    return {"mean": float(train.points.mean()),
            "position_mean": train.groupby("position").points.mean().to_dict()}


def predict_benchmarks(frame, fitted):
    """Use frozen training means and histories already available at each cutoff."""
    result = frame.copy()
    result["zero"] = 0.0
    result["train_mean"] = fitted["mean"]
    result["train_position_mean"] = result.position.map(
        fitted["position_mean"],
    ).fillna(fitted["mean"])
    result["recent_five"] = result.recent_points_5.fillna(result.train_position_mean)
    if not np.isfinite(result[PREDICTIONS].to_numpy()).all():
        raise ValueError("Non-finite benchmark prediction")
    return result


def benchmark_metrics(predictions, by=()):
    """Evaluate development predictions without exposing reserved target partitions."""
    if predictions.empty or not predictions.split.eq("validation").all():
        raise ValueError("Development scoring accepts validation records only")
    group_key = by[0] if len(by) == 1 else list(by)
    groups = predictions.groupby(group_key) if by else [((), predictions)]
    rows = []
    for key, frame in groups:
        values = key if isinstance(key, tuple) else (key,)
        identifiers = dict(zip(by, values))
        for model in PREDICTIONS:
            error = frame[model] - frame.points
            rows.append({**identifiers, "model": model, "records": len(frame),
                         "MAE": error.abs().mean(), "RMSE": np.sqrt(error.pow(2).mean()),
                         "bias": error.mean()})
    return pd.DataFrame(rows)
