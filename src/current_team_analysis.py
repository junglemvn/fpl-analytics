"""Current team form and observed player production in pre-match opponent context."""

import numpy as np
import pandas as pd

from .current_player_analysis import club_fixture_windows
from .data_preparation import MATCH_KEY, PLAYER_KEY, require_keys
from .eda_preparation import TEAM_FIXTURE_KEY
from .team_analysis import (
    TEAM_SEASON_KEY, rank_team_results, summarize_team_results, team_results, validate_team_results,
)


def prematch_opponent_context(matches):
    """Use only earlier completed fixtures from this season, without current-match leakage."""
    require_keys(matches, TEAM_FIXTURE_KEY, "team context fixtures")
    if matches.kickoff_time.isna().any():
        raise ValueError("Pre-match context requires recorded kickoff times")
    ordered = matches.sort_values(["kickoff_time", "fpl_fixture_id"], kind="stable").copy()
    keys = ["season", "fpl_team_id"]
    if ordered.duplicated(keys + ["kickoff_time"]).any():
        raise ValueError("A club has multiple fixtures at the same kickoff")
    grouped = ordered.groupby(keys, sort=False)
    ordered["prior_played"] = grouped.cumcount()
    for field in ["points", "xg", "xga"]:
        values = ordered[field].astype(float)
        valid = values.notna().astype(int)
        sums = values.fillna(0).groupby([ordered[key] for key in keys]).cumsum()
        counts = valid.groupby([ordered[key] for key in keys]).cumsum()
        denominator = (counts - valid).replace(0, np.nan)
        ordered[f"prior_{field}_per_match"] = (sums - values.fillna(0)) / denominator
        ordered[f"prior_{field}_fixtures"] = counts - valid
    fields = [field for field in ordered if field.startswith("prior_")]
    opponents = ordered[TEAM_FIXTURE_KEY + ["team"] + fields].rename(columns={
        "fpl_team_id": "opponent_fpl_team_id", "team": "opponent",
        **{field: f"opponent_{field}" for field in fields},
    })
    result = ordered.drop(columns=fields).merge(
        opponents, on=["season", "fpl_fixture_id", "opponent_fpl_team_id"],
        how="left", validate="one_to_one",
    )
    if result.opponent.isna().any():
        raise ValueError("Team fixtures lack an opposing record")
    return result.sort_values(TEAM_FIXTURE_KEY).reset_index(drop=True)


def current_team_progress(players, fixtures, teams, coverage, window_size=5):
    """Retain all completed results, then compare recent and earlier disjoint club windows."""
    seasons = players.season.unique()
    if len(seasons) != 1:
        raise ValueError("Current team progress requires one season")
    matches = team_results(players, fixtures, teams, coverage)
    validation = validate_team_results(matches, fixtures)
    context = prematch_opponent_context(matches)
    windows = club_fixture_windows(fixtures, seasons[0], window_size)
    linked = context.merge(windows[TEAM_FIXTURE_KEY], on=TEAM_FIXTURE_KEY,
                           how="left", validate="one_to_one", indicator=True)
    recent = linked.loc[linked._merge.eq("both")].drop(columns="_merge")
    earlier = linked.loc[linked._merge.eq("left_only")].drop(columns="_merge")
    full = rank_team_results(summarize_team_results(matches))
    recent_profiles = summarize_team_results(recent)
    earlier_profiles = summarize_team_results(earlier)
    summary = full.copy()
    columns = ["played", "points_per_match", "xg_per_match", "xga_per_match",
               "goals_for_per_match", "goals_against_per_match", "clean_sheet_rate"]
    for prefix, frame in [("recent", recent_profiles), ("earlier", earlier_profiles)]:
        summary = summary.merge(
            frame[["season", "fpl_team_id"] + columns].rename(columns={
                field: f"{prefix}_{field}" for field in columns
            }), on=["season", "fpl_team_id"], how="left", validate="one_to_one",
        )
    summary["recent_minus_earlier_ppg"] = (
        summary.recent_points_per_match - summary.earlier_points_per_match
    )
    venues = summarize_team_results(matches, TEAM_SEASON_KEY + ["was_home"])
    return {"matches": matches, "context": context, "profiles": summary, "venues": venues,
            "recent": recent, "earlier": earlier, "validation": validation}


def player_team_context(players, recent_records, team_progress, minimum_prior_games=3):
    """Join dated club form and minute-weighted opponents; preserve all snapshot players."""
    require_keys(players, PLAYER_KEY, "current context players")
    require_keys(recent_records, MATCH_KEY, "recent context records")
    if minimum_prior_games < 1:
        raise ValueError("Opponent context requires a positive prior-match cutoff")
    fields = ["opponent_prior_points_per_match", "opponent_prior_xg_per_match",
              "opponent_prior_xga_per_match"]
    fixture_context = team_progress["context"]
    valid = fixture_context.opponent_prior_played.ge(minimum_prior_games)
    fixture_context = fixture_context.copy()
    for metric in ["points", "xg", "xga"]:
        enough = fixture_context[f"opponent_prior_{metric}_fixtures"].ge(minimum_prior_games)
        field = f"opponent_prior_{metric}_per_match"
        fixture_context[field] = fixture_context[field].where(valid & enough)
    linked = recent_records.merge(
        fixture_context[TEAM_FIXTURE_KEY + fields], on=TEAM_FIXTURE_KEY,
        how="left", validate="many_to_one", indicator=True,
    )
    if not linked._merge.eq("both").all():
        raise ValueError("Recent player records lack corresponding team fixtures")
    linked = linked.drop(columns="_merge")
    played = linked.loc[linked.minutes.gt(0)]
    result = players.copy()
    for field in fields:
        measured = played.loc[played[field].notna()].copy()
        measured["weighted"] = measured[field] * measured.minutes
        grouped = measured.groupby(PLAYER_KEY).agg(
            weighted=("weighted", "sum"), context_minutes=("minutes", "sum"),
        )
        grouped[f"faced_{field.removeprefix('opponent_prior_')}"] = (
            grouped.weighted / grouped.context_minutes
        )
        result = result.merge(
            grouped.drop(columns="weighted").rename(columns={
                "context_minutes": f"{field}_minutes",
            }), on=PLAYER_KEY, how="left", validate="one_to_one",
        )
    club_fields = ["recent_points_per_match", "recent_xg_per_match", "recent_xga_per_match"]
    clubs = team_progress["profiles"][["season", "fpl_team_id"] + club_fields].rename(
        columns={"fpl_team_id": "snapshot_fpl_team_id",
                 **{field: f"team_{field}" for field in club_fields}},
    )
    return result.merge(clubs, on=["season", "snapshot_fpl_team_id"],
                        how="left", validate="many_to_one")
