"""Snapshot-aware current-player progress without invented eligibility or missed fixtures."""

import numpy as np
import pandas as pd

from .data_preparation import MATCH_KEY, PLAYER_KEY, require_keys
from .eda_preparation import TEAM_FIXTURE_KEY
from .player_analysis import performance_profiles


def club_fixture_windows(fixtures, season, size=5):
    """Select each club's latest completed fixtures, excluding future matches."""
    if size < 1:
        raise ValueError("A recent fixture window must have positive size")
    require_keys(fixtures, ["season", "fpl_fixture_id"], "current fixture calendar")
    complete = fixtures.loc[fixtures.season.eq(season) & fixtures.finished]
    calendar = pd.concat([
        complete[["season", "fpl_fixture_id", "kickoff_time", f"{side}_fpl_team_id"]]
        .rename(columns={f"{side}_fpl_team_id": "fpl_team_id"})
        for side in ("home", "away")
    ], ignore_index=True)
    calendar = calendar.sort_values(["season", "fpl_team_id", "kickoff_time", "fpl_fixture_id"])
    result = calendar.groupby(["season", "fpl_team_id"], sort=False).tail(size).copy()
    result["club_fixture_order"] = result.groupby(["season", "fpl_team_id"]).cumcount() + 1
    require_keys(result, TEAM_FIXTURE_KEY, "recent current-club fixtures")
    return result.reset_index(drop=True)


def recent_current_club_records(matches, snapshot, windows):
    """Retain observed rows in the current club's window; absent rows remain unknown."""
    require_keys(matches, MATCH_KEY, "current player records")
    require_keys(snapshot, PLAYER_KEY, "current player snapshot")
    require_keys(windows, TEAM_FIXTURE_KEY, "recent fixture windows")
    current_club = snapshot[PLAYER_KEY + ["fpl_team_id"]].rename(columns={
        "fpl_team_id": "snapshot_fpl_team_id",
    })
    linked = matches.merge(current_club, on=PLAYER_KEY, how="left", validate="many_to_one")
    if linked.snapshot_fpl_team_id.isna().any():
        raise ValueError("Current player records have no corresponding snapshot")
    current = linked.loc[linked.fpl_team_id.eq(linked.snapshot_fpl_team_id)]
    return current.merge(
        windows[TEAM_FIXTURE_KEY + ["club_fixture_order"]],
        on=TEAM_FIXTURE_KEY, how="inner", validate="many_to_one",
    ).sort_values(["season", "fpl_player_id", "club_fixture_order"]).reset_index(drop=True)


def current_player_progress(matches, snapshot, teams, fixtures, coverage, window_size=5):
    """Combine observed production, dated snapshot context and recent-club exposure."""
    require_keys(matches, MATCH_KEY, "current production")
    require_keys(snapshot, PLAYER_KEY, "current player snapshot")
    require_keys(teams, ["season", "fpl_team_id"], "club reference")
    seasons = snapshot.season.unique()
    if len(seasons) != 1 or not matches.season.eq(seasons[0]).all():
        raise ValueError("Current progress must use one snapshot season and its histories")
    profiles = performance_profiles(matches, coverage)
    attributes = snapshot[PLAYER_KEY + [
        "fpl_team_id", "web_name", "now_cost", "selected_by_percent", "status", "can_select",
        "removed", "chance_of_playing_next_round", "retrieved_at_utc",
    ]].rename(columns={
        "fpl_team_id": "snapshot_fpl_team_id", "web_name": "display_name",
        "selected_by_percent": "snapshot_ownership_pct", "retrieved_at_utc": "snapshot_at_utc",
    })
    result = attributes.merge(profiles, on=PLAYER_KEY, how="left", validate="one_to_one")
    if result.records.isna().any():
        raise ValueError("Snapshot players without histories require an explicit coverage review")
    club_names = teams[["season", "fpl_team_id", "name"]].rename(columns={
        "fpl_team_id": "snapshot_fpl_team_id", "name": "current_team",
    })
    result = result.merge(club_names, on=["season", "snapshot_fpl_team_id"],
                          how="left", validate="many_to_one")
    if result.current_team.isna().any():
        raise ValueError("A current player has no current-club reference")
    result["snapshot_price_m"] = result.now_cost / 10
    result["snapshot_points_per_million"] = (
        result.official_points / result.snapshot_price_m.where(result.snapshot_price_m.gt(0))
    )
    windows = club_fixture_windows(fixtures, seasons[0], window_size)
    recent_records = recent_current_club_records(matches, snapshot, windows)
    recent = performance_profiles(recent_records, coverage)
    recent_fields = [
        "records", "appearances", "minutes", "starts", "official_points", "official_points_per90",
        "points_per_record", "goals", "assists", "xg", "xa", "xgi_per90", "sixty_plus_rate",
    ]
    recent = recent[PLAYER_KEY + recent_fields].rename(columns={
        name: f"recent_{name}" for name in recent_fields
    })
    result = result.merge(recent, on=PLAYER_KEY, how="left", validate="one_to_one")
    window_counts = windows.groupby(["season", "fpl_team_id"]).size().rename("club_window_size")
    window_counts = window_counts.reset_index().rename(columns={
        "fpl_team_id": "snapshot_fpl_team_id",
    })
    result = result.merge(window_counts, on=["season", "snapshot_fpl_team_id"],
                          how="left", validate="many_to_one")
    result["recent_records"] = result.recent_records.fillna(0).astype("Int64")
    result["recent_missing_records"] = result.club_window_size - result.recent_records
    if result.recent_missing_records.lt(0).any() or result.club_window_size.isna().any():
        raise ValueError("Recent player records do not match their current-club fixture window")
    result["recent_vs_season_points90"] = (
        result.recent_official_points_per90 - result.official_points_per90
    )
    require_keys(result, PLAYER_KEY, "current player progress")
    comparisons = []
    captured = snapshot.set_index(PLAYER_KEY)
    observed = result.set_index(PLAYER_KEY)
    for metric, field in [("official_points", "total_points"), ("minutes", "minutes"),
                          ("goals", "goals_scored"), ("assists", "assists"), ("starts", "starts")]:
        actual = observed[metric].reindex(captured.index).astype(float)
        reference = captured[field].astype(float)
        comparisons.append({"metric": metric, "players": len(reference),
                            "disagreements": int((~np.isclose(actual, reference)).sum())})
    validation = pd.DataFrame(comparisons)
    if validation.disagreements.any():
        raise ValueError("Current production disagrees with the captured cumulative snapshot")
    return {"players": result, "recent_records": recent_records, "windows": windows,
            "validation": validation}
