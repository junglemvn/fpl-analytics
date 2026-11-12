"""Fixture-derived historical team performance and player-contribution concentration."""

import numpy as np
import pandas as pd

from .data_preparation import MATCH_KEY, require_keys
from .eda_preparation import TEAM_FIXTURE_KEY


TEAM_SEASON_KEY = ["season", "fpl_team_id", "team", "short_name"]


def team_results(player_matches, fixtures, teams, coverage):
    """Use completed fixture scores and coverage-checked player xG, once per club-fixture."""
    require_keys(player_matches, MATCH_KEY, "historical team player records")
    require_keys(fixtures, ["season", "fpl_fixture_id"], "fixtures")
    require_keys(teams, ["season", "fpl_team_id"], "team references")
    require_keys(coverage, ["season", "fpl_fixture_id"], "expected-stat coverage")
    complete = fixtures.loc[
        fixtures.finished & fixtures.season.isin(player_matches.season.unique())
    ].copy()
    if complete.empty:
        raise ValueError("No completed fixtures are available for team analysis")
    if player_matches[["goals_scored", "own_goals"]].isna().any().any():
        raise ValueError("Player goal reconciliation requires complete recorded goal counts")
    scores = complete[["team_h_score", "team_a_score"]]
    if scores.isna().any().any() or scores.lt(0).any().any():
        raise ValueError("Completed fixtures need nonnegative final scores")
    if scores.ne(np.floor(scores)).any().any():
        raise ValueError("Fixture scores must be integer goal counts")
    if complete.home_fpl_team_id.eq(complete.away_fpl_team_id).any():
        raise ValueError("A club cannot play itself")
    sides = []
    for side, opponent in [("home", "away"), ("away", "home")]:
        own_score = "team_h_score" if side == "home" else "team_a_score"
        other_score = "team_a_score" if side == "home" else "team_h_score"
        columns = ["season", "fpl_fixture_id", "kickoff_time", "event",
                   f"{side}_fpl_team_id", f"{opponent}_fpl_team_id", own_score, other_score]
        part = complete[columns].rename(columns={
            f"{side}_fpl_team_id": "fpl_team_id",
            f"{opponent}_fpl_team_id": "opponent_fpl_team_id",
            own_score: "goals_for", other_score: "goals_against", "event": "GW",
        })
        sides.append(part.assign(was_home=side == "home"))
    result = pd.concat(sides, ignore_index=True)
    result[["goals_for", "goals_against"]] = result[["goals_for", "goals_against"]].astype("Int64")
    require_keys(result, TEAM_FIXTURE_KEY, "historical team fixtures")
    played = player_matches.loc[player_matches.minutes.gt(0)]
    attacks = played.groupby(TEAM_FIXTURE_KEY).agg(
        appearances=("minutes", "size"), measured_xg=("expected_goals", "count"),
        xg=("expected_goals", lambda values: values.sum(min_count=1)),
    )
    goals = player_matches.groupby(TEAM_FIXTURE_KEY).agg(
        player_goals=("goals_scored", "sum"), own_goals=("own_goals", "sum"),
    )
    result = result.merge(attacks.join(goals), on=TEAM_FIXTURE_KEY,
                          how="left", validate="one_to_one").merge(
        coverage[["season", "fpl_fixture_id", "suspect_zero_block"]],
        on=["season", "fpl_fixture_id"], how="left", validate="many_to_one",
    )
    if result[["appearances", "suspect_zero_block"]].isna().any().any():
        raise ValueError("Completed team fixtures lack player records or coverage evidence")
    available = result.measured_xg.eq(result.appearances) & ~result.suspect_zero_block
    result["xg"] = result.xg.where(available).astype("Float64")
    opponent_values = result[[*TEAM_FIXTURE_KEY, "xg", "own_goals"]].rename(columns={
        "fpl_team_id": "opponent_fpl_team_id", "xg": "xga",
        "own_goals": "opponent_own_goals",
    })
    result = result.merge(
        opponent_values, on=["season", "fpl_fixture_id", "opponent_fpl_team_id"],
        how="left", validate="one_to_one",
    ).merge(
        teams[["season", "fpl_team_id", "name", "short_name"]].rename(columns={"name": "team"}),
        on=["season", "fpl_team_id"], how="left", validate="many_to_one",
    )
    if result.team.isna().any():
        raise ValueError("Team results contain an unknown club")
    if not result.player_goals.add(result.opponent_own_goals).eq(result.goals_for).all():
        raise ValueError("Player goals plus opponent own goals do not match team fixture scores")
    result["wins"] = result.goals_for.gt(result.goals_against).astype("Int64")
    result["draws"] = result.goals_for.eq(result.goals_against).astype("Int64")
    result["losses"] = result.goals_for.lt(result.goals_against).astype("Int64")
    result["points"] = 3 * result.wins + result.draws
    result["goal_difference"] = result.goals_for - result.goals_against
    result["scored"] = result.goals_for.gt(0).astype("Int64")
    result["clean_sheet"] = result.goals_against.eq(0).astype("Int64")
    result["goals_on_xg_fixtures"] = result.goals_for.where(result.xg.notna())
    result["conceded_on_xga_fixtures"] = result.goals_against.where(result.xga.notna())
    result["paired_xgd"] = (result.xg - result.xga).where(result.xg.notna() & result.xga.notna())
    if len(result) != 2 * len(complete):
        raise ValueError("Team result joins changed the completed fixture calendar")
    return result.sort_values(TEAM_FIXTURE_KEY).reset_index(drop=True)


# Preserve the historical entry point used by earlier working notebooks.
historical_team_results = team_results


def summarize_team_results(matches, by=TEAM_SEASON_KEY):
    """Retain expected-stat exposure and pair actual/expected goals on the same fixtures."""
    result = matches.groupby(list(by)).agg(
        played=("fpl_fixture_id", "size"), wins=("wins", "sum"), draws=("draws", "sum"),
        losses=("losses", "sum"), points=("points", "sum"),
        goals_for=("goals_for", "sum"), goals_against=("goals_against", "sum"),
        scoring_matches=("scored", "sum"), clean_sheets=("clean_sheet", "sum"),
        xg=("xg", lambda values: values.sum(min_count=1)),
        xga=("xga", lambda values: values.sum(min_count=1)),
        xg_fixtures=("xg", "count"), xga_fixtures=("xga", "count"),
        goals_on_xg_fixtures=("goals_on_xg_fixtures", lambda values: values.sum(min_count=1)),
        conceded_on_xga_fixtures=("conceded_on_xga_fixtures", lambda x: x.sum(min_count=1)),
        xgd=("paired_xgd", lambda values: values.sum(min_count=1)),
        xgd_fixtures=("paired_xgd", "count"),
    ).reset_index()
    result["goal_difference"] = result.goals_for - result.goals_against
    for metric in ["points", "goals_for", "goals_against"]:
        result[f"{metric}_per_match"] = result[metric] / result.played
    for metric, field in [("win_rate", "wins"), ("scoring_rate", "scoring_matches"),
                          ("clean_sheet_rate", "clean_sheets")]:
        result[metric] = result[field] / result.played
    result["unbeaten_rate"] = (result.wins + result.draws) / result.played
    result["xg_per_match"] = result.xg / result.xg_fixtures.replace(0, np.nan)
    result["xga_per_match"] = result.xga / result.xga_fixtures.replace(0, np.nan)
    result["goals_minus_xg"] = result.goals_on_xg_fixtures - result.xg
    result["conceded_minus_xga"] = result.conceded_on_xga_fixtures - result.xga
    if not (result.wins + result.draws + result.losses).eq(result.played).all():
        raise ValueError("Team W/D/L does not equal played fixtures")
    return result


def rank_team_results(profiles):
    """Order results-based points, GD and GF; retain shared ranks for remaining ties."""
    require_keys(profiles, ["season", "fpl_team_id"], "team-season results")
    result = profiles.sort_values(
        ["season", "points", "goal_difference", "goals_for", "fpl_team_id"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    keys = result[["season", "points", "goal_difference", "goals_for"]]
    ties = keys.ne(keys.shift()).any(axis=1).cumsum()
    order = result.groupby("season").cumcount() + 1
    result["results_rank"] = order.groupby(ties).transform("min").astype("Int64")
    return result


def contribution_concentration(player_matches, profiles):
    """Attribute credited G+A to match-time clubs, keeping transferred-player stints separate."""
    require_keys(player_matches, MATCH_KEY, "player contribution records")
    require_keys(profiles, ["season", "fpl_team_id"], "team contribution denominators")
    ordered = player_matches.sort_values(["kickoff_time", "fpl_fixture_id"], kind="stable")
    players = ordered.groupby(["season", "fpl_team_id", "fpl_player_id"]).agg(
        name=("name", "last"), goals=("goals_scored", "sum"), assists=("assists", "sum"),
    ).reset_index()
    players["goal_involvements"] = players.goals + players.assists
    key = ["season", "fpl_team_id"]
    total = players.groupby(key).goal_involvements.sum()
    players = players.sort_values(
        key + ["goal_involvements", "fpl_player_id"], ascending=[True, True, False, True],
    )
    top3 = players.groupby(key, sort=False).head(3).groupby(key).agg(
        top3_involvements=("goal_involvements", "sum"),
        top3_players=("name", lambda values: "; ".join(values)),
    )
    top3["club_involvements"] = total
    top3["top3_involvement_share"] = top3.top3_involvements / total.where(total.gt(0))
    top3["leading_scorer_goals"] = players.groupby(key).goals.max()
    result = profiles[key + ["team", "short_name", "goals_for"]].merge(
        top3, on=key, how="left", validate="one_to_one",
    )
    result["leading_scorer_goal_share"] = (
        result.leading_scorer_goals / result.goals_for.where(result.goals_for.gt(0))
    )
    return {"teams": result, "player_stints": players}


def validate_team_results(matches, fixtures):
    """Check source totals, two-sided conservation and results-based point allocation."""
    rows = []
    for season, frame in matches.groupby("season"):
        source = fixtures.loc[fixtures.season.eq(season) & fixtures.finished]
        draws = source.team_h_score.eq(source.team_a_score).sum()
        expected_goals = source[["team_h_score", "team_a_score"]].sum().sum()
        checks = {
            "two_team_records": len(frame) == 2 * len(source),
            "wins_equal_losses": frame.wins.sum() == frame.losses.sum(),
            "draw_records": frame.draws.sum() == 2 * draws,
            "goals_for_match_source": frame.goals_for.sum() == expected_goals,
            "goals_against_match_source": frame.goals_against.sum() == expected_goals,
            "results_points": frame.points.sum() == 3 * (len(source) - draws) + 2 * draws,
            "one_result_per_team": (frame.wins + frame.draws + frame.losses).eq(1).all(),
        }
        rows.extend({"season": season, "check": name, "passed": bool(passed)}
                    for name, passed in checks.items())
    result = pd.DataFrame(rows)
    if not result.passed.all():
        raise ValueError("Historical team results failed source-total validation")
    return result
