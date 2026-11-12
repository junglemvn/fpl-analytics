"""Compact historical comparisons with explicit denominators and coverage."""

import numpy as np
import pandas as pd

from .team_analysis import summarize_team_results


def seasonal_point_patterns(components):
    """Compare awarded sources per appearance, keeping scoring regimes separate."""
    played = components.loc[components.minutes.gt(0)].copy()
    played["Attacking"] = played.goals + played.assists
    played["Goalkeeping"] = played.saves + played.penalty_saves
    fields = {
        "appearance": "Appearance", "Attacking": "Attacking",
        "clean_sheets": "Clean sheets", "defensive": "Defensive",
        "Goalkeeping": "Goalkeeping", "bonus": "Bonus",
    }
    grouped = played.groupby(["season", "position"])
    detail = grouped[list(fields)].mean().rename(columns=fields)
    detail["appearances"] = grouped.size()
    detail["mean_official_points"] = grouped.P_off.mean()
    detail["largest_source"] = detail[list(fields.values())].idxmax(axis=1)
    secondary = [name for name in fields.values() if name != "Appearance"]
    detail["largest_other_source"] = detail[secondary].idxmax(axis=1)
    rows = []
    for position, frame in detail.reset_index().groupby("position"):
        other = frame.largest_other_source.value_counts()
        rows.append({
            "position": position, "seasons": len(frame),
            "appearance_largest": int(frame.largest_source.eq("Appearance").sum()),
            "largest_other_source": other.index[0], "other_source_repeats": int(other.iloc[0]),
            "min_points_per_appearance": frame.mean_official_points.min(),
            "max_points_per_appearance": frame.mean_official_points.max(),
        })
    return {"detail": detail.reset_index(), "recurrence": pd.DataFrame(rows)}


def seasonal_team_patterns(matches, profiles):
    """Relate team rates within seasons; pair xGD and points on covered fixtures only."""
    rows = []
    for season, frame in matches.groupby("season"):
        all_profiles = profiles.loc[profiles.season.eq(season)].copy()
        all_profiles["gd_per_match"] = all_profiles.goal_difference / all_profiles.played
        paired = frame.loc[frame.paired_xgd.notna()]
        expected = summarize_team_results(paired)
        expected["xgd_per_match"] = expected.xgd / expected.played
        row = {"season": season, "clubs": len(all_profiles),
               "all_team_fixtures": len(frame), "paired_team_fixtures": len(paired)}
        for field, label in [("goals_for_per_match", "scored_rho"),
                             ("goals_against_per_match", "conceded_rho"),
                             ("gd_per_match", "gd_rho")]:
            row[label] = _rank_correlation(all_profiles, field)
        row["xgd_rho"] = _rank_correlation(expected, "xgd_per_match")
        rows.append(row)
    return pd.DataFrame(rows)


def _rank_correlation(frame, field):
    sample = frame[["points_per_match", field]].dropna()
    if len(sample) < 3 or not sample.nunique().gt(1).all():
        return np.nan
    return sample.rank().corr().iloc[0, 1]
