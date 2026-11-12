"""Historical point accounting and player-performance analysis."""

import numpy as np
import pandas as pd

from .data_preparation import MATCH_KEY, PLAYER_KEY, require_keys
from .defensive_contributions import dc_points
from .eda_preparation import player_profiles


HISTORICAL_SEASONS = ["2022_2023", "2023_2024", "2024_2025", "2025_2026"]
PROFILE_KEY = PLAYER_KEY + ["position"]
POINT_COMPONENTS = [
    "appearance", "goals", "assists", "clean_sheets", "saves",
    "penalty_saves", "defensive", "bonus", "deductions",
]
COMPONENT_RULE = "historical_components_v1_official_residual"


def point_components(matches):
    """Account for documented awards, preserving official points and any residual."""
    require_keys(matches, MATCH_KEY, "historical point records")
    if not matches.season.isin(HISTORICAL_SEASONS).all():
        raise ValueError("Point decomposition supports the four reviewed historical seasons")
    goal_weights = matches.position.map({"GK": 6, "DEF": 6, "MID": 5, "FWD": 4})
    goal_weights = goal_weights.mask(
        matches.position.eq("GK") & matches.season.ge("2024_2025"), 10,
    )
    clean_weights = matches.position.map({"GK": 4, "DEF": 4, "MID": 1, "FWD": 0})
    if goal_weights.isna().any():
        raise ValueError("Unknown FPL position in point decomposition")
    result = matches[MATCH_KEY + ["name", "position", "minutes", "P_off", "P_adj"]].copy()
    result["appearance"] = matches.minutes.gt(0).astype(int) + matches.minutes.ge(60)
    result["goals"] = matches.goals_scored * goal_weights
    result["assists"] = 3 * matches.assists
    result["clean_sheets"] = matches.clean_sheets * clean_weights
    result["saves"] = matches.saves // 3
    result["penalty_saves"] = 5 * matches.penalties_saved
    result["defensive"] = dc_points(
        matches.defensive_contribution, matches.position,
    ).where(matches.season.eq("2025_2026"), 0)
    result["bonus"] = matches.bonus
    conceded = (matches.goals_conceded // 2).where(matches.position.isin(["GK", "DEF"]), 0)
    result["deductions"] = (
        -matches.yellow_cards - 3 * matches.red_cards - 2 * matches.own_goals
        - 2 * matches.penalties_missed - conceded
    )
    result[POINT_COMPONENTS] = result[POINT_COMPONENTS].astype("Int64")
    result["documented_total"] = result[POINT_COMPONENTS].sum(
        min_count=len(POINT_COMPONENTS), axis=1,
    ).astype("Int64")
    result["official_residual"] = result.P_off - result.documented_total
    result["adjustment"] = result.P_adj - result.P_off
    return result


def point_summary(matches, by=("position",)):
    """Describe official points with explicit all-record and appearance denominators."""
    views = {"All records": matches, "Appearances": matches.loc[matches.minutes.gt(0)]}
    parts = []
    for scope, frame in views.items():
        summary = frame.groupby(list(by)).P_off.agg(
            records="size", mean="mean", median="median", sd="std",
            q25=lambda values: values.quantile(.25), q75=lambda values: values.quantile(.75),
            zero_rate=lambda values: values.eq(0).mean(),
            negative_rate=lambda values: values.lt(0).mean(),
            six_plus_rate=lambda values: values.ge(6).mean(),
            ten_plus_rate=lambda values: values.ge(10).mean(),
        ).reset_index()
        parts.append(summary.assign(scope=scope))
    return pd.concat(parts, ignore_index=True)


def expected_coverage(matches):
    """Flag fixture-wide all-zero attacking expectations with recorded playing time."""
    require_keys(matches, MATCH_KEY, "expected-stat coverage")
    result = matches.groupby(["season", "fpl_fixture_id"]).agg(
        minutes=("minutes", "sum"), GW=("GW", "first"),
        xg=("expected_goals", lambda values: values.sum(min_count=1)),
        xa=("expected_assists", lambda values: values.sum(min_count=1)),
    ).reset_index()
    result["suspect_zero_block"] = (
        result.xg.eq(0) & result.xa.eq(0) & result.minutes.gt(0)
    ).fillna(False)
    return result


def event_rates(matches):
    """Compute event frequencies among positive-minute appearances in each source period."""
    played = matches.loc[matches.minutes.gt(0)].copy()
    played["period"] = np.where(
        played.season.eq("2025_2026"), "2025/26 supplied DC", "2022/23–2024/25 reconstructed DC",
    )
    flags = played[["period", "position"]].copy()
    flags["sixty_plus"] = played.minutes.ge(60)
    flags["attacking_return"] = (played.goals_scored + played.assists).gt(0)
    flags["clean_sheet"] = played.clean_sheets.gt(0).astype("boolean").mask(
        played.position.eq("FWD"), pd.NA,
    )
    flags["defensive_hit"] = dc_points(
        played.defensive_contribution, played.position,
    ).eq(2).mask(played.position.eq("GK"), pd.NA)
    flags["save_reward"] = played.saves.ge(3).astype("boolean").where(
        played.position.eq("GK"), pd.NA,
    )
    flags["bonus_award"] = played.bonus.gt(0)
    keys = ["period", "position"]
    rates = flags.groupby(keys).mean()
    rates["appearances"] = flags.groupby(keys).size()
    rates["defensive_valid_appearances"] = flags.groupby(keys).defensive_hit.count()
    return rates.reset_index()


def performance_profiles(matches, coverage):
    """Aggregate player-season production with metric-specific expected-stat exposure."""
    require_keys(matches, MATCH_KEY, "player-performance records")
    require_keys(coverage, ["season", "fpl_fixture_id"], "expected-stat coverage")
    ordered = matches.sort_values(["kickoff_time", "fpl_fixture_id"], kind="stable")
    result = player_profiles(ordered).set_index(PROFILE_KEY)
    grouped = ordered.groupby(PROFILE_KEY)
    fields = ["P_off", "P_adj", "goals_scored", "assists", "clean_sheets", "saves",
              "penalties_saved", "goals_conceded", "bonus", "yellow_cards", "red_cards"]
    totals = grouped[fields].sum(min_count=1).rename(columns={
        "P_off": "official_points", "P_adj": "adjusted_points", "goals_scored": "goals",
    })
    result = result.join(totals)
    result["goal_involvements"] = result.goals + result.assists
    exposure = result.minutes.where(result.minutes.gt(0))
    for field in ["official_points", "goals", "assists", "goal_involvements",
                  "saves", "goals_conceded"]:
        result[f"{field}_per90"] = 90 * result[field] / exposure
    result["points_per_record"] = result.official_points / result.records
    played = ordered.loc[ordered.minutes.gt(0)]
    summary = played.groupby(PROFILE_KEY).P_off.agg(
        points_per_appearance="mean", median_points="median",
        points_q25=lambda values: values.quantile(.25),
        points_q75=lambda values: values.quantile(.75),
        six_plus_rate=lambda values: values.ge(6).mean(),
        ten_plus_rate=lambda values: values.ge(10).mean(),
    )
    result = result.join(summary)
    result["points_iqr"] = result.points_q75 - result.points_q25
    result["clubs"] = grouped.team.agg(lambda values: ", ".join(dict.fromkeys(values)))
    price = ordered.loc[ordered.value.notna()].groupby(PROFILE_KEY).tail(1)
    price = price.set_index(PROFILE_KEY)[["value", "kickoff_time"]]
    result["last_recorded_price_m"] = price.value / 10
    result["price_fixture_kickoff"] = price.kickoff_time
    valid_price = result.last_recorded_price_m.where(result.last_recorded_price_m.gt(0))
    result["points_per_million"] = result.official_points / valid_price
    checked = ordered.merge(
        coverage[["season", "fpl_fixture_id", "suspect_zero_block"]],
        on=["season", "fpl_fixture_id"], how="left", validate="many_to_one",
    )
    if checked.suspect_zero_block.isna().any():
        raise ValueError("Missing expected-stat coverage for player records")
    for field, label in [("expected_goals", "xg"), ("expected_assists", "xa"),
                         ("expected_goal_involvements", "xgi")]:
        valid = checked.loc[
            checked.minutes.gt(0) & checked[field].notna() & ~checked.suspect_zero_block
        ].groupby(PROFILE_KEY)
        result[label] = valid[field].sum(min_count=1)
        result[f"{label}_minutes"] = valid.minutes.sum().reindex(result.index, fill_value=0)
        valid_minutes = result[f"{label}_minutes"].replace(0, np.nan)
        result[f"{label}_per90"] = 90 * result[label] / valid_minutes
    result["G_minus_xG"] = (result.goals - result.xg).where(result.xg_minutes.eq(result.minutes))
    result["A_minus_xA"] = (result.assists - result.xa).where(result.xa_minutes.eq(result.minutes))
    defensive = played.assign(dc_hit=dc_points(
        played.defensive_contribution, played.position,
    ).eq(2).mask(played.position.eq("GK"), pd.NA))
    result["defensive_hit_rate"] = defensive.groupby(PROFILE_KEY).dc_hit.mean()
    result["defensive_hits"] = defensive.groupby(PROFILE_KEY).dc_hit.sum(min_count=1)
    attacks = played.assign(attacking_return=(played.goals_scored + played.assists).gt(0))
    result["attacking_return_rate"] = attacks.groupby(PROFILE_KEY).attacking_return.mean()
    eligible = played.loc[played.minutes.ge(60) & played.position.ne("FWD")].copy()
    eligible["clean_sheet"] = eligible.clean_sheets.gt(0)
    result["clean_sheet_rate_60plus"] = eligible.groupby(PROFILE_KEY).clean_sheet.mean()
    return result.reset_index()


def rank_players(profiles, metric, minimum_minutes=900):
    """Rank each season/position cohort, sharing competition ranks for equal values."""
    result = profiles.loc[profiles.minutes.ge(minimum_minutes) & profiles[metric].notna()].copy()
    result["rank"] = result.groupby(["season", "position"])[metric].rank(
        method="min", ascending=False,
    ).astype("Int64")
    result["ranking"] = metric
    return result.sort_values(["season", "position", "rank", "fpl_player_id"])


def fit_expected_actual(profiles, expected="xg", actual="goals", minimum_minutes=900):
    """Fit an intercept-bearing descriptive OLS line on a fully observed outfield cohort."""
    require_keys(profiles, PROFILE_KEY, "regression player profiles")
    exposure_field = f"{expected.split('_per90')[0]}_minutes"
    result = profiles.loc[
        profiles.minutes.ge(minimum_minutes) & profiles.position.ne("GK")
        & profiles[exposure_field].eq(profiles.minutes)
        & profiles[[expected, actual]].notna().all(axis=1)
    ].copy()
    x = result[expected].to_numpy(dtype=float)
    y = result[actual].to_numpy(dtype=float)
    if len(x) < 3 or np.ptp(x) == 0:
        raise ValueError("OLS requires at least three players and nonconstant expectations")
    intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    result["fitted"] = intercept + slope * x
    result["fit_residual"] = y - result.fitted
    variance = np.square(y - y.mean()).sum()
    r_squared = 1 - np.square(result.fit_residual).sum() / variance if variance > 0 else np.nan
    statistics = pd.DataFrame([{
        "expected": expected, "actual": actual, "minimum_minutes": minimum_minutes,
        "players": len(x), "intercept": intercept, "slope": slope, "r_squared": r_squared,
    }])
    return {"players": result, "statistics": statistics}


def point_associations(profiles, minimum_minutes=900):
    """Describe outfield rank associations, never forecasting importance."""
    metrics = ["xg_per90", "xa_per90", "creativity_per90", "threat_per90", "sixty_plus_rate"]
    cohort = profiles.loc[
        profiles.minutes.ge(minimum_minutes) & profiles.season.ne("2022_2023")
        & profiles.position.ne("GK")
    ]
    rows = []
    for (season, position), frame in cohort.groupby(["season", "position"]):
        for metric in metrics:
            exposure = f"{metric.removesuffix('_per90')}_minutes"
            sample = frame.loc[frame[exposure].eq(frame.minutes)] if exposure in frame else frame
            sample = sample[[metric, "official_points_per90"]].dropna()
            varying = sample.nunique().gt(1).all()
            coefficient = (
                sample.rank().corr().iloc[0, 1] if len(sample) >= 3 and varying else np.nan
            )
            rows.append({"season": season, "position": position, "metric": metric,
                         "players": len(sample), "spearman": coefficient})
    return pd.DataFrame(rows)


def plot_expected_fit(fit, expected, actual, ax, colors):
    """Draw positional observations, an equality reference and a descriptive OLS line."""
    frame = fit["players"]
    stats = fit["statistics"].iloc[0]
    for position, group in frame.groupby("position", sort=True):
        ax.scatter(group[expected], group[actual], color=colors[position],
                   label=position, alpha=.65, s=24)
    upper = 1.06 * frame[expected].max()
    grid = np.linspace(0, frame[expected].max(), 100)
    ax.plot(grid, grid, color="0.5", linestyle="--", linewidth=1, label="Equality reference")
    ax.plot(grid, stats.intercept + stats.slope * grid, color="0.15", linewidth=1.5,
            label=f"OLS · R²={stats.r_squared:.2f}")
    padding = max(upper * .03, .02)
    upper_y = 1.06 * max(frame[actual].max(), (stats.intercept + stats.slope * grid).max())
    ax.set(xlim=(-padding, upper), ylim=(min(-padding, stats.intercept - padding), upper_y))
    return ax
