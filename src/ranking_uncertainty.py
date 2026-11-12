"""Conditional gameweek-bootstrap stability of observed points-per-90 rankings."""

import numpy as np
import pandas as pd

from .data_preparation import MATCH_KEY, PLAYER_KEY, require_keys


def bootstrap_rankings(matches, cohort, draws=2000, seed=20261110):
    """Resample shared GW blocks within a season, retaining the observed position cohorts.

    Missing rows contribute neither points nor exposure. Draws without positive exposure
    for every member of a position cohort are excluded for that entire cohort. These
    intervals describe resampling sensitivity, not latent ability or future production.
    """
    require_keys(matches, MATCH_KEY, "ranking records")
    require_keys(cohort, PLAYER_KEY, "ranking cohort")
    if cohort.season.nunique() != 1 or draws < 100:
        raise ValueError("Ranking stability requires one season and at least 100 draws")
    season = cohort.season.iloc[0]
    source = matches.loc[matches.season.eq(season)]
    if source.GW.isna().any() or source[["P_off", "minutes"]].isna().any().any():
        raise ValueError("Ranking records require gameweeks, points and exposure")
    weeks = sorted(source.GW.unique())
    if len(weeks) < 2 or source.minutes.lt(0).any() or cohort.minutes.le(0).any():
        raise ValueError("Ranking stability needs multiple gameweeks and positive exposure")
    weights = np.random.default_rng(seed).multinomial(
        len(weeks), np.full(len(weeks), 1 / len(weeks)), size=draws,
    )
    summaries, comparisons = [], []
    for position, players in cohort.groupby("position", sort=True):
        players = players.sort_values("fpl_player_id").reset_index(drop=True)
        ids = players.fpl_player_id.to_list()
        records = source.loc[source.fpl_player_id.isin(ids)]
        totals = records.groupby(["GW", "fpl_player_id"])[["P_off", "minutes"]].sum()
        matrices = [
            totals[field].unstack("fpl_player_id").reindex(index=weeks, columns=ids)
            .fillna(0).to_numpy(dtype=float)
            for field in ["P_off", "minutes"]
        ]
        points, minutes = matrices
        if not (np.allclose(points.sum(axis=0), players.official_points.astype(float))
                and np.allclose(minutes.sum(axis=0), players.minutes.astype(float))):
            raise ValueError("Ranking cohort totals differ from their observed records")
        observed_rate = 90 * points.sum(axis=0) / minutes.sum(axis=0)
        if not np.allclose(observed_rate, players.official_points_per90.astype(float)):
            raise ValueError("Ranking rates differ from their point and minute totals")
        sampled_minutes = weights @ minutes
        valid = (sampled_minutes > 0).all(axis=1)
        if valid.sum() < draws / 2:
            raise ValueError("Too few draws retain exposure for the full position cohort")
        rates = 90 * (weights[valid] @ points) / sampled_minutes[valid]
        ranks = pd.DataFrame(rates).rank(axis=1, method="min", ascending=False).to_numpy()
        players["rank"] = players.official_points_per90.rank(
            method="min", ascending=False,
        ).astype("Int64")
        for values, prefix in [(rates, "rate"), (ranks, "rank")]:
            players[f"{prefix}_low"], players[f"{prefix}_high"] = np.quantile(
                values, [.025, .975], axis=0,
            )
        players["top3_resample_share"] = (ranks <= 3).mean(axis=0)
        players["valid_draws"], players["draws"] = int(valid.sum()), draws
        players["gameweeks"] = len(weeks)
        summaries.append(players)
        order = players.sort_values(["rank", "fpl_player_id"]).index[:2]
        if len(order) < 2:
            continue
        first, second = [players.index.get_loc(index) for index in order]
        gap = rates[:, first] - rates[:, second]
        low, high = np.quantile(gap, [.025, .975])
        comparisons.append({
            "season": season, "position": position,
            "leader_id": ids[first], "runner_up_id": ids[second],
            "leader": players.iloc[first]["name"], "runner_up": players.iloc[second]["name"],
            "observed_gap": players.iloc[first].official_points_per90
                            - players.iloc[second].official_points_per90,
            "gap_low": low, "gap_high": high,
            "leader_ahead_resample_share": (gap > 1e-10).mean(),
            "tied_resample_share": (np.abs(gap) <= 1e-10).mean(),
            "valid_draws": int(valid.sum()), "draws": draws,
        })
    return {"players": pd.concat(summaries, ignore_index=True),
            "leader_comparisons": pd.DataFrame(comparisons)}
