"""Prepare EDA copies without changing Notebook 1's working tables."""

import json
from pathlib import Path
import tarfile

import pandas as pd

from .data_preparation import MATCH_KEY, PLAYER_KEY, require_keys
from .fotmob_identity import (
    PLAYER_NAME_ALIASES, build_fixture_identity_map, build_player_identity_map,
    build_team_identity_map, external_match_inputs, load_fotmob_identity_lookups,
)


TEAM_FIXTURE_KEY = ["season", "fpl_fixture_id", "fpl_team_id"]
EXTERNAL_TEAM_KEY = ["season", "fotmob_fixture_id", "fotmob_team_id"]
ICT_FIELDS = ["influence", "creativity", "threat", "ict_index"]
DEFENSIVE_FIELDS = [
    "clearances_blocks_interceptions", "tackles", "recoveries", "defensive_contribution",
]


def profile_team_starts(matches, column="starts"):
    """Distinguish all-zero start blocks from credible eleven-start team records."""
    require_keys(matches, MATCH_KEY, "player fixtures")
    return matches.groupby(TEAM_FIXTURE_KEY).agg(
        records=(column, "size"), known_starts=(column, "count"),
        starters=(column, lambda values: values.sum(min_count=1)),
        minutes=("minutes", "sum"), GW=("GW", "first"),
    ).reset_index()


def build_start_bridges(tables, cache_path):
    """Reuse Notebook 1's name, fixture and match-time-club evidence."""
    matches = tables["player_matches"]
    lookups = load_fotmob_identity_lookups(cache_path, matches.season.unique())
    teams = build_team_identity_map({
        season: frame.rename(columns={"fpl_team_id": "id"})
        for season, frame in tables["teams"].groupby("season")
    }, lookups["team_seasons"])
    fixture_input, player_input = external_match_inputs(matches, tables["fixtures"], teams)
    fixtures = build_fixture_identity_map(fixture_input, lookups["fixtures"])["mapping"]
    aliases = [
        (f"{row.first_name} {row.second_name}".strip(), alias.strip())
        for row in tables["current_players"].itertuples(index=False)
        for alias in (row.web_name, row.known_name)
        if isinstance(alias, str) and alias.strip()
    ]
    players = build_player_identity_map(
        player_input, fixtures, lookups["player_matches"], lookups["identity_issues"],
        PLAYER_NAME_ALIASES, aliases,
    )["mapping"]
    return {"teams": teams, "fixtures": fixtures, "players": players,
            "fixture_lookup": lookups["fixtures"]}


def load_lineup_starters(cache_path, fixture_lookup):
    """Read explicit starter lists; incomplete lineups cannot establish non-starts."""
    fixtures = fixture_lookup.loc[fixture_lookup.finished & ~fixture_lookup.cancelled]
    require_keys(fixtures, ["source_member"], "FotMob cache members")
    require_keys(fixtures, ["season", "fotmob_fixture_id"], "FotMob fixtures")
    members = fixtures.set_index("source_member").to_dict("index")
    starters, teams, seen = [], [], set()
    with tarfile.open(Path(cache_path), "r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in members:
                continue
            if member.name in seen:
                raise ValueError(f"Repeated cache member: {member.name}")
            seen.add(member.name)
            fixture = members[member.name]
            payload = json.load(archive.extractfile(member))
            if int(payload["general"]["matchId"]) != fixture["fotmob_fixture_id"]:
                raise ValueError(f"Fixture identity changed in {member.name}")
            lineup = payload.get("content", {}).get("lineup") or {}
            for side in ("home", "away"):
                expected_team = fixture[f"{side}_fotmob_team_id"]
                team = lineup.get(f"{side}Team") or {}
                entries = team.get("starters") or []
                ids = [player.get("id") for player in entries]
                valid_ids = all(type(value) is int and value > 0 for value in ids)
                complete = (team.get("id") == expected_team and len(ids) == 11
                            and valid_ids and len(set(ids)) == 11)
                context = {
                    "season": fixture["season"],
                    "fotmob_fixture_id": fixture["fotmob_fixture_id"],
                    "fotmob_team_id": expected_team, "source_member": member.name,
                }
                teams.append({**context, "listed_starters": len(ids),
                              "lineup_complete": complete})
                if complete:
                    starters.extend({**context, "fotmob_player_id": player["id"]}
                                    for player in entries)
    if seen != set(members):
        raise ValueError(f"Missing {len(set(members) - seen)} selected cache members")
    columns = EXTERNAL_TEAM_KEY + ["source_member", "fotmob_player_id"]
    return {"starters": pd.DataFrame(starters, columns=columns), "teams": pd.DataFrame(teams)}


def repair_starts(matches, lineups, bridges):
    """Validate healthy overlap and fill only all-zero team-fixture start gaps."""
    require_keys(matches, MATCH_KEY, "FPL player fixtures")
    if not matches.starts.isin([0, 1]).all():
        raise ValueError("Source starts must be complete and binary")
    fixture_map = bridges["fixtures"].loc[
        bridges["fixtures"].mapping_status.eq("matched"),
        ["season", "fotmob_fixture_id", "fpl_fixture_id"],
    ]
    team_map = bridges["teams"][["season", "fotmob_team_id", "fpl_team_id"]]
    player_map = bridges["players"].loc[
        bridges["players"].mapping_status.eq("matched"),
        ["season", "fotmob_player_id", "fpl_player_id"],
    ]
    for frame, key in [(fixture_map, ["season", "fotmob_fixture_id"]),
                       (fixture_map, ["season", "fpl_fixture_id"]),
                       (team_map, ["season", "fotmob_team_id"]),
                       (team_map, ["season", "fpl_team_id"]),
                       (player_map, ["season", "fotmob_player_id"]),
                       (player_map, PLAYER_KEY)]:
        require_keys(frame, key, "start repair bridge")

    def translate(frame):
        return frame.merge(fixture_map, on=["season", "fotmob_fixture_id"],
                           how="left", validate="many_to_one").merge(
            team_map, on=["season", "fotmob_team_id"], how="left", validate="many_to_one",
        )

    starters = translate(lineups["starters"]).merge(
        player_map, on=["season", "fotmob_player_id"], how="left", validate="many_to_one",
    )
    record_key = MATCH_KEY + ["fpl_team_id"]
    records = pd.MultiIndex.from_frame(matches[record_key])
    starters["record_present"] = pd.MultiIndex.from_frame(starters[record_key]).isin(records)
    mapped = starters.loc[starters.record_present].copy()
    require_keys(mapped, MATCH_KEY, "mapped FotMob starters")
    counts = mapped.groupby(TEAM_FIXTURE_KEY).size().rename("mapped_starters")
    teams = translate(lineups["teams"]).merge(
        counts, on=TEAM_FIXTURE_KEY, how="left", validate="one_to_one",
    )
    teams["mapped_starters"] = teams.mapped_starters.fillna(0).astype("Int64")
    teams["usable"] = teams.lineup_complete & teams.mapped_starters.eq(11)
    require_keys(teams, TEAM_FIXTURE_KEY, "mapped lineup teams")

    comparison = matches[MATCH_KEY + ["fpl_team_id", "name", "minutes", "starts"]].merge(
        teams[TEAM_FIXTURE_KEY + ["usable", "source_member"]],
        on=TEAM_FIXTURE_KEY, how="left", validate="many_to_one", sort=False,
    )
    comparison["usable"] = comparison.usable.fillna(False).astype(bool)
    comparison["lineup_start"] = pd.Series(
        pd.MultiIndex.from_frame(comparison[record_key]).isin(
            pd.MultiIndex.from_frame(mapped[record_key])
        ), index=comparison.index, dtype="Int64",
    ).where(comparison.usable)
    profile = profile_team_starts(matches)
    gaps = profile.loc[profile.starters.eq(0) & profile.minutes.gt(0)]
    gap_keys = pd.MultiIndex.from_frame(gaps[TEAM_FIXTURE_KEY])
    comparison["start_gap"] = pd.MultiIndex.from_frame(
        comparison[TEAM_FIXTURE_KEY]
    ).isin(gap_keys)
    healthy = profile.loc[profile.starters.eq(11), TEAM_FIXTURE_KEY]
    healthy_rows = pd.MultiIndex.from_frame(comparison[TEAM_FIXTURE_KEY]).isin(
        pd.MultiIndex.from_frame(healthy)
    )
    comparison["compared"] = healthy_rows & comparison.usable
    comparison["disagreement"] = (
        comparison.compared & comparison.starts.ne(comparison.lineup_start).fillna(False)
    )
    if comparison.loc[comparison.start_gap, "lineup_start"].isna().any():
        raise ValueError("Some all-zero start gaps lack eleven validated mapped starters")
    if comparison.disagreement.any():
        examples = comparison.loc[comparison.disagreement, MATCH_KEY].head().to_dict("records")
        raise ValueError(f"FotMob starter lists disagree with credible FPL starts: {examples}")

    result = matches.copy()
    result["starts_eda"] = matches.starts.astype("Int64")
    repaired = comparison.set_index(MATCH_KEY).lineup_start.reindex(
        pd.MultiIndex.from_frame(matches[MATCH_KEY])
    )
    affected = pd.MultiIndex.from_frame(matches[TEAM_FIXTURE_KEY]).isin(gap_keys)
    result.loc[affected, "starts_eda"] = repaired.loc[affected].to_numpy()
    audit = comparison.loc[comparison.start_gap].copy()
    audit["changed"] = audit.starts.ne(audit.lineup_start)
    validation = comparison.groupby("season").agg(
        compared_rows=("compared", "sum"), disagreements=("disagreement", "sum"),
    )
    if not profile_team_starts(result, "starts_eda").starters.eq(11).all():
        raise ValueError("Repaired FPL team fixtures do not each contain eleven starts")
    pd.testing.assert_frame_equal(result[matches.columns], matches)
    return {
        "matches": result, "audit": audit, "validation": validation, "gaps": gaps,
        "lineup_teams": teams, "starter_review": starters.loc[~starters.record_present],
    }


def mask_ict_gaps(matches):
    """Set only all-zero team index blocks with recorded playing time to NA."""
    require_keys(matches, MATCH_KEY, "player fixtures")
    totals = matches.groupby(TEAM_FIXTURE_KEY)[ICT_FIELDS + ["minutes"]].sum(min_count=1)
    zero_indices = totals[ICT_FIELDS].eq(0).fillna(False).all(axis=1)
    gaps = totals.loc[zero_indices & totals.minutes.gt(0)]
    affected = pd.MultiIndex.from_frame(matches[TEAM_FIXTURE_KEY]).isin(gaps.index)
    result = matches.copy()
    result[ICT_FIELDS] = result[ICT_FIELDS].astype("Float64")
    result.loc[affected, ICT_FIELDS] = pd.NA
    audit = matches.loc[affected, MATCH_KEY + ["fpl_team_id"] + ICT_FIELDS].copy()
    audit["reason"] = "all_zero_team_indices_with_positive_minutes"
    return {"matches": result, "gaps": gaps.reset_index(), "audit": audit}


def player_profiles(matches):
    """Summarize observed participation and activity with matched minute exposure."""
    key = PLAYER_KEY + ["position"]
    ordered = matches.sort_values(["kickoff_time", "fpl_fixture_id"], kind="stable")
    result = ordered.groupby(key).agg(
        name=("name", "last"), records=("minutes", "size"),
        appearances=("minutes", lambda values: values.gt(0).sum()),
        minutes=("minutes", "sum"), starts=("starts_eda", "sum"),
        sixty_plus=("minutes", lambda values: values.ge(60).sum()),
    )
    result["appearance_rate"] = result.appearances / result.records
    result["start_rate"] = result.starts / result.records
    result["sixty_plus_rate"] = result.sixty_plus / result.records
    played = ordered.loc[ordered.minutes.gt(0)]
    for field in ["creativity", "threat", *DEFENSIVE_FIELDS]:
        valid = played.loc[played[field].notna()]
        exposure = valid.groupby(key).minutes.sum()
        totals = valid.groupby(key)[field].sum(min_count=1)
        result[f"{field}_per90"] = 90 * totals / exposure
        result[f"{field}_minutes"] = exposure
    return result.reset_index()


def team_activity(matches, fixtures, teams):
    """Join additive activity to the completed two-sided fixture calendar."""
    complete = fixtures.loc[fixtures.finished]
    calendar = pd.concat([
        complete[["season", "fpl_fixture_id", "kickoff_time", f"{side}_fpl_team_id"]]
        .rename(columns={f"{side}_fpl_team_id": "fpl_team_id"}).assign(was_home=side == "home")
        for side in ("home", "away")
    ], ignore_index=True)
    require_keys(calendar, TEAM_FIXTURE_KEY, "team fixture calendar")
    counts = matches.groupby(TEAM_FIXTURE_KEY).agg(
        records=("minutes", "size"), appearances=("minutes", lambda values: values.gt(0).sum()),
        minutes=("minutes", "sum"), starts=("starts_eda", "sum"),
    )
    fields = ["creativity", "threat", *DEFENSIVE_FIELDS]
    totals = matches.groupby(TEAM_FIXTURE_KEY)[fields].sum(min_count=1)
    result = calendar.merge(counts.join(totals), on=TEAM_FIXTURE_KEY,
                            how="left", validate="one_to_one").merge(
        teams[["season", "fpl_team_id", "name"]].rename(columns={"name": "team"}),
        on=["season", "fpl_team_id"], how="left", validate="many_to_one",
    ).sort_values(["season", "fpl_team_id", "kickoff_time", "fpl_fixture_id"])
    gaps = result.groupby(["season", "fpl_team_id"]).kickoff_time.diff()
    result["rest_days"] = gaps.dt.total_seconds() / 86400
    if result[["team", "records"]].isna().any().any():
        raise ValueError("Completed team fixtures are missing names or player records")
    return result.reset_index(drop=True)
