"""Extract external defensive actions and validate estimates against FPL counts"""

import ast
import json
from math import isfinite
from numbers import Real
import tarfile

import pandas as pd

from .data_preparation import MATCH_KEY, PLAYER_KEY, require_keys
from .fotmob_identity import reviewed_player_records


STAT_FIELDS = {
    "minutes_played": "external_minutes",
    "clearances": "external_clearances",
    "shot_blocks": "external_shot_blocks",
    "interceptions": "external_interceptions",
    "matchstats.headers.tackles": "external_tackles",
    "tackles": "external_tackles",
    "recoveries": "external_recoveries",
}
STAT_COLUMNS = list(dict.fromkeys(STAT_FIELDS.values()))
CBI_FIELDS = ["external_clearances", "external_shot_blocks", "external_interceptions"]
SOURCE_KEY = ["season", "fotmob_player_id", "fotmob_fixture_id", "fotmob_team_id"]
DC_RULES = "2026_2027_DC_only"
ADJUSTED_RULES = "2026_2027_DC_plus_verified_corrections_v1"
HISTORICAL_SEASONS = ("2022_2023", "2023_2024", "2024_2025")
THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12}
ENRICHMENT_COLUMNS = STAT_COLUMNS + [
    "external_cbi", "external_defensive_count", "DC_estimated", "defensive_status",
    "defensive_source", "defensive_source_member", "defensive_rules",
]
FPL_DEFENSIVE_FIELDS = {
    "clearances_blocks_interceptions": "external_cbi",
    "defensive_contribution": "external_defensive_count",
    "recoveries": "external_recoveries",
    "tackles": "external_tackles",
}


def load_defensive_actions(cache_path, fixture_lookup):
    """Read selected completed fixtures, quarantining invalid IDs and metric conflicts"""
    fixtures = fixture_lookup.loc[fixture_lookup.finished & ~fixture_lookup.cancelled].copy()
    require_keys(fixtures, ["source_member"], "FotMob fixture members")
    require_keys(fixtures, ["fotmob_fixture_id"], "FotMob fixtures")
    members = fixtures.set_index("source_member").to_dict("index")
    rows, issues, coverage, repairs, seen = [], [], [], [], set()
    with tarfile.open(cache_path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in members:
                continue
            if member.name in seen:
                raise ValueError(f"Repeated archive member: {member.name}")
            seen.add(member.name)
            fixture = members[member.name]
            payload = json.load(archive.extractfile(member))
            if int(payload["general"]["matchId"]) != fixture["fotmob_fixture_id"]:
                raise ValueError(f"Fixture ID changed in {member.name}")
            players, reviewed = reviewed_player_records(payload, member.name)
            repairs.extend(reviewed)
            coverage.append({"season": fixture["season"], "fotmob_fixture_id": fixture["fotmob_fixture_id"],
                             "source_member": member.name, "player_records": len(players)})
            for player in players:
                record = {
                    "season": fixture["season"], "fotmob_fixture_id": fixture["fotmob_fixture_id"],
                    "fotmob_player_id": player.get("id"), "fotmob_team_id": player.get("teamId"),
                    "source_player_name": player["name"], "source_member": member.name,
                    "external_is_goalkeeper": player.get("isGoalkeeper"),
                }
                values = {}
                for group in player.get("stats") or []:
                    for stat in group.get("stats", {}).values():
                        key, value = stat.get("key"), stat.get("stat", {}).get("value")
                        if key in STAT_FIELDS and value is not None:
                            values.setdefault(STAT_FIELDS[key], []).append((key, value))
                for field in STAT_COLUMNS:
                    observations = values.get(field, [])
                    numbers = [v for _, v in observations]
                    valid = all(isinstance(v, Real) and not isinstance(v, bool) and isfinite(v) and v >= 0 and v == int(v) for v in numbers)
                    distinct = set(numbers) if valid else set()
                    record[field] = next(iter(distinct)) if valid and len(distinct) == 1 else pd.NA
                    if numbers and (not valid or len(distinct) != 1):
                        issues.append({**record, "field": field, "observations": tuple(observations),
                                       "issue": "conflicting_or_invalid_stat"})
                record["source_stat_keys"] = tuple(sorted({key for pairs in values.values() for key, _ in pairs}))
                rows.append(record)
    if seen != set(members):
        raise ValueError(f"Missing {len(set(members) - seen)} selected archive members")
    actions = pd.DataFrame(rows)
    if actions.empty:
        raise ValueError("No selected FotMob player statistics")
    for column in SOURCE_KEY[1:] + STAT_COLUMNS:
        actions[column] = pd.to_numeric(actions[column], errors="raise").astype("Int64")
    actions["external_is_goalkeeper"] = actions.external_is_goalkeeper.astype("boolean")
    invalid = actions.fotmob_player_id.isna() | actions.fotmob_player_id.le(0)
    source_issues, actions = actions.loc[invalid].copy(), actions.loc[~invalid].copy()
    require_keys(actions, SOURCE_KEY[:-1], "FotMob defensive records")
    if actions.fotmob_team_id.isna().any():
        raise ValueError("Missing match-time FotMob team")
    return {"actions": actions.reset_index(drop=True), "source_issues": source_issues.reset_index(drop=True),
            "stat_issues": pd.DataFrame(issues), "fixture_coverage": pd.DataFrame(coverage),
            "source_repairs": pd.DataFrame(repairs)}


def dc_points(count, position):
    """Apply the frozen per-fixture 0-or-2 rule without turning unknown counts into zero"""
    if not position.isin(["GK", *THRESHOLDS]).all():
        raise ValueError("Unknown FPL position")
    points = pd.Series(pd.NA, index=count.index, dtype="Int64")
    eligible = count.notna() & position.isin(THRESHOLDS)
    points.loc[eligible] = count.loc[eligible].ge(position.loc[eligible].map(THRESHOLDS)).astype("Int64") * 2
    points.loc[position.eq("GK")] = 0
    return points


def attach_defensive_actions(matches, actions, fixture_map, player_map, team_map):
    """Attach counts through technical bridges, retaining only FPL working keys"""
    matches = matches.drop(columns=ENRICHMENT_COLUMNS, errors="ignore")
    require_keys(matches, MATCH_KEY, "FPL player matches")
    require_keys(actions, SOURCE_KEY[:-1], "FotMob defensive actions")
    fixtures = fixture_map.loc[fixture_map.mapping_status.eq("matched"), ["season", "fpl_fixture_id", "fotmob_fixture_id"]]
    players = player_map.loc[player_map.mapping_status.eq("matched"), PLAYER_KEY + ["fotmob_player_id"]]
    teams = team_map[["season", "fpl_team_id", "fotmob_team_id"]]
    require_keys(fixtures, ["season", "fpl_fixture_id"], "fixture bridge")
    require_keys(players, PLAYER_KEY, "player bridge")
    require_keys(teams, ["season", "fpl_team_id"], "team bridge")
    joined = matches[MATCH_KEY + ["name", "fpl_team_id", "position", "minutes"]].merge(
        fixtures, on=["season", "fpl_fixture_id"], how="left", validate="many_to_one"
    ).merge(players, on=PLAYER_KEY, how="left", validate="many_to_one").merge(
        teams, on=["season", "fpl_team_id"], how="left", validate="many_to_one"
    ).merge(actions, on=SOURCE_KEY, how="left", validate="many_to_one")
    count = joined[CBI_FIELDS].sum(axis=1, min_count=3).astype("Int64")
    joined["external_cbi"] = count
    count = count + joined.external_tackles
    count += joined.external_recoveries.where(joined.position.isin(["MID", "FWD"]), 0)
    played, recorded = joined.minutes.gt(0), joined.external_minutes.gt(0).fillna(False)
    joined["defensive_status"] = "missing_actions"
    joined.loc[count.notna() & played & recorded, "defensive_status"] = "complete"
    joined.loc[joined.external_minutes.isna(), "defensive_status"] = "missing_source_minutes"
    joined.loc[joined.source_member.isna(), "defensive_status"] = "no_source_record"
    joined.loc[joined.fotmob_player_id.isna(), "defensive_status"] = "unmapped_player"
    joined.loc[joined.fotmob_fixture_id.isna(), "defensive_status"] = "unmapped_fixture"
    joined.loc[joined.fotmob_team_id.isna(), "defensive_status"] = "unmapped_team"
    joined.loc[~played, "defensive_status"] = "not_played"
    conflict = (played & joined.external_minutes.eq(0).fillna(False)) | (~played & recorded)
    conflict |= played & recorded & joined.external_is_goalkeeper.ne(joined.position.eq("GK")).fillna(False)
    joined.loc[conflict, "defensive_status"] = "participation_or_position_conflict"
    joined.loc[joined.position.eq("GK") & ~conflict, "defensive_status"] = "not_applicable_GK"
    joined["external_defensive_count"] = count.where(joined.defensive_status.eq("complete"))
    joined.loc[joined.position.eq("GK") & ~conflict, "external_defensive_count"] = 0
    joined["DC_estimated"] = dc_points(joined.external_defensive_count, joined.position)
    joined.loc[conflict, "DC_estimated"] = pd.NA
    joined["defensive_source"] = "fotmob"
    joined["defensive_rules"] = DC_RULES
    joined = joined.rename(columns={"source_member": "defensive_source_member"})
    enriched = matches.merge(joined[MATCH_KEY + ENRICHMENT_COLUMNS], on=MATCH_KEY, how="left", validate="one_to_one", sort=False)
    if len(enriched) != len(matches):
        raise ValueError("Defensive attachment changed the row count")
    coverage = joined.assign(
        positive_minute_rows=played, positive_outfield_rows=played & joined.position.ne("GK"),
        complete_outfield_rows=played & joined.position.ne("GK") & joined.defensive_status.eq("complete"),
    ).groupby("season").agg(
        player_fixture_rows=("minutes", "size"), source_records=("defensive_source_member", "count"),
        positive_minute_rows=("positive_minute_rows", "sum"), positive_outfield_rows=("positive_outfield_rows", "sum"),
        complete_outfield_rows=("complete_outfield_rows", "sum"),
    )
    return {"matches": enriched, "coverage": coverage,
            "review": joined.loc[~joined.defensive_status.isin(["complete", "not_played", "not_applicable_GK"])]}


def fill_historical_defensive_fields(matches, diagnostics, seasons=HISTORICAL_SEASONS):
    """Fill existing historical FPL fields, preserving official seasons and separate provenance"""
    if any(int(season[:4]) >= 2025 for season in seasons):
        raise ValueError("FPL defensive measurements from 2025/26 onward must remain unchanged")
    frame = matches.drop(columns=ENRICHMENT_COLUMNS, errors="ignore")
    require_keys(frame, MATCH_KEY, "FPL player matches")
    require_keys(diagnostics, MATCH_KEY, "defensive comparison records")
    if not set(FPL_DEFENSIVE_FIELDS).issubset(frame.columns):
        raise ValueError("Expected the shared FPL schema with existing defensive columns")
    aligned = diagnostics.set_index(MATCH_KEY).reindex(pd.MultiIndex.from_frame(frame[MATCH_KEY]))
    aligned.index = frame.index
    eligible = frame.season.isin(seasons) & aligned.defensive_status.isin(["complete", "missing_actions", "not_applicable_GK"])
    filled = pd.DataFrame(False, index=frame.index, columns=list(FPL_DEFENSIVE_FIELDS))
    def fill(field, values):
        mask = eligible & frame[field].isna() & values.notna()
        frame.loc[mask, field] = values.loc[mask].to_numpy(dtype="int64")
        filled[field] = mask
    for field, external in FPL_DEFENSIVE_FIELDS.items():
        if field != "defensive_contribution":
            fill(field, aligned[external])
    count = frame.clearances_blocks_interceptions + frame.tackles
    count += frame.recoveries.where(frame.position.isin(["MID", "FWD"]), 0)
    fill("defensive_contribution", count.where(frame.position.ne("GK"), 0))
    audit = frame.loc[filled.any(axis=1), MATCH_KEY].join(filled.add_prefix("filled_"))
    audit["source_member"] = aligned.loc[audit.index, "defensive_source_member"]
    audit["method"] = aligned.loc[audit.index, "defensive_status"]
    return {"matches": frame, "audit": audit, "summary": filled.groupby(frame.season).sum()}


def calculate_adjusted_points(matches):
    """Apply the versioned target policy in docs/fpl_rules.md without rewriting P_off"""
    require_keys(matches, MATCH_KEY, "adjusted-point records")
    if not matches.season.isin([*HISTORICAL_SEASONS, "2025_2026", "2026_2027"]).all():
        raise ValueError("Verify and version scoring rules before adding another season")
    frame = matches.copy()
    historical = frame.season.isin(HISTORICAL_SEASONS)
    award = dc_points(frame.defensive_contribution.astype("Int64"), frame.position)
    award = award.mask(award.isna() & frame.minutes.eq(0).fillna(False), 0)
    correction = pd.Series(
        pd.MultiIndex.from_frame(frame[MATCH_KEY]).isin([("2024_2025", 123, 239)]),
        index=frame.index,
    ) & frame.P_off.eq(0).fillna(False)
    if correction.any():
        row = frame.loc[correction]
        events = ["goals_scored", "assists", "own_goals", "penalties_missed",
                  "yellow_cards", "red_cards", "bonus"]
        if not (row.minutes.eq(17).all() and row.position.eq("FWD").all()
                and row[events].eq(0).all().all()):
            raise ValueError("Ferguson scoring evidence changed; review the point correction")
    correction = correction.astype("Int64")
    adjustment = award.where(historical, 0)
    frame["P_adj"] = (frame.P_off + adjustment + correction).astype("Float64")
    frame["P_adj_rules"] = ADJUSTED_RULES
    frame["P_adj_status"] = "official_preserved"
    frame.loc[historical, "P_adj_status"] = "historical_DC_estimate"
    frame.loc[correction.gt(0), "P_adj_status"] = "historical_DC_and_verified_correction"
    frame.loc[frame.P_adj.isna(), "P_adj_status"] = "incomplete"
    audit = frame[MATCH_KEY + ["P_off", "P_adj", "P_adj_status", "P_adj_rules"]].assign(
        DC_already_included=award.where(~historical, 0), DC_target=award,
        DC_adjustment=adjustment, verified_point_correction=correction,
    )
    summary = audit.groupby("season").agg(
        rows=("P_adj", "size"), adjusted_rows=("P_adj", "count"),
        DC_points_added=("DC_adjustment", "sum"),
        verified_point_corrections=("verified_point_correction", "sum"),
        P_off_total=("P_off", "sum"), P_adj_total=("P_adj", lambda s: s.sum(min_count=1)),
    )
    summary["P_adj_total"] = summary.P_adj_total.where(summary.rows.eq(summary.adjusted_rows))
    return {"matches": frame, "audit": audit, "summary": summary}


def fixture_defensive_reference(fixtures, season):
    """Read supplied fixture counts without interpreting omitted players as zero"""
    rows = []
    for fixture in fixtures.loc[fixtures.season.eq(season)].itertuples(index=False):
        stats = ast.literal_eval(fixture.stats) if isinstance(fixture.stats, str) else fixture.stats
        for entry in stats:
            if entry["identifier"] == "defensive_contribution":
                rows.extend({"season": season, "fpl_player_id": p["element"],
                             "fpl_fixture_id": fixture.fpl_fixture_id, "fixture_defensive_count": p["value"]}
                            for side in ("h", "a") for p in entry[side])
    reference = pd.DataFrame(rows, columns=MATCH_KEY + ["fixture_defensive_count"])
    require_keys(reference, MATCH_KEY, "FPL fixture defensive counts")
    return reference


def validate_defensive_actions(matches, fixtures=None, season="2025_2026"):
    """Compare outfield counts and rule-derived awards, retaining all disagreements"""
    overlap = matches.loc[matches.season.eq(season) & matches.minutes.gt(0) & matches.position.ne("GK")].copy()
    overlap["DC_reference"] = dc_points(overlap.defensive_contribution.astype("Int64"), overlap.position)
    metrics = {"external_cbi": "clearances_blocks_interceptions", "external_tackles": "tackles",
               "external_recoveries": "recoveries", "external_defensive_count": "defensive_contribution", "DC_estimated": "DC_reference"}
    summary, differences = [], pd.DataFrame(index=overlap.index)
    for external, reference in metrics.items():
        available = overlap.defensive_status.eq("complete") & overlap[[external, reference]].notna().all(axis=1)
        delta = (overlap[external] - overlap[reference]).where(available).astype("Int64")
        differences[external + "_difference"] = delta
        compared, equal = int(available.sum()), int(delta.eq(0).sum())
        summary.append({"metric": external, "compared": compared, "equal": equal, "mismatches": compared - equal,
                        "agreement": equal / compared if compared else pd.NA, "absolute_error_sum": delta.abs().sum(min_count=1)})
    overlap = pd.concat([overlap, differences], axis=1)
    overlap["near_threshold"] = (
        overlap.defensive_contribution.sub(overlap.position.map(THRESHOLDS)).abs().le(2)
        | overlap.external_defensive_count.sub(overlap.position.map(THRESHOLDS)).abs().le(2)
    )
    valid = overlap.loc[overlap.defensive_status.eq("complete") & overlap.DC_reference.notna() & overlap.DC_estimated.notna()]
    awards = []
    for position, group in [("ALL", valid), *valid.groupby("position")]:
        actual, predicted = group.DC_reference.eq(2), group.DC_estimated.eq(2)
        tp, fp, fn = int((actual & predicted).sum()), int((~actual & predicted).sum()), int((actual & ~predicted).sum())
        awards.append({"position": position, "compared": len(group), "reference_awards": int(actual.sum()),
                       "estimated_awards": int(predicted.sum()), "correct_positive": tp, "false_positive": fp,
                       "false_negative": fn, "precision": tp / (tp + fp) if tp + fp else pd.NA,
                       "recall": tp / (tp + fn) if tp + fn else pd.NA})
    disagreement = differences.ne(0).fillna(False).any(axis=1)
    fixture_summary, fixture_review = pd.DataFrame(), pd.DataFrame()
    if fixtures is not None:
        reference = overlap.merge(fixture_defensive_reference(fixtures, season), on=MATCH_KEY, how="left", validate="one_to_one")
        present = reference.fixture_defensive_count.notna()
        inconsistent = present & reference.fixture_defensive_count.ne(reference.defensive_contribution)
        missing = ~present & reference.defensive_contribution.gt(0)
        fixture_summary = pd.DataFrame([{"compared": int(present.sum()), "mismatches": int(inconsistent.sum()),
                                         "positive_counts_without_fixture_entry": int(missing.sum())}])
        fixture_review = reference.loc[inconsistent | missing]
    return {"summary": pd.DataFrame(summary), "awards": pd.DataFrame(awards), "overlap": overlap,
            "disagreements": overlap.loc[disagreement],
            "award_disagreements": overlap.loc[overlap.DC_estimated.ne(overlap.DC_reference).fillna(False)],
            "unavailable": overlap.loc[overlap.defensive_status.ne("complete") | overlap.DC_reference.isna()],
            "fixture_reference_summary": fixture_summary, "fixture_reference_review": fixture_review}
