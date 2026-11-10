"""Collect FPL players, teams, fixtures and histories."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]  # Project folder
API = "https://fantasy.premierleague.com/api"  # API base address
DEFAULT_SEASON = "2026_2027"  # Season to collect
# Historical player columns
SUMMARY_COLUMNS = [
    "first_name", "second_name", "goals_scored", "assists", "total_points", "minutes",
    "goals_conceded", "creativity", "influence", "threat", "bonus", "bps", "ict_index",
    "clean_sheets", "red_cards", "yellow_cards", "selected_by_percent", "now_cost",
    "element_type", "value_per_m",
]


def utc_now():
    """Current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def source_path(path):
    """Use project-relative provenance paths when possible"""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def create_session():
    """Reuse connections and retry errors."""
    session = requests.Session()  # HTTP connection pool
    retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET"], respect_retry_after_header=True)
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"Accept": "application/json", "User-Agent": "FPL-collector/0.2"})
    return session


def fetch_json(session, route, destination, receipts):
    """Download, save and log JSON."""
    # Request record
    receipt = {"url": f"{API}/{route}/", "started_at_utc": utc_now(),
               "http_status": None, "status": "failed", "path": None}
    if route.startswith("element-summary/"):
        receipt["fpl_player_id"] = int(route.split("/")[-1])
    receipts.append(receipt)
    try:
        response = session.get(receipt["url"], timeout=(5, 20))  # Server response
        receipt.update(http_status=response.status_code,
                       content_type=response.headers.get("Content-Type"))
        response.raise_for_status()
        raw = response.content  # Original response bytes
        receipt["retrieved_at_utc"] = utc_now()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        receipt.update(path=source_path(destination),
                       sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        payload = json.loads(raw)  # Parsed Python objects
        receipt["status"] = "received_json"
        return payload, receipt
    except (requests.RequestException, ValueError) as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["completed_at_utc"] = utc_now()


def validate_references(bootstrap, fixtures, season):
    """Check season and linked IDs."""
    if not isinstance(bootstrap, dict) or not isinstance(bootstrap.get("elements"), list):
        raise ValueError("Bootstrap must contain an elements list; inspect the saved response")
    content_url = bootstrap.get("game_config", {}).get("settings", {}).get("static_content_url", "")
    marker = f"/{season[:4]}_{season[-2:]}/"
    if marker not in content_url:
        raise ValueError(f"Cannot verify requested season {season} from bootstrap metadata")
    if not isinstance(fixtures, list) or not all(isinstance(f, dict) for f in fixtures):
        raise ValueError("Fixtures must be a list of objects")
    players = {p["id"]: p for p in bootstrap["elements"]}  # Player lookup
    games = {f["id"]: f for f in fixtures}  # Fixture lookup
    teams = {t["id"] for t in bootstrap["teams"]}  # Valid team IDs
    if len(players) != len(bootstrap["elements"]) or len(games) != len(fixtures):
        raise ValueError("Repeated player or fixture IDs")
    if any(f["team_h"] not in teams or f["team_a"] not in teams for f in fixtures):
        raise ValueError("Fixture refers to an unknown team")
    return players, games


def prepare_history(payload, player_id, fixtures, season, receipt):
    """Build one player's match table."""
    if not isinstance(payload, dict) or not isinstance(payload.get("history"), list):
        raise ValueError("Player summary must contain a history list")
    records, differences = [], []  # Rows and mismatches
    for row in payload["history"]:
        required = {"fixture", "round", "minutes", "total_points", "was_home", "opponent_team"}
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("Unexpected history fields; inspect the saved player response")
        if row.get("element", player_id) != player_id:
            raise ValueError("History player ID differs from requested player")
        if row["fixture"] not in fixtures or type(row["was_home"]) is not bool:
            raise ValueError("Unknown fixture or invalid was_home value")
        fixture = fixtures[row["fixture"]]  # Match details
        # Match-time teams
        club = fixture["team_h"] if row["was_home"] else fixture["team_a"]
        opponent = fixture["team_a"] if row["was_home"] else fixture["team_h"]
        if row["opponent_team"] != opponent:
            raise ValueError("Player opponent differs from the fixture")
        for field, expected in [("round", fixture.get("event")),
                                ("team_h_score", fixture.get("team_h_score")),
                                ("team_a_score", fixture.get("team_a_score"))]:
            if field in row and row[field] != expected:
                differences.append({"fpl_player_id": player_id, "fpl_fixture_id": row["fixture"],
                                    "field": field, "history": row[field], "fixture": expected})
        if row.get("kickoff_time") and fixture.get("kickoff_time"):
            if pd.to_datetime(row["kickoff_time"], utc=True) != pd.to_datetime(fixture["kickoff_time"], utc=True):
                differences.append({"fpl_player_id": player_id, "fpl_fixture_id": row["fixture"],
                                    "field": "kickoff_time", "history": row["kickoff_time"],
                                    "fixture": fixture["kickoff_time"]})
        record = dict(row)  # Source row copy
        record.pop("element", None)
        for old, new in [("fixture", "fpl_fixture_id"), ("round", "GW"),
                         ("total_points", "P_off"), ("opponent_team", "opponent_fpl_team_id")]:
            record[new] = record.pop(old)
        record.update(season=season, fpl_player_id=player_id, fpl_team_id=club,
                      source_file=receipt["path"], retrieved_at_utc=receipt["retrieved_at_utc"],
                      fixture_finished=fixture["finished"],
                      fixture_finished_provisional=fixture.get("finished_provisional"))
        records.append(record)
    keys = ["season", "fpl_player_id", "fpl_fixture_id"]  # Unique row identity
    # Player-match DataFrame
    frame = pd.DataFrame(records) if records else pd.DataFrame(columns=keys + ["minutes", "P_off"])
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Missing or duplicate player-fixture keys")
    # Whole-number columns
    integer_fields = keys[1:] + ["GW", "minutes", "P_off", "starts", "goals_scored", "assists",
                                "clean_sheets", "goals_conceded", "fpl_team_id", "opponent_fpl_team_id",
                                "clearances_blocks_interceptions", "tackles", "recoveries",
                                "defensive_contribution", "value", "selected"]
    for field in integer_fields:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="raise").astype("Int64")
    for field in frame.columns:
        if field.startswith("expected_"):
            frame[field] = pd.to_numeric(frame[field], errors="raise").astype("Float64")
    if frame.minutes.isna().any() or frame.P_off.isna().any() or frame.minutes.lt(0).any():
        raise ValueError("Missing points/minutes or negative minutes")
    if "kickoff_time" in frame:
        frame["kickoff_time"] = pd.to_datetime(frame.kickoff_time, utc=True, errors="raise")
    return frame, differences


def reconcile(frame, player):
    """Compare match and season totals."""
    checks = []  # Comparison results
    fields = ["minutes", "total_points", "goals_scored", "assists", "starts",
              "clearances_blocks_interceptions", "tackles", "recoveries", "defensive_contribution",
              "expected_goals", "expected_assists"]
    for field in fields:
        column = "P_off" if field == "total_points" else field
        if field not in player or column not in frame:
            continue
        complete = bool(frame[column].notna().all())
        observed = float(frame[column].sum()) if complete else None
        reference = float(player[field])
        # Two-decimal rounding allowance
        tolerance = 0.005 * (len(frame) + 1) + 1e-9 if field.startswith("expected_") else 0
        checks.append({"fpl_player_id": player["id"], "field": field,
                       "history_total": observed, "bootstrap_total": reference,
                       "tolerance": tolerance,
                       "matches": complete and abs(observed - reference) <= tolerance})
    return checks


def prepare_player_summaries(bootstrap, season, receipt):
    """Build historical-format player summaries."""
    source = pd.DataFrame(bootstrap["elements"])  # Current player records
    if source.empty or "id" not in source or source.id.isna().any() or source.id.duplicated().any():
        raise ValueError("Bootstrap player IDs must be present and unique")
    required = set(SUMMARY_COLUMNS) - {"value_per_m"}
    if not required.issubset(source.columns):
        raise ValueError(f"Missing summary fields: {sorted(required - set(source.columns))}")
    # Position ID labels
    positions = {p["id"]: p["singular_name_short"].replace("GKP", "GK")
                 for p in bootstrap["element_types"]}
    position = source.element_type.map(positions)
    if position.isna().any() or not position.isin(["GK", "DEF", "MID", "FWD", "AM"]).all():
        raise ValueError("Unknown player position in bootstrap")
    player_rows = position.ne("AM")  # Exclude manager entities
    cleaned = source.loc[player_rows, SUMMARY_COLUMNS[:-1]].copy()
    cleaned["element_type"] = position.loc[player_rows]
    integers = ["goals_scored", "assists", "total_points", "minutes", "goals_conceded",
                "bonus", "bps", "clean_sheets", "red_cards", "yellow_cards", "now_cost"]
    decimals = ["creativity", "influence", "threat", "ict_index", "selected_by_percent"]
    for field in integers:
        cleaned[field] = pd.to_numeric(cleaned[field], errors="raise").astype("Int64")
    for field in decimals:
        cleaned[field] = pd.to_numeric(cleaned[field], errors="raise").astype("Float64")
    if cleaned[SUMMARY_COLUMNS[:-1]].isna().any().any() or cleaned.minutes.lt(0).any():
        raise ValueError("Missing summary measurements or negative minutes")
    # Price in £millions
    price_millions = cleaned.now_cost.astype("Float64").div(10)
    cleaned["value_per_m"] = cleaned.total_points.div(price_millions.where(price_millions.gt(0))).round(1)
    # Full player snapshot
    snapshot = source.rename(columns={"id": "fpl_player_id", "team": "fpl_team_id"}).copy()
    snapshot["position"] = position
    snapshot["season"] = season
    snapshot["source_file"] = receipt["path"]
    snapshot["retrieved_at_utc"] = receipt.get("retrieved_at_utc")
    # Batch-local row links
    row_numbers = {pid: i for i, pid in enumerate(source.loc[player_rows, "id"])}
    snapshot["cleaned_csv_row"] = snapshot.fpl_player_id.map(row_numbers).astype("Int64")
    return cleaned.reset_index(drop=True), snapshot


def prepare_team_table(bootstrap):
    """Keep native team fields."""
    teams = pd.DataFrame(bootstrap["teams"])  # Team DataFrame
    required = {"id", "code", "name", "short_name"}
    if teams.empty or not required.issubset(teams.columns):
        raise ValueError("Bootstrap must contain team IDs, codes and names")
    teams["id"] = pd.to_numeric(teams.id, errors="raise").astype("Int64")
    if teams.id.isna().any() or teams.id.duplicated().any() or teams.name.isna().any():
        raise ValueError("Missing or duplicate team identity")
    return teams


def save_player_tables(output, bootstrap, season, receipt):
    """Save summaries and full attributes."""
    cleaned, snapshot = prepare_player_summaries(bootstrap, season, receipt)
    cleaned.to_csv(output / f"fpl_cleaned_players_{season}.csv", index=False)
    for column in snapshot:
        if snapshot[column].dtype == object:
            snapshot[column] = snapshot[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value)
    snapshot.to_csv(output / "current_player_snapshot.csv", index=False)
    return cleaned, snapshot


def export_tables(output, bootstrap, fixtures, season, receipt, manifest, frames, checks):
    """Save prepared CSV tables."""
    output.mkdir(parents=True, exist_ok=False)
    cleaned, snapshot = save_player_tables(output, bootstrap, season, receipt)
    prepare_team_table(bootstrap).to_csv(output / f"fpl_teams_{season}.csv", index=False)
    fixture_table = pd.DataFrame(fixtures)
    if "stats" in fixture_table:
        fixture_table["stats"] = fixture_table.stats.map(json.dumps)
    fixture_table.to_csv(output / f"fpl_fixtures_{season}.csv", index=False)
    manifest.update(prepared_output=str(output.relative_to(ROOT)), player_summary_rows=len(cleaned),
                    prepared_at_utc=utc_now(),
                    summary_columns=SUMMARY_COLUMNS, summary_population="all_current_bootstrap_players",
                    excluded_manager_rows=len(snapshot) - len(cleaned),
                    snapshot_retrieved_at_utc=receipt.get("retrieved_at_utc"),
                    summary_period="season_to_date_at_source_snapshot",
                    value_per_m_definition="round(total_points / (now_cost / 10), 1)")
    if frames:
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(output / "current_player_matches.csv", index=False)
        pd.DataFrame(checks).to_csv(output / "reconciliation.csv", index=False)
        manifest["player_match_rows"] = len(table)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def collect_snapshot(table, season=DEFAULT_SEASON, from_batch=None):
    """Collect player or team totals."""
    validate_season_name(season)
    if table not in ["players", "teams"]:
        raise ValueError("Snapshot table must be players or teams")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # Batch name
    batch = Path(from_batch).resolve() if from_batch else ROOT / "data/raw/fpl_api" / season / stamp
    if not from_batch:
        batch.mkdir(parents=True)
    name = f"{batch.name}_{table}_{stamp}" if from_batch else stamp
    output = ROOT / "data/processed/fpl_self_collected" / season / name
    manifest = {"season": season, "table": table, "started_at_utc": utc_now(),
                "mode": "saved_snapshot" if from_batch else "live_snapshot",
                "status": "failed", "requests": []}  # Collection log
    try:
        if from_batch:
            original = json.loads((batch / "manifest.json").read_text())
            if original["season"] != season:
                raise ValueError("Saved batch season differs from requested season")
            receipt = next((r for r in original["requests"] if r.get("status") == "received_json"
                            and r["url"] == f"{API}/bootstrap-static/"), None)
            raw = (batch / "bootstrap-static.json").read_bytes()
            if receipt is None or hashlib.sha256(raw).hexdigest() != receipt.get("sha256"):
                raise ValueError("Missing or mismatched bootstrap checksum")
            receipt = {**receipt, "path": source_path(batch / "bootstrap-static.json")}
            bootstrap = json.loads(raw)  # Saved API data
            manifest.update(source_batch=str(batch), requests=[receipt])
        else:
            with create_session() as session:
                bootstrap, receipt = fetch_json(session, "bootstrap-static", batch / "bootstrap-static.json", manifest["requests"])
        validate_references(bootstrap, [], season)
        output.mkdir(parents=True, exist_ok=False)
        if table == "players":
            cleaned, snapshot = save_player_tables(output, bootstrap, season, receipt)
            manifest.update(rows=len(cleaned), columns=SUMMARY_COLUMNS,
                            snapshot_rows=len(snapshot), summary_population="all_current_bootstrap_players",
                            summary_period="season_to_date_at_source_snapshot",
                            all_current_histories_complete=False)
        else:
            teams = prepare_team_table(bootstrap)
            teams.to_csv(output / f"fpl_teams_{season}.csv", index=False)
            manifest.update(rows=len(teams), columns=teams.columns.tolist())
        manifest.update(status="completed", snapshot_retrieved_at_utc=receipt.get("retrieved_at_utc"),
                        prepared_output=str(output.relative_to(ROOT)))
        print(f"Saved {manifest['rows']} {table} rows")
        print(f"Output: {output}")
        return 0
    except (OSError, requests.RequestException, ValueError, KeyError, TypeError) as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Collection failed: {manifest['error']}")
        return 1
    finally:
        manifest["finished_at_utc"] = utc_now()
        if output.exists():
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if not from_batch:
            (batch / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def prepare_saved_batch(batch, season, player_ids=None, limit=None, summary_only=False):
    """Prepare archived API responses."""
    batch = Path(batch).resolve()
    original = json.loads((batch / "manifest.json").read_text())  # Original collection log
    if original["season"] != season:
        raise ValueError("Saved batch season differs from --season")
    payloads, receipts = {}, {}
    for route in ["bootstrap-static", "fixtures"]:
        path = batch / f"{route}.json"
        raw = path.read_bytes()
        receipt = next((r for r in original["requests"] if r.get("status") == "received_json"
                        and r["url"] == f"{API}/{route}/"), None)
        if receipt is None or hashlib.sha256(raw).hexdigest() != receipt.get("sha256"):
            raise ValueError(f"No verified receipt for {path.name}")
        payloads[route], receipts[route] = json.loads(raw), {**receipt, "path": source_path(path)}
    bootstrap, fixtures = payloads["bootstrap-static"], payloads["fixtures"]
    players, games = validate_references(bootstrap, fixtures, season)
    available = sorted(int(p.stem) for p in (batch / "element-summary").glob("*.json"))  # Saved player IDs
    selected = list(dict.fromkeys(player_ids)) if player_ids else available[:limit]
    if summary_only:
        selected = []
    if any(pid not in players for pid in selected):
        raise ValueError("Selected history ID is absent from this batch's bootstrap")
    report = {"season": season, "collector_version": 2, "mode": "prepare_saved_batch",
              "prepared_at_utc": utc_now(), "source_batch": str(batch),
              "discovered_players": len(players), "requested_player_ids": selected,
              "history_scope": "saved_responses_only", "coverage": []}
    frames, checks, differences = [], [], []
    for pid in selected:
        entry = {"fpl_player_id": pid, "status": "failed"}
        report["coverage"].append(entry)
        try:
            path = batch / "element-summary" / f"{pid}.json"
            raw = path.read_bytes()
            receipt = next((r for r in original["requests"] if r.get("status") == "received_json"
                            and r["url"] == f"{API}/element-summary/{pid}/"), None)
            if receipt is None or hashlib.sha256(raw).hexdigest() != receipt.get("sha256"):
                raise ValueError("Missing or mismatched player-response checksum")
            receipt = {**receipt, "path": source_path(path)}
            frame, conflicts = prepare_history(json.loads(raw), pid, games, season, receipt)
            player_checks = reconcile(frame, players[pid])
            frames.append(frame)
            checks.extend(player_checks)
            differences.extend(conflicts)
            entry.update(status="successful_empty_history" if frame.empty else "validated_history", history_rows=len(frame))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
    failures = any(p["status"] == "failed" for p in report["coverage"])
    mismatches = any(not c["matches"] for c in checks) or bool(differences)
    report.update(status="partial_failure" if failures else "completed_with_discrepancies" if mismatches else "completed",
                  reconciliation=checks, context_differences=differences,
                  histories_validated=sum(p["status"] != "failed" for p in report["coverage"]),
                  all_current_histories_complete=len(selected) == len(players) and not failures and not mismatches)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "data/processed/fpl_self_collected" / season / f"{batch.name}_prepared_{stamp}"
    export_tables(output, bootstrap, fixtures, season, receipts["bootstrap-static"], report, frames, checks)
    print(f"Prepared all {report['player_summary_rows']} player summaries and {report.get('player_match_rows', 0)} saved match records")
    print(f"Output: {output}")
    print(f"Histories validated: {report['histories_validated']} / {len(players)} current players")
    return 1 if failures else 2 if mismatches else 0


def select_player_ids(players, player_ids=None, limit=None, summary_only=False):
    """Choose IDs from bootstrap."""
    if summary_only:
        return []
    selected = list(dict.fromkeys(player_ids)) if player_ids else sorted(players)
    if limit is not None:
        if limit < 1:
            raise ValueError("History limit must be positive")
        selected = selected[:limit]
    if any(pid not in players for pid in selected):
        raise ValueError("Requested player ID not found in this fresh bootstrap")
    return selected


def run(season=DEFAULT_SEASON, player_ids=None, limit=None, summary_only=False):
    """Collect requested player histories."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # Unique batch name
    batch = ROOT / "data/raw/fpl_api" / season / stamp
    batch.mkdir(parents=True)
    # Collection audit log
    manifest = {"season": season, "scope": "current_season_collection", "collector_version": 2, "started_at_utc": utc_now(),
                "status": "failed", "requests": [], "requested_player_ids": player_ids}
    try:
        with create_session() as session:
            bootstrap, bootstrap_receipt = fetch_json(session, "bootstrap-static", batch / "bootstrap-static.json", manifest["requests"])
            fixtures, _ = fetch_json(session, "fixtures", batch / "fixtures.json", manifest["requests"])
            players, games = validate_references(bootstrap, fixtures, season)
            selected = select_player_ids(players, player_ids, limit, summary_only)
            manifest.update(discovered_players=len(players), requested_player_ids=selected,
                            history_scope="not_requested" if summary_only else "all_current_players" if len(selected) == len(players) else "selected_players")
            print(f"Discovered {len(players)} players; collecting {len(selected)} player histories. "
                  f"Season-statistics export covers all {len(players)} players.", flush=True)
            frames, checks, differences, coverage = [], [], [], []  # Tables and audits
            for pid in selected:
                entry = {"fpl_player_id": pid, "name": players[pid].get("web_name"), "status": "failed"}
                coverage.append(entry)
                try:
                    payload, receipt = fetch_json(session, f"element-summary/{pid}",
                                                   batch / "element-summary" / f"{pid}.json", manifest["requests"])
                    frame, conflicts = prepare_history(payload, pid, games, season, receipt)
                    player_checks = reconcile(frame, players[pid])
                    frames.append(frame)
                    checks.extend(player_checks)
                    differences.extend(conflicts)
                    entry.update(status="successful_empty_history" if frame.empty else "validated_history",
                                 history_rows=len(frame), history_fields=sorted({k for row in payload["history"] for k in row}),
                                 response_sections=list(payload))
                    print(f"Player {pid} ({len(coverage)}/{len(selected)}): {len(frame)} history records", flush=True)
                except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"Player {pid}: FAILED ({exc})", flush=True)
                time.sleep(1)  # Request pause
        manifest["coverage"] = coverage
        manifest["context_differences"] = differences
        manifest["reconciliation"] = checks
        failures = any(p["status"] == "failed" for p in coverage)
        mismatches = any(not c["matches"] for c in checks) or bool(differences)
        manifest["status"] = "partial_failure" if failures else "completed_with_discrepancies" if mismatches else "completed"
        manifest["reconciliation_note"] = "Totals use the initial bootstrap; API revisions or cutoff changes may cause differences."
        manifest.update(histories_validated=sum(p["status"] != "failed" for p in coverage),
                        all_current_histories_complete=len(selected) == len(players) and not failures and not mismatches)
        output = ROOT / "data/processed/fpl_self_collected" / season / stamp
        export_tables(output, bootstrap, fixtures, season, bootstrap_receipt, manifest, frames, checks)
        print(f"Prepared all {manifest['player_summary_rows']} current player summaries")
        print(f"Output: {output}")
        return 1 if failures else 2 if mismatches else 0
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Collection failed: {manifest['error']}")
        return 1
    finally:
        manifest["finished_at_utc"] = utc_now()
        (batch / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print(f"Batch status: {manifest['status']}")
        print(f"Manifest: {batch / 'manifest.json'}")


def validate_season_name(season):
    """Check consecutive season years."""
    if not re.fullmatch(r"\d{4}_\d{4}", season) or int(season[5:]) != int(season[:4]) + 1:
        raise ValueError("Season must contain consecutive years, e.g. 2026_2027")
    return season


def parse_snapshot_arguments(table, argv=None):
    """Read snapshot collection options."""
    parser = argparse.ArgumentParser(description=f"Collect all current FPL {table}")
    parser.add_argument("--season", type=validate_season_name, default=DEFAULT_SEASON,
                        help=f"Season to verify; default: {DEFAULT_SEASON}")
    parser.add_argument("--from-batch", type=Path, help="Prepare a saved bootstrap without network access")
    return parser.parse_args(argv)


def parse_arguments(argv=None):
    """Read Terminal options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=validate_season_name, default=DEFAULT_SEASON,
                        help=f"Season to verify; default: {DEFAULT_SEASON}")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all-players", action="store_true",
                           help="Collect every player discovered in bootstrap (the default)")
    selection.add_argument("--player-ids", type=int, nargs="+", help="Fetch only these histories; summaries still cover all players")
    selection.add_argument("--limit", type=int, help="Fetch only the first N histories; default is all players")
    selection.add_argument("--summary-only", action="store_true", help="Export all player totals without requesting histories")
    parser.add_argument("--from-batch", type=Path, help="Prepare a saved batch offline; use available histories only")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1 or args.player_ids and any(pid < 1 for pid in args.player_ids):
        parser.error("Player IDs and --limit must be positive")
    return args


if __name__ == "__main__":
    args = parse_arguments()  # Run settings
    if args.from_batch:
        try:
            result = prepare_saved_batch(args.from_batch, args.season, args.player_ids, args.limit, args.summary_only)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Saved batch preparation failed: {exc}")
            result = 1
    else:
        result = run(args.season, args.player_ids, args.limit, args.summary_only)
    raise SystemExit(result)
