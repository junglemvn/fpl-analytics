"""Inspect local FPL files and prepare copies with native FPL keys"""

import hashlib
import json
import re

import pandas as pd


FAMILIES = {"players": "fpl_cleaned_players", "fixtures": "fpl_fixtures", "teams": "fpl_teams"}
PLAYER_KEY = ["season", "fpl_player_id"]
MATCH_KEY = PLAYER_KEY + ["fpl_fixture_id"]


def require_keys(frame, keys, context="table"):
    """Reject missing or repeated keys without changing records"""
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Missing or repeated {keys} in {context}")


def read_fpl_sources(raw_dir):
    """Load separate CSV/JSON tables and record encodings and JSON structure"""
    tables, inventory, structure = {}, [], []
    paths = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".json"})
    if not paths:
        raise FileNotFoundError(f"No CSV or JSON files in {raw_dir}")
    for path in paths:
        encoding = "utf-8"
        if path.suffix.lower() == ".csv":
            try:
                frame = pd.read_csv(path, low_memory=False, encoding=encoding)
            except UnicodeDecodeError:
                encoding = "cp1252"
                frame = pd.read_csv(path, low_memory=False, encoding=encoding)
            tables[path.name] = frame
        else:
            payload = json.loads(path.read_text(encoding=encoding))
            sections = {"root": payload} if isinstance(payload, list) else payload
            for section, value in sections.items():
                structure.append({
                    "filename": path.name, "section": section, "type": type(value).__name__,
                    "items": len(value) if isinstance(value, (list, dict)) else pd.NA,
                })
                if isinstance(value, list) and (section == "root" or value and all(isinstance(v, dict) for v in value)):
                    name = path.name if section == "root" else f"{path.name} / {section}"
                    tables[name] = pd.DataFrame(value)
        season = re.search(r"\d{4}_\d{4}", path.stem)
        inventory.append({
            "filename": path.name, "format": path.suffix[1:],
            "season": season.group() if season else "snapshot; inspect metadata", "encoding": encoding,
        })
    return tables, pd.DataFrame(inventory), pd.DataFrame(structure)


def comparison_copy(frame):
    """Serialize nested cells only in a temporary comparison copy"""
    result = frame.copy()
    for column in frame.select_dtypes(include="object"):
        result[column] = frame[column].map(
            lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (list, dict)) else v
        )
    return result


def profile_columns(frame, detailed=False, comparison=None):
    """Profile missingness and types, optionally adding ranges and sample values"""
    comparison = comparison_copy(frame) if comparison is None else comparison
    profile = pd.DataFrame({
        "column": frame.columns, "dtype": frame.dtypes.astype(str).to_numpy(),
        "non_null": frame.notna().sum().to_numpy(), "nulls": frame.isna().sum().to_numpy(),
        "null_share": frame.isna().mean().to_numpy(),
        "unique_non_null": comparison.nunique(dropna=True).to_numpy(),
    })
    if detailed:
        numeric = set(frame.select_dtypes(include="number").columns)
        profile["numeric_min"] = [frame[c].min() if c in numeric else pd.NA for c in frame]
        profile["numeric_max"] = [frame[c].max() if c in numeric else pd.NA for c in frame]
        profile["example_values"] = [", ".join(map(repr, comparison[c].dropna().unique()[:3])) for c in frame]
    return profile


def profile_tables(tables):
    """Report exact duplicates separately from repeated candidate keys"""
    quality, profiles = [], []
    for name, frame in tables.items():
        comparison = comparison_copy(frame)
        key = ["element", "fixture"] if name.startswith("fpl_merged_gw_") else ["id"] if "id" in frame else []
        quality.append({
            "table": name, "rows": len(frame), "columns": len(frame.columns),
            "null_cells": int(frame.isna().sum().sum()),
            "extra_exact_duplicates": int(comparison.duplicated().sum()),
            "candidate_key": ", ".join(key) if key else "not supplied / not assigned",
            "null_key_rows": int(frame[key].isna().any(axis=1).sum()) if key else pd.NA,
            "extra_key_duplicates": int(comparison.duplicated(key).sum()) if key else pd.NA,
        })
        profile = profile_columns(frame, comparison=comparison)
        profile.insert(0, "table", name)
        profiles.append(profile)
    return pd.DataFrame(quality), pd.concat(profiles, ignore_index=True)


def profile_gameweeks(tables):
    """Count gameweeks and distinct fixtures without collapsing double gameweeks"""
    rows = []
    for name, frame in tables.items():
        if name.startswith("fpl_merged_gw_"):
            fixtures = frame.groupby(["element", "GW"]).fixture.nunique()
            rows.append({
                "filename": name, "rows": len(frame), "columns": len(frame.columns),
                "first_GW": frame.GW.min(), "last_GW": frame.GW.max(), "distinct_GWs": frame.GW.nunique(),
                "player_GW_groups_with_multiple_fixtures": int(fixtures.gt(1).sum()),
                "max_distinct_fixtures_per_player_GW": int(fixtures.max()),
                "extra_exact_duplicates": int(frame.duplicated().sum()),
                "extra_player_fixture_keys": int(frame.duplicated(["element", "fixture"]).sum()),
            })
    return pd.DataFrame(rows)


def prepare_fpl_tables(tables, seasons):
    """Clean separate season summaries, fixtures and teams without inventing IDs"""
    prepared, changes = {}, []
    integer_columns = {"fixtures": ["id", "event", "team_h", "team_a"], "teams": ["id"]}
    for season in seasons:
        for family, prefix in FAMILIES.items():
            filename = f"{prefix}_{season}.csv"
            raw = tables[filename]
            frame = raw.copy() if family == "players" else raw.drop_duplicates().copy()
            duplicates = len(raw) - len(frame)
            managers = int(frame.element_type.eq("AM").sum()) if family == "players" else 0
            if family == "players":
                frame = frame.loc[frame.element_type.ne("AM")].copy()
                if frame.minutes.isna().any() or frame.minutes.lt(0).any():
                    raise ValueError(f"Missing or negative minutes in {filename}")
            else:
                columns = integer_columns[family]
                frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise").astype("Int64")
                require_keys(frame, ["id"], filename)
            if family == "fixtures":
                frame["kickoff_time"] = pd.to_datetime(frame.kickoff_time, utc=True, errors="raise")
            prepared[family, season] = frame.reset_index(drop=True)
            changes.append({
                "table": family, "season": season, "input_rows": len(raw), "output_rows": len(frame),
                "exact_duplicates_removed": duplicates, "manager_rows_removed": managers,
            })
    return prepared, pd.DataFrame(changes)


def stack_seasons(prepared, family, seasons):
    """Stack prepared copies with explicit season and source provenance"""
    return pd.concat([
        prepared[family, season].assign(season=season, source_file=f"{FAMILIES[family]}_{season}.csv")
        for season in seasons
    ], ignore_index=True, sort=False)


def prepare_player_matches(tables, prepared, seasons, adjustment_rules):
    """Prepare player-fixture views using FPL player, fixture and match-time team keys"""
    views, changes = {}, []
    for season in seasons:
        filename = f"fpl_merged_gw_{season}.csv"
        raw = tables[filename]
        frame = raw.drop_duplicates()
        duplicates, managers = len(raw) - len(frame), int(frame.position.eq("AM").sum())
        frame = frame.loc[frame.position.ne("AM")].drop(columns="xP").rename(columns={
            "element": "fpl_player_id", "fixture": "fpl_fixture_id",
            "opponent_team": "opponent_fpl_team_id", "total_points": "P_off",
        }).copy()
        ids = ["fpl_player_id", "fpl_fixture_id", "opponent_fpl_team_id"]
        frame[ids] = frame[ids].apply(pd.to_numeric, errors="raise").astype("Int64")
        if season == "2024_2025":
            # Verified against FotMob match 4506621
            ferguson = frame.fpl_player_id.eq(123) & frame.fpl_fixture_id.eq(239)
            frame.loc[ferguson, "minutes"] = 17
        frame["kickoff_time"] = pd.to_datetime(frame.kickoff_time, utc=True, errors="raise")
        if not pd.api.types.is_bool_dtype(frame.was_home):
            raise ValueError(f"Expected Boolean was_home in {filename}")
        fixtures = prepared["fixtures", season].set_index("id")
        home, away = frame.fpl_fixture_id.map(fixtures.team_h), frame.fpl_fixture_id.map(fixtures.team_a)
        frame["fpl_team_id"] = home.where(frame.was_home, away).astype("Int64")
        opponent = away.where(frame.was_home, home)
        teams = set(prepared["teams", season].id)
        if (opponent.isna().any() or not opponent.eq(frame.opponent_fpl_team_id).all()
                or not frame.fpl_team_id.isin(teams).all() or not opponent.isin(teams).all()):
            raise ValueError(f"Invalid fixture/team relationship in {filename}")
        if frame.P_off.isna().any() or frame.minutes.isna().any() or frame.minutes.lt(0).any():
            raise ValueError(f"Invalid points/minutes in {filename}")
        frame = frame.assign(season=season, source_file=filename, P_adj=pd.NA,
                             P_adj_status="not_calculated", P_adj_rules=adjustment_rules)
        frame["P_adj"] = frame.P_adj.astype("Float64")
        require_keys(frame, MATCH_KEY, filename)
        views[season] = frame.reset_index(drop=True)
        changes.append({
            "season": season, "source_rows": len(raw), "player_rows": len(frame),
            "exact_duplicates_removed": duplicates, "manager_rows_removed": managers,
            "calculated_P_adj_rows": int(frame.P_adj.notna().sum()),
        })
    return views, pd.DataFrame(changes)


def prepare_current_snapshot(directory, season, filenames=("fpl_bootstrap_latest.json", "fpl_fixtures_latest.json")):
    """Prepare saved FPL totals and fixtures without inventing player histories"""
    bootstrap_path, fixture_path = (directory / name for name in filenames)
    bootstrap = json.loads(bootstrap_path.read_text())
    content_url = bootstrap["game_config"]["settings"]["static_content_url"]
    if f"/{season[:4]}_{season[-2:]}/" not in content_url:
        raise ValueError("Bootstrap season differs from the requested season")
    frames = {name: pd.DataFrame(bootstrap[name]) for name in ["elements", "teams", "events"]}
    frames["fixtures"] = pd.DataFrame(json.loads(fixture_path.read_text()))
    for name, frame in frames.items():
        frame["id"] = frame.id.astype("Int64")
        frame["season"] = season
        frame["source_file"] = f"{directory.name}/{fixture_path.name if name == 'fixtures' else bootstrap_path.name}"
        require_keys(frame, ["season", "id"], name)
    players, fixtures, events = frames["elements"], frames["fixtures"], frames["events"]
    numeric = [
        "price_change_percent", "ep_next", "ep_this", "form", "points_per_game",
        "selected_by_percent", "value_form", "value_season", "influence", "creativity",
        "threat", "ict_index", "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded",
    ]
    players[numeric] = players[numeric].apply(pd.to_numeric, errors="raise")
    players[["team", "element_type"]] = players[["team", "element_type"]].astype("Int64")
    positions = {p["id"]: p["singular_name_short"].replace("GKP", "GK") for p in bootstrap["element_types"]}
    players["position"] = players.element_type.map(positions)
    players["news_added"] = pd.to_datetime(players.news_added, utc=True, errors="raise")
    fixtures[["event", "team_h", "team_a"]] = fixtures[["event", "team_h", "team_a"]].astype("Int64")
    fixtures["kickoff_time"] = pd.to_datetime(fixtures.kickoff_time, utc=True, errors="raise")
    events["deadline_time"] = pd.to_datetime(events.deadline_time, utc=True, errors="raise")
    if (players.position.isna().any() or not players.team.isin(frames["teams"].id).all()
            or not fixtures[["team_h", "team_a"]].isin(frames["teams"].id.to_numpy()).all().all()
            or not fixtures.event.dropna().isin(events.id).all()):
        raise ValueError("Invalid current player/team/event reference")
    metadata = pd.DataFrame([{
        "season": season, "retrieved_at_utc": pd.NaT,
        "price_updated_at_utc": pd.to_datetime(bootstrap["game_config"]["status"]["price_change_last_updated"], utc=True),
        "last_finished_kickoff_utc": fixtures.loc[fixtures.finished, "kickoff_time"].max(),
        "last_started_kickoff_utc": fixtures.loc[fixtures.started.fillna(False), "kickoff_time"].max(),
        "last_checked_GW": events.loc[events.finished & events.data_checked, "id"].max(),
        "players": len(players), "finished_fixtures": int(fixtures.finished.sum()),
        "started_unfinished_fixtures": int((fixtures.started.fillna(False) & ~fixtures.finished).sum()),
        "player_history_available": False,
    }])
    metadata["retrieved_at_utc"] = pd.to_datetime(metadata.retrieved_at_utc, utc=True)
    return {
        "current_players": players.drop(columns="opta_code").rename(columns={"id": "fpl_player_id", "team": "fpl_team_id"}),
        "current_teams": frames["teams"].rename(columns={"id": "fpl_team_id"}),
        "current_fixtures": fixtures.rename(columns={"id": "fpl_fixture_id", "team_h": "home_fpl_team_id", "team_a": "away_fpl_team_id"}),
        "current_events": events, "current_snapshot_metadata": metadata,
    }


def join_player_metadata(matches, players, teams):
    """Join descriptive fields by native keys without copying snapshot statistics"""
    for frame, key in [(matches, MATCH_KEY), (players, PLAYER_KEY), (teams, ["season", "fpl_team_id"])]:
        require_keys(frame, key)
    metadata = players.assign(name=players.first_name.str.cat(players.second_name, sep=" ").str.strip())[
        PLAYER_KEY + ["name", "position"]
    ]
    clubs = teams[["season", "fpl_team_id", "name"]].rename(columns={"name": "team"})
    frame = matches.drop(columns=["name", "position", "team"], errors="ignore").merge(
        metadata, on=PLAYER_KEY, how="left", validate="many_to_one",
    ).merge(clubs, on=["season", "fpl_team_id"], how="left", validate="many_to_one")
    if len(frame) != len(matches) or frame[["name", "position", "team"]].isna().any().any():
        raise ValueError("Missing player/club metadata or changed match row count")
    return frame


def prepare_collected_batch(batch, season):
    """Verify archived responses and join current histories without changing points"""
    from .collect_fpl import API, prepare_history, reconcile, source_path, validate_references

    batch = batch.resolve()
    manifest = json.loads((batch / "manifest.json").read_text())
    if (manifest["season"] != season or manifest["status"] != "completed"
            or not manifest.get("all_current_histories_complete")):
        raise ValueError("Select a complete, reconciled collection batch for this season")
    receipts = {r["url"]: r for r in manifest["requests"] if r["status"] == "received_json"}

    def response(route):
        receipt = receipts[f"{API}/{route}/"]
        path = batch / f"{route}.json"
        raw = path.read_bytes()
        if len(raw) != receipt["bytes"] or hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
            raise ValueError(f"Response checksum/size mismatch: {path}")
        return json.loads(raw), {**receipt, "path": source_path(path)}

    bootstrap, bootstrap_receipt = response("bootstrap-static")
    fixtures, fixture_receipt = response("fixtures")
    players, games = validate_references(bootstrap, fixtures, season)
    frames, checks, differences = [], [], []
    for player_id in sorted(players):
        payload, receipt = response(f"element-summary/{player_id}")
        frame, conflicts = prepare_history(payload, player_id, games, season, receipt)
        frames.append(frame)
        checks.extend(reconcile(frame, players[player_id]))
        differences.extend(conflicts)
    reconciliation = pd.DataFrame(checks)
    if differences or not reconciliation["matches"].all():
        raise ValueError("Collected histories differ from their fixture/bootstrap references")
    tables = prepare_current_snapshot(batch, season, ("bootstrap-static.json", "fixtures.json"))
    for name, frame in tables.items():
        receipt = fixture_receipt if name == "current_fixtures" else bootstrap_receipt
        frame["retrieved_at_utc"] = pd.Timestamp(receipt["retrieved_at_utc"])
        if name != "current_snapshot_metadata":
            frame["source_file"] = receipt["path"]
    matches = join_player_metadata(pd.concat(frames, ignore_index=True), tables["current_players"], tables["current_teams"])
    numeric = ["influence", "creativity", "threat", "ict_index"]
    matches[numeric] = matches[numeric].apply(pd.to_numeric, errors="raise")
    matches["retrieved_at_utc"] = pd.to_datetime(matches.retrieved_at_utc, utc=True, errors="raise")
    context_review = validate_fpl_relations(matches, tables["current_fixtures"], tables["current_teams"])
    if not context_review.empty:
        raise ValueError("Unresolved current fixture context")
    tables["current_player_matches"] = matches
    metadata = tables["current_snapshot_metadata"]
    metadata["player_history_available"] = True
    metadata["histories_collected"] = len(players)
    metadata["player_fixture_rows"] = len(matches)
    metadata["collection_batch"] = batch.name
    requests = pd.DataFrame(manifest["requests"]).rename(columns={"path": "recorded_path"})
    requests["path"] = [source_path(batch / f"{url.removeprefix(API + '/').rstrip('/')}.json") for url in requests.url]
    return {"tables": tables, "reconciliation": reconciliation, "context_review": context_review,
            "requests": requests}


def validate_fpl_relations(matches, fixtures, teams):
    """Validate native keys and match-time clubs, returning fixture-context differences"""
    for frame, key in [(matches, MATCH_KEY), (fixtures, ["season", "fpl_fixture_id"]),
                       (teams, ["season", "fpl_team_id"])]:
        require_keys(frame, key)
    team_keys = pd.MultiIndex.from_frame(teams[["season", "fpl_team_id"]])
    for frame, columns in [(fixtures, ["home_fpl_team_id", "away_fpl_team_id"]),
                           (matches, ["fpl_team_id", "opponent_fpl_team_id"])]:
        for column in columns:
            if not pd.MultiIndex.from_arrays([frame.season, frame[column]]).isin(team_keys).all():
                raise ValueError(f"Unresolved team reference: {column}")
    context = {"GW": "event", "round": "event", "kickoff_time": "kickoff_time",
               "team_h_score": "team_h_score", "team_a_score": "team_a_score"}
    joined = matches.merge(fixtures, on=["season", "fpl_fixture_id"], how="left",
                           validate="many_to_one", suffixes=("", "_fixture"), indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Unresolved player-fixture reference")
    home, away = joined.home_fpl_team_id, joined.away_fpl_team_id
    if not (joined.fpl_team_id.eq(home.where(joined.was_home, away))
            & joined.opponent_fpl_team_id.eq(away.where(joined.was_home, home))).all():
        raise ValueError("Player club/opponent differs from the fixture")
    reviews = []
    for field, reference in context.items():
        if field not in matches:
            continue
        reference += "_fixture" if reference in matches else ""
        left, right = joined[field], joined[reference]
        different = ~(left.eq(right).fillna(False) | (left.isna() & right.isna()))
        reviews.append(joined.loc[different, MATCH_KEY].assign(
            field=field, match_value=left[different].astype(str), fixture_value=right[different].astype(str),
        ))
    return pd.concat(reviews, ignore_index=True)


def reconcile_player_totals(matches, summaries):
    """Compare aggregate source totals without joining ID-less summaries by name"""
    fields = ["minutes", "total_points", "goals_scored", "assists", "bonus", "bps", "yellow_cards", "red_cards"]
    source = summaries.groupby("season")[fields].sum().rename_axis(columns="metric").stack().rename("season_summary")
    observed = matches.rename(columns={"P_off": "total_points"}).groupby("season")[fields].sum()
    observed = observed.rename_axis(columns="metric").stack().rename("fixture_records")
    result = pd.concat([source, observed], axis=1).reset_index()
    return result.assign(difference=result.fixture_records - result.season_summary)


def export_prepared_data(directory, tables, audits, sources, project_root, rules):
    """Save typed local EDA tables and a reproducible source receipt"""
    directory.mkdir(parents=True, exist_ok=True)
    for name, frames in [("fpl_tables", tables), ("fpl_audits", audits)]:
        pd.to_pickle(frames, directory / f"{name}.pkl")
    manifest = {
        "schema_version": "fpl_preparation_v2", "prepared_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "P_adj_rules": rules, "current_player_histories_available": "current_player_matches" in tables,
        "current_points": "P_off preserved from collected FPL history without recalculation",
        "tables": {name: {"rows": len(frame), "columns": len(frame.columns)} for name, frame in tables.items()},
        "sources": [],
    }
    for path in sorted(set(sources)):
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["sources"].append({"path": str(path.relative_to(project_root)), "sha256": checksum})
    (directory / "fpl_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return pd.DataFrame(manifest["tables"]).T.rename_axis("table")
