"""Match external records for enrichment while retaining FPL working keys"""

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import re
import tarfile
import unicodedata

import pandas as pd


# Explicit club-name equivalents for the selected EPL seasons; no fuzzy matching
CLUB_ALIASES = {
    "afcbournemouth": "bournemouth",
    "brightonhovealbion": "brighton",
    "brightonandhovealbion": "brighton",
    "ipswichtown": "ipswich",
    "leedsunited": "leeds",
    "leicestercity": "leicester",
    "lutontown": "luton",
    "manchestercity": "mancity",
    "manchesterunited": "manutd",
    "newcastleunited": "newcastle",
    "nottinghamforest": "nottmforest",
    "sheffieldunited": "sheffieldutd",
    "tottenhamhotspur": "spurs",
    "westhamunited": "westham",
    "wolverhamptonwanderers": "wolves",
}


# Explicit spelling/short-name equivalents inspected for this archive These are
# name evidence only: shared fixtures, the same match-time club and a unique
# target identity are still required Keep unfamiliar names unresolved
PLAYER_NAME_ALIASES = (
    ("Benjamin White", "Ben White"),
    ("Eddie Nketiah", "Edward Nketiah"),
    ("Marcus Oliveira Alencar", "Marquinhos"),
    ("Josh Dasilva", "Joshua Dasilva"),
    ("Josh Onomah", "Joshua Onomah"),
    ("Dan Burn", "Daniel Burn"),
    ("Joe Willock", "Joseph Willock"),
    ("Tino Livramento", "Valentino Livramento"),
    ("Joseph Hodge", "Joe Hodge"),
    ("Joseph Whitworth", "Joe Whitworth"),
    ("Alexandre Moreno Lopera", "Álex Moreno"),
    ("Mykhailo Mudryk", "Mykhaylo Mudryk"),
    ("Odeluga Offiah", "Odel Offiah"),
    ("Kamari Doyle", "Kami Doyle"),
    ("Yegor Yarmoliuk", "Yehor Yarmoliuk"),
    ("Yegor Yarmolyuk", "Yehor Yarmoliuk"),
    ("Yegor Yarmoliuk", "Ehor Yarmolyuk"),
    ("Yegor Yarmolyuk", "Ehor Yarmolyuk"),
    ("Joe Gomez", "Joseph Gomez"),
    ("Wes Foderingham", "Wesley Foderingham"),
    ("Olu Aina", "Ola Aina"),
    ("Oliver Arblaster", "Ollie Arblaster"),
    ("Illia Zabarnyi", "Ilya Zabarnyi"),
    ("Vitalii Mykolenko", "Vitaliy Mykolenko"),
    ("Carlos Alcaraz Durán", "Charly Alcaraz"),
    ("Will Smallbone", "William Smallbone"),
    ("Josh King", "Joshua King"),
    ("Sam Szmodics", "Sammie Szmodics"),
    ("Matt O'Riley", "Matthew O'Riley"),
    ("Ollie Scarles", "Oliver Scarles"),
    ("Dan Neil", "Daniel Neil"),
    ("Maximilian Kilman", "Max Kilman"),
    ("Yéremy Pino Santos", "Yéremi Pino"),
    ("Jorge Luiz Frello Filho", "Jorginho"),
    ("Rodrigo Moreno", "Rodrigo"),
    ("Fabio Henrique Tavares", "Fabinho"),
    ("Diogo Teixeira da Silva", "Diogo Jota"),
    ("Rodrigo Hernandez", "Rodri"),
    ("Rodrigo 'Rodri' Hernandez", "Rodri"),
    ("Rodrigo 'Rodri' Hernandez Cascante", "Rodri"),
    ("Frederico Rodrigues de Paula Santos", "Fred"),
    ("Joelinton Cássio Apolinário de Lira", "Joelinton"),
    ("Carlos Ribeiro Dias", "Cafú"),
    ("Lyanco Silveira Neves Vojnovic", "Lyanco"),
    ("Emerson Leite de Souza Junior", "Emerson Royal"),
    ("Richarlison de Andrade", "Richarlison"),
    ("Jonathan Castro Otto", "Jonny Otto"),
    ("Emerson Palmieri dos Santos", "Emerson"),
    ("Norberto Murara Neto", "Neto"),
    ("Carlos Henrique Casimiro", "Casemiro"),
    ("Lucas Tolentino Coelho de Lima", "Lucas Paquetá"),
    ("Antony Matheus dos Santos", "Antony"),
    ("Willian Borges da Silva", "Willian"),
    ("Danilo dos Santos de Oliveira", "Danilo"),
    ("Mateus Cardoso Lemos Martins", "Tetê"),
    ("Felipe Augusto de Almeida Monteiro", "Felipe"),
    ("Victor da Silva", "Vitinho"),
    ("Norberto Bercique Gomes Betuncal", "Beto"),
    ("Murillo Santiago Costa dos Santos", "Murillo"),
    ("Murillo Costa dos Santos", "Murillo"),
    ("Anssumane Fati Vieira", "Ansu Fati"),
    ("Ederson Santana de Moraes", "Ederson"),
    ("Francisco Jorge Tomás Oliveira", "Chiquinho"),
    ("Sávio 'Savinho' Moreira de Oliveira", "Savinho"),
    ("Sávio 'Savinho' Moreira de Oliveira", "Sávio"),
    ("Sávio Moreira de Oliveira", "Savinho"),
    ("Sávio Moreira de Oliveira", "Sávio"),
    ("João Pedro Ferreira Silva", "Jota Silva"),
    ("João Pedro Ferreira da Silva", "Jota Silva"),
    ("André Trindade da Costa Neto", "André"),
    ("Welington Damascena Santos", "Welington"),
    ("Pablo Felipe Pereira de Jesus", "Pablo"),
)

# Match-specific repairs reviewed against the same FotMob lineups and events
PLAYER_RECORD_REPAIRS = {
    (4193455, "538210"): (0, "Andre Brooks", 1396784),
    (4193476, "538210"): (0, "Andre Brooks", 1396784),
    (4193491, "501284"): (0, "Dakota Ochsenham", 1130797),
    (4193511, "501284"): (0, "Dakota Ochsenham", 1130797),
    (4193500, "1553377"): (1553377, "Alex Matos", 1202031),
}


def reviewed_player_records(payload, source_member):
    """Apply reviewed identity repairs to copies, preserving original cells in an audit"""
    match_id = int(payload["general"]["matchId"])
    records, repairs = [], []
    content = payload.get("content", {})
    for key, player in (content.get("playerStats") or {}).items():
        repair = PLAYER_RECORD_REPAIRS.get((match_id, str(key)))
        if repair:
            old_id, old_name, player_id = repair
            lineup = content.get("lineup", {})
            candidates = [
                (team["id"], p)
                for team in (lineup.get("homeTeam", {}), lineup.get("awayTeam", {}))
                for p in team.get("starters", []) + team.get("subs", [])
                if p.get("id") == player_id
            ]
            if len(candidates) != 1:
                raise ValueError(f"Missing or ambiguous reviewed lineup in {source_member}")
            team_id, actor = candidates[0]
            if player.get("teamId") != team_id:
                raise ValueError(f"Reviewed player club changed in {source_member}")
            if (player.get("id"), player.get("name")) == (player_id, actor["name"]):
                records.append(player)
                continue
            if (player.get("id"), player.get("name")) != (old_id, old_name):
                raise ValueError(f"Reviewed source identity changed in {source_member}")
            events = actor.get("performance", {}).get("substitutionEvents", [])
            entries = tuple(e["time"] for e in events if e.get("type") == "subIn")
            if bool(player.get("stats")) != bool(entries):
                raise ValueError(f"Reviewed participation changed in {source_member}")
            repairs.append({
                "source_member": source_member, "fotmob_fixture_id": match_id,
                "source_player_key": str(key), "reported_player_id": old_id,
                "reported_name": old_name, "fotmob_player_id": player_id,
                "player_name": actor["name"], "fotmob_team_id": team_id,
                "substitution_minutes": entries,
            })
            player = {**player, "id": player_id, "name": actor["name"]}
        records.append(player)
    return records, repairs


def _plain_name(value):
    text = str(value).casefold().translate(str.maketrans({"ø": "o", "ł": "l", "đ": "dj", "ı": "i", "æ": "ae", "œ": "oe"}))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def club_name_key(value):
    plain = _plain_name(value)
    return CLUB_ALIASES.get(plain, plain)


def _aliases(values):
    return tuple(sorted(set(values.dropna())))


def load_fotmob_identity_lookups(cache_path, seasons):
    """Extract identities/context only from cached EPL matches in Aug–July seasons

    Opta IDs and player performance-stat groups are deliberately not extracted
    Cancelled/abandoned fixtures remain in the fixture lookup with their status
    """
    seasons = set(seasons)
    fixture_rows, player_rows, identity_issues, source_repairs = [], [], [], []
    counters = Counter()
    with tarfile.open(Path(cache_path), "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            counters["archive_jsons"] += 1
            payload = json.load(archive.extractfile(member))
            general = payload.get("general", {})
            date = str(general.get("matchTimeUTCDate", ""))
            if not re.match(r"^\d{4}-\d{2}-\d{2}", date):
                counters["entries_without_usable_date"] += 1
                continue
            year, month = int(date[:4]), int(date[5:7])
            start_year = year if month >= 8 else year - 1
            season = f"{start_year}_{start_year + 1}"
            if season not in seasons or str(general.get("leagueId")) != "47":
                continue

            kickoff = pd.Timestamp(date)
            kickoff = kickoff.tz_localize("UTC") if kickoff.tzinfo is None else kickoff.tz_convert("UTC")
            home, away = general["homeTeam"], general["awayTeam"]
            match_id = int(general["matchId"])
            status = payload.get("header", {}).get("status", {})
            reason = status.get("reason") or {}
            if not isinstance(reason, dict):
                reason = {"long": str(reason)}
            fixture_rows.append({
                "season": season, "fotmob_fixture_id": match_id, "kickoff_time": kickoff,
                "home_fotmob_team_id": int(home["id"]), "home_team_name": home["name"],
                "away_fotmob_team_id": int(away["id"]), "away_team_name": away["name"],
                "finished": bool(general.get("finished", False)),
                "cancelled": bool(status.get("cancelled", False)),
                "status_reason": reason.get("long"), "source_member": member.name,
            })

            players = payload.get("content", {}).get("playerStats", {})
            if not isinstance(players, dict) or not players:
                counters["selected_matches_without_player_records"] += 1
                continue
            players, repairs = reviewed_player_records(payload, member.name)
            source_repairs.extend(repairs)
            for player in players:
                reported_id = player.get("id")
                try:
                    player_id = int(reported_id)
                except (ValueError, TypeError):
                    player_id = None
                team_id = player.get("teamId")
                team_id = int(team_id) if team_id is not None else None
                if team_id not in (int(home["id"]), int(away["id"])):
                    raise ValueError(f"Unresolved match-time player club in {member.name}: {player['id']}")
                record = {
                    "season": season, "fotmob_player_id": player_id,
                    "fotmob_fixture_id": match_id, "fotmob_team_id": team_id,
                    "player_name": player["name"], "kickoff_time": kickoff,
                    "source_member": member.name,
                }
                if player_id is None or player_id <= 0:
                    record["fotmob_player_id"] = pd.NA
                    record["reported_fotmob_player_id"] = reported_id
                    record["issue"] = "missing_or_nonpositive_player_id"
                    identity_issues.append(record)
                    counters["selected_player_records_with_unusable_id"] += 1
                    continue
                player_rows.append(record)

    fixtures = pd.DataFrame(fixture_rows)
    player_matches = pd.DataFrame(player_rows)
    if fixtures.empty or player_matches.empty:
        raise ValueError("No matching FotMob fixture/player identities found")
    if fixtures.duplicated(["fotmob_fixture_id"]).any():
        raise ValueError("Repeated FotMob fixture ID; inspect archive entries")
    if player_matches.duplicated(["fotmob_player_id", "fotmob_fixture_id"]).any():
        examples = player_matches.loc[player_matches.duplicated(["fotmob_player_id", "fotmob_fixture_id"], keep=False)].head(8).to_dict("records")
        raise ValueError(f"Repeated FotMob player-fixture identity: {examples}")

    for frame in (fixtures, player_matches):
        for column in frame.columns:
            if column.endswith("_id"):
                frame[column] = frame[column].astype("Int64")

    team_observations = pd.concat([
        fixtures[["season", "home_fotmob_team_id", "home_team_name", "kickoff_time"]].rename(
            columns={"home_fotmob_team_id": "fotmob_team_id", "home_team_name": "team_name"}),
        fixtures[["season", "away_fotmob_team_id", "away_team_name", "kickoff_time"]].rename(
            columns={"away_fotmob_team_id": "fotmob_team_id", "away_team_name": "team_name"}),
    ], ignore_index=True)

    def registry(observations, identifier, name):
        ordered = observations.sort_values("kickoff_time", kind="stable")
        result = ordered.groupby(identifier).agg(
            display_name=(name, "last"), name_aliases=(name, _aliases),
            first_seen_utc=("kickoff_time", "min"), last_seen_utc=("kickoff_time", "max"),
            seasons_seen=("season", _aliases),
        ).reset_index()
        return result.rename(columns={"display_name": name})

    players = registry(player_matches, "fotmob_player_id", "player_name")
    teams = registry(team_observations, "fotmob_team_id", "team_name")
    team_seasons = team_observations[["season", "fotmob_team_id", "team_name"]].drop_duplicates()
    if not set(player_matches.fotmob_team_id).issubset(set(teams.fotmob_team_id)):
        raise ValueError("Player club missing from FotMob team registry")
    return {
        "players": players, "teams": teams,
        "fixtures": fixtures.sort_values(["kickoff_time", "fotmob_fixture_id"]).reset_index(drop=True),
        "player_matches": player_matches.sort_values(["kickoff_time", "fotmob_fixture_id", "fotmob_player_id"]).reset_index(drop=True),
        "team_seasons": team_seasons.reset_index(drop=True),
        "identity_issues": pd.DataFrame(identity_issues),
        "source_repairs": pd.DataFrame(source_repairs),
        "extraction_summary": dict(counters),
    }


def build_team_identity_map(fpl_teams_by_season, fotmob_team_seasons):
    """Match every FPL team-season to FotMob using explicit club-name equivalents"""
    source_rows = []
    for season, frame in fpl_teams_by_season.items():
        part = frame[["id", "name"]].rename(columns={"id": "fpl_team_id", "name": "fpl_team_name"}).copy()
        part["season"] = season
        source_rows.append(part)
    source = pd.concat(source_rows, ignore_index=True)
    source["club_name_key"] = source.fpl_team_name.map(club_name_key)
    target = fotmob_team_seasons.copy()
    target["club_name_key"] = target.team_name.map(club_name_key)
    ambiguous = target.groupby(["season", "club_name_key"]).fotmob_team_id.nunique()
    if ambiguous.gt(1).any():
        raise ValueError("Ambiguous FotMob club-name mapping")
    target = target.sort_values("team_name").drop_duplicates(["season", "club_name_key"])
    result = source.merge(target, on=["season", "club_name_key"], how="left", validate="one_to_one", indicator=True)
    if not result._merge.eq("both").all():
        missing = result.loc[result._merge.ne("both"), ["season", "fpl_team_name"]].to_dict("records")
        raise ValueError(f"Unmatched FPL clubs: {missing}")
    result = result.drop(columns="_merge").rename(columns={"team_name": "fotmob_team_name"})
    result["fpl_team_id"] = result.fpl_team_id.astype("Int64")
    result["fotmob_team_id"] = result.fotmob_team_id.astype("Int64")
    result["mapping_method"] = [
        "normalized_name" if _plain_name(a) == _plain_name(b) else "explicit_club_alias"
        for a,b in zip(result.fpl_team_name, result.fotmob_team_name)
    ]
    result["mapping_status"] = "validated_team_name"
    if result.duplicated(["season", "fpl_team_id"]).any() or result.duplicated(["season", "fotmob_team_id"]).any():
        raise ValueError("Team mapping must be one-to-one within each season")
    if len(result) != len(source):
        raise ValueError("Team mapping changed row count")
    return result[["season", "fpl_team_id", "fpl_team_name", "fotmob_team_id", "fotmob_team_name", "mapping_method", "mapping_status"]]


def external_match_inputs(player_matches, fixtures, team_identity_map):
    """Translate team keys in temporary matching copies, leaving working tables intact"""
    key = ["season", "fpl_team_id"]
    if team_identity_map[key + ["fotmob_team_id"]].isna().any().any() or team_identity_map.duplicated(key).any():
        raise ValueError("Team bridge must have complete, unique FPL keys")
    lookup = team_identity_map.set_index(key).fotmob_team_id
    fixture_input = fixtures[["season", "fpl_fixture_id", "kickoff_time", "home_fpl_team_id", "away_fpl_team_id"]].copy()
    player_input = player_matches[["season", "fpl_player_id", "fpl_fixture_id", "fpl_team_id", "name"]].copy()
    translations = (
        (fixture_input, {"home_fpl_team_id": "home_fotmob_team_id", "away_fpl_team_id": "away_fotmob_team_id"}),
        (player_input, {"fpl_team_id": "fotmob_team_id"}),
    )
    for frame, columns in translations:
        for source, target in columns.items():
            index = pd.MultiIndex.from_arrays([frame.season, frame[source]])
            frame[target] = lookup.reindex(index).array.astype("Int64")
            if frame[target].isna().any():
                raise ValueError(f"Unmatched FPL club in {source}")
    return fixture_input, player_input


def build_fixture_identity_map(fpl_fixture_team_links, fotmob_fixtures, tolerance_minutes=30):
    """Match unique home/away fixtures in the same season, preserving both times

    Completed, non-cancelled records only A small kickoff discrepancy is allowed;
    larger differences (including possible rescheduling) remain review cases
    """
    if tolerance_minutes < 0:
        raise ValueError("Kickoff tolerance cannot be negative")
    source = fpl_fixture_team_links.copy().rename(columns={"kickoff_time": "fpl_kickoff_utc"})
    source_key = ["season", "fpl_fixture_id"]
    pair = ["season", "home_fotmob_team_id", "away_fotmob_team_id"]
    if source[source_key + pair[1:]].isna().any().any() or source.duplicated(source_key).any():
        raise ValueError("FPL fixture identities/teams must be present with unique source keys")
    source["fpl_kickoff_utc"] = pd.to_datetime(source.fpl_kickoff_utc, utc=True, errors="raise")
    target = fotmob_fixtures.loc[
        fotmob_fixtures.finished & ~fotmob_fixtures.cancelled,
        pair + ["fotmob_fixture_id", "kickoff_time"],
    ].rename(columns={"kickoff_time": "fotmob_kickoff_utc"}).copy()
    if target.fotmob_fixture_id.isna().any() or target.fotmob_fixture_id.duplicated().any():
        raise ValueError("FotMob fixture IDs must be present and unique")
    target["fotmob_kickoff_utc"] = pd.to_datetime(target.fotmob_kickoff_utc, utc=True, errors="raise")
    candidates = source.merge(target, on=pair, how="left", validate="many_to_many")
    candidates["kickoff_difference_minutes"] = (
        candidates.fotmob_kickoff_utc - candidates.fpl_kickoff_utc
    ).dt.total_seconds().div(60)
    eligible = candidates.loc[candidates.kickoff_difference_minutes.abs().le(tolerance_minutes)].copy()
    counts = eligible.groupby(source_key).size()
    approved = eligible.merge(counts.rename("candidate_count"), on=source_key, validate="many_to_one")
    approved = approved.loc[approved.candidate_count.eq(1)].copy()
    # Never allow two FPL fixtures to acquire the same FotMob fixture identity
    collisions = approved.fotmob_fixture_id.duplicated(keep=False)
    approved = approved.loc[~collisions]
    selected = approved[source_key + ["fotmob_fixture_id", "fotmob_kickoff_utc", "kickoff_difference_minutes"]]
    result = source.merge(selected, on=source_key, how="left", validate="one_to_one")
    result["fotmob_fixture_id"] = result.fotmob_fixture_id.astype("Int64")
    result["mapping_status"] = "unresolved"
    result.loc[result.fotmob_fixture_id.notna(), "mapping_status"] = "matched"
    result["mapping_method"] = "review_required"
    result.loc[result.fotmob_fixture_id.notna(), "mapping_method"] = "same_teams_kickoff_tolerance"
    result.loc[result.kickoff_difference_minutes.eq(0), "mapping_method"] = "same_teams_exact_kickoff"
    candidates["within_tolerance"] = candidates.kickoff_difference_minutes.abs().le(tolerance_minutes)
    return {"mapping": result, "review": candidates.merge(
        result.loc[result.mapping_status.ne("matched"), source_key], on=source_key, validate="many_to_one"
    )}


@lru_cache(maxsize=50000)
def _name_evidence(source_name, target_name):
    if _plain_name(source_name) and _plain_name(source_name) == _plain_name(target_name):
        return "normalized_name"
    def tokens(name):
        text = name.casefold().translate(str.maketrans({"ø": "o", "ł": "l", "đ": "dj", "ı": "i", "æ": "ae", "œ": "oe"}))
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
        return frozenset(re.findall(r"[a-z0-9]+", text))
    left, right = tokens(source_name), tokens(target_name)
    if left == right and left:
        return "same_name_tokens"
    # Require two meaningful shared tokens; a surname alone is insufficient
    particles = {"da", "de", "di", "do", "dos", "das", "van", "von", "der", "den", "la", "le", "el", "al"}
    shorter = left if len(left) <= len(right) else right
    if shorter and (left <= right or right <= left) and len(shorter - particles) >= 2:
        return "contained_name_tokens"
    return None


def build_player_identity_map(working_matches, fixture_identity_map, fotmob_player_matches,
                              fotmob_identity_issues=None, reviewed_name_aliases=(), source_name_aliases=()):
    """Propose source-player identities from names plus shared fixture/team evidence

    Source aliases are full-name/display-name pairs from the saved FPL bootstrap;
    they supply labels only, never snapshot IDs, clubs or performance values
    Explicit aliases must be reviewed name pairs Similarity scores, equal numeric
    IDs and squad membership without a shared fixture never approve a mapping
    Return the bridge and its evidence; do not alter the working observations
    """
    source_key = ["season", "fpl_player_id"]
    match_key = source_key + ["fpl_fixture_id"]
    if working_matches[match_key + ["fotmob_team_id", "name"]].isna().any().any():
        raise ValueError("Source player identities, clubs and names must be present")
    if working_matches.duplicated(match_key).any():
        raise ValueError("Repeated source player-fixture key")
    bridge = fixture_identity_map.loc[fixture_identity_map.mapping_status.eq("matched"),
                                    ["season", "fpl_fixture_id", "fotmob_fixture_id"]]
    if bridge.duplicated(["season", "fpl_fixture_id"]).any() or bridge.fotmob_fixture_id.duplicated().any():
        raise ValueError("Fixture bridge must be one-to-one")
    source = working_matches.drop(columns="fotmob_fixture_id", errors="ignore").merge(
        bridge, on=["season", "fpl_fixture_id"], how="inner", validate="many_to_one"
    )
    target = fotmob_player_matches.copy()
    if target[["fotmob_player_id", "fotmob_fixture_id", "fotmob_team_id", "player_name"]].isna().any().any():
        raise ValueError("Unusable target identity belongs in the issues table")
    if target.fotmob_player_id.le(0).any() or target.duplicated(["fotmob_player_id", "fotmob_fixture_id"]).any():
        raise ValueError("FotMob player-fixture identities must be positive and unique")
    def grouped(frame, keys, name):
        return frame.groupby(keys).agg(
            names=(name, _aliases), fixtures=("fotmob_fixture_id", lambda s: frozenset(s)),
        ).reset_index()
    left = grouped(source, source_key + ["fotmob_team_id"], "name")
    right = grouped(target, ["season", "fotmob_team_id", "fotmob_player_id"], "player_name")
    pairs = left.merge(right, on=["season", "fotmob_team_id"], suffixes=("_fpl", "_fotmob"))
    aliases = {(_plain_name(a), _plain_name(b)) for a, b in reviewed_name_aliases}
    source_alias_lookup = {}
    for full_name, display_name in source_name_aliases:
        source_alias_lookup.setdefault(_plain_name(full_name), set()).add(display_name)
    @lru_cache(maxsize=None)
    def name_method(a, b):
        method = _name_evidence(a, b)
        if not method and (_plain_name(a), _plain_name(b)) in aliases:
            method = "reviewed_name_alias"
        if not method:
            for display_name in source_alias_lookup.get(_plain_name(a), ()):
                if _name_evidence(display_name, b):
                    return "FPL_snapshot_name_alias"
        return method
    rows = []
    for p in pairs.itertuples(index=False):
        common = p.fixtures_fpl & p.fixtures_fotmob
        if not common:
            continue
        evidence = []
        for a in p.names_fpl:
            for b in p.names_fotmob:
                method = name_method(a, b)
                if method:
                    evidence.append((method, a, b))
        if evidence:
            rows.append({
                "season": p.season, "fpl_player_id": p.fpl_player_id,
                "fotmob_player_id": p.fotmob_player_id, "fotmob_team_id": p.fotmob_team_id,
                "shared_fixture_ids": tuple(sorted(common)), "shared_fixtures": len(common),
                "name_evidence": tuple(sorted(set(evidence))),
            })
    columns = source_key + ["fotmob_player_id", "fotmob_team_id", "shared_fixture_ids", "shared_fixtures", "name_evidence"]
    candidates = pd.DataFrame(rows, columns=columns)
    registry = working_matches.groupby(source_key).agg(name_aliases=("name", _aliases)).reset_index()
    mappings = []
    for (season, player_id), group in candidates.groupby(source_key):
        ids = tuple(sorted(group.fotmob_player_id.unique()))
        methods = tuple(sorted({e[0] for names in group.name_evidence for e in names}))
        shared = tuple(sorted({f for fixtures in group.shared_fixture_ids for f in fixtures}))
        mappings.append({
            "season": season, "fpl_player_id": player_id,
            "fotmob_player_id": ids[0] if len(ids) == 1 else pd.NA,
            "candidate_fotmob_ids": ids, "candidate_count": len(ids),
            "shared_fixtures": len(shared), "evidence_fixture_ids": shared,
            "mapping_method": ", ".join(methods),
            "mapping_status": "matched" if len(ids) == 1 else "ambiguous_candidates",
        })
    mapping_columns = source_key + ["fotmob_player_id", "candidate_fotmob_ids", "candidate_count", "shared_fixtures",
                                     "evidence_fixture_ids", "mapping_method", "mapping_status"]
    result = registry.merge(pd.DataFrame(mappings, columns=mapping_columns), on=source_key, how="left", validate="one_to_one")
    result["fotmob_player_id"] = result.fotmob_player_id.astype("Int64")
    result["mapping_status"] = result.mapping_status.fillna("no_shared_fixture_name_match")
    result["candidate_count"] = result.candidate_count.fillna(0).astype("Int64")
    result["shared_fixtures"] = result.shared_fixtures.fillna(0).astype("Int64")
    for column in ("candidate_fotmob_ids", "evidence_fixture_ids"):
        result[column] = result[column].map(lambda value: value if isinstance(value, tuple) else ())
    result["mapping_method"] = result.mapping_method.fillna("none")
    matched = result.loc[result.mapping_status.eq("matched")]
    conflict_index = matched.loc[matched.duplicated(["season", "fotmob_player_id"], keep=False)].index
    result.loc[conflict_index, "mapping_status"] = "conflicting_source_identities"
    result.loc[conflict_index, "fotmob_player_id"] = pd.NA

    # Unusable ID-zero records can explain gaps, but must never establish identity
    issue_rows = []
    if fotmob_identity_issues is not None and not fotmob_identity_issues.empty:
        issues = fotmob_identity_issues.rename(columns={"player_name": "issue_player_name"})
        joined = source.merge(issues[["season", "fotmob_fixture_id", "fotmob_team_id", "issue_player_name", "issue"]],
                              on=["season", "fotmob_fixture_id", "fotmob_team_id"], how="inner")
        for row in joined.itertuples(index=False):
            if name_method(row.name, row.issue_player_name):
                issue_rows.append({"season": row.season, "fpl_player_id": row.fpl_player_id,
                                   "fpl_name": row.name, "source_name": row.issue_player_name,
                                   "fotmob_fixture_id": row.fotmob_fixture_id, "issue": row.issue})
    issue_links = pd.DataFrame(issue_rows, columns=source_key + ["fpl_name", "source_name", "fotmob_fixture_id", "issue"])
    if not issue_links.empty:
        issue_keys = pd.MultiIndex.from_frame(issue_links[source_key])
        affected = pd.MultiIndex.from_frame(result[source_key]).isin(issue_keys)
        result.loc[affected & result.mapping_status.eq("no_shared_fixture_name_match"), "mapping_status"] = "unusable_fotmob_source_id"
    return {"mapping": result, "candidates": candidates, "source_id_issues": issue_links}


def player_matching_report(player_matches, result):
    """Summarize external matching without applying provider IDs to FPL records"""
    key = ["season", "fpl_player_id"]
    mapping = result["mapping"]
    matched = mapping.loc[mapping.mapping_status.eq("matched")]
    if (mapping.duplicated(key).any() or len(mapping) != len(player_matches[key].drop_duplicates())
            or matched.fotmob_player_id.isna().any() or matched.shared_fixtures.lt(1).any()
            or matched.duplicated(["season", "fotmob_player_id"]).any()):
        raise ValueError("Invalid player bridge coverage or matched keys")
    coverage = player_matches[key + ["fpl_fixture_id", "minutes"]].merge(
        mapping[key + ["mapping_status"]], on=key, how="left", validate="many_to_one"
    )
    if coverage.mapping_status.isna().any():
        raise ValueError("FPL player missing from the external matching report")
    coverage["matched_rows"] = coverage.mapping_status.eq("matched")
    coverage["positive_minute_rows"] = coverage.minutes.gt(0)
    coverage["matched_positive_minute_rows"] = coverage.matched_rows & coverage.positive_minute_rows
    summary = coverage.groupby("season").agg(
        player_fixture_rows=("fpl_player_id", "size"), proposed_matched_rows=("matched_rows", "sum"),
        positive_minute_rows=("positive_minute_rows", "sum"),
        matched_positive_minute_rows=("matched_positive_minute_rows", "sum"),
    )
    minutes = player_matches.groupby(key).minutes.sum().rename("recorded_minutes")
    review = mapping.loc[mapping.mapping_status.ne("matched")].merge(minutes, on=key, validate="one_to_one")
    if review.fotmob_player_id.notna().any():
        raise ValueError("Unresolved external matches must not have an assigned target")
    conflicts = review.loc[review.mapping_status.isin(["ambiguous_candidates", "conflicting_source_identities"]), key]
    return {"summary": summary, "review": review, "conflicts": result["candidates"].merge(conflicts, on=key, validate="many_to_one")}
