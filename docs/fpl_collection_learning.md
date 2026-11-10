# FPL collection and preparation

Run scripts from the project root using the Python interpreter selected in VS Code. The default season is `2026_2027`; live responses are checked against that season.

## Collection commands

| Command | Purpose |
|---|---|
| `python src/collect_players.py` | All player season-to-date totals and a rich player snapshot |
| `python src/collect_teams.py` | All native FPL team records |
| `python src/collect_fpl.py --all-players` | Bootstrap, fixtures and every current player's history |
| `python src/collect_fpl.py --player-ids 16 295` | A small history sample; summaries still cover all players |
| `python src/collect_fpl.py --summary-only` | All cumulative summaries without history requests |

The full collector uses `bootstrap-static/`, `fixtures/` and `element-summary/{player_id}/`. The smaller collectors each need only bootstrap. Requests use timeouts, bounded retries and a one-second pause between player histories. Failures are recorded separately from successful empty histories.

Raw responses and their receipts go into:

```text
data/raw/fpl_api/<season>/<batch_utc>/
  bootstrap-static.json
  fixtures.json
  element-summary/<player_id>.json
  manifest.json
```

Each receipt records its URL, HTTP status, UTC retrieval time and response checksum. Prepared CSVs go into the corresponding `data/processed/fpl_self_collected/` batch. Collection does not overwrite earlier batches.

## Existing complete batch

Batch `20261110T210622560387Z` collected all 672 current players on 10 November 2026, approximately 21:06–21:18 UTC. It contains 674 verified response files, 6,562 history rows and 7,392 passing reconciliation checks. Those rows cover 100 finalized fixtures through GW 10 and retain zero-minute observations and negative points.

To prepare that saved batch again without network access:

```sh
python src/collect_fpl.py --from-batch data/raw/fpl_api/2026_2027/20261110T210622560387Z
```

Both smaller scripts accept the same `--from-batch` argument. Saved-response processing retains original retrieval times and verifies checksums. Live requests require working network access; offline preparation does not.

## Tables and joins

| Export | Row level |
|---|---|
| `fpl_cleaned_players_<season>.csv` | Player cumulative totals in the historical 20-column layout |
| `current_player_snapshot.csv` | Rich dated player attributes with native IDs |
| `fpl_teams_<season>.csv` | Native team records |
| `fpl_fixtures_<season>.csv` | Native fixtures; nested statistics serialized as JSON |
| `current_player_matches.csv` | Player-fixture observations with official `P_off` |

The compact historical-format summary has no ID. Its companion snapshot preserves IDs and a batch-local `cleaned_csv_row` locator. Do not join by names or use a row locator across batches.

Notebook 1 reads a complete raw batch through `prepare_collected_batch` in `src/data_preparation.py`. It verifies receipts, normalizes numeric fields and UTC timestamps, and reconciles history totals with the captured bootstrap. It adds names/positions by `(season, fpl_player_id)` and club names by `(season, fpl_team_id)`, using the club derived from each fixture. Fixture references use `(season, fpl_fixture_id)`. All joins preserve observation counts and native FPL IDs.

Snapshots and player-match observations remain separate. Price units remain unchanged: `now_cost` and `value` use tenths of £1m. Snapshot ownership is a percentage; history `selected` is a count. Current official points are not recalculated. Historical point columns and their adjustment scope are documented in [the rules reference](fpl_rules.md).

The team table contains reference attributes; zero-valued `played`, `points` or `win` fields are not a verified league table. Derive descriptive team results from finalized fixture scores during EDA.

## Next collection improvements

Preserve a season registry for players missing from later bootstrap responses, support resuming interrupted live batches, and capture snapshots before deadlines. Refresh results after matches and finalization, retaining corrections as new raw versions. When a collection spans changing results, compare compatible cutoffs and refresh reference snapshots rather than forcing totals to agree. Scheduling and PostgreSQL loading remain later work.

## Terminology and future exports

Use **self-collected API responses** for the raw JSON, **parsed records** for decoded fields, and **prepared tables** for cleaned outputs. The joined history is a **player-fixture table**. Scraping usually refers to extracting information from web pages; the current collector requests structured JSON.

Existing `fpl_*` exports retain source-compatible filenames. For future canonical exports, use `players_<season>.csv` for player-season records with FPL IDs, `teams_<season>.csv`, `fixtures_<season>.csv`, and `player_matches_<season>.csv` for joined player-fixture observations. Keep dated snapshot metadata and separate original/adjusted point definitions. These names describe proposed future exports; existing files are not renamed.
