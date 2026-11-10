# FPL learning and analysis

Collect and prepare FPL data for player and team EDA, then forecasting. PostgreSQL and SQL learning are later work.

## Repository navigation

```text
src/                            Collection, parsing, joins and validation
notebooks/                      Preparation, then EDA and modelling
  01_data_preparation.ipynb
tests/                          Preparation and defensive-scoring checks
docs/
  fpl_rules.md                   Scoring definitions and target policy
  fpl_collection_learning.md     Collection commands and table definitions
.vscode/extensions.json          Shared editor recommendations
requirements.txt                Python dependencies
.gitignore                      Local data, caches and machine settings
```

External archive: **`/epl-premier-league-dataset`**, retained locally. It is the downloaded [source dataset](https://huggingface.co/datasets/Toibinguz/epl-premier-league-dataset); its location is configured by `ARCHIVE_DIR` in Notebook 1.

## Self-collected data navigation

**Self-collected API responses** is the appropriate term for the raw JSON. Parsing converts those responses into records; cleaning and joining produce **prepared tables**. The combined player history is a **joined player-fixture table**.

These data paths are local and excluded from Git:

```text
data/raw/fpl_api/<season>/<batch_utc>/
  bootstrap-static.json
  fixtures.json
  element-summary/<player_id>.json
  manifest.json

data/processed/fpl_self_collected/<season>/<batch>/
  fpl_cleaned_players_<season>.csv    Player season-to-date totals
  current_player_snapshot.csv        Dated attributes with native IDs
  fpl_teams_<season>.csv              Team reference table
  fpl_fixtures_<season>.csv           Fixture context and results
  current_player_matches.csv         Parsed player-fixture history
  reconciliation.csv
  manifest.json

data/processed/
  fpl_tables.pkl                     Joined and checked EDA tables
  fpl_audits.pkl                     Quality and collection evidence
  fpl_manifest.json                  Coverage and source checksums
```

For future canonical CSV exports, use `players_<season>.csv`, `teams_<season>.csv`, `fixtures_<season>.csv` and `player_matches_<season>.csv`. Describe their grain explicitly and retain FPL IDs. Existing `fpl_*` filenames remain available for source-format compatibility; current files have not been renamed.

## Current preparation

[Notebook 1](notebooks/01_data_preparation.ipynb) prepares 113,260 historical player-fixture records for 2022/23–2025/26. Complete API batch `20261110T210622560387Z` supplies 6,562 current records for 672 players across 100 finalized fixtures through GW 10.

Names and positions join by `(season, fpl_player_id)`; match-time club names join by `(season, fpl_team_id)`. Match records use `(season, fpl_player_id, fpl_fixture_id)`. Snapshots, teams and fixtures stay separate. FotMob matching evidence remains in audits.

Historical `P_off` preserves original awards and `P_adj` retains the existing adjustment. Current `P_off` is preserved exactly as collected. Consult [the rules reference](docs/fpl_rules.md) before changing scoring.

Run Notebook 1 from the project or `notebooks/` directory. Select another complete raw batch with `COLLECTION_BATCH`. Load the resulting dictionary with `pd.read_pickle("data/processed/fpl_tables.pkl")` for EDA. Preparation preserves raw responses and replaces generated outputs.

## Collection and development

- `src/collect_fpl.py`: full collection and offline parsing
- `src/collect_players.py`, `src/collect_teams.py`: smaller snapshot collectors
- `src/data_preparation.py`: joins, validation and exports
- `src/fotmob_identity.py`, `src/defensive_contributions.py`: historical enrichment
- `src/download_dataset.py`: archive download utility

See [the collection guide](docs/fpl_collection_learning.md) for commands. Use the same Python environment for scripts and notebooks, with dependencies from `requirements.txt`. Editor interpreter settings and Mac-specific setup helpers remain local.

Next: Notebook 2 for player/team EDA, followed by forecasting baselines and chronological evaluation. Add model code, curated reports and PostgreSQL work to the repository when implemented; retain raw batches, generated datasets and binary model artifacts outside Git.

## Copyright and use

Copyright © 2026 Zhangir Bayanov. All rights reserved.

The original code is available for personal, noncommercial use under the repository's license. Commercial use requires the copyright holder's permission. [License terms](LICENSE)
