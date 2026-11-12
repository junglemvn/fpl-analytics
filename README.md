# FPL learning and analysis

Collect, prepare and explore FPL player, team and fixture data. Preparation and the initial EDA are complete. Next: pre-deadline forecasting and constrained squad selection. PostgreSQL and SQL learning remain later work.

## Repository navigation

```text
src/                            Collection, preparation and reusable EDA calculations
notebooks/
  01_data_preparation.ipynb       Completed preparation and validation
  02_eda.ipynb                   Completed player/team EDA and findings
docs/
  fpl_rules.md                   Scoring definitions and target policy
  fpl_collection_learning.md     Collection commands and table definitions
  notebook_style.md              Structure, wording, code and validation conventions
  02_eda_plan.md                 EDA scope, treatments and validation results
.vscode/extensions.json          Shared editor recommendations
requirements.txt                Python dependencies
.gitignore                      Local data, caches and machine settings
README.md                       Project status and navigation
LICENSE                         Personal, noncommercial code-use terms
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
  fpl_tables.pkl                     Prepared player, team and fixture tables
  fpl_audits.pkl                     Quality and collection evidence
  fpl_manifest.json                  Coverage and source checksums
  fpl_eda_tables.pkl                 Profiles, rankings, team context and stability summaries
  fpl_eda_audits.pkl                 Quality checks, treatments and ranking diagnostics
  fpl_eda_manifest.json              Coverage, treatments and input/code checksums
```

For future canonical CSV exports, use `players_<season>.csv`, `teams_<season>.csv`, `fixtures_<season>.csv` and `player_matches_<season>.csv`. Describe their grain explicitly and retain FPL IDs. Existing `fpl_*` filenames remain available for source-format compatibility; current files have not been renamed.

## Preparation complete

[Notebook 1](notebooks/01_data_preparation.ipynb) prepares 113,260 historical player-fixture records for 2022/23–2025/26. Complete API batch `20261110T210622560387Z` supplies 6,562 current records for 672 players across 100 finalized fixtures through GW 10.

Names and positions join by `(season, fpl_player_id)`; match-time club names join by `(season, fpl_team_id)`. Match records use `(season, fpl_player_id, fpl_fixture_id)`. Snapshots, teams and fixtures stay separate. FotMob matching evidence remains in audits.

Historical `P_off` preserves original awards and `P_adj` retains the existing adjustment. Current `P_off` is preserved exactly as collected. Consult [the rules reference](docs/fpl_rules.md) before changing scoring.

Run Notebook 1 from the project or `notebooks/` directory. Select another complete raw batch with `COLLECTION_BATCH`. Load the resulting dictionary with `pd.read_pickle("data/processed/fpl_tables.pkl")` for EDA. Preparation preserves raw responses and replaces generated outputs.

## EDA complete

[Notebook 2](notebooks/02_eda.ipynb) retains all 119,822 player-fixture records across the four completed seasons and the captured 2026/27 season through GW10. It repairs 2,992 missing starts in `starts_eda`, marks 4,732 suspect ICT cells as unavailable and preserves original defensive measurements. Notebook 1's outputs remain unchanged.

The completed analysis covers:

- Participation, fantasy point components and patterns across historical seasons
- Positional player rankings, expected versus actual returns and dated price context
- Historical team attack/defence, venue splits and contribution concentration
- Current player/team progress, recent fixture windows and prior-only opponent context
- Gameweek-resampling diagnostics showing uncertainty in close player rankings

Start with the notebook's findings and reading path. Full tables, audits and provenance are saved separately in `fpl_eda_tables.pkl`, `fpl_eda_audits.pkl` and `fpl_eda_manifest.json`. These are descriptive outputs; forecasting features still need pre-deadline cutoffs. See the [EDA scope](docs/02_eda_plan.md) and [notebook conventions](docs/notebook_style.md).

Club emblems load from `logos/`, with coverage checks for all five seasons and image checksums in the EDA manifest. No separate logo archive is required.

## Collection and development

- `src/collect_fpl.py`: full collection and offline parsing
- `src/collect_players.py`, `src/collect_teams.py`: smaller snapshot collectors
- `src/data_preparation.py`: joins, validation and exports
- `src/eda_preparation.py`: validated starter repair, ICT masks and EDA summaries
- `src/player_analysis.py`: historical point accounting, player rankings and descriptive fits
- `src/team_analysis.py`: fixture-derived team results, venue splits and contribution concentration
- `src/plot_annotations.py`: reproducible chart label selection and placement
- `src/current_player_analysis.py`: snapshot-aware progress and current-club fixture windows
- `src/current_team_analysis.py`: current team form and prior-only opponent context
- `src/analysis_synthesis.py`: recurring point-source and team patterns across seasons
- `src/ranking_uncertainty.py`: shared-gameweek resampling and ranking stability
- `src/fotmob_identity.py`, `src/defensive_contributions.py`: historical enrichment
- `src/download_dataset.py`: archive download utility

See [the collection guide](docs/fpl_collection_learning.md) for commands. Use the same Python environment for scripts and notebooks, with dependencies from `requirements.txt`. Editor interpreter settings and Mac-specific setup helpers remain local.

## Next stage

1. Define the next-gameweek points target and create lagged player, team and opponent features
2. Compare simple benchmarks, regularized regression and boosted trees using chronological validation
3. Evaluate team scoreline and clean-sheet probabilities, plus player participation forecasts
4. Optimise squad, XI, captain and bench under budget, position and club limits
5. Extend to several gameweeks, availability updates and transfer decisions

Save dated prices and availability with future collections. The current snapshot cannot establish what was known before every historical deadline. Consult the rules reference for the limits of historical `P_adj`; it harmonises defensive rewards and reviewed corrections, not every scoring-rule change.

## GitHub scope

Upload the code, both notebooks, documentation, required `logos/` assets and root files shown above. Include `.vscode/extensions.json` as the shared editor configuration. Saved notebook charts and concise tables can remain for readers to inspect without the local datasets.

Keep `tests/`, raw batches, processed tables, the external archive, caches, virtual environments, local editor settings and machine-specific setup helpers outside Git. The current `.gitignore` covers these paths. Add modelling notebooks, model code and curated reports when implemented; keep generated model binaries local.

## Copyright and use

Copyright © 2026 Zhangir Bayanov. All rights reserved.

The original code is available for personal, noncommercial use under the repository's license. Commercial use requires the copyright holder's permission. [License terms](LICENSE)
