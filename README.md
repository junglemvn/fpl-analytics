# FPL learning and analysis

Collect FPL data, prepare player/team histories and forecast player-fixture fantasy points.
Preparation, EDA and the frozen model's historical evaluation are complete. The next stage is
constrained squad selection. The public repository contains preparation/analysis code,
chronological benchmarks and a results notebook; final model training, tuned settings and
weights remain private. PostgreSQL and deployment remain later work.

## Repository navigation

```text
src/                            Public collection, preparation, EDA and benchmark helpers
notebooks/
  01_data_preparation.ipynb       Completed preparation and validation
  02_eda.ipynb                   Completed player/team EDA and findings
  03_data_split_and_benchmarks.ipynb  Chronological split, benchmarks and validation errors
  04_model_results.ipynb          Public development and reserved-test results
reports/public/
  model_results.json             Aggregate results used by the public notebook
logos/                          All 27 club emblems used by the historical/current data
docs/
  fpl_rules.md                   Scoring definitions and target policy
  fpl_collection_learning.md     Collection commands and table definitions
.vscode/extensions.json          Shared editor recommendations
requirements.txt                Python dependencies
.gitignore                      Local data, caches and machine settings
README.md                       Project status and navigation
LICENSE                         MIT License
```

External archive: **`/epl-premier-league-dataset`**, retained locally. It is the downloaded [source dataset](https://huggingface.co/datasets/Toibinguz/epl-premier-league-dataset); its location is configured by `ARCHIVE_DIR` in Notebook 1.

## Self-collected data navigation

**Self-collected API responses** is the appropriate term for the raw JSON. Parsing converts those responses into records; cleaning and joining produce **prepared tables**. The combined player history is a **joined player-fixture table**.

These self-collected data and derived-table paths are local and excluded from Git:

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
  forecasts/<season>_GW<gw>_<batch>/
    fixture_expectations_v4.csv       Player-fixture expected points
    gameweek_expectations_v4.csv      Roster-preserving gameweek totals and prices
    forecast_receipt_v4.json          Source batch, deadline, coverage and model version
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

The notebook opens with its main findings and links to the supporting analysis. Full tables, audits and provenance are saved separately in `fpl_eda_tables.pkl`, `fpl_eda_audits.pkl` and `fpl_eda_manifest.json`. These are descriptive outputs; forecasting features still need pre-deadline cutoffs.

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
- `src/model_preparation.py`: chronological partitions, cutoff-aware histories and benchmarks
- `src/fotmob_identity.py`, `src/defensive_contributions.py`: historical enrichment
- `src/download_dataset.py`: archive download utility

See [the collection guide](docs/fpl_collection_learning.md) for commands. Use the same Python environment for scripts and notebooks, with dependencies from `requirements.txt`. Editor interpreter settings and Mac-specific setup helpers remain local.

## Forecast evaluation complete

[Notebook 3](notebooks/03_data_split_and_benchmarks.ipynb) establishes season partitions,
cutoff-aware histories and point benchmarks. Development fitting uses 2022/23–2023/24 and
validation uses 2024/25. The final frozen model and benchmark are refitted on
2022/23–2024/25 before evaluation on the original 2025/26 reserved partition.

[The public results notebook](notebooks/04_model_results.ipynb) loads aggregate results from
[the public summary](reports/public/model_results.json). It reports feature experiments,
availability coverage and grouped errors without exposing the fitted model or tuned settings.

| 2025/26 reserved evaluation | Player-fixtures | RMSE | MAE |
|---|---:|---:|---:|
| Frozen XGBoost v4 | 29,747 | 1.9014 | 0.9604 |
| Recent-five benchmark | 29,747 | 2.1182 | 1.0581 |

Both forecasts cover identical records. Model parameters remain fixed while eligible earlier
histories advance. Archived FPL status and round-matched playing chance supply availability;
current status is not joined backward onto historical fixtures. Historical cutoffs use a
first-kickoff-minus-24-hour proxy and results enter history after a fixed 24-hour delay.
These assumptions and corrected source data limit exact historical replay.

The test season appeared in descriptive EDA but was reserved during predictive selection.
The scoring receipt now records completed evaluation; the original preparation manifest
preserves its earlier unscored state. Current-season monitoring is separate. Historical
`P_adj` harmonises defensive rewards and reviewed corrections, not every scoring-rule change.
See [the scoring reference](docs/fpl_rules.md).

The initial GW12 forecast export contains 672 fixture forecasts and 672 roster totals,
captured on 21 November 2026 at 16:45 UTC. Its match history ends at GW10; refresh after
GW11 results become available before treating it as a deadline-time recommendation.
These are forecast inputs for selection, not evidence of realised squad performance.

## Next stage: squad selection

Notebook 5 is planned and has not been created. It will consume forecasts rather than fit
another model:

1. Validate gameweek totals against fixture forecasts and the dated roster/price snapshot
2. Select a legal squad, XI and captain under budget, position and club constraints
3. Verify the objective and legality independently, including small exhaustive examples
4. Compare XGBoost and recent-five selections under identical decision rules
5. Save recommendations before deadlines and evaluate subsequent outcomes

The initial objective covers starting-XI points and normal captaincy. Transfers, automatic
substitutions, chips and multiweek planning follow later. Fresh data updates features and
predictions; retraining and new research versions are separate processes. Any refit using
2025/26 outcomes requires later evaluation rather than reusing the earlier test claim.

## Copyright and use

Copyright © 2026 Zhangir Bayanov.

The original code is available under the [MIT License](LICENSE), permitting use, modification
and redistribution, including commercial use. Third-party datasets and assets retain their
own terms.
