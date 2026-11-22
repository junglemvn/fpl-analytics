# FPL performance forecasting and squad selection

An end-to-end research workflow for Fantasy Premier League: collect and validate football
data, explore player and team performance, forecast player-fixture points, and select legal
squads while managing transfers and budget.

Preparation, exploratory analysis, historical model evaluation and a retrospective GW1–10
manager replay are complete. The next stage is a database-backed application with automated
updates and a Telegram interface.

## Notebooks

| Notebook | Content |
|---|---|
| [01 — Data preparation](notebooks/01_data_preparation.ipynb) | Schemas, missing values, duplicates, joins and scoring preparation |
| [02 — Exploratory analysis](notebooks/02_eda.ipynb) | Player performance, positional patterns, team strength and ranking stability |
| [03 — Splits and benchmarks](notebooks/03_data_split_and_benchmarks.ipynb) | Chronological partitions, information cutoffs and recent-five benchmarks |
| [04 — Model results](notebooks/04_model_results.ipynb) | Forecast evaluation and subgroup diagnostics |
| [05 — Squad results](notebooks/05_squad_results.ipynb) | Persistent squads, rolling decisions, retrospective scores and a GW11 recommendation |

The results notebooks run from the included JSON summaries. Preparation and EDA require
local source data. Model training, tuned settings and fitted weights are not distributed;
the published code covers collection, preparation, benchmarks and generic optimisation.

## Data and forecasting

The prepared history contains 113,260 player-fixture records from 2022/23–2025/26, plus
6,562 collected records through GW10 of 2026/27. FPL supplies player, team, fixture and
availability data. FotMob provides additional historical defensive statistics for the
reviewed scoring adjustments.

Records retain season-scoped FPL player, team and fixture IDs. Historical official points
remain in `P_off`; reviewed adjustments are stored separately in `P_adj`. Current-season
official points are preserved. See [the scoring policy](docs/fpl_rules.md).

Forecasts use eligible earlier player histories, team/opponent context, venue and dated
availability. Results enter history after a conservative 24-hour delay. Archived status
and playing chance are matched to the prediction boundary and applicable round. Historical
deadlines use a first-kickoff-minus-24-hour proxy where official deadlines are unavailable.

The final forecasters were fitted on 2022/23–2024/25 and evaluated on the original 2025/26
reserved partition with their parameters fixed:

| Forecast | Player-fixtures | RMSE | MAE |
|---|---:|---:|---:|
| Frozen XGBoost | 29,747 | 1.9014 | 0.9604 |
| Recent-five benchmark | 29,747 | 2.1182 | 1.0581 |

The test season appeared in descriptive EDA but was excluded from predictive fitting and
tuning. Later-collected statistics, approximate publication timing and historical scoring
differences limit exact reconstruction of past information.

## Rolling squad replay

Six virtual managers combine three forecast sources with standard selection rules or a
maximum of two starting defenders per club. Each starts with £100m and carries purchase
prices, bank and holdings forward. A four-GW optimiser compares holding with one free
transfer per deadline, then chooses legal lineups and captains. Realised scoring includes
automatic substitutions and captain-to-vice fallback. No chips or paid transfers were used.

| GW1–10 net points | Standard | Defender cap |
|---|---:|---:|
| Recent-five | 299 | 279 |
| Frozen XGBoost | 307 | 367 |
| Preseason start → XGBoost | 593 | 593 |

The preseason-start policy uses verified prior-season performance for its opening forecast,
then frozen XGBoost from GW2 onward. The sum of the archived weekly manager averages is
563. That population includes chip users, and aggregate transfer-hit treatment is unverified.
Different initial squads mean these results compare complete policies rather than isolate
XGBoost's contribution. Standard lineups already met the defender cap; tied selections can
produce different continuing paths. Ten retrospective weeks are development evidence.

The saved GW11 recommendation remains unscored and retrospective. It is a transfer-feasible
continuation, not a fresh draft or a claim of a whole-season optimum.

## Repository structure

```text
notebooks/                 Preparation, EDA, benchmarks and public results
src/                       Collection, analysis, benchmarks and generic squad optimisation
reports/public/
  model_results.json       Selected forecast evaluation
  squad_results.json       Selected replay results and GW11 recommendation
docs/
  fpl_rules.md                 Scoring definitions and target policy
  fpl_collection_learning.md   Collection commands and table definitions
logos/                     Club emblems used by the analysis
.vscode/extensions.json    Optional editor recommendations
requirements.txt          Python dependencies
LICENSE                   MIT License
```

Raw API responses and prepared tables are stored locally:

```text
data/raw/fpl_api/<season>/<batch_utc>/
  bootstrap-static.json
  fixtures.json
  element-summary/<player_id>.json
  manifest.json
data/processed/fpl_self_collected/<season>/<batch>/
data/processed/fpl_tables.pkl
data/processed/fpl_eda_tables.pkl
```

The external [source dataset](https://huggingface.co/datasets/Toibinguz/epl-premier-league-dataset)
is referred to as `/epl-premier-league-dataset` and configured through `ARCHIVE_DIR` in
Notebook 1. Datasets and fitted models are excluded from the repository.

## Run the public reports

Use Python 3.11 and run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open `04_model_results.ipynb` or `05_squad_results.ipynb`. Both read the supplied public
summaries without requiring the local datasets or fitted model. The selected tables and
figures reproduce reporting outputs; they do not reproduce model training.

Collection and preparation instructions are in [the collection guide](docs/fpl_collection_learning.md).
`src/fpl_manager.py` contains reusable holdings, selling-price and rolling optimisation
logic. `src/public_squad_report.py` exports selected fields from verified local review
outputs into the public replay summary.

## Next stage

1. Connect collection, validation, forecasting and selection through one repeatable command
2. Introduce PostgreSQL in Supabase for dated football records and per-user manager state
3. Package the Python application with Docker and schedule data/prediction updates
4. Add Telegram squad and transfer commands, followed by grounded LLM explanations

Appearance/minutes forecasting remains a separate research experiment. Database integration,
deployment and the Telegram application are planned.

## License

Copyright © 2026 Zhangir Bayanov.

Original project code is available under the [MIT License](LICENSE). Third-party datasets
and assets retain their own terms. This is an independent project, unaffiliated with
Fantasy Premier League or the Premier League.
