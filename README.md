# FPL learning and product roadmap

## Direction

Build a personal football analysis tool while learning data modelling, SQL, statistics and boosting. The first useful product is an offline historical stats explorer. A Telegram interface, forecasts and squad optimization are successive extensions.

Use the downloaded dataset at `data/raw/epl-premier-league-dataset/`. Preserve those files as original inputs. Begin with one source and one season, then expand after the first product works.

Start with a brief structural audit, then alternate preparation and exploration. Initial EDA helps discover what needs cleaning; keys and validation make subsequent joins trustworthy. Complete enterprise data engineering is not a prerequisite for learning boosting.

## What exists now

Verified locally on 2026-11-08:

- `raw_data/fpl_api/fpl_merged_gw_2024_2025.csv`: 27,605 rows and 49 columns; `(element, fixture)` is unique. It contains 322 `AM` records for 20 managers. Excluding those leaves 27,283 rows for 784 players.
- `raw_data/fpl_api/fpl_fixtures_2024_2025.csv`: 380 fixtures. All gameweek-file fixture references exist here.
- `raw_data/fpl_api/fpl_teams_2024_2025.csv`: 20 teams.
- The player-match fields `element`, `fixture`, `GW`, `position`, `expected_goals`, `expected_assists`, `minutes` and `value` have no missing entries in the inspected season. This is not a complete quality audit.
- `fpl_cleaned_players_2024_2025.csv` lacks player IDs. It cannot be joined on an assumed `id` column.
- The existing `01_data_preparation.ipynb` is an earlier scaffold. Its saved execution only produced an inventory, and its paths/player-ID assumptions require revision before using the downloaded data.
- `data/pl_goals.sqlite` is an earlier scraping prototype database with no imported player statistics. It is separate from the proposed curated FPL database.

## Global milestones

| Milestone | Useful capability | What to learn | Completion condition |
|---|---|---|---|
| 1. Historical stats explorer | Filter and compare players, inspect team results, export tables | Python/pandas, table grain, SQL keys and joins, basic EDA | Reproducible one-season report with reconciled totals and documented definitions |
| 2. Personal Telegram stats interface | Request the same summaries from Telegram | Reusable functions, API handlers, configuration, simple deployment | A personal bot returns tested database queries and clearly states historical coverage |
| 3. Baseline point forecasts | Predict player points for later historical gameweeks | Lagged features, supervised targets, chronological validation | Beats or honestly fails to beat simple baselines on held-out periods |
| 4. Boosting experiments | Compare a simple boosted model, CatBoost and XGBoost | Additive models, gradients, regularization, categorical processing and tuning | Comparable temporal folds, feature experiments and a recorded model decision |
| 5. Squad selection | Recommend a legal squad/lineup from expected points | Binary optimization, constraints, captaincy and bench effects | Constraints pass and historical decisions are compared with baselines |
| 6. Current-data operation | Refresh inputs and return dated recommendations | Collection, validation, snapshots, scheduling and monitoring | The bot reports freshness and refuses unsupported/current claims from stale inputs |

A historical stats bot can be useful before a forecast exists. A bot is a presentation layer, not the forecasting model. Current recommendations require separately verified current-season inputs and scoring rules.

Clustering, team-result models, newcomer handling, external leagues and injury NLP are research extensions. Add each when it answers a concrete question; none blocks the first product. PostgreSQL is an optional later database-learning/deployment milestone.

## First product: offline FPL stats explorer

### User stories

1. Show a leaderboard for 2024/25, filtered by position and an adjustable minimum-minutes threshold.
2. Compare two players' minutes, points, goals, assists, xG, xA and per-90 production.
3. Show the selected player's points/minutes history and underlying attacking production over time.
4. Show a club's results and goals for/against from the fixture table.
5. Export the filtered tables and figures, with source, season and date-range labels.

Start with a notebook containing tables, filters as editable variables, and plots. A separate dashboard framework is optional after these queries work. Historical outputs must be labelled as retrospective, not live recommendations.

### Scope and definitions

- One season: 2024/25.
- One source family: FPL files.
- Include GK, DEF, MID and FWD; report and exclude `AM` manager rows from player analysis.
- Keep zero-minute records. Define appearances separately as rows with minutes greater than zero.
- Use `xGI = xG + xA` and `attacking_gap = xGI - (goals + assists)`. Positive gap means underperformance relative to expectation; negative means overperformance.
- Compute aggregated rates as `90 * sum(statistic) / sum(minutes)`. A zero denominator produces an unavailable rate. A configurable minimum of 450 minutes is a starting display filter, not an optimal statistical threshold.
- Goals and assists remain separate alongside the combined gap. Document the source's assist definitions before interpreting xA differences as skill.
- Compute team results and goals from fixtures once, not repeatedly from player rows.
- Defer price ranking initially. Later use a dated recorded price, rather than interpreting a season-end price as historical information.
- Full-season analysis is the first report. Any later date-range feature filters by actual kickoff timestamps, including rescheduled fixtures; displays must make the included period explicit.

### Tables

Proposed curated database: `data/processed/fpl.sqlite` (not created by this roadmap).

| Table | Row level | Key and important relationships |
|---|---|---|
| `teams` | Club in a season | `(season, team_id)` |
| `players` | Player in a season | `(season, player_id)`; derive identity from gameweek rows after checking name/position consistency |
| `fixtures` | Fixture in a season | `(season, fixture_id)`; home/away teams reference `teams` |
| `player_matches` | Player in a fixture | `(season, player_id, fixture_id)`; references players and fixtures |

Use `element` as the FPL season-scoped player identifier and `fixture` as the fixture reference. Derive a player's match-time club from the fixture's home/away teams and `was_home`; avoid a permanent club field on the player dimension. Names are attributes, not keys. Retain source filenames and document source IDs/column mappings.

Treat unexpected multiple names or positions for one player ID as a validation question rather than silently retaining an arbitrary record. Later multi-source or multi-season identity mappings are separate work.

### Proposed deliverables

- `notebooks/01_data_preparation.ipynb`: revise the existing notebook to audit and load the actual local files.
- `notebooks/02_stats_explorer.ipynb`: descriptive analysis, leaderboards, comparisons and plots.
- `src/prepare_fpl.py`: reusable parsing, validation and database loading once the notebook logic is understood.
- `src/stats_queries.py`: leaderboard, player-history, comparison and team-summary query functions.
- `sql/fpl_schema.sql`: table definitions, keys and indexes.
- `data/processed/fpl.sqlite`: curated local tables.
- `data/processed/player_summary_2024_2025.csv`: exported analysis table, generated from the curated data.
- `reports/figures/`: figures with explicit season and metric labels.
- `docs/data_dictionary.md`: definitions, units, keys, provenance and known limitations.

These are planned artifacts. Only the roadmap and updated README are delivered in the planning task.

### Acceptance checks

- Source counts reconcile: 27,605 original records = 27,283 player records + 322 manager records.
- Player-match primary keys are complete and unique; required player/fixture/team foreign keys resolve.
- Joins preserve the expected player-match row count.
- A fresh database rebuild works, and rerunning does not append duplicate observations.
- Sum of player-summary points equals sum of retained player-match points.
- A selected player's summary matches their underlying records; per-90 calculations use aggregate minutes.
- Team match counts and goals reconcile with the fixture table.
- Missing values, zeros, excluded records and source coverage are visible in the report.
- The stats explorer runs top-to-bottom and exports its tables without live web requests.

## Local plan: first six sessions

Use sessions rather than rigid dates; continue until each deliverable is understood.

| Session | Product work | Deliverable |
|---|---|---|
| 1 | Inspect the three FPL files, identify row levels/keys, examine rows and missingness | Short data dictionary and input-quality table |
| 2 | Normalize identifiers/types, handle manager rows, establish player/fixture/team relationships | Validated DataFrames and checked joins |
| 3 | Define SQLite tables, load one season and practise joins/aggregation | Curated database, repeatable load and five SQL queries |
| 4 | Explore points, minutes, xG/xA and the attacking gap; inspect distributions | Descriptive notebook and first player leaderboard |
| 5 | Add player comparison/history and team summaries | Working notebook stats explorer |
| 6 | Reconcile totals, document limitations, export outputs and rerun from scratch | MVP acceptance checklist and reproducible report |

Product development can pause for a concept or debugging exercise. A completed small deliverable matters more than finishing one numbered session per day.

## Daily theory and practice: boosting learning in parallel

Suggested 60–90 minute rhythm, adjustable to available time:

- 15–25 minutes: one theory concept and notes in your own words.
- 15–25 minutes: a small numerical or Python exercise directly testing it.
- 25–40 minutes: the next bounded project task.
- 5 minutes: record what changed, what you can explain and one unresolved question.

Do not delay boosting theory until the data pipeline is complete. Early exercises use tiny arrays/toy data so errors are easy to inspect. Later exercises use the validated FPL histories.

| Learning session | Theory | Hands-on exercise |
|---|---|---|
| 1 | Supervised regression, target, prediction and squared loss | Predict an array's mean and calculate its squared error |
| 2 | Regression trees and split selection | Evaluate a few candidate splits by their sum of squared errors |
| 3 | Additive models and residuals | Fit one shallow tree to squared-error residuals and update predictions |
| 4 | Learning rate and repeated boosting steps | Implement a short squared-error boosting loop; plot training loss |
| 5 | Gradients and loss functions | Derive the negative gradient for squared error; explain why other losses need different pseudo-residuals |
| 6 | Tree complexity and regularization | Vary depth/leaf size and learning rate; compare train/validation curves |
| 7 | Temporal validation and leakage | Define whole-gameweek folds; show how same-match goals leak a points target |
| 8 | Baselines versus boosted trees | Compare lagged-points/linear baselines with a first supervised FPL forecast |
| 9 | XGBoost and CatBoost | Compare their documented objectives, categorical handling and complexity controls on the same folds |
| 10 | Feature experiments, tuning and interpretation | Add/remove feature groups, use a bounded search, record accuracy and stability |

Sessions 7–10 may span multiple days and depend on the required validated data. Feature importance is not a manually assigned percentage weight. Tune and select features within development periods, then evaluate a locked model on an untouched later period.

For forecasting, the target is actual player points in the subsequent fixture/gameweek. Features use only information available at the prediction cutoff. An eventual six-week forecast needs horizon-specific evaluation rather than assuming future realized form is known.

## Telegram extension

After query functions work, a thin Telegram interface can provide illustrative commands such as:

- `/top MID` → historical leaderboard for the configured season and minutes filter.
- `/player <id>` → dated player summary.
- `/compare <id1> <id2>` → comparison.
- `/team <id>` → historical team results.
- `/coverage` → dataset season, covered period and collection/import provenance.

Start with deterministic database queries. The first bot does not require an LLM or forecasting model. It should use an allowed personal chat/user, keep its token outside versioned code and present source coverage with outputs. A laptop process only responds while running; continuous availability is a later hosting decision.

## Roles of the tools

- Terminal: run scripts and SQL commands, inspect logs, manage the Python environment.
- `.py` files: reusable data preparation and query functions; later bot handlers.
- `.ipynb` files: learning exercises, EDA, plots and documented model experiments.
- SQLite or a SQLite viewer: inspect relationships and query the curated data.
- Scrapy session: separate collection-learning exercises; the offline MVP does not depend on live scraping.

## References

- [Scikit-learn: gradient-boosted trees](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [XGBoost: introduction to boosted trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
- [CatBoost: categorical features](https://catboost.ai/docs/en/features/categorical-features)
- [Telegram: first bot tutorial](https://core.telegram.org/bots/tutorial)
- [Local dataset findings](web_scraping_dataset_findings.md)
