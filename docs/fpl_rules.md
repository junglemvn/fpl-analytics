# FPL rules reference — 2026/27

Checked **10 November 2026**. Start here for project scoring or selection work. Recheck official sources before changing seasons or implementing a rule not covered below, then version the change explicitly.

Canonical source: [FPL rules](https://fantasy.premierleague.com/en/help/rules). The page was not text-readable in this run; the linked official Premier League explanations supplied the checks below. This is a working summary, not a replacement for the detailed rules.

## Player points

| Event | Points |
|---|---:|
| On-field appearance below 60 minutes | 1 |
| At least 60 minutes | 2 |
| Goal: GK / DEF / MID / FWD | 10 / 6 / 5 / 4 |
| Fantasy assist | 3 |
| Clean sheet: GK or DEF / MID | 4 / 1 |
| Each complete group of three goalkeeper saves | 1 |
| Penalty saved | 5 |
| Penalty missed | −2 |
| Bonus award | 1–3 |
| Each pair of goals conceded: GK or DEF | −1 |
| Yellow / red card | −1 / −3 |
| Own goal | −2 |
| Qualifying defensive contribution | 2 |

Official minutes exclude stoppage time. Clean-sheet rewards require at least 60 minutes and no goals conceded while on the pitch; a goalkeeper/defender sent off remains liable for later goals conceded. Fantasy assists follow FPL definitions. Keep awarded `bonus` separate from raw `bps`; `P_off` already contains bonus. Card deductions can produce zero or negative points despite participation.

Sources: [scoring](https://www.premierleague.com/en/news/2174909/), [detailed rules](https://www.premierleague.com/es/news/4661029).

## Defensive contribution

Use each row's FPL position and calculate per fixture:

- DEF: clearances + blocked shots + interceptions + tackles, threshold **10**
- MID/FWD: those actions + recoveries, threshold **12**
- GK: no defensive award
- Reaching the threshold gives **2 points once**, even at twice the threshold
- Sum fixture awards afterward for double gameweeks or season totals

In the local FotMob cache, defender blocks use `shot_blocks`, and tackles use `matchstats.headers.tackles` or `tackles`. Repeated metric displays are not additional actions. Provider agreement is imperfect, so historical reconstruction remains an estimate.

Source: [official defensive scoring](https://www.premierleague.com/en/news/4361991/whats-new-in-202526-fantasy-defensive-contributions). The category began in 2025/26 and remains in 2026/27: [season changes](https://www.premierleague.com/en/news/4679873).

## Squad and manager decisions

- Initial squad: 15 players within £100m, comprising 2 GK, 5 DEF, 5 MID and 3 FWD; at most three from one club
- XI: one goalkeeper, at least three defenders and at least one forward
- Captain normally doubles points; vice-captain covers a non-playing captain
- Automatic substitutions respect bench order, position and formation; a card can count as participation
- One free transfer arrives each gameweek, with up to five banked; extra transfers normally cost four manager points each
- Selling profit retains half a price rise, rounded down to £0.1m; price losses apply fully
- One chip per gameweek; Wildcard, Free Hit, Bench Boost and Triple Captain each have an allocation in both season halves

Manager deductions and captain/chip multipliers do not belong in the individual player-fixture target.

Sources: [squad](https://www.premierleague.com/en/news/2174419), [lineups and substitutions](https://www.premierleague.com/es/news/4661029), [transfers and selling prices](https://www.premierleague.com/en/news/2174907/1000), [2026/27 chip allocations](https://www.premierleague.com/en/news/4679873).

## Adjusted target used in Notebook 1

Version: **`2026_2027_DC_plus_verified_corrections_v1`**.

```text
P_adj = P_off − DC_already_included + DC_target + verified_point_correction
```

- **2022/23–2024/25:** add defensive awards from the prepared historical counts; the old defensive component is zero
- **2025/26–2026/27:** preserve official FPL defensive scoring, so the net defensive adjustment is zero under these unchanged thresholds
- **Zero-minute rows with unobserved actions:** use zero defensive award without filling action measurements with zero; retain any known qualifying count
- **Positive-minute historical rows with unknown counts:** keep `P_adj` missing and mark incomplete
- **Ferguson, 2024/25, FPL player 123, fixture 239:** minutes are corrected to 17; add one verified appearance point only while imported `P_off` is zero and the reviewed scoring events remain zero
- Preserve `P_off`, FPL keys and all other observations; keep calculation components in a separate audit

The Ferguson evidence is in local FotMob member `cache_matches/4506621.json`: `minutes_played=17`, `fantasy_points=1`, lineup `fantasyScore="1"` and substitution at minute 73. The FPL export records zero points and no goal, assist, card, own goal, missed penalty or bonus for that row. If the source already reports one point, no extra appearance point is added.

This target retains historical bonus, assist definitions and historical FPL positions. The current BPS weights differ from older seasons; recomputing every component would require a separate event-level reconstruction and fixture-wide bonus ranking. See [2026/27 BPS changes](https://www.premierleague.com/en/news/4679946/). The present target harmonizes defensive rewards and the reviewed data correction only.

## Prepared point columns

Historical tables retain `P_off` and `P_adj` so the original and adjusted awards remain distinguishable. The joined self-collected current-season table keeps official `P_off` directly; preparation does not recalculate it or create another defensive adjustment. A later modelling view can name its selected target `points`, using historical `P_adj` and current `P_off`, without renaming the stored source measurements.
