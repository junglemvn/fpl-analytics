"""Persistent FPL holdings and a sparse rolling-horizon selection model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, vstack

from .squad_selection import POSITION_COUNTS, VARIANTS


FORMATION = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
POINT_TOLERANCE = 1e-6
INITIAL_BUDGET = 1000


def selling_price(bought, current):
    return current if current <= bought else bought + (current - bought) // 2


@dataclass(frozen=True)
class ManagerState:
    purchases: tuple[tuple[int, int], ...]
    bank: int
    free_transfers: int


@dataclass
class Decision:
    state: ManagerState
    lineup: pd.DataFrame
    expected_points: float
    current_points: float
    incoming: tuple[int, ...]
    outgoing: tuple[int, ...]
    bought: int
    sold: int
    hits: int
    stages: list[dict]


def legal_formation(positions):
    counts = positions.value_counts()
    return len(positions) == 11 and all(
        low <= counts.get(pos, 0) <= high for pos, (low, high) in FORMATION.items()
    )


def _decision_inputs(roster, forecast, source, state, week, hold_only):
    rows = roster.sort_values("fpl_player_id").set_index("fpl_player_id", verify_integrity=True)
    if rows.empty or not rows.position.isin(POSITION_COUNTS).all():
        raise ValueError("Forecast roster has invalid positions")
    if rows.fpl_team_id.isna().any() or not rows.price_tenths.gt(0).all() or not (
        rows.price_tenths.eq(np.floor(rows.price_tenths)).all()
    ) or rows.selection_allowed.isna().any():
        raise ValueError("Forecast roster has invalid club, price or eligibility")
    weekly = forecast.pivot(index="fpl_player_id", columns="GW", values=source).sort_index(axis=1)
    if set(weekly.index) != set(rows.index) or not forecast.coverage.isin(
        ["complete", "verified_blank"]
    ).all():
        raise ValueError("Forecast does not cover the dated roster")
    weekly = weekly.reindex(rows.index)
    if list(weekly.columns) != list(range(week, min(week + 3, 19) + 1)) or not (
        np.isfinite(weekly.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Incomplete rolling forecast horizon")

    ownership = {} if state is None else dict(state.purchases)
    if state is not None and (
        len(state.purchases) != 15 or len(ownership) != 15
        or not set(ownership).issubset(rows.index) or state.bank < 0
        or not 0 <= state.free_transfers <= 5
    ):
        raise ValueError("Invalid holdings, bank, transfer balance or missing owned player")
    rows["owned"] = rows.index.isin(ownership)
    if state is None:
        spend = rows.price_tenths.astype(int)
        budget, transfer_limit = INITIAL_BUDGET, 15
    else:
        spend = rows.price_tenths.astype(int).copy()
        for player_id, bought in ownership.items():
            spend.at[player_id] = selling_price(bought, spend.at[player_id])
        budget = state.bank + spend.loc[rows.owned].sum()
        transfer_limit = 0 if hold_only else min(1, state.free_transfers)

    return rows, weekly, ownership, spend.to_numpy(), int(budget), transfer_limit


def _problem(rows, weekly, spend, budget, transfer_limit, variant):
    n, weeks = len(rows), weekly.shape[1]
    width = n * (1 + 2 * weeks)
    squad = np.arange(n)
    xi = np.arange(n, width).reshape(weeks, 2, n)[:, 0]
    captain = xi + n
    vector_rows, vector_cols, values, lower, upper = [], [], [], [], []

    def constraint(indices, low=0, high=np.inf, value=1):
        vector_rows.extend([len(lower)] * len(indices))
        vector_cols.extend(indices)
        values.extend(np.broadcast_to(value, np.shape(indices)))
        lower.append(low)
        upper.append(high)

    position = rows.position.to_numpy()
    clubs = list(rows.groupby("fpl_team_id").indices.values())
    constraint(squad, 15, 15)
    constraint(squad, 0, budget, spend)
    constraint(squad[~rows.owned.to_numpy()], 0, transfer_limit)
    for pos, count in POSITION_COUNTS.items():
        constraint(squad[position == pos], count, count)
    for club in clubs:
        constraint(squad[club], 0, 3)
    for starters, captains in zip(xi, captain, strict=True):
        constraint(starters, 11, 11)
        constraint(captains, 1, 1)
        for pos, (minimum, maximum) in FORMATION.items():
            constraint(starters[position == pos], minimum, maximum)
        if VARIANTS[variant] is not None:
            for club in clubs:
                constraint(starters[club[position[club] == "DEF"]], 0, VARIANTS[variant])
        # Vectorised two-entry inequalities for x <= s and c <= x
        for before, after in ((squad, starters), (starters, captains)):
            indices = np.arange(len(lower), len(lower) + n)
            vector_rows.extend(np.repeat(indices, 2))
            vector_cols.extend(np.column_stack([before, after]).ravel())
            values.extend(np.tile([-1, 1], n))
            lower.extend([-np.inf] * n)
            upper.extend([0] * n)

    matrix = coo_matrix((values, (vector_rows, vector_cols)), shape=(len(lower), width)).tocsr()
    points = weekly.to_numpy(dtype=float).T
    objective, transfers, bench, cost = np.zeros((4, width))
    objective[xi] = -points
    objective[captain] = -points
    transfers[squad] = ~rows.owned.to_numpy()
    bench[squad] = -points.sum(axis=0)
    bench[xi] = points
    cost[squad] = spend
    enabled = np.ones(width)
    enabled[squad] = (rows.owned | rows.selection_allowed).astype(int)
    # Players already excluded cannot occupy lineup or captain variables
    enabled[xi] = enabled[squad]
    enabled[captain] = enabled[squad]
    return matrix, lower, upper, Bounds(0, enabled), (objective, transfers, bench, cost), xi, captain


def _optimise(problem, initial, hold_only, time_limit):
    matrix, low, high, bounds, vectors, _, _ = problem
    labels = ("points", "transfers", "bench", "spend")
    priorities = (0, 2, 3) if initial or hold_only else (0, 1, 2, 3)
    solution, stages = None, []
    for index in priorities:
        objective = vectors[index]
        result = milp(
            objective, bounds=bounds, integrality=np.ones(len(objective)),
            constraints=LinearConstraint(matrix, low, high),
            options={"time_limit": time_limit, "mip_rel_gap": 0.0},
        )
        record = {
            "stage": labels[index], "status": int(result.status), "message": result.message,
            "gap": getattr(result, "mip_gap", None), "bound": getattr(result, "mip_dual_bound", None),
            "objective": result.fun,
        }
        stages.append(record)
        if result.x is None:
            if solution is None:
                return None, stages
            break
        rounded = np.rint(result.x)
        values = matrix @ rounded
        if (not np.allclose(result.x, rounded, atol=1e-6, rtol=0)
                or not np.all(rounded <= bounds.ub)
                or not np.all(values >= np.asarray(low) - 1e-5)
                or not np.all(values <= np.asarray(high) + 1e-5)):
            raise ValueError("MILP returned a solution violating its discrete constraints")
        solution = rounded.astype(bool)
        if result.status != 0:
            break
        tolerance = POINT_TOLERANCE if index in (0, 2) else 0
        matrix = vstack([matrix, csr_matrix(objective.reshape(1, -1))], format="csr")
        low = [*low, -np.inf]
        high = [*high, float(objective @ solution) + tolerance]
    return solution, stages


def _lineups(rows, weekly, selection, starters, captains):
    players = rows.index[selection]
    frames = []
    for line, captain, week in zip(starters, captains, weekly.columns, strict=True):
        lineup = rows.loc[players].copy()
        lineup["GW"] = week
        lineup["expected_points"] = weekly.loc[players, week]
        lineup["is_starter"] = players.isin(rows.index[line])
        lineup["is_captain"] = players.isin(rows.index[captain])
        lineup["is_vice_captain"] = False
        lineup["bench_order"] = 0
        ranked = lineup.sort_values("expected_points", ascending=False, kind="stable")
        vice = ranked.loc[ranked.is_starter & ~ranked.is_captain].index[0]
        lineup.loc[vice, "is_vice_captain"] = True
        bench = ranked.loc[~ranked.is_starter & ranked.position.ne("GK")].index
        lineup.loc[bench, "bench_order"] = range(1, 4)
        frames.append(lineup.reset_index())
    return pd.concat(frames, ignore_index=True)


def _check_lineups(lineup, variant):
    for _, rows in lineup.groupby("GW"):
        xi = rows.loc[rows.is_starter]
        bench = rows.loc[~rows.is_starter]
        if (len(rows) != 15 or rows.fpl_player_id.nunique() != 15
                or rows.position.value_counts().to_dict() != POSITION_COUNTS
                or rows.groupby("fpl_team_id").size().max() > 3
                or not legal_formation(xi.position)
                or xi.is_captain.sum() != 1 or rows.is_captain.sum() != 1
                or xi.is_vice_captain.sum() != 1 or rows.is_vice_captain.sum() != 1
                or (xi.is_captain & xi.is_vice_captain).any()
                or set(bench.loc[bench.position.ne("GK"), "bench_order"]) != {1, 2, 3}):
            raise ValueError("Selected lineup failed independent FPL checks")
        if VARIANTS[variant] is not None and (
            xi.loc[xi.position.eq("DEF")].groupby("fpl_team_id").size() > VARIANTS[variant]
        ).any():
            raise ValueError("Starting defender concentration exceeds the selected cap")


def solve_origin(roster, forecast, source, week, variant, state=None, hold_only=False,
                 time_limit=30):
    """Return a legal origin decision and its new immutable manager state."""
    rows, weekly, ownership, spend, budget, limit = _decision_inputs(
        roster, forecast, source, state, week, hold_only,
    )
    problem = _problem(rows, weekly, spend, budget, limit, variant)
    solution, stages = _optimise(problem, state is None, hold_only, time_limit)
    if solution is None:
        return None, stages
    _, _, _, _, vectors, xi, captain = problem
    squad = solution[:len(rows)]
    owned_ids = set(ownership)
    picked = set(rows.index[squad])
    incoming, outgoing = tuple(sorted(picked - owned_ids)), tuple(sorted(owned_ids - picked))
    purchases = {player: ownership.get(player, int(rows.at[player, "price_tenths"]))
                 for player in sorted(picked)}
    bought = int(rows.loc[list(incoming), "price_tenths"].sum())
    sold = sum(selling_price(ownership[player], rows.at[player, "price_tenths"])
               for player in outgoing)
    bank = int((INITIAL_BUDGET if state is None else state.bank) + sold - bought)
    moves = 0 if state is None else len(incoming)
    free = 1 if state is None else min(5, state.free_transfers - moves + 1)
    if (bank < 0 or not rows.loc[list(incoming), "selection_allowed"].all()
            or (state is not None and (moves != len(outgoing) or moves > limit))):
        raise ValueError("Decision failed independent transfer, eligibility or funding checks")
    new_state = ManagerState(tuple(purchases.items()), bank, free)
    lineup = _lineups(rows, weekly, squad, solution[xi], solution[captain])
    _check_lineups(lineup, variant)
    lineup["purchase_price_tenths"] = lineup.fpl_player_id.map(purchases)
    lineup["points_with_captain"] = lineup.expected_points * (
        lineup.is_starter.astype(int) + lineup.is_captain.astype(int)
    )
    expected = float(lineup.points_with_captain.sum())
    if not np.isclose(expected, -(vectors[0] @ solution), atol=1e-7, rtol=0):
        raise ValueError("Horizon objective disagrees with selected lineups")
    current = float(lineup.loc[lineup.GW.eq(week), "points_with_captain"].sum())
    return Decision(
        new_state, lineup, expected, current, incoming, outgoing, bought, int(sold), 0, stages,
    ), stages
