"""Export selected replay results without model weights, settings or raw inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


LABELS = {
    "recent_five": "Recent-five",
    "xgboost": "Frozen XGBoost",
    "xgboost_preseason_v1": "Preseason → XGBoost",
    "max_expected": "Standard",
    "defender_diversified": "Defender cap",
}
POLICY = ["source", "variant"]


def _records(frame):
    return json.loads(frame.replace(LABELS).round(4).to_json(orient="records"))


def build_summary(review):
    """Read a verified local review and return only the public reporting fields."""
    review = Path(review)
    receipt_path = review / "review_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["schema"] != "fpl_manager_replay_review_v1" or receipt["scored_GWs"] != list(
        range(1, 11)
    ):
        raise ValueError("Expected the frozen GW1–10 manager review")
    for name, digest in receipt["output_sha256"].items():
        if sha256((review / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Review output changed: {name}")

    def table(name):
        return pd.read_csv(review / f"{name}.csv")

    totals = table("manager_totals")
    weekly = table("weekly_comparison")
    benchmark = table("benchmark_series")
    accuracy = table("forecast_review").query("slice == 'overall'")
    opening = table("opening_XI")
    actions = table("gw11_actions")
    lineups = table("gw11_lineups")
    selected = actions.loc[
        actions.source.eq("xgboost_preseason_v1") & actions.variant.eq("max_expected")
    ].iloc[0]
    lineup = lineups.loc[
        lineups.source.eq(selected.source) & lineups.variant.eq(selected.variant)
    ].copy()
    lineup["position_order"] = lineup.position.map({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3})
    lineup = lineup.sort_values(
        ["is_starter", "position_order", "bench_order"], ascending=[False, True, True],
    )
    if (len(totals) != 6 or len(weekly) != 60 or len(lineup) != 15
            or not weekly.net_points.eq(weekly.gross_points - weekly.transfer_hits).all()
            or weekly.groupby(POLICY).GW.agg(list).map(sorted).tolist() != [
                list(range(1, 11)) for _ in range(6)
            ]):
        raise ValueError("Incomplete or inconsistent public replay tables")
    summed = weekly.groupby(POLICY).net_points.sum().sort_index()
    pd.testing.assert_series_equal(
        summed, totals.set_index(POLICY).net_points.sort_index(), check_names=False,
    )

    return {
        "schema_version": "fpl_public_squad_results_v1",
        "season": "2026/27",
        "evaluation": "Retrospective GW1–10 replay; GW11 recommendation remains unscored",
        "source_review_sha256": sha256(receipt_path.read_bytes()).hexdigest(),
        "protocol": [
            {"setting": "Initial squad", "value": "£100m; 15 players; standard FPL composition"},
            {"setting": "Decision horizon", "value": "Four gameweeks, capped at GW19"},
            {"setting": "Transfers", "value": "Hold or at most one free move per deadline"},
            {"setting": "Scoring", "value": "Official points, legal autosubs and captain/vice fallback"},
            {"setting": "Availability boundary", "value": "Dated captures; results after kickoff + 24h"},
            {"setting": "Policies", "value": "Standard; maximum two starting DEF per club"},
            {"setting": "Chips and paid transfers", "value": "Unused in this replay"},
        ],
        "manager_totals": _records(totals[[
            *POLICY, "net_points", "moves", "captain_points", "auto_sub_points", "vs_average",
        ]]),
        "weekly": _records(weekly[[
            *POLICY, "GW", "gross_points", "transfer_hits", "net_points", "cumulative_net",
        ]]),
        "benchmark": _records(benchmark[["GW", "manager_average", "manager_maximum"]]),
        "forecast_accuracy": _records(accuracy[["source", "records", "MAE", "RMSE", "bias"]]),
        "opening": _records(opening[[*POLICY, "XI_with_minutes", "XI_60_plus", "net_points"]]),
        "gw11": {
            "source": LABELS[selected.source], "variant": LABELS[selected.variant],
            "as_of_utc": selected.as_of_utc, "deadline_utc": selected.deadline_utc,
            "formation": selected.formation, "incoming": selected.incoming,
            "outgoing": selected.outgoing, "captain": selected.captain, "vice": selected.vice,
            "bank_m": round(float(selected.bank_tenths) / 10, 1),
            "expected_points": round(float(selected.expected_GW), 2),
            "expected_horizon_points": round(float(selected.expected_horizon), 2),
            "lineup": _records(lineup[["web_name", "position", "role", "expected_points"]]),
        },
        "limitations": [
            "Ten retrospective gameweeks are development evidence, not a prospective test.",
            "The field averages include chip users; aggregate transfer-hit treatment is unverified.",
            "Opening squads differ, so policy results do not isolate the forecasting model's effect.",
            "The standard lineups already met the defender cap; tied solutions can follow different paths.",
            "Later-collected statistics may contain revisions; the 24-hour delay approximates publication time.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/public/squad_results.json"))
    args = parser.parse_args()
    summary = build_summary(args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
