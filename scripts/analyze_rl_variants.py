#!/usr/bin/env python3
"""Analyze paired RL-variant screening results without external dependencies."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def mcnemar_exact(left: list[bool], right: list[bool]) -> tuple[int, int, float]:
    gains = sum((not a) and b for a, b in zip(left, right, strict=True))
    losses = sum(a and (not b) for a, b in zip(left, right, strict=True))
    discordant = gains + losses
    if discordant == 0:
        return gains, losses, 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / 2**discordant
    return gains, losses, min(1.0, 2 * tail)


def paired_bootstrap(left: list[bool], right: list[bool], seed: int = 20260827) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(left)
    values = []
    for _ in range(20_000):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(sum(float(right[i]) - float(left[i]) for i in indices) / n)
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]


def holm(rows: list[dict]) -> None:
    ordered = sorted(rows, key=lambda row: row["p_raw"])
    running = 0.0
    count = len(rows)
    for rank, row in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * row["p_raw"])
        running = max(running, adjusted)
        row["p_holm"] = running


def load_eval(path: Path) -> dict:
    payload = json.loads(path.read_text())
    payload["solved_vector"] = [episode["evaluation"]["passed"] for episode in payload["episodes"]]
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eval_dir = args.screen_dir / "evaluations"
    names = ["baseline", "partial", "invalid-zero", "invalid-hard", "temp-low", "temp-high", "group-4", "group-16"]
    pretrained = load_eval(eval_dir / "pretrained-dev.json")
    evaluations = {name: load_eval(eval_dir / f"{name}-dev.json") for name in names}
    baseline = evaluations["baseline"]

    comparisons = []
    for name in names[1:]:
        gains, losses, p_raw = mcnemar_exact(baseline["solved_vector"], evaluations[name]["solved_vector"])
        low, high = paired_bootstrap(baseline["solved_vector"], evaluations[name]["solved_vector"])
        comparisons.append({"name": name, "gains": gains, "losses": losses, "p_raw": p_raw, "ci": [low, high]})
    holm(comparisons)
    comparison_by_name = {row["name"]: row for row in comparisons}

    rows = []
    for name in names:
        run = json.loads((args.screen_dir / name / "training-receipt.json").read_text())
        summary = evaluations[name]["summary"]
        groups = run["groups_detail"]
        rollouts = run["groups"] * run["group_size"]
        rows.append({
            "name": name,
            "solved": summary["tasks_solved"],
            "total": summary["tasks_total"],
            "test_accuracy": summary["test_accuracy"],
            "valid": summary["valid_programs"],
            "train_exact": sum(row["exact_rollouts"] for row in groups),
            "train_valid": sum(row["valid_rollouts"] for row in groups),
            "rollouts": rollouts,
            "updates": sum(row["updated"] for row in groups),
            "seconds": run["elapsed_seconds"],
        })

    winner = max(rows, key=lambda row: (row["solved"], row["test_accuracy"], row["valid"], -row["seconds"]))
    lines = [
        "# RL Variant Screening",
        "",
        "Primary metric: greedy pass@1 on 20 fixed dev tasks across five families. All RL arms used 128 training rollouts from the same pretrained checkpoint.",
        "",
        f"Pretrained checkpoint: **{pretrained['summary']['tasks_solved']}/{pretrained['summary']['tasks_total']}** tasks solved.",
        "",
        "| Arm | Dev solved | Test accuracy | Valid programs | Train exact | Updates | Seconds | vs baseline gains/losses | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        comparison = comparison_by_name.get(row["name"])
        paired = "reference" if comparison is None else f"{comparison['gains']}/{comparison['losses']}"
        p_value = "—" if comparison is None else f"{comparison['p_holm']:.4f}"
        lines.append(
            f"| {row['name']} | {row['solved']}/{row['total']} | {row['test_accuracy']:.1%} | "
            f"{row['valid']}/{row['total']} | {row['train_exact']}/{row['rollouts']} | {row['updates']} | "
            f"{row['seconds']:.1f} | {paired} | {p_value} |"
        )
    lines += [
        "",
        f"Screening winner: **{winner['name']}**. This is a dev-selected result, not confirmation evidence.",
        "",
        "Statistical note: raw paired McNemar tests compare each arm with baseline on the same tasks; Holm correction covers seven comparisons. The bootstrap interval is retained in `screening-analysis.json`. Training-seed uncertainty is not measured by this screen and must be checked when repeating the winner.",
    ]
    args.output.write_text("\n".join(lines) + "\n")
    (args.screen_dir / "screening-analysis.json").write_text(json.dumps({
        "pretrained": pretrained["summary"],
        "arms": rows,
        "comparisons_vs_baseline": comparisons,
        "winner": winner["name"],
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"winner": winner["name"], "report": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
