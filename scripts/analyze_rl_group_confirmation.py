#!/usr/bin/env python3
"""Analyze group-size confirmation across held-out tasks and training seeds."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


SEEDS = (31415, 27182, 16180)


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    payload["by_task"] = {
        episode["task_id"]: bool(episode["evaluation"]["passed"])
        for episode in payload["episodes"]
    }
    return payload


def mcnemar(left: dict[str, bool], right: dict[str, bool]) -> tuple[int, int, float]:
    assert left.keys() == right.keys()
    gains = sum(not left[key] and right[key] for key in left)
    losses = sum(left[key] and not right[key] for key in left)
    n = gains + losses
    if n == 0:
        return gains, losses, 1.0
    tail = sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2**n
    return gains, losses, min(1.0, 2 * tail)


def task_bootstrap(differences: list[float], seed: int = 20260827) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(differences)
    values = []
    for _ in range(50_000):
        values.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    values.sort()
    return values[1_250], values[48_749]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    eval_dir = args.run_dir / "evaluations"
    pretrained = load(eval_dir / "pretrained-confirm.json")
    rows = []
    task_differences: dict[str, list[int]] = {}
    for seed in SEEDS:
        baseline = load(eval_dir / f"baseline-seed-{seed}-confirm.json")
        group4 = load(eval_dir / f"group-4-seed-{seed}-confirm.json")
        gains, losses, p_value = mcnemar(baseline["by_task"], group4["by_task"])
        for task_id in baseline["by_task"]:
            task_differences.setdefault(task_id, []).append(
                int(group4["by_task"][task_id]) - int(baseline["by_task"][task_id])
            )
        rows.append({
            "seed": seed,
            "baseline": baseline["summary"]["tasks_solved"],
            "group4": group4["summary"]["tasks_solved"],
            "difference": group4["summary"]["tasks_solved"] - baseline["summary"]["tasks_solved"],
            "gains": gains,
            "losses": losses,
            "mcnemar_p": p_value,
        })

    mean_baseline = sum(row["baseline"] for row in rows) / len(rows)
    mean_group4 = sum(row["group4"] for row in rows) / len(rows)
    per_task_mean_differences = [sum(values) / len(values) for values in task_differences.values()]
    ci = task_bootstrap(per_task_mean_differences)
    all_positive = all(row["difference"] > 0 for row in rows)
    lines = [
        "# RL Group-Size Confirmation",
        "",
        "Untouched confirmation set: 40 executable tasks (8 per family). Each arm uses the same pretrained checkpoint, training data, learning rate, temperature, and 128-rollout budget. Only group size and the resulting number of updates differ.",
        "",
        f"Pretrained checkpoint: **{pretrained['summary']['tasks_solved']}/40**.",
        "",
        "| Training seed | Group 8 | Group 4 | Difference | Paired gains/losses | Exact McNemar p |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['baseline']}/40 | {row['group4']}/40 | {row['difference']:+d} | "
            f"{row['gains']}/{row['losses']} | {row['mcnemar_p']:.4f} |"
        )
    lines += [
        "",
        f"Mean across three training seeds: **group 8 {mean_baseline:.1f}/40; group 4 {mean_group4:.1f}/40; difference {mean_group4 - mean_baseline:+.1f} tasks**.",
        f"Task-bootstrap 95% CI for the seed-averaged pass-rate difference: **[{ci[0]:+.1%}, {ci[1]:+.1%}]**.",
        "",
        "Interpretation: " + (
            "group 4 won under every tested training seed, which supports a reproducible directional effect. "
            if all_positive else
            "group 4 did not win under every training seed, so the direction is not robust. "
        ) + "With only three training seeds and five task families, training-seed and family-level uncertainty remain substantial; task-level p-values must not be presented as proof of broad coding improvement.",
    ]
    args.output.write_text("\n".join(lines) + "\n")
    result = {
        "pretrained": pretrained["summary"],
        "seeds": rows,
        "mean_baseline": mean_baseline,
        "mean_group4": mean_group4,
        "mean_difference": mean_group4 - mean_baseline,
        "task_bootstrap_ci_pass_rate": ci,
        "group4_won_every_seed": all_positive,
    }
    (args.run_dir / "confirmation-analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
