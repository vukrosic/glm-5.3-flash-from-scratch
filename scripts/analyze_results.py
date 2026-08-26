#!/usr/bin/env python3
"""Turn immutable experiment receipts into summary statistics and charts."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


TARGET_FAMILIES = ("increment", "double", "even")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def exact_two_sided_binomial(gained: int, lost: int) -> float:
    total = gained + lost
    if total == 0:
        return 1.0
    tail = min(gained, lost)
    probability = sum(math.comb(total, index) for index in range(tail + 1)) / 2**total
    return min(1.0, 2.0 * probability)


def targeted_sample_summary(payload: dict) -> dict:
    rows = [payload["by_family"][family] for family in TARGET_FAMILIES]
    exact = sum(row["exact_rollouts"] for row in rows)
    rollouts = sum(row["rollouts"] for row in rows)
    solved = sum(row["tasks_solved_at_k"] for row in rows)
    tasks = sum(row["tasks"] for row in rows)
    return {
        "exact_rollouts": exact,
        "rollouts": rollouts,
        "exact_rollout_rate": exact / rollouts,
        "tasks_solved_at_8": solved,
        "tasks": tasks,
        "pass_at_8": solved / tasks,
    }


def targeted_greedy_summary(payload: dict) -> dict:
    episodes = [row for row in payload["episodes"] if row["family"] in TARGET_FAMILIES]
    solved = sum(row["evaluation"]["passed"] for row in episodes)
    tests = sum(row["evaluation"]["tests_passed"] for row in episodes)
    tests_total = sum(row["evaluation"]["tests_total"] for row in episodes)
    return {
        "tasks_solved": solved,
        "tasks": len(episodes),
        "pass_at_1": solved / len(episodes),
        "tests_passed": tests,
        "tests_total": tests_total,
    }


def paired_task_differences(before: dict, after: dict) -> list[float]:
    before_by_id = {row["task_id"]: row for row in before["episodes"]}
    differences = []
    for after_episode in after["episodes"]:
        if after_episode["family"] not in TARGET_FAMILIES:
            continue
        before_episode = before_by_id[after_episode["task_id"]]
        before_exact = sum(row["evaluation"]["passed"] for row in before_episode["samples"])
        after_exact = sum(row["evaluation"]["passed"] for row in after_episode["samples"])
        differences.append((after_exact - before_exact) / len(after_episode["samples"]))
    return differences


def bootstrap_interval(values: list[float], *, samples: int = 50_000, seed: int = 20260826) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(values)
    means = sorted(sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def add_labels(axis, bars, suffix="%"):
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, default=Path("artifacts/receipts/runs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    runs = args.receipts
    args.output.mkdir(parents=True, exist_ok=True)
    charts = args.output / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    pretrain_receipt = load(runs / "glm53-coding-pretrain-001/training-receipt.json")
    rl_receipt = load(runs / "glm53-executable-rloo-diverse-001/training-receipt.json")
    sampled_before = load(runs / "confirm-pass8-pretrain-0100.json")
    sampled_after = load(runs / "confirm-pass8-rl-0096.json")
    greedy_before = load(runs / "confirm-greedy-pretrain-0100.json")
    greedy_after = load(runs / "confirm-greedy-rl-0096.json")

    target_sample_before = targeted_sample_summary(sampled_before)
    target_sample_after = targeted_sample_summary(sampled_after)
    target_greedy_before = targeted_greedy_summary(greedy_before)
    target_greedy_after = targeted_greedy_summary(greedy_after)

    before_greedy = {row["task_id"]: row["evaluation"]["passed"] for row in greedy_before["episodes"]}
    after_greedy = {row["task_id"]: row["evaluation"]["passed"] for row in greedy_after["episodes"]}
    gained = sum((not before_greedy[key]) and after_greedy[key] for key in before_greedy)
    lost = sum(before_greedy[key] and (not after_greedy[key]) for key in before_greedy)
    differences = paired_task_differences(sampled_before, sampled_after)
    ci_low, ci_high = bootstrap_interval(differences)

    summary = {
        "experiment": "GLM-5.3-Flash From Scratch: executable-reward coding RL",
        "architecture": {
            "total_parameters": pretrain_receipt["parameter_counts"]["total"],
            "active_parameters_per_token_estimate": pretrain_receipt["parameter_counts"]["active_per_token_estimate"],
            "layers": pretrain_receipt["config"]["layers"],
            "attention_pattern": "3 linear : 1 gathered sparse",
            "experts": pretrain_receipt["config"]["experts"],
            "top_k": pretrain_receipt["config"]["top_k"],
            "residual_streams": pretrain_receipt["config"]["streams"],
        },
        "pretraining": {
            "selected_checkpoint_step": 100,
            "tokens_at_selected_checkpoint": 345995,
            "full_run_tokens": pretrain_receipt["tokens_seen"],
            "full_run_seconds": pretrain_receipt["elapsed_seconds"],
            "peak_vram_gib": pretrain_receipt["peak_vram_gib"],
        },
        "reinforcement_learning": {
            "algorithm": rl_receipt["algorithm"],
            "groups": rl_receipt["groups"],
            "group_size": rl_receipt["group_size"],
            "rollouts": rl_receipt["groups"] * rl_receipt["group_size"],
            "updates": sum(row["updated"] for row in rl_receipt["groups_detail"]),
            "exact_training_rollouts": sum(row["exact_rollouts"] for row in rl_receipt["groups_detail"]),
            "families": list(TARGET_FAMILIES),
            "seconds": rl_receipt["elapsed_seconds"],
            "peak_vram_gib": rl_receipt["peak_vram_gib"],
            "selected_checkpoint_group": 96,
        },
        "untouched_confirmation": {
            "targeted_greedy_before": target_greedy_before,
            "targeted_greedy_after": target_greedy_after,
            "targeted_sampled_before": target_sample_before,
            "targeted_sampled_after": target_sample_after,
            "all_families_before": sampled_before["summary"],
            "all_families_after": sampled_after["summary"],
            "by_family_before": sampled_before["by_family"],
            "by_family_after": sampled_after["by_family"],
        },
        "statistics": {
            "greedy_paired_gains": gained,
            "greedy_paired_losses": lost,
            "greedy_exact_mcnemar_p": exact_two_sided_binomial(gained, lost),
            "sampled_target_absolute_gain": target_sample_after["exact_rollout_rate"] - target_sample_before["exact_rollout_rate"],
            "sampled_target_task_bootstrap_95_percent": [ci_low, ci_high],
        },
        "limitations": [
            "The confirmation tasks use unseen function identities but the same eight operation families and prompt templates.",
            "RL clearly learned increment and double, but not even.",
            "Non-target sampled accuracy regressed for square and list_sum, showing interference.",
            "This is a scaled educational implementation, not a reproduction of the 320B multimodal model.",
        ],
    }
    (args.output / "results-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    navy, blue, green, orange, gray = "#0f172a", "#2563eb", "#059669", "#ea580c", "#94a3b8"

    figure, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=160)
    figure.patch.set_facecolor("#f8fafc")
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)

    pretrain_steps = [0, 100, 200, 400]
    pretrain_scores = [
        100 * load(runs / "pretrain-pilot-001/dev-random.json")["summary"]["pass_at_1"],
        100 * load(runs / "glm53-coding-pretrain-001/dev-0100.json")["summary"]["pass_at_1"],
        100 * load(runs / "glm53-coding-pretrain-001/dev-0200.json")["summary"]["pass_at_1"],
        100 * load(runs / "glm53-coding-pretrain-001/dev-0400.json")["summary"]["pass_at_1"],
    ]
    axes[0].plot(pretrain_steps, pretrain_scores, color=blue, marker="o", linewidth=3, markersize=8)
    axes[0].set(title="1. Pretraining learns code", xlabel="Pretraining steps", ylabel="Dev tasks solved (greedy)", ylim=(-5, 110))
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    for x_value, y_value in zip(pretrain_steps, pretrain_scores):
        axes[0].annotate(f"{y_value:.0f}%", (x_value, y_value), xytext=(0, 10), textcoords="offset points", ha="center", fontweight="bold")

    rl_groups = [0, 24, 48, 72, 96]
    rl_scores = []
    for group in rl_groups:
        payload = load(runs / f"glm53-executable-rloo-diverse-001/dev-pass8-{group:04d}.json")
        rl_scores.append(100 * payload["summary"]["exact_rollout_rate"])
    axes[1].plot(rl_groups, rl_scores, color=green, marker="o", linewidth=3, markersize=8)
    axes[1].set(title="2. RL improves the policy", xlabel="RLOO prompt groups", ylabel="Dev exact rollout rate", ylim=(0, 55))
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    for x_value, y_value in zip(rl_groups, rl_scores):
        axes[1].annotate(f"{y_value:.1f}%", (x_value, y_value), xytext=(0, 10), textcoords="offset points", ha="center", fontweight="bold")

    labels = ["Greedy\ntrained skills", "Pass@8\ntrained skills", "Pass@8\nall skills"]
    before_values = [
        100 * target_greedy_before["pass_at_1"],
        100 * target_sample_before["pass_at_8"],
        100 * sampled_before["summary"]["pass_at_k"],
    ]
    after_values = [
        100 * target_greedy_after["pass_at_1"],
        100 * target_sample_after["pass_at_8"],
        100 * sampled_after["summary"]["pass_at_k"],
    ]
    positions = list(range(len(labels)))
    width = 0.36
    before_bars = axes[2].bar([value - width / 2 for value in positions], before_values, width, label="Before RL", color=gray)
    after_bars = axes[2].bar([value + width / 2 for value in positions], after_values, width, label="After RL", color=orange)
    axes[2].set(title="3. Untouched confirmation", ylabel="Tasks solved", ylim=(0, 100), xticks=positions, xticklabels=labels)
    axes[2].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axes[2].legend(frameon=False, loc="upper left")
    add_labels(axes[2], before_bars)
    add_labels(axes[2], after_bars)

    figure.suptitle("GLM-5.3-Flash From Scratch: a tiny executable-reward experiment", fontsize=20, fontweight="bold", color=navy, y=0.97)
    figure.text(0.5, 0.035, "25.7M parameters • synthetic coding only • RTX 3080 Ti • exact unit-test verifier", ha="center", fontsize=11, color="#475569")
    figure.tight_layout(rect=(0.02, 0.07, 0.98, 0.92))
    figure.savefig(charts / "main-results.png", bbox_inches="tight")
    plt.close(figure)

    families = ["increment", "double", "even", "square", "list_sum", "absolute", "nonnegative", "reverse"]
    before_rates = [100 * sampled_before["by_family"][family]["exact_rollouts"] / sampled_before["by_family"][family]["rollouts"] for family in families]
    after_rates = [100 * sampled_after["by_family"][family]["exact_rollouts"] / sampled_after["by_family"][family]["rollouts"] for family in families]
    figure, axis = plt.subplots(figsize=(14, 8), dpi=160)
    figure.patch.set_facecolor("#f8fafc")
    axis.set_facecolor("white")
    positions = list(range(len(families)))
    before_bars = axis.bar([value - width / 2 for value in positions], before_rates, width, label="Before RL", color=gray)
    after_bars = axis.bar([value + width / 2 for value in positions], after_rates, width, label="After RL", color=blue)
    labels = [f"{family}*" if family in TARGET_FAMILIES else family for family in families]
    axis.set(title="Where RL helped—and where it regressed", ylabel="Exact sampled completions", xticks=positions, xticklabels=labels, ylim=(0, 110))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)
    axis.legend(frameon=False, loc="upper right")
    add_labels(axis, before_bars)
    add_labels(axis, after_bars)
    figure.text(0.5, 0.025, "* trained with executable reward; 64 samples per family on unseen function identities", ha="center", color="#475569")
    figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.96))
    figure.savefig(charts / "family-results.png", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=160)
    figure.patch.set_facecolor("#f8fafc")
    prompt = "# Return two times x.\ndef double_ywkaoot(x):"
    cards = [
        (axes[0], "BEFORE RL", prompt + "\n    return x * * 0 0", "FAIL • 0/3 tests", "#fee2e2", "#b91c1c"),
        (axes[1], "AFTER RL", prompt + "\n    return x * 2", "PASS • 3/3 tests", "#dcfce7", "#047857"),
    ]
    for axis, title, code, result, background, accent in cards:
        axis.set_facecolor(background)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(accent)
            spine.set_linewidth(2)
        axis.text(0.05, 0.90, title, transform=axis.transAxes, fontsize=18, fontweight="bold", color=accent)
        axis.text(0.05, 0.68, code, transform=axis.transAxes, fontsize=15, family="monospace", va="top", color=navy, linespacing=1.6)
        axis.text(0.05, 0.14, result, transform=axis.transAxes, fontsize=18, fontweight="bold", color=accent)
    figure.suptitle("Same unseen prompt. Same verifier. Reward changed the output.", fontsize=22, fontweight="bold", color=navy)
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.91))
    figure.savefig(charts / "before-after-generation.png", bbox_inches="tight")
    plt.close(figure)

    print(json.dumps(summary["untouched_confirmation"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
