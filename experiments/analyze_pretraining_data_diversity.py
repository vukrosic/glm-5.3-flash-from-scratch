#!/usr/bin/env python3
"""Aggregate the matched pretraining diversity runs into a report and chart."""
from __future__ import annotations

import itertools
import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    50: ROOT / "artifacts/experiments/pretraining-data-diversity-050steps-10seed/results.json",
    100: ROOT / "artifacts/experiments/pretraining-data-diversity-002/results.json",
    200: ROOT / "artifacts/experiments/pretraining-data-diversity-200steps-10seed/results.json",
}
OUTPUT = ROOT / "artifacts/experiments/pretraining-data-diversity-summary"


def bootstrap_ci(values: list[float], *, seed: int = 42, draws: int = 50_000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(values) for _ in values) for _ in range(draws))
    return means[int(draws * 0.025) - 1], means[int(draws * 0.975) - 1]


def paired_permutation_p(differences: list[float]) -> float:
    observed = abs(statistics.mean(differences))
    null = [
        abs(sum(sign * value for sign, value in zip(signs, differences)) / len(differences))
        for signs in itertools.product((-1, 1), repeat=len(differences))
    ]
    return sum(value >= observed - 1e-15 for value in null) / len(null)


def summarize() -> dict:
    checkpoints = []
    for steps, path in INPUTS.items():
        receipt = json.loads(path.read_text())
        by_diversity = {8: {}, 88: {}}
        for run in receipt["runs"]:
            diversity = run["diversity"]
            if diversity in by_diversity:
                by_diversity[diversity][run["seed"]] = run["after"]["target_byte_accuracy"]
        seeds = sorted(set(by_diversity[8]) & set(by_diversity[88]))
        repeated = [by_diversity[8][seed] for seed in seeds]
        diverse = [by_diversity[88][seed] for seed in seeds]
        differences = [right - left for left, right in zip(repeated, diverse)]
        checkpoints.append({
            "steps": steps,
            "seeds": seeds,
            "repeated_8": {
                "values": repeated,
                "mean": statistics.mean(repeated),
                "ci95": bootstrap_ci(repeated, seed=steps + 8),
            },
            "diverse_88": {
                "values": diverse,
                "mean": statistics.mean(diverse),
                "ci95": bootstrap_ci(diverse, seed=steps + 88),
            },
            "paired_difference": {
                "values": differences,
                "mean": statistics.mean(differences),
                "ci95": bootstrap_ci(differences, seed=steps),
                "exact_permutation_p_two_sided": paired_permutation_p(differences),
            },
        })
    return {
        "research_question": "At matched updates, when does greater pretraining-data diversity improve unseen compositional code prediction?",
        "metric": "teacher-forced argmax accuracy over held-out target-expression bytes",
        "model_parameters": 248_412,
        "seeds_per_checkpoint": 10,
        "checkpoints": checkpoints,
        "important_limit": "All conditions scored 0% exact-expression accuracy, so this is partial token learning rather than solved code generation.",
    }


def write_chart(summary: dict, path: Path) -> None:
    width, height = 1120, 650
    left, top, plot_width, plot_height = 110, 100, 850, 400
    x_positions = {50: left + 70, 100: left + plot_width / 2, 200: left + plot_width - 70}
    colors = {8: "#d97757", 88: "#9eb67b"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#171714"/>',
        '<text x="110" y="45" fill="#f4f0e8" font-family="Arial" font-size="29" font-weight="700">Data diversity helps only after enough training</text>',
        '<text x="110" y="72" fill="#aaa69d" font-family="Arial" font-size="15">Tiny 248K-parameter GLM-style model · 10 paired seeds · unseen program structures</text>',
    ]
    for percent in range(20, 71, 10):
        y = top + plot_height * (0.70 - percent / 100) / 0.50
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#353530"/>')
        parts.append(f'<text x="95" y="{y + 5:.1f}" fill="#aaa69d" text-anchor="end" font-family="Arial" font-size="13">{percent}%</text>')
    for diversity, key, label in ((8, "repeated_8", "8 repeated structures"), (88, "diverse_88", "88 unique structures")):
        points = []
        for checkpoint in summary["checkpoints"]:
            x = x_positions[checkpoint["steps"]]
            mean = checkpoint[key]["mean"]
            low, high = checkpoint[key]["ci95"]
            y = top + plot_height * (0.70 - mean) / 0.50
            y_low = top + plot_height * (0.70 - low) / 0.50
            y_high = top + plot_height * (0.70 - high) / 0.50
            points.append((x, y))
            parts.append(f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="{colors[diversity]}" stroke-width="3"/>')
            parts.append(f'<line x1="{x - 7:.1f}" y1="{y_high:.1f}" x2="{x + 7:.1f}" y2="{y_high:.1f}" stroke="{colors[diversity]}" stroke-width="3"/>')
            parts.append(f'<line x1="{x - 7:.1f}" y1="{y_low:.1f}" x2="{x + 7:.1f}" y2="{y_low:.1f}" stroke="{colors[diversity]}" stroke-width="3"/>')
        path_data = " ".join(("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}" for index, (x, y) in enumerate(points))
        parts.append(f'<path d="{path_data}" fill="none" stroke="{colors[diversity]}" stroke-width="5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{colors[diversity]}" stroke="#f4f0e8" stroke-width="2"/>')
        legend_y = 45 + diversity
        parts.append(f'<line x1="790" y1="{legend_y}" x2="825" y2="{legend_y}" stroke="{colors[diversity]}" stroke-width="5"/>')
        parts.append(f'<text x="835" y="{legend_y + 5}" fill="#f4f0e8" font-family="Arial" font-size="14">{label}</text>')
    for steps, x in x_positions.items():
        parts.append(f'<text x="{x:.1f}" y="530" fill="#f4f0e8" text-anchor="middle" font-family="Arial" font-size="17">{steps} updates</text>')
    final = summary["checkpoints"][-1]
    delta = final["paired_difference"]
    parts.extend([
        '<rect x="675" y="555" width="390" height="65" rx="9" fill="#24241f" stroke="#4a4a42"/>',
        f'<text x="695" y="581" fill="#f4f0e8" font-family="Arial" font-size="16" font-weight="700">At 200 updates: +{delta["mean"] * 100:.1f} points</text>',
        f'<text x="695" y="605" fill="#aaa69d" font-family="Arial" font-size="14">paired permutation p = {delta["exact_permutation_p_two_sided"]:.4f}</text>',
        '<text x="110" y="585" fill="#aaa69d" font-family="Arial" font-size="14">Metric: held-out target-byte accuracy. Error bars: bootstrap 95% CI.</text>',
        '<text x="110" y="609" fill="#d5a26f" font-family="Arial" font-size="14">Exact unseen expressions remained 0% in every condition.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts) + "\n")


def write_seed_chart(summary: dict, path: Path) -> None:
    checkpoint = summary["checkpoints"][-1]
    differences = checkpoint["paired_difference"]["values"]
    seeds = checkpoint["seeds"]
    width, height = 1120, 560
    left, top, plot_width, plot_height = 100, 100, 920, 320
    minimum, maximum = -0.05, 0.07

    def y_for(value: float) -> float:
        return top + plot_height * (maximum - value) / (maximum - minimum)

    zero = y_for(0.0)
    bar_width = plot_width / len(seeds) * 0.58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#171714"/>',
        '<text x="100" y="45" fill="#f4f0e8" font-family="Arial" font-size="29" font-weight="700">Nine of ten seeds improved at 200 updates</text>',
        '<text x="100" y="72" fill="#aaa69d" font-family="Arial" font-size="15">Each bar is 88 unique structures minus 8 repeated structures on the same initialization seed</text>',
    ]
    for tick in (-0.04, -0.02, 0.0, 0.02, 0.04, 0.06):
        y = y_for(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#{"696960" if tick == 0 else "353530"}" stroke-width="{2 if tick == 0 else 1}"/>')
        parts.append(f'<text x="85" y="{y + 5:.1f}" fill="#aaa69d" text-anchor="end" font-family="Arial" font-size="13">{tick * 100:+.0f}</text>')
    for index, (seed, difference) in enumerate(zip(seeds, differences)):
        center = left + (index + 0.5) * plot_width / len(seeds)
        y = y_for(difference)
        height_value = abs(zero - y)
        top_value = min(zero, y)
        color = "#9eb67b" if difference >= 0 else "#d97757"
        parts.append(f'<rect x="{center - bar_width / 2:.1f}" y="{top_value:.1f}" width="{bar_width:.1f}" height="{height_value:.1f}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{center:.1f}" y="{top_value - 8:.1f}" fill="#f4f0e8" text-anchor="middle" font-family="Arial" font-size="13">{difference * 100:+.1f}</text>')
        parts.append(f'<text x="{center:.1f}" y="455" fill="#aaa69d" text-anchor="middle" font-family="Arial" font-size="13">{seed}</text>')
    delta = checkpoint["paired_difference"]
    parts.extend([
        '<text x="28" y="310" fill="#aaa69d" font-family="Arial" font-size="14" transform="rotate(-90 28 310)">Accuracy difference · percentage points</text>',
        '<text x="100" y="495" fill="#aaa69d" font-family="Arial" font-size="13">Initialization seed</text>',
        f'<text x="100" y="530" fill="#f4f0e8" font-family="Arial" font-size="17" font-weight="700">Mean +{delta["mean"] * 100:.1f} points · exact paired permutation p = {delta["exact_permutation_p_two_sided"]:.4f}</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts) + "\n")


def write_report(summary: dict, path: Path) -> None:
    rows = []
    for checkpoint in summary["checkpoints"]:
        rows.append(
            f'| {checkpoint["steps"]} | {checkpoint["repeated_8"]["mean"] * 100:.1f}% | '
            f'{checkpoint["diverse_88"]["mean"] * 100:.1f}% | '
            f'{checkpoint["paired_difference"]["mean"] * 100:+.1f} points | '
            f'{checkpoint["paired_difference"]["exact_permutation_p_two_sided"]:.4f} |'
        )
    text = """# Pretraining Data Diversity Experiment

## Research question

At matched training updates, when does seeing more unique program structure improve prediction on unseen compositional code?

## Design

- Model: 248,412 parameters, byte-level, miniature GLM-style hybrid-attention MoE.
- Conditions: repeatedly sample 8 structures versus train across 88 unique structures.
- Held out: 32 expression structures never used for optimization.
- Fixed: initialization within each paired seed, optimizer, batch size, updates, tokenizer, and evaluation set.
- Replication: 10 paired seeds at 50, 100, and 200 updates.
- Primary metric: teacher-forced argmax accuracy over bytes in the held-out target expression.
- Test: exact two-sided paired sign-flip permutation test over seed-level differences.

## Results

| Updates | 8 repeated | 88 diverse | Paired difference | p value |
|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

At 50 updates, diversity had not helped. At 100 updates, the estimate turned positive but was not statistically secure. At 200 updates, the diverse condition improved held-out target-byte accuracy by 3.2 percentage points, with exact paired permutation p = 0.0137.

## Conclusion

For this tiny model and synthetic code-composition task, data diversity only became useful after enough optimization. A larger corpus is not automatically better when the training budget is too small to absorb it.

## Limits

- Every condition remained at 0% exact-expression accuracy. The model learned partial token structure, not reliable code generation.
- Training was update-matched, not perfectly token-matched. Average token counts differed by less than about 1.3% at 100 updates.
- This is a synthetic miniature and does not establish the same crossover for frontier-scale pretraining.
- The 50 and 100 update comparisons were not significant under the prespecified paired permutation test.
"""
    path.write_text(text)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = summarize()
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_chart(summary, OUTPUT / "learning-curve.svg")
    write_seed_chart(summary, OUTPUT / "paired-seeds-200-updates.svg")
    write_report(summary, OUTPUT / "REPORT.md")
    print(json.dumps({"status": "complete", "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
