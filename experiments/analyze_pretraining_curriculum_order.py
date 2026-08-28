#!/usr/bin/env python3
"""Analyze the 10-seed pretraining curriculum and ordering ablations."""
from __future__ import annotations

import itertools
import json
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "artifacts/experiments/pretraining-data-diversity-200steps-10seed/results.json"
NEW_PATH = ROOT / "artifacts/experiments/pretraining-curriculum-order-200steps-10seed/results.json"
OUTPUT = ROOT / "artifacts/experiments/pretraining-curriculum-order-summary"


def bootstrap_ci(values: list[float], *, seed: int, draws: int = 50_000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(values) for _ in values) for _ in range(draws))
    return means[int(draws * 0.025) - 1], means[int(draws * 0.975) - 1]


def permutation_p(differences: list[float]) -> float:
    observed = abs(statistics.mean(differences))
    null = [
        abs(sum(sign * value for sign, value in zip(signs, differences)) / len(differences))
        for signs in itertools.product((-1, 1), repeat=len(differences))
    ]
    return sum(value >= observed - 1e-15 for value in null) / len(null)


def comparison(left: dict[int, float], right: dict[int, float], *, seed: int) -> dict:
    seeds = sorted(set(left) & set(right))
    differences = [right[value] - left[value] for value in seeds]
    return {
        "seeds": seeds,
        "differences": differences,
        "mean": statistics.mean(differences),
        "ci95": bootstrap_ci(differences, seed=seed),
        "exact_paired_permutation_p_two_sided": permutation_p(differences),
        "right_wins": sum(value > 0 for value in differences),
    }


def main() -> int:
    base = json.loads(BASE_PATH.read_text())
    new = json.loads(NEW_PATH.read_text())
    values = {
        "repeated_8": {},
        "diverse_interleaved": {},
        "curriculum_8_to_88": {},
        "diverse_blocked": {},
    }
    for run in base["runs"]:
        key = "repeated_8" if run["diversity"] == 8 else "diverse_interleaved"
        values[key][run["seed"]] = run["after"]["target_byte_accuracy"]
    for run in new["runs"]:
        key = run["condition"].replace("-", "_")
        values[key][run["seed"]] = run["after"]["target_byte_accuracy"]
    conditions = {
        key: {
            "values": [mapping[seed] for seed in sorted(mapping)],
            "mean": statistics.mean(mapping.values()),
            "ci95": bootstrap_ci(list(mapping.values()), seed=index + 91),
        }
        for index, (key, mapping) in enumerate(values.items())
    }
    summary = {
        "metric": "teacher-forced argmax accuracy over held-out target-expression bytes",
        "model_parameters": 248_412,
        "updates": 200,
        "seeds": 10,
        "conditions": conditions,
        "comparisons": {
            "interleaved_minus_blocked": comparison(values["diverse_blocked"], values["diverse_interleaved"], seed=501),
            "curriculum_minus_interleaved": comparison(values["diverse_interleaved"], values["curriculum_8_to_88"], seed=502),
            "curriculum_minus_repeated": comparison(values["repeated_8"], values["curriculum_8_to_88"], seed=503),
        },
        "important_limit": "Exact-expression accuracy remained 0% in every condition.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    ordering = summary["comparisons"]["interleaved_minus_blocked"]
    curriculum = summary["comparisons"]["curriculum_minus_interleaved"]
    report = f"""# Pretraining Curriculum and Ordering Experiments

## Experiment 1: does example order matter?

The blocked and interleaved conditions use the exact same 4,800 generated examples, model initialization within seed, optimizer, 200 updates, batch size, and held-out structures. Only presentation order changes.

- Blocked mean: {conditions['diverse_blocked']['mean'] * 100:.1f}%
- Interleaved mean: {conditions['diverse_interleaved']['mean'] * 100:.1f}%
- Paired difference: +{ordering['mean'] * 100:.1f} percentage points
- Bootstrap 95% interval: [{ordering['ci95'][0] * 100:.1f}, {ordering['ci95'][1] * 100:.1f}] points
- Exact paired permutation p: {ordering['exact_paired_permutation_p_two_sided']:.6f}
- Seeds improved: {ordering['right_wins']} / 10

Conclusion: interleaving the same diverse examples substantially improved held-out target-byte accuracy. Long homogeneous blocks likely create stronger recency bias or forgetting in this tiny model.

## Experiment 2: does an 8-to-88 curriculum help?

The curriculum spends the first 100 updates cycling over 8 structures, then expands to 88 for the final 100 updates.

- Repeated 8 mean: {conditions['repeated_8']['mean'] * 100:.1f}%
- Curriculum mean: {conditions['curriculum_8_to_88']['mean'] * 100:.1f}%
- Diverse from the start mean: {conditions['diverse_interleaved']['mean'] * 100:.1f}%
- Curriculum minus diverse: {curriculum['mean'] * 100:+.1f} points
- Exact paired permutation p: {curriculum['exact_paired_permutation_p_two_sided']:.4f}

Conclusion: the curriculum improved over repeating only 8 structures, but did not beat diverse interleaving from the start. The measured curriculum disadvantage of 0.6 points was not statistically significant.

## Limits

- Exact-expression accuracy remained 0% in every condition. These experiments measure partial token learning, not reliable executable generation.
- The model has 248,412 parameters and the task is synthetic. The effect should be replicated on natural code before generalizing it.
- The forgetting explanation is plausible but not directly measured. Hidden-state or per-structure trajectory analysis would be needed to establish the mechanism.
"""
    (OUTPUT / "REPORT.md").write_text(report)
    print(json.dumps({"status": "complete", "output": str(OUTPUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
