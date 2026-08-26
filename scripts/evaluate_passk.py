#!/usr/bin/env python3
"""Sampled executable evaluation for measuring an RL policy."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm53_flash import ByteTokenizer
from glm53_flash.evaluator import evaluate_source
from glm53_flash.runtime import generate_group, load_checkpoint, sha256_file
from glm53_flash.tasks import frozen_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "final", "confirm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--families", default="")
    parser.add_argument("--seed", type=int, default=8675309)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    device = torch.device("cuda")
    tokenizer = ByteTokenizer()
    model = load_checkpoint(args.checkpoint, device)
    tasks = frozen_tasks(args.split, per_family=args.per_family)
    selected_families = {value.strip() for value in args.families.split(",") if value.strip()}
    if selected_families:
        tasks = [task for task in tasks if task.family in selected_families]
        if {task.family for task in tasks} != selected_families:
            raise ValueError("unknown or missing requested family")
    episodes = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for task_index, task in enumerate(tasks):
        generations = generate_group(
            model,
            tokenizer,
            task,
            group_size=args.samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            sample=True,
            seed=args.seed + task_index * 104729,
        )
        samples = []
        for sample_index, generation in enumerate(generations):
            evaluation = evaluate_source(task, task.prompt + generation["completion"]).to_dict()
            samples.append({"sample": sample_index, **generation, "evaluation": evaluation})
        episodes.append({
            "task_id": task.task_id,
            "family": task.family,
            "prompt": task.prompt,
            "reference_completion": task.reference_completion,
            "solved_at_k": any(row["evaluation"]["passed"] for row in samples),
            "samples": samples,
        })
        print(json.dumps({
            "task": task.task_id,
            "exact": sum(row["evaluation"]["passed"] for row in samples),
            "valid": sum(row["evaluation"]["status"] != "invalid" for row in samples),
        }), flush=True)

    torch.cuda.synchronize()
    rollouts = [sample for episode in episodes for sample in episode["samples"]]
    tests_passed = sum(row["evaluation"]["tests_passed"] for row in rollouts)
    tests_total = sum(row["evaluation"]["tests_total"] for row in rollouts)
    summary = {
        "tasks_solved_at_k": sum(row["solved_at_k"] for row in episodes),
        "tasks_total": len(episodes),
        "pass_at_k": sum(row["solved_at_k"] for row in episodes) / len(episodes),
        "exact_rollouts": sum(row["evaluation"]["passed"] for row in rollouts),
        "rollouts_total": len(rollouts),
        "exact_rollout_rate": sum(row["evaluation"]["passed"] for row in rollouts) / len(rollouts),
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "test_accuracy": tests_passed / tests_total,
        "valid_rollouts": sum(row["evaluation"]["status"] != "invalid" for row in rollouts),
        "token_cap_hits": sum(row["hit_token_cap"] for row in rollouts),
    }
    by_family = {}
    for family in sorted({row["family"] for row in episodes}):
        family_episodes = [row for row in episodes if row["family"] == family]
        family_rollouts = [sample for row in family_episodes for sample in row["samples"]]
        by_family[family] = {
            "tasks_solved_at_k": sum(row["solved_at_k"] for row in family_episodes),
            "tasks": len(family_episodes),
            "exact_rollouts": sum(row["evaluation"]["passed"] for row in family_rollouts),
            "rollouts": len(family_rollouts),
            "valid_rollouts": sum(row["evaluation"]["status"] != "invalid" for row in family_rollouts),
        }
    payload = {
        "schema_version": "1.0",
        "status": "complete",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint / "model.pt"),
        "inference": {
            "samples_per_task": args.samples,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "families": sorted(selected_families) if selected_families else "all",
        },
        "summary": summary,
        "by_family": by_family,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
