#!/usr/bin/env python3
"""Greedy executable evaluation of a frozen coding split."""
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
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--families", default="")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for index, task in enumerate(tasks):
        generated = generate_group(
            model, tokenizer, task, group_size=1, max_new_tokens=args.max_new_tokens,
            temperature=1.0, sample=False, seed=args.seed + index,
        )[0]
        source = task.prompt + generated["completion"]
        evaluation = evaluate_source(task, source).to_dict()
        episodes.append({
            "task_id": task.task_id,
            "family": task.family,
            "prompt": task.prompt,
            "reference_completion": task.reference_completion,
            "generated_completion": generated["completion"],
            "generated_tokens": generated["tokens"],
            "hit_token_cap": generated["hit_token_cap"],
            "evaluation": evaluation,
        })
        print(json.dumps({"task": task.task_id, "passed": evaluation["passed"], "tests": evaluation["tests_passed"]}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    cap_hits = sum(row["hit_token_cap"] for row in episodes)
    solved = sum(row["evaluation"]["passed"] for row in episodes)
    tests_passed = sum(row["evaluation"]["tests_passed"] for row in episodes)
    tests_total = sum(row["evaluation"]["tests_total"] for row in episodes)
    summary = {
        "tasks_solved": solved,
        "tasks_total": len(episodes),
        "pass_at_1": solved / len(episodes),
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "test_accuracy": tests_passed / tests_total,
        "valid_programs": sum(row["evaluation"]["status"] != "invalid" for row in episodes),
        "token_cap_hits": cap_hits,
    }
    by_family = {}
    for family in sorted({row["family"] for row in episodes}):
        rows = [row for row in episodes if row["family"] == family]
        by_family[family] = {
            "solved": sum(row["evaluation"]["passed"] for row in rows),
            "tasks": len(rows),
            "tests_passed": sum(row["evaluation"]["tests_passed"] for row in rows),
            "tests_total": sum(row["evaluation"]["tests_total"] for row in rows),
        }
    payload = {
        "schema_version": "1.0",
        "status": "complete",
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint / "model.pt"),
        "inference": {
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "families": sorted(selected_families) if selected_families else "all",
        },
        "summary": summary,
        "by_family": by_family,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if device.type == "cuda" else None,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
