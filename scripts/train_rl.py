#!/usr/bin/env python3
"""RLOO post-training with executable unit-test rewards."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm53_flash import ByteTokenizer
from glm53_flash.evaluator import evaluate_source
from glm53_flash.runtime import generate_group, load_checkpoint, save_checkpoint, sha256_file
from glm53_flash.tasks import CodingTask, frozen_tasks


def schedule(tasks: list[CodingTask], groups: int, seed: int) -> list[CodingTask]:
    result = []
    epoch = 0
    while len(result) < groups:
        block = list(tasks)
        random.Random(seed + epoch * 1009).shuffle(block)
        result.extend(block)
        epoch += 1
    return result[:groups]


def leave_one_out(rewards: list[float]) -> list[float]:
    total = sum(rewards)
    return [reward - (total - reward) / (len(rewards) - 1) for reward in rewards]


def reward_for(
    task: CodingTask,
    completion: str,
    mode: str,
    *,
    invalid_penalty: float,
    exact_bonus: float,
) -> tuple[float, dict]:
    evaluation = evaluate_source(task, task.prompt + completion).to_dict()
    if mode == "binary":
        reward = 1.0 if evaluation["passed"] else (invalid_penalty if evaluation["status"] == "invalid" else 0.0)
    else:
        penalty = invalid_penalty if evaluation["status"] == "invalid" else 0.0
        bonus = exact_bonus if evaluation["passed"] else 0.0
        reward = float(evaluation["pass_fraction"]) + bonus + penalty
    return reward, evaluation


def completion_log_probabilities(
    model,
    prompt: list[int],
    completions: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Score a rollout group in one model call instead of one call per sample."""
    if not completions or any(not completion for completion in completions):
        raise ValueError("empty completion")
    maximum = max(map(len, completions))
    rows = [prompt + completion + [0] * (maximum - len(completion)) for completion in completions]
    sequence = torch.tensor(rows, dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, _ = model(sequence[:, :-1])
    start = len(prompt) - 1
    selected_logits = logits[:, start : start + maximum].float()
    targets = sequence[:, len(prompt) :]
    token_log_probabilities = torch.log_softmax(selected_logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    lengths = torch.tensor([len(completion) for completion in completions], device=device)
    mask = torch.arange(maximum, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return (token_log_probabilities * mask).sum(dim=1) / lengths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--checkpoints", default="8,16,32")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-scope", choices=("all", "last-block", "last-block-head"), default="all")
    parser.add_argument("--reward-mode", choices=("case-fraction", "binary"), default="case-fraction")
    parser.add_argument("--invalid-penalty", type=float, default=-0.1)
    parser.add_argument("--exact-bonus", type=float, default=0.0)
    parser.add_argument("--families", default="")
    parser.add_argument("--tasks-per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.group_size < 2:
        raise ValueError("RLOO requires group size at least two")
    checkpoints = {int(value) for value in args.checkpoints.split(",") if value}
    if max(checkpoints) != args.groups:
        raise ValueError("final checkpoint must equal groups")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    tokenizer = ByteTokenizer()
    model = load_checkpoint(args.initial_checkpoint, device)
    if args.train_scope in {"last-block", "last-block-head"}:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layers[-1].parameters():
            parameter.requires_grad = True
        for parameter in model.final_norm.parameters():
            parameter.requires_grad = True
        if args.train_scope == "last-block-head":
            for parameter in model.embedding.parameters():
                parameter.requires_grad = True
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=0.0)
    tasks = frozen_tasks("rl", per_family=args.tasks_per_family)
    selected_families = {value.strip() for value in args.families.split(",") if value.strip()}
    if selected_families:
        tasks = [task for task in tasks if task.family in selected_families]
        if {task.family for task in tasks} != selected_families:
            raise ValueError("unknown or missing requested family")
    task_schedule = schedule(tasks, args.groups, args.seed)
    schedule_ids = [task.task_id for task in task_schedule]
    (args.output / "schedule.json").write_text(json.dumps(schedule_ids, indent=2) + "\n")
    saved = [dict(group=0, **save_checkpoint(model, args.output / "checkpoint-0000", {
        "stage": "pre_rl", "source_checkpoint": str(args.initial_checkpoint),
        "source_sha256": sha256_file(args.initial_checkpoint / "model.pt"),
    }))]
    rows = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for group_index, task in enumerate(task_schedule, 1):
        group_started = time.perf_counter()
        generations = generate_group(
            model, tokenizer, task, group_size=args.group_size,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            sample=True, seed=args.seed + group_index * 17,
        )
        rewards, evaluations = [], []
        for generation in generations:
            reward, evaluation = reward_for(
                task,
                generation["completion"],
                args.reward_mode,
                invalid_penalty=args.invalid_penalty,
                exact_bonus=args.exact_bonus,
            )
            rewards.append(reward)
            evaluations.append(evaluation)
        advantages = leave_one_out(rewards)
        spread = max(rewards) - min(rewards)
        updated = spread > 1e-8 and all(generation["token_ids"] for generation in generations)
        loss_value = None
        gradient_norm = None
        if updated:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prompt = tokenizer.encode(task.prompt, bos=True)
            log_probabilities = completion_log_probabilities(
                model,
                prompt,
                [generation["token_ids"] for generation in generations],
                device,
            )
            advantage_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
            objective = -(advantage_tensor * log_probabilities).mean()
            if not bool(torch.isfinite(objective).item()):
                raise FloatingPointError("non-finite RL objective")
            objective.backward()
            gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_value = float(objective.detach())
            gradient_norm = float(gradient_norm_tensor.detach())
        row = {
            "group": group_index,
            "task_id": task.task_id,
            "family": task.family,
            "rewards": rewards,
            "advantages": advantages,
            "reward_spread": spread,
            "updated": updated,
            "policy_loss": loss_value,
            "gradient_norm": gradient_norm,
            "exact_rollouts": sum(evaluation["passed"] for evaluation in evaluations),
            "valid_rollouts": sum(evaluation["status"] != "invalid" for evaluation in evaluations),
            "token_cap_hits": sum(generation["hit_token_cap"] for generation in generations),
            "seconds": round(time.perf_counter() - group_started, 3),
            "samples": [dict(**generation, reward=reward, evaluation=evaluation) for generation, reward, evaluation in zip(generations, rewards, evaluations, strict=True)],
        }
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("group", "task_id", "rewards", "updated", "exact_rollouts", "seconds")}, sort_keys=True), flush=True)
        if group_index in checkpoints:
            saved.append(dict(group=group_index, **save_checkpoint(model, args.output / f"checkpoint-{group_index:04d}", {
                "stage": "executable_reward_rl", "group": group_index, "seed": args.seed,
            })))
    if device.type == "cuda":
        torch.cuda.synchronize()
    receipt = {
        "schema_version": "1.0",
        "status": "complete",
        "stage": "executable_reward_rl",
        "algorithm": "RLOO",
        "initial_checkpoint": str(args.initial_checkpoint),
        "initial_checkpoint_sha256": sha256_file(args.initial_checkpoint / "model.pt"),
        "seed": args.seed,
        "groups": args.groups,
        "group_size": args.group_size,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.learning_rate,
        "train_scope": args.train_scope,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "reward_mode": args.reward_mode,
        "invalid_penalty": args.invalid_penalty,
        "exact_bonus": args.exact_bonus,
        "device": device.type,
        "families": sorted(selected_families) if selected_families else "all",
        "tasks_per_family": args.tasks_per_family,
        "schedule_sha256": hashlib.sha256((args.output / "schedule.json").read_bytes()).hexdigest(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if device.type == "cuda" else None,
        "checkpoints": saved,
        "groups_detail": rows,
    }
    (args.output / "training-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "groups", "elapsed_seconds", "peak_vram_gib")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
