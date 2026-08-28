#!/usr/bin/env python3
"""Fast pretraining ablations for curriculum and example ordering."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm53_flash import ByteTokenizer, GLM53FlashFromScratch, ModelConfig
from experiments.pretraining_data_diversity import all_structures, evaluate, render_example


def model_config(sequence_length: int) -> ModelConfig:
    return ModelConfig(
        dim=48,
        layers=3,
        heads=3,
        expert_hidden=96,
        experts=4,
        top_k=1,
        streams=2,
        sparse_window=24,
        sparse_stride=24,
        max_sequence_length=sequence_length,
    )


def static_items(pool: list[tuple[str, ...]], *, examples: int, mode: str) -> list[tuple[tuple[str, ...], int]]:
    items = [(pool[index % len(pool)], index) for index in range(examples)]
    if mode == "blocked":
        rank = {structure: index for index, structure in enumerate(pool)}
        items.sort(key=lambda item: (rank[item[0]], item[1]))
    elif mode != "interleaved":
        raise ValueError(f"unknown static mode: {mode}")
    return items


def curriculum_items(
    train_pool: list[tuple[str, ...]],
    *,
    examples: int,
    switch_example: int,
) -> list[tuple[tuple[str, ...], int]]:
    repeated_pool = train_pool[:8]
    items = []
    for index in range(examples):
        pool = repeated_pool if index < switch_example else train_pool
        items.append((pool[index % len(pool)], index))
    return items


def batch_from_items(
    tokenizer: ByteTokenizer,
    items: list[tuple[tuple[str, ...], int]],
    *,
    seed: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rows = []
    for structure, example_id in items:
        text, _ = render_example(structure, index=example_id, seed=seed)
        ids = tokenizer.encode(text, bos=True, eos=True)
        if len(ids) > sequence_length + 1:
            raise ValueError(f"example is {len(ids)} tokens, limit is {sequence_length + 1}")
        ids += [tokenizer.pad_id] * (sequence_length + 1 - len(ids))
        rows.append(ids)
    values = torch.tensor(rows, dtype=torch.long, device=device)
    labels = values[:, 1:].clone()
    labels[labels == tokenizer.pad_id] = -100
    return values[:, :-1], labels, int((labels != -100).sum())


def train_condition(
    *,
    condition: str,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    sequence_length: int,
    device: torch.device,
    train_pool: list[tuple[str, ...]],
    eval_pool: list[tuple[str, ...]],
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = ByteTokenizer()
    config = model_config(sequence_length)
    model = GLM53FlashFromScratch(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    total_examples = steps * batch_size
    if condition == "curriculum-8-to-88":
        items = curriculum_items(
            train_pool,
            examples=total_examples,
            switch_example=(steps // 2) * batch_size,
        )
    elif condition == "diverse-interleaved":
        items = static_items(train_pool, examples=total_examples, mode="interleaved")
    elif condition == "diverse-blocked":
        items = static_items(train_pool, examples=total_examples, mode="blocked")
    else:
        raise ValueError(f"unknown condition: {condition}")
    before = evaluate(model, tokenizer, eval_pool, seed=seed, device=device)
    curve = []
    tokens_seen = 0
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        batch_items = items[step * batch_size:(step + 1) * batch_size]
        inputs, labels, tokens = batch_from_items(
            tokenizer,
            batch_items,
            seed=seed,
            sequence_length=sequence_length,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, usage = model(inputs)
        language_loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100
        )
        balance = ((usage.mean(dim=0) - 1.0 / config.experts) ** 2).mean() * config.experts
        loss = language_loss + 0.01 * balance
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens_seen += tokens
        if step == 0 or (step + 1) % 20 == 0:
            curve.append({"step": step + 1, "training_loss": float(language_loss), "tokens_seen": tokens_seen})
    after = evaluate(model, tokenizer, eval_pool, seed=seed, device=device)
    return {
        "condition": condition,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "tokens_seen": tokens_seen,
        "elapsed_seconds": time.perf_counter() - started,
        "parameter_counts": model.parameter_counts(),
        "before": before,
        "after": after,
        "training_curve": curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", default="curriculum-8-to-88,diverse-interleaved,diverse-blocked")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",")]
    structures = all_structures()
    train_pool, eval_pool = structures[:88], structures[88:120]
    device = torch.device(args.device)
    rows = []
    started = time.perf_counter()
    for condition in conditions:
        for seed in seeds:
            row = train_condition(
                condition=condition,
                seed=seed,
                steps=args.steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                sequence_length=args.sequence_length,
                device=device,
                train_pool=train_pool,
                eval_pool=eval_pool,
            )
            rows.append(row)
            print(json.dumps({
                "condition": condition,
                "seed": seed,
                "accuracy": row["after"]["target_byte_accuracy"],
                "exact": row["after"]["target_exact_accuracy"],
                "seconds": row["elapsed_seconds"],
            }), flush=True)
    receipt = {
        "schema_version": "1.0",
        "status": "complete",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "conditions": conditions,
        "seeds": seeds,
        "device": args.device,
        "elapsed_seconds": time.perf_counter() - started,
        "held_out_structures": [list(value) for value in eval_pool],
        "runs": rows,
    }
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
