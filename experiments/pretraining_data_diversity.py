#!/usr/bin/env python3
"""Matched miniature pretraining study: repeated versus diverse program structures."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm53_flash import ByteTokenizer, GLM53FlashFromScratch, ModelConfig


OPS = ("add", "sub", "mul", "neg", "abs")
PHRASES = {
    "add": ("Add {c}.", "Increase it by {c}."),
    "sub": ("Subtract {c}.", "Decrease it by {c}."),
    "mul": ("Multiply by {c}.", "Scale it by {c}."),
    "neg": ("Negate it.", "Flip its sign."),
    "abs": ("Take its absolute value.", "Make the result nonnegative."),
}


def all_structures() -> list[tuple[str, ...]]:
    """Balanced compositional structures, each containing multiple primitives."""
    values = []
    for length in (3,):
        for structure in itertools.product(OPS, repeat=length):
            if len(set(structure)) >= 2:
                values.append(structure)
    random.Random(1701).shuffle(values)
    return values


def render_example(structure: tuple[str, ...], *, index: int, seed: int) -> tuple[str, int]:
    structure_key = sum((position + 1) * (OPS.index(operation) + 1) * 101 for position, operation in enumerate(structure))
    rng = random.Random(seed * 1_000_003 + index * 9176 + structure_key)
    constants = [rng.choice((2, 3, 4, 5)) for _ in structure]
    instructions = []
    expression = "x"
    value = rng.randint(-5, 5)
    for operation, constant in zip(structure, constants):
        instructions.append(rng.choice(PHRASES[operation]).format(c=constant))
        if operation == "add":
            expression = f"({expression} + {constant})"
            value += constant
        elif operation == "sub":
            expression = f"({expression} - {constant})"
            value -= constant
        elif operation == "mul":
            expression = f"({expression} * {constant})"
            value *= constant
        else:
            if operation == "neg":
                expression = f"(-{expression})"
                value = -value
            else:
                expression = f"abs({expression})"
                value = abs(value)
    prompt = f"Task: start with x. {' '.join(instructions)}\nPython: "
    completion = f"{expression}\n"
    return prompt + completion, len(ByteTokenizer().encode(prompt, bos=True))


def make_batch(
    tokenizer: ByteTokenizer,
    structures: list[tuple[str, ...]],
    *,
    step: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rows = []
    for offset in range(batch_size):
        index = step * batch_size + offset
        structure = structures[index % len(structures)]
        text, _ = render_example(structure, index=index, seed=seed)
        ids = tokenizer.encode(text, bos=True, eos=True)
        if len(ids) > sequence_length + 1:
            raise ValueError(f"example is {len(ids)} tokens, limit is {sequence_length + 1}")
        ids += [tokenizer.pad_id] * (sequence_length + 1 - len(ids))
        rows.append(ids)
    values = torch.tensor(rows, dtype=torch.long, device=device)
    labels = values[:, 1:].clone()
    labels[labels == tokenizer.pad_id] = -100
    return values[:, :-1], labels, int((labels != -100).sum())


@torch.no_grad()
def evaluate(
    model: GLM53FlashFromScratch,
    tokenizer: ByteTokenizer,
    structures: list[tuple[str, ...]],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    exact = 0
    examples = []
    for index, structure in enumerate(structures):
        text, prompt_length = render_example(structure, index=100_000 + index, seed=seed + 999)
        ids = tokenizer.encode(text, bos=True, eos=True)
        values = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
        logits, _ = model(values[:, :-1])
        labels = values[:, 1:]
        start = prompt_length - 1
        target_logits = logits[:, start:, :]
        target_labels = labels[:, start:]
        token_losses = F.cross_entropy(
            target_logits.reshape(-1, model.config.vocab_size),
            target_labels.reshape(-1),
            reduction="none",
        )
        losses.extend(token_losses.tolist())
        predicted = target_logits.argmax(dim=-1)
        correct += int((predicted == target_labels).sum())
        total += target_labels.numel()
        is_exact = bool(torch.equal(predicted, target_labels))
        exact += int(is_exact)
        if len(examples) < 3:
            examples.append({
                "structure": list(structure),
                "target": tokenizer.decode(target_labels[0].tolist()),
                "teacher_forced_argmax": tokenizer.decode(predicted[0].tolist()),
                "exact": is_exact,
            })
    return {
        "target_loss": sum(losses) / len(losses),
        "target_perplexity": math.exp(min(20.0, sum(losses) / len(losses))),
        "target_byte_accuracy": correct / total,
        "target_exact_accuracy": exact / len(structures),
        "examples": examples,
    }


def train_arm(
    *,
    diversity: int,
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
    config = ModelConfig(
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
    model = GLM53FlashFromScratch(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    tokenizer = ByteTokenizer()
    structures = train_pool[:diversity]
    curve = []
    tokens_seen = 0
    started = time.perf_counter()
    before = evaluate(model, tokenizer, eval_pool, seed=seed, device=device)
    model.train()
    for step in range(steps):
        inputs, labels, tokens = make_batch(
            tokenizer,
            structures,
            step=step,
            batch_size=batch_size,
            sequence_length=sequence_length,
            seed=seed,
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
        if step == 0 or (step + 1) % 10 == 0:
            curve.append({"step": step + 1, "training_loss": float(language_loss), "tokens_seen": tokens_seen})
    after = evaluate(model, tokenizer, eval_pool, seed=seed, device=device)
    return {
        "diversity": diversity,
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


def write_svg(path: Path, rows: list[dict], diversities: list[int]) -> None:
    width, height = 960, 560
    plot_left, plot_top, plot_width, plot_height = 90, 75, 790, 370
    colors = {diversity: color for diversity, color in zip(diversities, ("#d97757", "#e0b45c", "#9eb67b", "#7aa2c9"))}
    grouped = {diversity: [r["after"]["target_byte_accuracy"] for r in rows if r["diversity"] == diversity] for diversity in diversities}
    means = {d: sum(v) / len(v) for d, v in grouped.items()}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#171714"/>',
        '<text x="90" y="40" fill="#f4f0e8" font-family="Arial" font-size="25" font-weight="700">Does more unique pretraining data improve unseen code?</text>',
        '<text x="90" y="64" fill="#aaa69d" font-family="Arial" font-size="14">Same model, optimizer, steps and batch size. Only unique program structures change.</text>',
    ]
    for tick in range(0, 11, 2):
        y = plot_top + plot_height * (1 - tick / 10)
        parts.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_width}" y2="{y:.1f}" stroke="#353530"/>')
        parts.append(f'<text x="78" y="{y + 5:.1f}" fill="#aaa69d" text-anchor="end" font-family="Arial" font-size="13">{tick * 10}%</text>')
    bar_width = 130
    gap = (plot_width - bar_width * len(diversities)) / (len(diversities) + 1)
    for i, diversity in enumerate(diversities):
        x = plot_left + gap * (i + 1) + bar_width * i
        mean = means[diversity]
        y = plot_top + plot_height * (1 - mean)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{plot_top + plot_height - y:.1f}" rx="8" fill="{colors[diversity]}"/>')
        for j, value in enumerate(grouped[diversity]):
            px = x + bar_width / 2 + (j - (len(grouped[diversity]) - 1) / 2) * 18
            py = plot_top + plot_height * (1 - value)
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="#171714" stroke="#f4f0e8" stroke-width="2"/>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 12:.1f}" fill="#f4f0e8" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">{mean * 100:.1f}%</text>')
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{plot_top + plot_height + 32}" fill="#f4f0e8" text-anchor="middle" font-family="Arial" font-size="16">{diversity} structures</text>')
    parts.extend([
        '<text x="18" y="270" fill="#aaa69d" font-family="Arial" font-size="14" transform="rotate(-90 18 270)">Held-out target-byte accuracy</text>',
        '<text x="90" y="515" fill="#aaa69d" font-family="Arial" font-size="13">Dots are independent seeds. Evaluation structures never appear in training.</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--diversities", default="8,32,88")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    device_name = "mps" if args.device == "auto" and torch.backends.mps.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    structures = all_structures()
    train_pool, eval_pool = structures[:88], structures[88:120]
    diversities = [int(value) for value in args.diversities.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    if max(diversities) > len(train_pool):
        raise ValueError("diversity exceeds training pool")
    rows = []
    started = time.perf_counter()
    for diversity in diversities:
        for seed in seeds:
            row = train_arm(
                diversity=diversity,
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
                "diversity": diversity,
                "seed": seed,
                "accuracy": row["after"]["target_byte_accuracy"],
                "exact": row["after"]["target_exact_accuracy"],
                "seconds": row["elapsed_seconds"],
            }), flush=True)
    receipt = {
        "schema_version": "1.0",
        "status": "complete",
        "research_question": "At fixed compute, does more unique program structure improve held-out compositional code prediction?",
        "independent_variable": "number of unique training expression structures",
        "controlled": ["model", "initialization within seed", "optimizer", "steps", "batch size", "tokenizer", "evaluation set"],
        "structural_holdout": [list(value) for value in eval_pool],
        "device": device.type,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "diversities": diversities,
        "seeds": seeds,
        "elapsed_seconds": time.perf_counter() - started,
        "runs": rows,
    }
    (args.output / "results.json").write_text(json.dumps(receipt, indent=2) + "\n")
    write_svg(args.output / "accuracy.svg", rows, diversities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
