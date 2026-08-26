#!/usr/bin/env python3
"""Short synthetic coding pretraining from random initialization."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm53_flash import ByteTokenizer, GLM53FlashFromScratch, ModelConfig
from glm53_flash.runtime import save_checkpoint
from glm53_flash.tasks import pretraining_text


def batch_for(tokenizer: ByteTokenizer, *, step: int, batch_size: int, sequence_length: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    rows = []
    for offset in range(batch_size):
        text = pretraining_text(step * batch_size + offset, seed=seed)
        ids = tokenizer.encode(text, bos=True, eos=True)
        if len(ids) > sequence_length + 1:
            raise ValueError(f"training example exceeds sequence length: {len(ids)}")
        ids += [tokenizer.pad_id] * (sequence_length + 1 - len(ids))
        rows.append(ids)
    values = torch.tensor(rows, dtype=torch.long, device=device)
    inputs, labels = values[:, :-1], values[:, 1:]
    labels = labels.masked_fill(labels == tokenizer.pad_id, -100)
    return inputs, labels, int((labels != -100).sum().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--checkpoints", default="100,200,400")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    checkpoints = {int(value) for value in args.checkpoints.split(",") if value}
    if max(checkpoints) != args.steps:
        raise ValueError("final checkpoint must equal steps")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    tokenizer = ByteTokenizer()
    config = ModelConfig(dim=args.dim, layers=args.layers, max_sequence_length=max(192, args.sequence_length))
    model = GLM53FlashFromScratch(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    counts = model.parameter_counts()
    initial = save_checkpoint(model, args.output / "checkpoint-0000", {
        "stage": "random_initialization", "seed": args.seed, "parameter_counts": counts,
    })
    rows = []
    checkpoint_rows = [dict(step=0, **initial)]
    tokens_seen = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        step_started = time.perf_counter()
        inputs, labels, tokens = batch_for(
            tokenizer, step=step - 1, batch_size=args.batch_size,
            sequence_length=args.sequence_length, seed=args.seed, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, usage = model(inputs)
            language_loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), labels.reshape(-1), ignore_index=-100)
            balance = ((usage.mean(dim=0) - 1.0 / config.experts) ** 2).mean() * config.experts
            loss = language_loss + 0.01 * balance
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("non-finite loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        tokens_seen += tokens
        row = {
            "step": step,
            "loss": float(language_loss.detach()),
            "router_balance_loss": float(balance.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "tokens": tokens,
            "tokens_seen": tokens_seen,
            "seconds": round(time.perf_counter() - step_started, 4),
        }
        rows.append(row)
        if step == 1 or step % 10 == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        if step in checkpoints:
            saved = save_checkpoint(model, args.output / f"checkpoint-{step:04d}", {
                "stage": "coding_pretraining", "seed": args.seed, "step": step,
                "tokens_seen": tokens_seen, "parameter_counts": counts,
            })
            checkpoint_rows.append(dict(step=step, **saved))
    torch.cuda.synchronize()
    receipt = {
        "schema_version": "1.0",
        "status": "complete",
        "stage": "coding_pretraining",
        "config": config.to_dict(),
        "parameter_counts": counts,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "learning_rate": args.learning_rate,
        "tokens_seen": tokens_seen,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "checkpoints": checkpoint_rows,
        "training_curve": rows,
    }
    (args.output / "training-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "tokens_seen", "elapsed_seconds", "peak_vram_gib")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
