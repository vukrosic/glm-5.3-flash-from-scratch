#!/usr/bin/env python3
"""Train and compare tiny RGB vision paths on generated held-out digits."""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glm53_flash import (  # noqa: E402
    ByteTokenizer,
    DirectPatchVisionLanguageModel,
    GLM53FlashFromScratch,
    MiniVisionConfig,
    ModelConfig,
    VisionLanguageModel,
    answer_only_labels,
)


SEGMENTS = {
    0: "abcdef",
    1: "bc",
    2: "abged",
    3: "abgcd",
    4: "fgbc",
    5: "afgcd",
    6: "afgecd",
    7: "abc",
    8: "abcdefg",
    9: "abfgcd",
}


class TrainableVisionModel(Protocol):
    visual_token_count: int
    inserted_token_count: int

    def __call__(
        self, images: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def parameters(self): ...

    def train(self, mode: bool = True): ...

    def eval(self): ...


def generated_rgb_digit_images(
    labels: torch.Tensor, *, seed: int, noise: float = 0.06
) -> torch.Tensor:
    """Render shifted seven-segment digits with label-independent random RGB colors."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.zeros((labels.numel(), 3, 32, 32), dtype=torch.float32)
    horizontal = {
        "a": (4, 7, 8, 24),
        "g": (14, 17, 8, 24),
        "d": (25, 28, 8, 24),
    }
    vertical = {
        "f": (6, 15, 5, 8),
        "b": (6, 15, 24, 27),
        "e": (16, 26, 5, 8),
        "c": (16, 26, 24, 27),
    }
    coordinates = horizontal | vertical
    for row, label in enumerate(labels.tolist()):
        shift_y = int(torch.randint(-2, 3, (1,), generator=generator).item())
        shift_x = int(torch.randint(-2, 3, (1,), generator=generator).item())
        color = 0.55 + 0.45 * torch.rand((3, 1, 1), generator=generator)
        intensity = 0.80 + 0.20 * torch.rand((), generator=generator)
        for segment in SEGMENTS[int(label)]:
            y0, y1, x0, x1 = coordinates[segment]
            y0, y1 = max(0, y0 + shift_y), min(32, y1 + shift_y)
            x0, x1 = max(0, x0 + shift_x), min(32, x1 + shift_x)
            images[row, :, y0:y1, x0:x1] = color * intensity
    images += noise * torch.rand(images.shape, generator=generator)
    return images.clamp_(0, 1)


def text_batch(
    tokenizer: ByteTokenizer, labels: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, int]:
    prompt = tokenizer.encode("Digit: ", bos=True)
    rows = [prompt + tokenizer.encode(str(int(label))) for label in labels]
    return torch.tensor(rows, dtype=torch.long, device=device), len(prompt)


@torch.no_grad()
def held_out_metrics(
    model: TrainableVisionModel,
    tokenizer: ByteTokenizer,
    *,
    examples: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    labels = torch.arange(examples) % 10
    images = generated_rgb_digit_images(labels, seed=seed).to(device)
    prompt = tokenizer.encode("Digit: ", bos=True)
    prompts = torch.tensor([prompt] * examples, dtype=torch.long, device=device)
    logits, _ = model(images, prompts)
    expected = torch.tensor(
        [tokenizer.byte_offset + ord(str(int(label))) for label in labels],
        dtype=torch.long,
        device=device,
    )
    answer_logits = logits[:, -1]
    predicted = answer_logits.argmax(dim=-1)
    samples = [
        {
            "target": str(int(labels[index])),
            "prediction": tokenizer.decode([int(predicted[index])]),
        }
        for index in range(min(10, examples))
    ]
    return {
        "accuracy": float((predicted == expected).float().mean()),
        "loss": float(F.cross_entropy(answer_logits, expected)),
        "correct": int((predicted == expected).sum()),
        "examples": examples,
        "samples": samples,
    }


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(name)


def build_model(
    architecture: str,
    *,
    language_config: ModelConfig,
    vision_config: MiniVisionConfig,
    device: torch.device,
) -> nn.Module:
    language_model = GLM53FlashFromScratch(language_config)
    if architecture == "faithful":
        model = VisionLanguageModel(language_model, vision_config)
    elif architecture == "direct_patch_baseline":
        model = DirectPatchVisionLanguageModel(language_model, patch_size=8)
    else:
        raise ValueError(f"unknown architecture: {architecture}")
    return model.to(device)


def train_one(
    architecture: str,
    *,
    args: argparse.Namespace,
    language_config: ModelConfig,
    vision_config: MiniVisionConfig,
    tokenizer: ByteTokenizer,
    device: torch.device,
) -> dict[str, object]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = build_model(
        architecture,
        language_config=language_config,
        vision_config=vision_config,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    evaluation_seed = 10_000 + args.seed
    before = held_out_metrics(
        model,
        tokenizer,
        examples=args.eval_examples,
        seed=evaluation_seed,
        device=device,
    )
    started = time.perf_counter()
    final_training_loss = None
    model.train()
    for step in range(args.steps):
        labels = (torch.arange(args.batch_size) + step) % 10
        images = generated_rgb_digit_images(labels, seed=args.seed + step).to(device)
        tokens, prompt_length = text_batch(tokenizer, labels, device)
        targets = answer_only_labels(
            tokens,
            prompt_length=prompt_length,
            visual_token_count=model.visual_token_count,
            boundary_token_count=model.inserted_token_count - model.visual_token_count,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, usage = model(images, tokens[:, :-1])
        language_loss = F.cross_entropy(
            logits.reshape(-1, language_config.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )
        balance = (
            (usage.mean(dim=0) - 1.0 / language_config.experts).square().mean()
            * language_config.experts
        )
        loss = language_loss + 0.01 * balance
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        final_training_loss = float(language_loss.detach())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    after = held_out_metrics(
        model,
        tokenizer,
        examples=args.eval_examples,
        seed=evaluation_seed,
        device=device,
    )
    return {
        "architecture": architecture,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "visual_token_count": model.visual_token_count,
        "inserted_token_count": model.inserted_token_count,
        "before": before,
        "after": after,
        "final_training_loss": final_training_loss,
        "wall_time_seconds": round(elapsed, 3),
    }


def receipt_path(output: Path) -> Path:
    return output if output.suffix == ".json" else output / "receipt.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("faithful", "baseline", "both"),
        default="both",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Cap run size for CI")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--eval-examples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, help="Optional JSON receipt path or directory")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.eval_examples < 10:
        raise ValueError("steps and batch size must be positive; eval examples must be at least 10")
    if args.smoke_test:
        args.steps = min(args.steps, 40)
        args.batch_size = min(args.batch_size, 20)
        args.eval_examples = min(args.eval_examples, 100)

    device = choose_device(args.device)
    if device.type == "cpu":
        torch.use_deterministic_algorithms(True)
    language_config = ModelConfig(
        vocab_size=263,
        dim=32,
        layers=1,
        heads=4,
        expert_hidden=48,
        experts=4,
        top_k=2,
        streams=2,
        sparse_window=8,
        sparse_stride=8,
        max_sequence_length=64,
    )
    vision_config = MiniVisionConfig(
        image_size=32,
        patch_size=4,
        hidden_size=24,
        depth=2,
        heads=4,
        intermediate_size=48,
        spatial_merge_size=2,
        projection_intermediate_size=64,
    )
    tokenizer = ByteTokenizer()
    architectures = {
        "faithful": ["faithful"],
        "baseline": ["direct_patch_baseline"],
        "both": ["faithful", "direct_patch_baseline"],
    }[args.architecture]
    run_started = time.perf_counter()
    results = [
        train_one(
            architecture,
            args=args,
            language_config=language_config,
            vision_config=vision_config,
            tokenizer=tokenizer,
            device=device,
        )
        for architecture in architectures
    ]
    receipt = {
        "schema_version": "1.0",
        "status": "complete",
        "dataset": "generated RGB seven-segment digits",
        "split": {
            "training_seed_schedule": "seed + step",
            "held_out_seed": 10_000 + args.seed,
            "held_out_reused_before_after": True,
            "downloaded_data": False,
        },
        "config": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "training_examples_seen_per_model": args.steps * args.batch_size,
            "eval_examples": args.eval_examples,
            "learning_rate": args.learning_rate,
            "language_model": language_config.to_dict(),
            "vision_model": asdict(vision_config),
        },
        "environment": {
            "device": device.type,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "results": results,
        "total_wall_time_seconds": round(time.perf_counter() - run_started, 3),
        "checkpoint_saved": False,
    }
    if args.output:
        destination = receipt_path(args.output)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
