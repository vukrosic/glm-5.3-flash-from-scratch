#!/usr/bin/env python3
"""Bounded held-out RGB digit pilot through the full 25.7M language model.

The script never saves model weights. It optionally initializes the 260-token
language model from a local pretraining checkpoint, expands the tied embedding
to three image-control tokens, and trains only the vision tower plus the last
language-model block, final norm, and tied embedding/output matrix.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glm53_flash import (  # noqa: E402
    ByteTokenizer,
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


def generated_rgb_digits(labels: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Generate shifted, noisy seven-segment RGB digits without downloads."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.zeros((labels.numel(), 3, 32, 32), dtype=torch.float32)
    coordinates = {
        "a": (4, 7, 8, 24),
        "g": (14, 17, 8, 24),
        "d": (25, 28, 8, 24),
        "f": (6, 15, 5, 8),
        "b": (6, 15, 24, 27),
        "e": (16, 26, 5, 8),
        "c": (16, 26, 24, 27),
    }
    for row, label in enumerate(labels.tolist()):
        shift_y = int(torch.randint(-2, 3, (1,), generator=generator))
        shift_x = int(torch.randint(-2, 3, (1,), generator=generator))
        color = 0.55 + 0.45 * torch.rand((3, 1, 1), generator=generator)
        intensity = 0.80 + 0.20 * torch.rand((), generator=generator)
        for segment in SEGMENTS[int(label)]:
            y0, y1, x0, x1 = coordinates[segment]
            y0, y1 = max(0, y0 + shift_y), min(32, y1 + shift_y)
            x0, x1 = max(0, x0 + shift_x), min(32, x1 + shift_x)
            images[row, :, y0:y1, x0:x1] = color * intensity
    images += 0.06 * torch.rand(images.shape, generator=generator)
    return images.clamp_(0, 1)


def build_full_model(
    *, checkpoint: Path | None, device: torch.device
) -> tuple[VisionLanguageModel, dict[str, object]]:
    config = ModelConfig(vocab_size=263, max_sequence_length=64)
    language_model = GLM53FlashFromScratch(config)
    initialization: dict[str, object] = {"source": "random", "checkpoint": None}
    if checkpoint is not None:
        checkpoint_config = ModelConfig(
            **json.loads((checkpoint / "config.json").read_text())
        )
        if checkpoint_config.vocab_size != 260:
            raise ValueError("expected a 260-token local language checkpoint")
        pretrained = GLM53FlashFromScratch(checkpoint_config)
        pretrained.load_state_dict(
            torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True)
        )
        state = pretrained.state_dict()
        expanded = language_model.state_dict()
        for name, tensor in state.items():
            if name in {"embedding.weight", "output.weight"}:
                expanded[name][:260].copy_(tensor)
            else:
                expanded[name].copy_(tensor)
        language_model.load_state_dict(expanded)
        initialization = {"source": "local_pretraining_checkpoint", "checkpoint": str(checkpoint)}

    vision_config = MiniVisionConfig(
        image_size=32,
        patch_size=4,
        hidden_size=24,
        depth=2,
        heads=4,
        intermediate_size=48,
        spatial_merge_size=2,
        projection_intermediate_size=128,
    )
    model = VisionLanguageModel(language_model, vision_config).to(device)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.vision_encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in model.language_model.layers[-1].parameters():
        parameter.requires_grad_(True)
    for parameter in model.language_model.final_norm.parameters():
        parameter.requires_grad_(True)
    model.language_model.embedding.weight.requires_grad_(True)
    return model, initialization


def text_rows(
    tokenizer: ByteTokenizer, labels: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, int]:
    prompt = tokenizer.encode("Digit: ", bos=True)
    rows = [prompt + tokenizer.encode(str(int(label))) for label in labels]
    return torch.tensor(rows, dtype=torch.long, device=device), len(prompt)


@torch.no_grad()
def evaluate(
    model: VisionLanguageModel,
    tokenizer: ByteTokenizer,
    *,
    examples: int,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    labels = torch.arange(examples) % 10
    images = generated_rgb_digits(labels, seed=seed).to(device)
    prompt = tokenizer.encode("Digit: ", bos=True)
    prompts = torch.tensor([prompt] * examples, dtype=torch.long, device=device)
    logits, _ = model(images, prompts)
    expected = torch.tensor(
        [tokenizer.byte_offset + ord(str(int(label))) for label in labels],
        dtype=torch.long,
        device=device,
    )
    answer_logits = logits[:, -1].float()
    predicted = answer_logits.argmax(dim=-1)
    correct = predicted == expected
    return {
        "accuracy": float(correct.float().mean()),
        "correct": int(correct.sum()),
        "examples": examples,
        "cross_entropy": float(F.cross_entropy(answer_logits, expected)),
        "predictions_first_20": [
            tokenizer.decode([int(token)]) for token in predicted[:20].cpu()
        ],
        "targets_first_20": [str(int(label)) for label in labels[:20]],
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--eval-examples", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--max-seconds", type=float, default=540.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.steps, args.batch_size, args.eval_examples) < 1:
        raise ValueError("steps, batch size, and eval examples must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    model, initialization = build_full_model(checkpoint=args.checkpoint, device=device)
    tokenizer = ByteTokenizer()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    language_parameters = sum(
        parameter.numel() for parameter in model.language_model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    held_out_seed = args.seed + 100_000
    before = evaluate(
        model, tokenizer, examples=args.eval_examples, seed=held_out_seed, device=device
    )
    started = time.perf_counter()
    losses: list[dict[str, float | int]] = []
    stop_reason = "step_cap"
    completed_steps = 0
    for step in range(args.steps):
        if time.perf_counter() - started >= args.max_seconds:
            stop_reason = "time_cap"
            break
        model.train()
        labels = (torch.arange(args.batch_size) + step * 3) % 10
        images = generated_rgb_digits(labels, seed=args.seed + step).to(device)
        tokens, prompt_length = text_rows(tokenizer, labels, device)
        targets = answer_only_labels(
            tokens,
            prompt_length=prompt_length,
            visual_token_count=model.visual_token_count,
            boundary_token_count=2,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, usages = model(images, tokens[:, :-1])
        language_loss = F.cross_entropy(
            logits.reshape(-1, model.language_model.config.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )
        balance = (
            (usages[-1] - 1.0 / model.language_model.config.experts)
            .square()
            .mean()
            * model.language_model.config.experts
        )
        loss = language_loss + 0.01 * balance
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        completed_steps = step + 1
        if step == 0 or (step + 1) % 5 == 0 or step + 1 == args.steps:
            losses.append(
                {
                    "step": step + 1,
                    "language_loss": round(float(language_loss.detach()), 6),
                    "gradient_norm": round(float(gradient_norm), 6),
                }
            )
    synchronize(device)
    elapsed = time.perf_counter() - started
    after = evaluate(
        model, tokenizer, examples=args.eval_examples, seed=held_out_seed, device=device
    )
    receipt = {
        "schema_version": "1.0",
        "status": "complete",
        "experiment": "full_25m_language_model_rgb_digit_integration",
        "claim_tested": "A miniature GLM-like vision encoder can learn a held-out RGB digit task while its tokens pass through the full 12-layer 25.7M language model.",
        "initialization": initialization,
        "environment": {
            "device": device.type,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "model": {
            "language_model_parameters": language_parameters,
            "total_multimodal_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "language_layers_executed": len(model.language_model.layers),
            "visual_tokens": model.visual_token_count,
            "trained_components": [
                "vision_encoder",
                "language_model.layers[-1]",
                "language_model.final_norm",
                "tied_embedding_output",
            ],
            "frozen_components": "language_model.layers[0:11]",
        },
        "data": {
            "source": "procedurally generated RGB seven-segment digits",
            "downloaded": False,
            "training_examples_seen": completed_steps * args.batch_size,
            "training_seed_schedule": "seed + step",
            "held_out_examples": args.eval_examples,
            "held_out_seed": held_out_seed,
            "held_out_reused_before_after": True,
        },
        "optimization": {
            "requested_steps": args.steps,
            "completed_steps": completed_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "max_seconds": args.max_seconds,
            "stop_reason": stop_reason,
            "wall_time_seconds": round(elapsed, 3),
            "loss_trace": losses,
        },
        "before": before,
        "after": after,
        "learned_at_all": after["correct"] > before["correct"],
        "checkpoint_saved": False,
        "limitations": [
            "One synthetic task and one seed; this is an integration pilot, not general vision evidence.",
            "The first eleven language-model layers were frozen but still executed in every forward and backward pass.",
            "Only a small number of procedural training images were seen.",
            "The held-out set changes image rendering randomness, not the ten semantic digit classes.",
            "No architecture comparison or statistical significance claim is made.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["learned_at_all"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
