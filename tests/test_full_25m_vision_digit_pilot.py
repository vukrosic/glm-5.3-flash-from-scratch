from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "full_25m_vision_digit_pilot.py"
SPEC = importlib.util.spec_from_file_location("full_25m_vision_digit_pilot", SCRIPT)
assert SPEC and SPEC.loader
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def test_full_language_model_is_attached_and_only_bounded_surface_is_trainable():
    model, initialization = PILOT.build_full_model(checkpoint=None, device=torch.device("cpu"))
    assert initialization["source"] == "random"
    assert len(model.language_model.layers) == 12
    assert sum(parameter.numel() for parameter in model.language_model.parameters()) == 25_731_168
    assert model.visual_token_count == 16
    assert not any(
        parameter.requires_grad
        for layer in model.language_model.layers[:-1]
        for parameter in layer.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.vision_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.language_model.layers[-1].parameters())


def test_procedural_train_and_held_out_seeds_change_pixels():
    labels = torch.arange(10)
    train = PILOT.generated_rgb_digits(labels, seed=73)
    held_out = PILOT.generated_rgb_digits(labels, seed=100_073)
    assert train.shape == (10, 3, 32, 32)
    assert float(train.min()) >= 0.0 and float(train.max()) <= 1.0
    assert not torch.equal(train, held_out)
