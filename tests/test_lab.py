from __future__ import annotations

import torch

from glm53_flash import ByteTokenizer, GLM53FlashFromScratch, ModelConfig
from glm53_flash.evaluator import evaluate_source
from glm53_flash.model import SparseAttention
from glm53_flash.tasks import frozen_tasks


def tiny_config() -> ModelConfig:
    return ModelConfig(dim=32, layers=4, heads=4, expert_hidden=48, experts=4, top_k=2, streams=4, sparse_window=8, sparse_stride=8, max_sequence_length=64)


def test_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "def f(x):\n    return x + 1\n"
    assert tokenizer.decode(tokenizer.encode(text, bos=True, eos=True)) == text


def test_all_frozen_references_pass():
    for split in ("dev", "final", "rl", "confirm"):
        for task in frozen_tasks(split, per_family=1):
            result = evaluate_source(task, task.reference_source)
            assert result.passed, task.task_id
            assert not evaluate_source(task, task.prompt + "\n    pass\n").passed


def test_split_function_identities_are_disjoint():
    identities = {
        split: {task.entry_point for task in frozen_tasks(split, per_family=8)}
        for split in ("dev", "rl", "final", "confirm")
    }
    for left_index, left in enumerate(identities):
        for right in list(identities)[left_index + 1 :]:
            assert identities[left].isdisjoint(identities[right])


def test_architecture_has_three_to_one_pattern():
    model = GLM53FlashFromScratch(tiny_config())
    sparse = [isinstance(layer.block.attention, SparseAttention) for layer in model.layers]
    assert sparse == [False, False, False, True]


def test_forward_shape_usage_and_causality():
    torch.manual_seed(0)
    model = GLM53FlashFromScratch(tiny_config()).eval()
    first = torch.randint(0, 260, (2, 12))
    second = first.clone()
    second[:, 8:] = torch.randint(0, 260, (2, 4))
    logits_a, usage = model(first)
    logits_b, _ = model(second)
    assert logits_a.shape == (2, 12, 260)
    assert usage.shape == (4, 4)
    assert torch.allclose(usage.sum(dim=-1), torch.ones(4))
    assert torch.allclose(logits_a[:, :8], logits_b[:, :8], atol=1e-5, rtol=1e-5)


def test_router_balance_loss_has_router_gradients():
    torch.manual_seed(0)
    model = GLM53FlashFromScratch(tiny_config())
    _, usage = model(torch.randint(0, 260, (2, 12)))
    balance = ((usage.mean(dim=0) - 1.0 / model.config.experts) ** 2).mean()

    balance.backward()

    for layer in model.layers:
        gradient = layer.block.moe.router.weight.grad
        assert gradient is not None
        assert torch.count_nonzero(gradient) > 0


def test_parameter_count_is_reported():
    counts = GLM53FlashFromScratch(tiny_config()).parameter_counts()
    assert 0 < counts["active_per_token_estimate"] < counts["total"]
