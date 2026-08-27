from __future__ import annotations

import torch
import torch.nn.functional as F

from glm53_flash import (
    ByteTokenizer,
    GLM53FlashFromScratch,
    ImageTokenIds,
    MiniVisionConfig,
    ModelConfig,
    VisionLanguageModel,
    answer_only_labels,
    build_multimodal_input_ids,
)


def tiny_model(*, image_size: int = 16) -> VisionLanguageModel:
    language_config = ModelConfig(
        vocab_size=263,
        dim=16,
        layers=1,
        heads=2,
        expert_hidden=24,
        experts=2,
        top_k=1,
        streams=2,
        sparse_window=8,
        sparse_stride=8,
        max_sequence_length=64,
    )
    vision_config = MiniVisionConfig(
        image_size=image_size,
        patch_size=4,
        hidden_size=8,
        depth=2,
        heads=2,
        intermediate_size=16,
        projection_intermediate_size=24,
    )
    return VisionLanguageModel(GLM53FlashFromScratch(language_config), vision_config)


def test_rgb_patch_transformer_merge_and_projector_shapes():
    model = tiny_model()
    images = torch.randn(3, 3, 16, 16)
    patches, grid = model.vision_encoder.patch_embedding(images)
    assert patches.shape == (3, 16, 8)
    assert grid == (4, 4)
    assert len(model.vision_encoder.blocks) == 2

    hidden = patches
    for block in model.vision_encoder.blocks:
        hidden = block(hidden)
        assert hidden.shape == (3, 16, 8)
    merged = model.vision_encoder.spatial_merger(
        model.vision_encoder.post_layernorm(hidden), grid
    )
    assert merged.shape == (3, 4, 16)
    projected = model.vision_encoder.projector(merged)
    assert projected.shape == (3, 4, 16)
    assert model.vision_encoder(images).shape == (3, 4, 16)


def test_explicit_image_token_order_and_placeholder_replacement():
    model = tiny_model()
    specials = ImageTokenIds()
    text = torch.tensor([[1, 10, 11]])
    multimodal = build_multimodal_input_ids(
        text, visual_token_count=model.visual_token_count, token_ids=specials
    )
    assert multimodal.tolist() == [
        [1, specials.start, specials.placeholder, specials.placeholder,
         specials.placeholder, specials.placeholder, specials.end, 10, 11]
    ]

    images = torch.randn(1, 3, 16, 16)
    visual = model.vision_encoder(images)
    embedded = model.embed_multimodal(images, multimodal)
    mask = multimodal == specials.placeholder
    assert torch.allclose(embedded[mask], visual.reshape(-1, 16))
    assert torch.allclose(embedded[:, 0], model.language_model.embedding(text[:, 0]))
    assert torch.allclose(
        embedded[:, 1], model.language_model.embedding(multimodal[:, 1])
    )


def test_logits_labels_and_gradients_cover_the_full_vision_path():
    torch.manual_seed(7)
    model = tiny_model()
    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode("Digit: ", bos=True)
    tokens = torch.tensor([prompt + tokenizer.encode("7")])
    logits, usage = model(torch.randn(1, 3, 16, 16), tokens[:, :-1])
    targets = answer_only_labels(
        tokens,
        prompt_length=len(prompt),
        visual_token_count=model.visual_token_count,
    )
    assert logits.shape == (1, model.inserted_token_count + tokens.shape[1] - 1, 263)
    assert usage.shape == (1, 2)
    assert int((targets != -100).sum()) == 1
    assert targets[0, -1:].tolist() == tokenizer.encode("7")

    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
    loss.backward()
    gradient_parameters = (
        model.vision_encoder.patch_embedding.proj.weight,
        model.vision_encoder.blocks[0].attention.qkv.weight,
        model.vision_encoder.spatial_merger.downsample.weight,
        model.vision_encoder.projector.down_proj.weight,
    )
    for parameter in gradient_parameters:
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
        assert float(parameter.grad.abs().sum()) > 0


def test_tiny_two_example_overfit_smoke():
    torch.manual_seed(11)
    model = tiny_model()
    tokenizer = ByteTokenizer()
    prompt = tokenizer.encode("Digit: ", bos=True)
    labels = torch.tensor([0, 1])
    tokens = torch.tensor([prompt + tokenizer.encode(str(int(label))) for label in labels])
    images = torch.zeros(2, 3, 16, 16)
    images[0, 0, 2:14, 2:6] = 1.0
    images[1, 1, 2:14, 10:14] = 1.0
    targets = answer_only_labels(
        tokens,
        prompt_length=len(prompt),
        visual_token_count=model.visual_token_count,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    model.train()
    with torch.no_grad():
        initial_logits, _ = model(images, tokens[:, :-1])
        initial_loss = F.cross_entropy(
            initial_logits.flatten(0, 1), targets.flatten(), ignore_index=-100
        )
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images, tokens[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_logits, _ = model(images, tokens[:, :-1])
        final_loss = F.cross_entropy(
            final_logits.flatten(0, 1), targets.flatten(), ignore_index=-100
        )
        expected = tokens[:, -1]
        predicted = final_logits[:, -1].argmax(dim=-1)
    assert float(final_loss) < float(initial_loss) * 0.25
    assert predicted.tolist() == expected.tolist()
