# Miniature GLM-5.3-Flash vision path

## Result

The architecture-faithful teaching path is implemented and learns on held-out generated RGB digits. It did not beat the simpler direct-patch baseline in this short, single-seed run.

| Model | Parameters | Before held-out accuracy | After held-out accuracy | Before loss | After loss | Training wall time |
|---|---:|---:|---:|---:|---:|---:|
| Faithful miniature | 60,924 | 0/200 | 119/200 | 5.580904 | 1.161805 | 2.517 s |
| Direct-patch baseline | 42,500 | 0/200 | 189/200 | 5.575836 | 0.960843 | 1.112 s |

Both arms used seed `42`, 120 optimizer steps, batch size 40, learning rate `0.002`, 4,800 generated training examples, and the same 200 held-out renderings from seed `10042`. The whole command took 4.285 seconds on CPU. No dataset was downloaded and no model checkpoint was saved. The machine used macOS arm64, Python 3.12.9, and PyTorch 2.5.1.

The baseline result is stronger by 70/200 examples in this run. It also has fewer parameters. This is a useful negative result: the added transformer, merger, and projector structure did not pay for itself under the tiny dimensions and 120-step budget. One synthetic dataset and one seed cannot establish a general architecture ranking.

## Exact commands

```bash
.venv/bin/python -m pytest -q

.venv/bin/python scripts/train_vision.py \
  --architecture both \
  --device cpu \
  --seed 42 \
  --steps 120 \
  --batch-size 40 \
  --eval-examples 200 \
  --learning-rate 0.002 \
  --output artifacts/receipts/vision-miniature-rgb.json
```

The complete machine-readable receipt is [`artifacts/receipts/vision-miniature-rgb.json`](artifacts/receipts/vision-miniature-rgb.json).

## Exact miniature configuration

| Setting | Value |
|---|---:|
| Input | 32 x 32 RGB |
| Patch size | 4 x 4 |
| Patch grid | 8 x 8, 64 tokens |
| Vision width | 24 |
| Vision blocks | 2 |
| Vision heads | 4 |
| Vision MLP width | 48 |
| Spatial merge | 2 x 2 rearrangement plus learned linear projection |
| Merged grid | 4 x 4, 16 tokens |
| Projector width | 64 |
| LM width | 32 |
| LM layers | 1 |
| LM vocabulary | 263: 260 byte-model IDs plus three image IDs |
| Multimodal order | `BOS <image_start> <image> x 16 <image_end> prompt` |
| Objective | answer-only autoregressive next-byte cross-entropy |

## Component audit

The production details below were checked against the [official pinned configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/04c4e9e95c5da8862dced7e5056455116f83a7e0/config.json) and the [pinned Hugging Face Transformers implementation](https://github.com/huggingface/transformers/blob/323f24345ed92f88e68a14aeef7db78ee2b52475/src/transformers/models/glm5_next/modeling_glm5_next.py).

| Official GLM feature | Miniature implementation | Status | Why |
|---|---|---|---|
| Three-channel RGB input | `[batch, 3, 32, 32]` tensors | Faithful | Input modality and channel semantics match. |
| 3D temporal-spatial patch convolution with temporal size 2 and spatial size 14 | 2D image-only convolution with patch size 4 | Simplified | It preserves learned non-overlapping RGB patch embedding but omits video time. |
| 24 vision blocks, width 1024, 16 heads | Two blocks, width 24, four heads | Scaled down | Depth and dimensions are reduced for fast CPU and MPS teaching runs. |
| Bidirectional vision attention with query/key RMSNorm | Full non-causal attention with per-head query/key RMSNorm | Faithful | Attention direction and normalization topology match. |
| 2D rotary vision positions | Learned 8 x 8 positional table | Simplified | Spatial order is represented, but not with the production rotary mechanism. |
| Pre-normalized attention and gated SwiGLU MLP residual blocks | RMSNorm, attention residual, RMSNorm, SwiGLU residual | Faithful | The block-level computation order matches. |
| Post-vision RMSNorm | Post-stack RMSNorm | Faithful | Normalization occurs before spatial downsampling. |
| Learned 2 x 2 strided convolution into LM output width | Non-overlapping 2 x 2 rearrangement plus a linear map from width 24 to width 32 | Simplified | This is mathematically the same local linear operation as the stride-2 convolution, expressed without Conv2d so PyTorch 2.5 MPS backward works. |
| Projection, LayerNorm, GELU, gated projection MLP | Same projector sequence with intermediate width 64 | Scaled down | The multimodal projection topology is retained with tiny dimensions. |
| Explicit image start, repeated placeholder, and image end token IDs | IDs 260, 261, and 262 around 16 placeholders | Faithful | Boundary and placeholder roles are explicit and independently testable. |
| Validate placeholder count, then scatter image features into token embeddings | Per-row count validation followed by masked scatter | Faithful | Integration semantics match the released model path. |
| Feed replaced embeddings into the autoregressive language model | Reuse `GLM53FlashFromScratch.forward_embeddings` | Faithful | Image and byte embeddings share one causal prediction stream. |
| Variable image grids and packed attention lengths | Fixed 32 x 32 images | Missing | Dynamic resolution and packed multimodal batches would obscure the small teaching path. |
| Video temporal patching and video boundaries | Image path only | Missing | The task requested a miniature vision path, not video support. |
| Frontier-scale pretraining and broad multimodal evaluation | Generated seven-segment digits and one-byte answers | Missing | This is a mechanism and learning smoke check, not a frontier capability claim. |

## Tests

`tests/test_vision.py` covers:

- RGB patch, transformer, merge, and projector shapes.
- Exact token order and placeholder replacement.
- Answer-label alignment.
- Nonzero finite gradients through patch embedding, vision attention, spatial merger, and projector.
- A deterministic two-example overfit check.

The complete repository suite passed: `10 passed`.
