# Full 25.7M Language-Model Vision Integration Pilot

## Question

Can the existing miniature GLM-like vision encoder learn a held-out RGB digit
task when its 16 visual tokens are passed through the complete 12-layer,
25.7M-parameter language model?

## Bounded setup

- Language initialization: local step-100 coding-pretraining checkpoint
- Language-model parameters after adding three image-control tokens: 25,731,168
- Complete multimodal model: 25,875,576 parameters
- Trainable surface: 2,335,136 parameters
- Trainable components: vision encoder, final LM block, final norm, and tied
  embedding/output matrix
- Frozen but executed components: LM blocks 1 through 11
- Device: Apple Silicon MPS
- Training: 40 updates, batch size 10, learning rate 0.001
- Training images seen: 400 procedurally generated 32 x 32 RGB digits
- Evaluation: the same fixed 100-image held-out set before and after training
- Held-out variation: unseen shifts, colors, intensities, and pixel noise
- Checkpoint saved: no

## Result

| Metric | Before | After |
|---|---:|---:|
| Held-out accuracy | 0/100 (0%) | 40/100 (40%) |
| Held-out cross-entropy | 8.3894 | 1.5550 |

Training cross-entropy fell from 8.3763 at update 1 to 1.5442 at update 40.
The bounded training loop took 37.007 seconds, excluding model construction and
the before/after evaluations.

This is a successful integration pilot: held-out image accuracy increased, and
the learned visual tokens were consumed by all 12 language-model layers.

## What this does not establish

- It is one synthetic task and one seed, not evidence of general vision ability.
- The ten semantic digit classes are shared across train and evaluation; only
  image renderings are held out.
- Most language-model parameters were frozen, although the full network was
  executed and gradients passed through it to the vision encoder.
- There is no matched baseline, repeated-seed uncertainty estimate, or
  architecture-ranking claim in this pilot.
- No trained weights were retained, so the run demonstrates learnability and
  integration rather than delivering a reusable vision checkpoint.

## Reproduction command

```bash
.venv/bin/python experiments/full_25m_vision_digit_pilot.py \
  --device auto \
  --steps 40 \
  --batch-size 10 \
  --eval-examples 100 \
  --learning-rate 0.001 \
  --max-seconds 540 \
  --checkpoint runs/mac-pretrain-before-after-001/checkpoint-0100 \
  --output artifacts/receipts/full-25m-vision-digit-pilot-seed73.json
```

The script refuses to overwrite an existing receipt and contains no checkpoint
saving path.
