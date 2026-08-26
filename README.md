# Build & Train GLM-5.3-Flash From Scratch

A readable 25.7M-parameter language model inspired by GLM-5.3-Flash, trained from random initialization and then improved with executable-reward reinforcement learning on one 12 GB GPU.

![Main experiment results](artifacts/charts/main-results.png)

## Result

The frozen confirmation set used unseen function names and was opened only after training duration was selected on dev.

| Confirmation metric | Before RL | After RL |
|---|---:|---:|
| Greedy pass@1, three trained families | 0/24 | **16/24** |
| Sampled pass@8, three trained families | 9/24 | **19/24** |
| Exact sampled completions, three trained families | 13/192 | **88/192** |
| Sampled pass@8, all eight families | 25/64 | **34/64** |

The paired greedy comparison had 16 gains and zero losses on the trained-family confirmation tasks (exact McNemar/binomial p = 0.0000305). The task-level bootstrap interval for the sampled absolute gain was +25.0 to +53.1 percentage points. Those statistics describe this synthetic benchmark; they do not establish broad coding ability.

![Same unseen prompt before and after RL](artifacts/charts/before-after-generation.png)

RL made `increment` and `double` reliable after those operation patterns had already appeared in pretraining. It did not make `even` reliable, and sampled performance regressed on the untrained `square` and `list_sum` families. That interference is part of the result, not hidden.

![Per-family results](artifacts/charts/family-results.png)

## What was built

The real GLM-5.3-Flash has 320B total parameters, 18B active parameters, 45 text layers, hybrid linear/sparse attention, MoE, four-stream mHC, IndexPool, a one-million-token context, and native multimodality. This repository implements a scaled text-only teaching model:

| Component | Released model | This project |
|---|---:|---:|
| Total parameters | 320B | 25,730,592 |
| Estimated active parameters/token | 18B | 9,805,344 |
| Layers | 45 | 12 |
| Attention pattern | 34 linear + 11 sparse | 9 linear + 3 gathered sparse |
| Routed experts | 288, top-8 | 8, top-2 |
| Shared experts | 1 | 1 |
| Residual streams | four-stream mHC | four learned streams, simplified |
| Context | up to 1M | 192 tokens |
| Modalities | text, image, video | text only |

The model uses the released model's high-level `linear, linear, linear, sparse` rhythm. Its ELU-feature linear attention is not GLM's exact KDA implementation; its local-plus-strided gathered attention is not DeepSeek Sparse Attention with an indexer; and its learned stream mixer is not the full manifold-constrained Sinkhorn formulation. This is an educational implementation, not a reproduction.

```text
bytes → embedding → four residual streams
                  ↓
       [linear attention + MoE] × 3
                  ↓
       [sparse attention + MoE] × 1
                  ↓
             repeat × 3
                  ↓
          RMSNorm → tied LM head
```

## Experiment

1. Generate a tiny synthetic Python corpus directly on the GPU; no dataset download is required.
2. Pretrain from random weights for 400 steps. The run consumed 1,381,760 tokens in 63.1 seconds and peaked at 2.139 GiB allocated VRAM.
3. Select checkpoint 100 using dev because checkpoint 400 had saturated the tiny benchmark. The selected checkpoint had seen 345,995 tokens.
4. Run RLOO policy-gradient post-training on 96 prompts across `increment`, `double`, and `even`, with 16 sampled completions per prompt.
5. Give the policy only executable reward: `1` for passing all hidden tests, `0` for valid but wrong code, and `-0.1` for invalid code.
6. Select RL checkpoint 96 from a monotonic dev curve, then open the separate confirmation split once.

The final RL run used 1,536 rollouts, performed 73 reward-bearing updates, took 155.5 seconds, and peaked at 0.936 GiB allocated VRAM on an RTX 3080 Ti.

## Run it

```bash
python3 -m venv /root/glm53-flash-venv
source /root/glm53-flash-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy matplotlib pytest

./run_gpu.sh -m pytest -q
```

Pretrain:

```bash
./run_gpu.sh scripts/train_pretrain.py \
  --output runs/glm53-coding-pretrain-001 \
  --steps 400 \
  --checkpoints 100,200,400 \
  --batch-size 32 \
  --sequence-length 128 \
  --learning-rate 3e-4 \
  --seed 42
```

Post-train with executable reward:

```bash
./run_gpu.sh scripts/train_rl.py \
  --initial-checkpoint runs/glm53-coding-pretrain-001/checkpoint-0100 \
  --output runs/glm53-executable-rloo-diverse-001 \
  --groups 96 \
  --checkpoints 24,48,72,96 \
  --group-size 16 \
  --tasks-per-family 32 \
  --families increment,double,even \
  --temperature 0.35 \
  --learning-rate 5e-5 \
  --train-scope last-block-head \
  --reward-mode binary
```

Evaluate the locked confirmation set:

```bash
./run_gpu.sh scripts/evaluate.py \
  --checkpoint runs/glm53-coding-pretrain-001/checkpoint-0100 \
  --split confirm --per-family 8 \
  --families increment,double,even \
  --output runs/confirm-greedy-pretrain-0100.json

./run_gpu.sh scripts/evaluate.py \
  --checkpoint runs/glm53-executable-rloo-diverse-001/checkpoint-0096 \
  --split confirm --per-family 8 \
  --families increment,double,even \
  --output runs/confirm-greedy-rl-0096.json
```

Every task, prompt, completion, unit-test count, seed, checkpoint hash, timing, and training rollout is preserved under [`artifacts/receipts`](artifacts/receipts). Model weights are intentionally excluded: rerunning both training stages takes roughly four minutes on the measured GPU.

## Repository map

- [`glm53_flash/model.py`](glm53_flash/model.py): hybrid attention, MoE, residual streams, and language model.
- [`glm53_flash/tasks.py`](glm53_flash/tasks.py): synthetic pretraining data and deterministic splits.
- [`glm53_flash/evaluator.py`](glm53_flash/evaluator.py): fail-closed AST and executable unit-test verifier.
- [`scripts/train_pretrain.py`](scripts/train_pretrain.py): random initialization and causal language-model pretraining.
- [`scripts/train_rl.py`](scripts/train_rl.py): batched RLOO with executable reward.
- [`scripts/evaluate.py`](scripts/evaluate.py): greedy pass@1 evaluation.
- [`scripts/evaluate_passk.py`](scripts/evaluate_passk.py): sampled policy evaluation.
- [`REPORT.md`](REPORT.md): complete method, analysis, negative pilots, and limitations.
- [`TUTORIAL.md`](TUTORIAL.md): the beginner course/video lesson.
- [`EXAMPLES.md`](EXAMPLES.md): exact before-and-after generations.

## Sources

- [GLM-5.3-Flash official announcement](https://z.ai/blog/glm-5.3-flash)
- [Official GLM-5.3-Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Official released configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)

## License

MIT. This repository is an independent educational implementation and is not affiliated with Z.ai.
