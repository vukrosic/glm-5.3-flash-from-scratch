# Executable-Reward Post-Training of a 25.7M-Parameter GLM-5.3-Flash-Inspired Model

## Abstract

We implemented a 25.7M-parameter hybrid-attention mixture-of-experts language model from random initialization, pretrained it on a synthetic Python corpus, and asked whether a short executable-reward reinforcement-learning run could improve its coding policy on unseen task identities. The operation families had appeared in pretraining; RL tested whether scalar execution feedback could make latent solutions more reliable without supplying reference completions. On a confirmation set that was not used for training or checkpoint selection, greedy accuracy across three RL-targeted operation families increased from 0/24 to 16/24. Sampled exact completion rate increased from 13/192 to 88/192, and pass@8 increased from 9/24 to 19/24. Gains came from `increment` and `double`; `even` did not improve greedily. Performance declined on two untrained families, demonstrating interference. The experiment establishes narrow within-family policy improvement—not general coding capability and not a reproduction of GLM-5.3-Flash.

## Research question

Can executable binary reward, without supplying reference completions during RL, measurably improve a barely pretrained small language model on unseen identities from simple coding-task families?

The primary comparison was fixed before the confirmation split was opened:

- policy before RL: pretraining checkpoint 100;
- policy after RL: checkpoint selected by sampled dev performance;
- primary families: `increment`, `double`, `even`;
- primary deterministic metric: greedy executable pass@1;
- secondary policy metrics: exact rollout rate and pass@8 at temperature 0.35;
- confirmation size: eight unseen task identities per family.

## Model

The model contains 25,730,592 parameters, with an architectural estimate of 9,805,344 active parameters per token. It has 12 layers, hidden size 192, six attention heads, eight routed experts with top-2 routing, one always-active shared expert, and four residual streams.

Every fourth layer uses true gathered causal attention over a local window plus strided anchors; the other layers use causal positive-feature linear attention. This preserves the released model's high-level 3:1 linear/sparse pattern at educational scale. The four-stream mixer, linear attention, and sparse attention are simplified mechanisms; they are not exact implementations of mHC, KDA, DSA, or IndexPool.

## Data and splits

All data was generated deterministically on the GPU. No external dataset was downloaded.

The corpus has eight operation families:

- increment an integer;
- double an integer;
- square an integer;
- absolute value;
- clamp to nonnegative;
- test evenness;
- reverse a string;
- sum a list.

Each sample contains a natural-language comment, a randomized function name, a function signature, and—during pretraining only—the reference body. Dev, RL, original final, and confirmation splits use separate deterministic seeds and therefore separate function identities. Descriptions come from two templates per family and test cases are family-specific, so this is within-distribution identity generalization rather than broad algorithmic generalization.

The verifier parses the complete candidate with Python's AST, rejects imports, attributes, loops, recursion, classes, and unsafe calls, executes it with restricted builtins, and compares exact output type and value on three unit tests. Invalid source fails closed.

## Pretraining

The model was randomly initialized and trained by next-byte prediction with AdamW, batch size 32, sequence length 128, and learning rate 3e-4.

| Step | Dev tasks solved | Dev pass@1 |
|---:|---:|---:|
| 0 | 0/8 | 0.0% |
| 100 | 2/8 | 25.0% |
| 200 | 7/8 | 87.5% |
| 400 | 8/8 | 100.0% |

The complete run consumed 1,381,760 tokens in 63.093 seconds and peaked at 2.139 GiB allocated VRAM. Checkpoint 100, after 345,995 tokens, was selected for RL because it showed genuine code learning but retained headroom. The selection was made on dev, not confirmation.

## Reinforcement learning

We used an on-policy REINFORCE estimator with a leave-one-out group baseline (RLOO). For each prompt, the policy sampled 16 completions. Rewards were:

```text
1.0   passes all three executable tests
0.0   parses and runs but is wrong
-0.1  invalid source
```

The model never received the reference completion during RL. It received only the prompt and scalar executable outcome.

The final run used:

- 96 unique RL prompts: 32 identities for each of three families;
- 16 rollouts per prompt, 1,536 total;
- temperature 0.35 and 48-token output cap;
- learning rate 5e-5;
- last hybrid block, final norm, and tied embedding/language head trainable;
- 2,190,152 trainable parameters;
- 73 groups with nonzero reward spread and therefore an optimizer update;
- 417 exact successful rollouts encountered during training;
- 155.482 seconds and 0.936 GiB peak allocated VRAM.

The rollout scorer was vectorized so all 16 completion log-probabilities required one model call rather than 16 sequential calls.

### Dev checkpoint selection

| RL groups | Exact sampled rollouts | Rate |
|---:|---:|---:|
| 0 | 4/96 | 4.2% |
| 24 | 23/96 | 24.0% |
| 48 | 27/96 | 28.1% |
| 72 | 32/96 | 33.3% |
| 96 | 44/96 | 45.8% |

The dev curve was monotonic, so checkpoint 96 was selected. Only then was the confirmation split opened.

## Confirmation results

### Targeted families

| Metric | Before RL | After RL | Change |
|---|---:|---:|---:|
| Greedy tasks solved | 0/24 | **16/24** | +16 tasks |
| Greedy hidden tests | 2/72 | **48/72** | +46 tests |
| Exact sampled rollouts | 13/192 | **88/192** | +39.1 points |
| Tasks solved at 8 samples | 9/24 | **19/24** | +10 tasks |

The paired greedy comparison had 16 gains and zero losses. An exact McNemar/binomial test gives p = 0.0000305 under task-level independence. A deterministic 50,000-resample task bootstrap placed the sampled absolute gain between +25.0 and +53.1 percentage points. Because prompts share templates and operation families, these values should not be interpreted as evidence over a broad task population.

### Per-family sampled exact rate

| Family | Trained with RL | Before | After |
|---|:---:|---:|---:|
| increment | yes | 6/64 | **40/64** |
| double | yes | 4/64 | **45/64** |
| even | yes | 3/64 | 3/64 |
| square | no | **52/64** | 37/64 |
| list_sum | no | **61/64** | 56/64 |
| absolute | no | 0/64 | 0/64 |
| nonnegative | no | 0/64 | 0/64 |
| reverse | no | 0/64 | 0/64 |

Across all eight families, pass@8 increased from 25/64 to 34/64, exact rollout rate from 126/512 to 181/512, and test accuracy from 394/1536 to 552/1536. The net improvement coexists with clear non-target interference.

## Exact behavior

For the unseen prompt:

```python
# Return two times x.
def double_ywkaoot(x):
```

the pre-RL policy's eight samples were all wrong; the first was:

```python
    return x * * 0 0
```

The post-RL policy generated the following in all eight samples:

```python
    return x * 2
```

The latter passed all three hidden tests. Full outputs are in [`EXAMPLES.md`](EXAMPLES.md) and the raw confirmation receipts.

## Pilot history and method development

The successful run was not the first RL attempt. All pilots are preserved.

1. Updating all parameters at temperature 0.8 and learning rate 2e-5 became unstable: dev fell from 2/8 before RL to 0/8 by group 16.
2. Updating all parameters at temperature 0.35 and learning rate 1e-6 was stable but left dev unchanged at 2/8.
3. Updating only the final block at learning rate 2e-5 was also unchanged at 2/8.
4. A 24-group focused run at learning rate 5e-5 moved dev from 2/8 to 3/8, but did not improve the original held-out greedy score.
5. The final run increased task-identity diversity, doubled group size, exposed the tied output head to training, batched policy scoring, and scaled to 96 groups. Its duration was selected on dev; a new confirmation split was then used once.

This history matters: reporting only the successful configuration would conceal the amount of method selection.

## Limitations

- The benchmark is intentionally tiny and synthetically templated.
- “Unseen” means new function identities from known operation families, not unseen algorithms.
- All operation families appeared during pretraining; RL amplified existing latent solutions rather than introducing unseen concepts.
- Only one final training seed and one confirmation seed were run.
- The same three unit cases are reused within each operation family.
- The RL update has no explicit KL penalty; interference was observed.
- `even` remained unsolved greedily despite receiving RL data.
- The architecture experiment does not isolate hybrid attention against a matched dense control.
- The result says nothing about multimodality, long context, frontier coding, or real GLM-5.3-Flash quality.

## Reproducibility

The repository includes source, deterministic data generation, verifier tests, schedules, full rollout receipts, checkpoint SHA-256 hashes, GPU/software receipt, raw generations, and chart-generation code. Weight files are omitted because they are large and the measured training stages complete in roughly four minutes on the recorded RTX 3080 Ti.
