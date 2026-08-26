# Course lesson: build, pretrain, and reinforce a language model from scratch

This is a 45–60 minute beginner lesson. The viewer should leave understanding one complete loop:

```text
architecture → random model → pretraining → executable environment
             → sampled rollouts → reward → policy update → held-out evaluation
```

The teaching outcome is not “we reproduced a frontier model.” It is: “I can see every part of a small language-model experiment, run it myself, and tell whether the model actually learned.”

## 0:00–3:00 — Show the result first

Open [`artifacts/charts/before-after-generation.png`](artifacts/charts/before-after-generation.png).

Say:

> We started with a randomly initialized 25.7-million-parameter model. After a minute of pretraining and two and a half minutes of reinforcement learning, the same unseen prompt changed from invalid Python to the correct function. On 24 confirmation tasks from the trained families, greedy accuracy moved from 0 to 16.

Immediately establish the boundary: this model borrows high-level architectural ideas from GLM-5.3-Flash but is not the 320B released model.

## 3:00–10:00 — Build the architecture

Open [`glm53_flash/model.py`](glm53_flash/model.py).

Teach four ideas:

1. Linear attention carries a compact causal prefix state.
2. Every fourth layer uses gathered sparse attention to retrieve selected prior positions.
3. The MoE router activates two of eight routed experts plus one shared expert.
4. Four learned residual streams give information multiple paths through the network.

Draw this:

```text
token bytes
    ↓
embedding copied into four streams
    ↓
linear → linear → linear → sparse
    ↓               each layer has MoE
repeat three times
    ↓
normalize → predict next byte
```

Then show the model receipt:

```text
25,730,592 total parameters
 9,805,344 estimated active parameters/token
12 layers
8 routed experts, top-2
4 residual streams
```

## 10:00–15:00 — Prove the code is real

Run:

```bash
./run_gpu.sh -m pytest -q
```

Explain the five checks:

- tokenizer round trip;
- every trusted reference solution passes;
- the 3:1 attention pattern exists;
- future tokens cannot affect earlier logits;
- active and total parameter counts are sensible.

The point is simple: never train for minutes or hours before proving the model and evaluator can work.

## 15:00–22:00 — Create data without downloading a dataset

Open [`glm53_flash/tasks.py`](glm53_flash/tasks.py).

One pretraining example looks like:

```python
# Complete this Python function.
# Return x plus one.
def increment_abcdxyz(x):
    return x + 1
```

The operation, wording, and function identity are generated deterministically. The split seeds are different:

```text
pretraining examples → next-byte learning
dev examples        → choose duration and settings
RL examples         → collect executable reward
confirmation        → final result, opened once
```

Stress that a separate confirmation set is what prevents us from calling training-set memorization a result.

## 22:00–30:00 — Pretrain from random weights

Run the exact command from [`README.md`](README.md).

Explain causal language modeling with one sentence: the model sees bytes up to position `t` and learns to predict byte `t+1`.

Show the first chart panel:

```text
step 0:   0/8 dev tasks
step 100: 2/8
step 200: 7/8
step 400: 8/8
```

Why use checkpoint 100 for RL? Checkpoint 400 already solves the tiny dev benchmark, so there is no visible headroom for post-training. Checkpoint 100 has learned Python structure but remains weak.

## 30:00–38:00 — Turn code execution into reward

Open [`glm53_flash/evaluator.py`](glm53_flash/evaluator.py).

The verifier does not ask another LLM whether code “looks good.” It runs the candidate under strict syntax restrictions and checks exact outputs.

```text
all tests pass  → +1.0
valid but wrong →  0.0
invalid Python  → -0.1
```

This is the key research mechanism: the computer can grade unlimited rollouts without writing a reference completion into the RL target.

## 38:00–47:00 — Reinforce successful generations

Open [`scripts/train_rl.py`](scripts/train_rl.py).

For each prompt:

1. sample 16 completions;
2. execute each completion;
3. subtract the mean reward of the other 15 samples;
4. increase probability for above-baseline samples;
5. decrease probability for below-baseline samples.

The leave-one-out advantage for sample `i` is:

```text
A_i = reward_i - mean(reward of every other sample)
```

The policy objective is the negative average of `A_i × log probability_i`. No calculus derivation is required for a first lesson: higher-than-peer reward makes a completion more likely next time.

Run the RL command from [`README.md`](README.md). On the measured GPU it took 155.5 seconds for 1,536 rollouts.

Show the dev curve. The policy's exact rollout rate rose monotonically from 4.2% before RL to 45.8% at group 96. This is why checkpoint 96—not an after-the-fact confirmation winner—was selected.

## 47:00–54:00 — Open the confirmation set

Run both greedy evaluation commands. Do not change decoding settings between checkpoints.

Show:

```text
before RL:  0/24 greedy tasks
after RL:  16/24 greedy tasks
```

Then run sampled evaluation and show:

```text
before RL: 13/192 exact rollouts, 9/24 pass@8
after RL:  88/192 exact rollouts, 19/24 pass@8
```

Open [`EXAMPLES.md`](EXAMPLES.md) and read one entire before/after generation aloud.

## 54:00–60:00 — Explain what failed

Open [`artifacts/charts/family-results.png`](artifacts/charts/family-results.png).

The honest conclusion has three parts:

1. Executable reward made the pretrained model's latent `increment` and `double` solutions reliable on new function identities.
2. It failed to teach `even`, because successful exploration stayed too rare.
3. It reduced probability on two untrained skills, showing catastrophic interference.

That leads directly to the next research experiment: compare the current focused RL run with KL-regularized RL or replay that mixes pretraining examples into each update, then test whether it preserves `square` and `list_sum` while retaining the new gains.

## Beginner exercises

1. Change `group_size` from 16 to 4 and predict what happens to reward diversity.
2. Train only the tied language head and compare speed and final accuracy.
3. Add a ninth operation family with three hidden tests.
4. Add KL regularization against checkpoint 100 and measure non-target regression.
5. Replace binary reward with fraction-of-tests-passed and look for reward shortcuts.

Every exercise has one variable, an executable verifier, and a before/after metric—the basic shape of credible AI research.
