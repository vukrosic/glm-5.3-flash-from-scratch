# Exact model generations

These are copied from the immutable confirmation receipts. No output was rewritten or repaired by hand.

## Example 1: double an integer

Prompt:

```python
# Complete this Python function.
# Return two times x.
def double_ywkaoot(x):
```

### Before RL — checkpoint 100

All eight sampled completions failed:

```text
sample 0  FAIL  "\n    return x * * 0 0\n"
sample 1  FAIL  "\n    return x * * 0 ======== elselselse x\n"
sample 2  FAIL  "\n    return x * x\n"
sample 3  FAIL  "\n    return x * x\n"
sample 4  FAIL  "\n    return x * x\n"
sample 5  FAIL  "\n    return x * * 0 0\n"
sample 6  FAIL  "\n    return x * * 0 0\n"
sample 7  FAIL  "\n    return x * * 0 0\n"
```

### After RL — checkpoint 96

All eight sampled completions passed all three tests:

```text
sample 0  PASS  "\n    return x * 2\n"
sample 1  PASS  "\n    return x * 2\n"
sample 2  PASS  "\n    return x * 2\n"
sample 3  PASS  "\n    return x * 2\n"
sample 4  PASS  "\n    return x * 2\n"
sample 5  PASS  "\n    return x * 2\n"
sample 6  PASS  "\n    return x * 2\n"
sample 7  PASS  "\n    return x * 2\n"
```

## Example 2: increment an integer

Prompt:

```python
# Complete this Python function.
# Return x plus one.
def increment_jgsmuqb(x):
```

### Before RL

```text
sample 0  FAIL  "\n    return x * 1\n"
sample 1  FAIL  "\n    return x * 1\n"
sample 2  FAIL  "\n    return x * * x\n"
sample 3  FAIL  "\n    return x * 1\n"
sample 4  FAIL  "\n    return x * x\n"
sample 5  FAIL  "\n    return x * 1\n"
sample 6  FAIL  "\n    return x * 1\n"
sample 7  FAIL  "\n    return x * 1\n"
```

### After RL

```text
sample 0  FAIL  "\n    return x * + x\n"
sample 1  FAIL  "\n    return x * 1\n"
sample 2  FAIL  "\n    return x * 1\n"
sample 3  PASS  "\n    return x + 1\n"
sample 4  PASS  "\n    return x + 1\n"
sample 5  FAIL  "\n    return x * 1\n"
sample 6  PASS  "\n    return x + 1\n"
sample 7  FAIL  "\n    return x * 1\n"
```

## Raw evidence

- Before sampled: [`artifacts/receipts/runs/confirm-pass8-pretrain-0100.json`](artifacts/receipts/runs/confirm-pass8-pretrain-0100.json)
- After sampled: [`artifacts/receipts/runs/confirm-pass8-rl-0096.json`](artifacts/receipts/runs/confirm-pass8-rl-0096.json)
- Before greedy: [`artifacts/receipts/runs/confirm-greedy-pretrain-0100.json`](artifacts/receipts/runs/confirm-greedy-pretrain-0100.json)
- After greedy: [`artifacts/receipts/runs/confirm-greedy-rl-0096.json`](artifacts/receipts/runs/confirm-greedy-rl-0096.json)
