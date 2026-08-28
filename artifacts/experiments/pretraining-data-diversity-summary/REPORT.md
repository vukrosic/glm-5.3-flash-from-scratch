# Pretraining Data Diversity Experiment

## Research question

At matched training updates, when does seeing more unique program structure improve prediction on unseen compositional code?

## Design

- Model: 248,412 parameters, byte-level, miniature GLM-style hybrid-attention MoE.
- Conditions: repeatedly sample 8 structures versus train across 88 unique structures.
- Held out: 32 expression structures never used for optimization.
- Fixed: initialization within each paired seed, optimizer, batch size, updates, tokenizer, and evaluation set.
- Replication: 10 paired seeds at 50, 100, and 200 updates.
- Primary metric: teacher-forced argmax accuracy over bytes in the held-out target expression.
- Test: exact two-sided paired sign-flip permutation test over seed-level differences.

## Results

| Updates | 8 repeated | 88 diverse | Paired difference | p value |
|---:|---:|---:|---:|---:|
| 50 | 27.8% | 25.9% | -1.9 points | 0.2109 |
| 100 | 43.5% | 45.7% | +2.2 points | 0.1172 |
| 200 | 57.0% | 60.2% | +3.2 points | 0.0137 |

At 50 updates, diversity had not helped. At 100 updates, the estimate turned positive but was not statistically secure. At 200 updates, the diverse condition improved held-out target-byte accuracy by 3.2 percentage points, with exact paired permutation p = 0.0137.

## Conclusion

For this tiny model and synthetic code-composition task, data diversity only became useful after enough optimization. A larger corpus is not automatically better when the training budget is too small to absorb it.

## Limits

- Every condition remained at 0% exact-expression accuracy. The model learned partial token structure, not reliable code generation.
- Training was update-matched, not perfectly token-matched. Average token counts differed by less than about 1.3% at 100 updates.
- This is a synthetic miniature and does not establish the same crossover for frontier-scale pretraining.
- The 50 and 100 update comparisons were not significant under the prespecified paired permutation test.
