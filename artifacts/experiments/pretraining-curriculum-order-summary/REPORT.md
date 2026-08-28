# Pretraining Curriculum and Ordering Experiments

## Experiment 1: does example order matter?

The blocked and interleaved conditions use the exact same 4,800 generated examples, model initialization within seed, optimizer, 200 updates, batch size, and held-out structures. Only presentation order changes.

- Blocked mean: 50.5%
- Interleaved mean: 60.2%
- Paired difference: +9.7 percentage points
- Bootstrap 95% interval: [7.6, 11.7] points
- Exact paired permutation p: 0.001953
- Seeds improved: 10 / 10

Conclusion: interleaving the same diverse examples substantially improved held-out target-byte accuracy. Long homogeneous blocks likely create stronger recency bias or forgetting in this tiny model.

## Experiment 2: does an 8-to-88 curriculum help?

The curriculum spends the first 100 updates cycling over 8 structures, then expands to 88 for the final 100 updates.

- Repeated 8 mean: 57.0%
- Curriculum mean: 59.6%
- Diverse from the start mean: 60.2%
- Curriculum minus diverse: -0.6 points
- Exact paired permutation p: 0.2148

Conclusion: the curriculum improved over repeating only 8 structures, but did not beat diverse interleaving from the start. The measured curriculum disadvantage of 0.6 points was not statistically significant.

## Limits

- Exact-expression accuracy remained 0% in every condition. These experiments measure partial token learning, not reliable executable generation.
- The model has 248,412 parameters and the task is synthetic. The effect should be replicated on natural code before generalizing it.
- The forgetting explanation is plausible but not directly measured. Hidden-state or per-structure trajectory analysis would be needed to establish the mechanism.
