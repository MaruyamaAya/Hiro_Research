# Publication Experiment Matrix

This file is the execution contract for the real-model study. A condition is
not considered complete until all required seeds, checkpoints, evaluations,
and provenance records exist.

## Fixed model and data

- Base model: platform-mounted Qwen3.5-9B, with the resolved checkpoint path and
  model-file hashes stored in every run directory.
- Training: prepared DAPO-Math-17K subset recorded in
  `data/manifests/real_math_manifest.json`.
- Calibration/selection: deterministic 5% DAPO validation split.
- Held-out: MATH-500 and GSM8K test; no held-out prompt may enter training.
- Primary inference budget: 1024 new tokens; secondary sensitivity budget: 2048.
- LoRA target modules and rank are fixed across all conditions.

## Conditions

| ID | Loss | Reward | Curriculum | Purpose |
|---|---|---|---|---|
| G-U | GRPO | outcome | uniform | original family baseline |
| D-U | Dr.GRPO | outcome | uniform | removes length/question bias |
| A-U | DAPO | outcome | uniform | token-level baseline |
| A-HD | DAPO | outcome | historical zero-gradient filtering | practical online approximation |
| A-H | DAPO | outcome | Hiro + historical filtering | primary method |
| A-C | DAPO | outcome | challenge only | Hiro ablation |
| A-P | DAPO | outcome | progress only | Hiro ablation |
| A-E | DAPO | outcome + effort | uniform | reward-hacking control |

## Run ladder

1. One seed, five optimizer steps: interface and exact-resume integration.
2. One seed, 20 steps: memory, completion length, verifier, and gradient audit.
3. One seed, 100--200 steps: hyperparameter calibration using training-only
   diagnostics and a validation split carved from training data.
4. Three independent seeds per frozen condition for the final comparison.

Final-test results must not be used to tune the method.

## Required logging

Every optimizer/checkpoint interval records:

- outcome reward, format reward, overlong penalty, and total reward;
- completion length, truncation ratio, valid-answer ratio;
- zero-variance/zero-gradient prompt-group ratio;
- KL, clipping low/high fractions, gradient norm, and learning rate;
- curriculum probability, attempts, pass rate, prior-window pass rate,
  signed progress, and coverage for every bucket;
- wall time, GPU model/count, peak allocated memory, and GPU-hours.

## Evaluation

Evaluate base and every saved checkpoint with identical prompts and seed lists.
Report:

- pass@1 and empirical pass@k;
- MATH-500 by official level and subject;
- GSM8K accuracy;
- completion tokens, truncation, invalid/ambiguous/no-answer rates;
- accuracy under both fixed inference budgets;
- easy-bucket forgetting and high-bucket gains;
- verifier disagreement on a manually labeled audit sample;
- effort baseline reward/accuracy divergence.

## Statistical analysis

- Treat training seed as the independent replicate.
- Report mean, standard deviation, and seed-level values.
- Use paired seed bootstrap confidence intervals for checkpoint-final and
  learning-curve-area differences.
- Use paired per-item bootstrap intervals for held-out accuracy differences,
  stratified by benchmark.
- Report all planned conditions, including failed or unstable runs.
- Do not claim superiority when the confidence interval includes zero; report
  the result as inconclusive.

## Publication gate

The paper may be called complete only when:

1. every final condition has at least three valid independent seeds;
2. exact resume has been tested on the deployed distributed stack;
3. all final checkpoints and curriculum states are on persistent storage;
4. the verifier audit includes both deterministic controls and manual labels;
5. all tables/figures regenerate from checked-in analysis scripts;
6. limitations include proxy-bucket validity, LoRA-only scope, model choice,
   compute scale, benchmark contamination risk, and verifier error;
7. the ACM two-column source compiles without errors.
