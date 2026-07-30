# Concrete Experiment Plan

## Phase A — Controlled HiroWorld benchmark

### Main conditions

- Extrinsic
- Novelty
- Surprise/prediction-error
- Suffering/effort
- Difficulty seeking
- Full Hiro reward

### Ablations

- Hiro without learning progress
- Hiro without challenge matching
- Hiro without safety constraint
- Hiro without external reward
- Hiro with additive effort

Each condition uses 50 common random seeds and 20,000 decisions. Report:

- final skill and learning progress,
- external return,
- door-choice ratios,
- noisy-TV ratio,
- meaningless-suffering ratio,
- catastrophic-choice rate,
- challenge-appropriateness error,
- curriculum trajectory,
- 95% bootstrap confidence intervals.

### Acceptance checks

The implementation is considered behaviorally valid only if:

1. surprise chooses Noisy TV substantially more than Hiro;
2. suffering chooses treadmill substantially more than Hiro;
3. difficulty seeking chooses impossible tasks substantially more than safe Hiro;
4. Hiro achieves higher challenge skill than extrinsic-only;
5. removing learning progress or safety causes the predicted failure mode.

## Phase B — Qwen3.5-9B preference representation

Generate controlled trajectory pairs crossing:

- learnable vs random uncertainty,
- productive vs meaningless effort,
- matched vs impossible difficulty,
- first failure vs repeated stagnant failure,
- mastered repetition vs frontier challenge.

Use held-out templates and held-out numeric ranges to prevent lexical
memorization.

### Conditions

1. Qwen3.5-9B zero-shot, outcome-only prompt.
2. Qwen3.5-9B zero-shot, history-aware prompt.
3. Qwen3.5-9B preference fine-tuning, outcome-only input.
4. Qwen3.5-9B preference fine-tuning, history-aware input.
5. Formula oracle upper bound and simple heuristic baselines.

### Metrics

- pairwise accuracy,
- per-category accuracy,
- calibration/Brier score,
- consistency under A/B order swapping,
- robustness to paraphrase,
- accuracy on counterfactual history pairs where the current outcome is
  identical but prior learning progress differs.

The primary hypothesis is that history-aware models outperform outcome-only
models especially on counterfactual-history pairs.

## Compute allocation

- Node 1: main preference fine-tuning/evaluation.
- Node 2: independent replicate, ablations, and generation/evaluation jobs.
- HiroWorld simulation runs on CPU while `gpu_hold` remains active.
- Every GPU command must be wrapped by `scripts/with_gpu_hold.sh`.
