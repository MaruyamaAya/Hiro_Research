# Hiro Reward in Modern LLM RL

## Resource conclusion

The available allocation is sufficient for serious Qwen3.5-9B RL experiments:

- two nodes;
- 8× H20 96GB per node;
- 16 GPUs / roughly 1.5 TB total VRAM.

Recommended scope:

- **Primary:** LoRA GRPO/DAPO-style RL on Qwen3.5-9B.
- **Secondary:** full-parameter short-run GRPO/FSDP validation.
- **Not initially recommended:** long-horizon full-parameter PPO with a
  separately trained 9B value model, because it consumes much more engineering
  time without directly improving the reward-design comparison.

## Research question

Does a productive-difficulty shaping signal change modern LLM RL in ways that
are not captured by outcome-only RL?

## Experimental axes

### RL algorithms

1. Outcome-only GRPO.
2. Difficulty-reward GRPO.
3. Effort/length-reward GRPO.
4. Hiro GRPO.
5. Hiro without learning progress.
6. Optional DPO/offline preference baseline.

### Task curriculum

Eight levels of verifiable arithmetic/algebra. Train and evaluation instances
are disjoint. Correctness is checked by a deterministic parser.

### Hiro shaping

For task level `d`, maintain an online recent success estimate:

`ability[d] = mean(correctness over recent attempts)`.

Use:

- outcome correctness;
- formatting validity;
- challenge match around 45% success;
- positive change in recent success;
- penalty for pathological verbosity.

The shaping signal is non-stationary and history-aware, unlike a standard
static outcome reward.

## Main measurements

- held-out accuracy by difficulty;
- area under the learning curve;
- samples required to reach each accuracy threshold;
- generalization to unseen numeric ranges and templates;
- response length and format validity;
- reward/accuracy divergence;
- difficulty-conditioned gradient and reward statistics;
- catastrophic forgetting on easy levels;
- variance across seeds;
- wall-clock time and GPU-hours.

## Predicted effects

- Outcome GRPO should maximize average benchmark accuracy but over-sample
  currently easy examples if sampling is adaptive.
- Difficulty reward may chase levels with almost no useful learning signal.
- Length/effort reward should produce verbosity hacking.
- Hiro reward should improve mid/high-level sample efficiency and curriculum
  smoothness, but may sacrifice some short-run easy-task accuracy.
- Removing learning progress should make challenge shaping less adaptive and
  increase time spent on stagnant levels.

## Initial allocation

- Node 1 GPUs 0–3: outcome GRPO.
- Node 1 GPUs 4–7: Hiro GRPO.
- Node 2 GPUs 0–3: difficulty/effort baselines.
- Node 2 GPUs 4–7: ablations and independent seeds.

Start with 20-step smoke runs, then 200-step pilots, then 3-seed full runs.
All GPU jobs must use `scripts/with_gpu_hold.sh`.
