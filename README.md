# Hiro Reward Research

Reproducible experiments for distinguishing productive difficulty from:

- easy repetition,
- irreducible noise ("Noisy TV"),
- meaningless effort,
- and dangerous/impossible tasks.

The repository has two experiment layers:

1. **HiroWorld simulation** — a controlled contextual-bandit benchmark used to
   validate the reward design and its ablations.
2. **Qwen3.5 preference study** — trajectory-pair evaluation/fine-tuning with
   Qwen3.5-9B, testing whether history-aware descriptions are necessary to
   represent Hiro-style preferences.
3. **Real-model RL prototype** — Qwen3.5-9B LoRA-GRPO/DAPO experiments with
   persistent checkpoints and a planned history-aware curriculum.

For the exact current status, remote artifact layout, known methodological
issues, and the next actions, read:

```text
docs/HANDOFF_STATUS.md
```

## Quick start

```bash
python3 -m hiro_world.run --config configs/main.json
python3 -m hiro_world.summarize --input results/main --output results/main_summary
python3 -m unittest discover -s tests -v
```

The real-model layer now includes a reusable symbolic/numeric answer verifier
and a held-out evaluator for the base model and saved LoRA checkpoints:

```bash
scripts/run_math_eval.sh \
  --data data/math_curriculum.jsonl \
  --checkpoint-root /persistent/run/checkpoints \
  --output /persistent/run/evaluation \
  --max-new-tokens 1024
```

The simulation only requires Python and NumPy. Plotting is optional and uses
Matplotlib when available.

## Server safety

The remote servers use `/root/gpu_hold` while no GPU experiment is active.
Always launch GPU work through:

```bash
scripts/with_gpu_hold.sh <experiment command...>
```

The wrapper stops the hold workers, runs the command, and restores the hold
workers on exit (including interruption/failure).
