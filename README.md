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
