# Generated data

Files in this directory are generated and intentionally not tracked.

```bash
python3 -m qwen_study.generate_pairs
python3 -m llm_rl.generate_math_curriculum
python3 -m llm_rl.prepare_real_math_data
```

The synthetic datasets are smoke-test fixtures only. They are not intended to
be the primary data for the final RL study.

`prepare_real_math_data` expects the pinned raw DAPO-Math-17K, MATH-500, and
GSM8K files described in `data/manifests/real_math_manifest.json`. Raw files and
generated JSONL outputs are ignored; the compact provenance manifest is tracked.

The prepared DAPO output contains deterministic `train` and `validation`
splits. MATH-500 and GSM8K remain final-test-only and must not be used for
checkpoint selection or hyperparameter tuning.
