# Hiro Research Handoff and Current Status

Last updated: 2026-07-30 (Asia/Shanghai)

This document is the operational handoff for continuing the project in a new
session. It records what has been implemented, what has actually run, where the
remote artifacts live, known flaws, and the next required work.

## 1. Research goal

The project studies whether a **productive-difficulty** objective can improve
modern reinforcement learning for language models without degenerating into:

- irreducible-noise seeking;
- rewarding verbosity or effort for its own sake;
- repeatedly choosing impossible tasks;
- static repetition after a task is mastered;
- unsafe or irreversible choices.

The intended final comparison is not merely a toy bandit. It should compare
real Qwen3.5-9B RL runs using paper-aligned GRPO-family methods and realistic,
verifiable reasoning data.

## 2. Repository contents

### Concept and experiment specification

- `hiro_reward_research_proposal.md`
- `docs/experiment_plan.md`
- `docs/real_llm_rl_plan.md`

### Controlled HiroWorld simulation

- `hiro_world/environment.py`
- `hiro_world/agents.py`
- `hiro_world/run.py`
- `hiro_world/summarize.py`
- `configs/main.json`

### Qwen preference study

- `qwen_study/generate_pairs.py`
- `qwen_study/evaluate_zero_shot.py`

### Real-model RL prototype

- `llm_rl/generate_math_curriculum.py`
- `llm_rl/math_verifier.py`
- `llm_rl/train_grpo.py`
- `llm_rl/evaluate_checkpoints.py`
- `scripts/launch_distributed_grpo.sh`
- `scripts/run_grpo.sh`
- `scripts/run_math_eval.sh`

### GPU lifecycle

- `scripts/with_gpu_hold.sh`
- `scripts/check_remote_hold.sh`
- `scripts/setup_qwen_env.sh`

`with_gpu_hold.sh` stops all hold workers before a GPU command and restores all
visible GPUs on every normal/error exit path. Do not run GPU training outside
this wrapper.

## 3. Compute environment

Two remote nodes are available. Each node has:

- 8 NVIDIA H20 GPUs;
- approximately 96 GB VRAM per GPU;
- 384 logical CPU cores;
- approximately 2.2 TiB RAM.

The mounted base model is Qwen3.5-9B:

```text
${TAIJI_BASIC_MODEL_PATH}
```

On the current jobs this resolves to the platform-mounted Qwen3.5-9B Hugging
Face checkpoint.

The research checkout on each node is:

```text
/root/Hiro_Research
```

The Python environment is:

```text
/root/hiro-env
```

Important environment versions observed during the latest run:

- PyTorch 2.10.0 + CUDA 12.8;
- Transformers 5.12.1;
- TRL 0.26.1;
- PEFT 0.18.1;
- Datasets 4.4.1.

The platform NVIDIA driver is compatible with CUDA 12.8 wheels but not CUDA
13.0 wheels. Do not let package upgrades replace the working PyTorch build with
a CUDA 13 build.

## 4. Persistent remote artifact root

All RL artifacts were written to persistent platform storage:

```text
${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl/
```

This is deliberately outside the container-local filesystem. Each run includes
the generated dataset, launch metadata, code snapshot, log, checkpoints, and
final LoRA adapter.

Existing run directories:

```text
outcome_pilot/
hiro_pilot/
outcome_pilot_v2/
hiro_pilot_v2/
```

Do not delete these runs. They are useful engineering records even though they
are not final scientific experiments.

## 5. Completed work

### 5.1 GPU hold deployment

The local `gpu_hold` package was deployed to:

```text
/root/gpu_hold
```

When idle, each node runs eight low-priority hold workers. Typical state:

- roughly 79 GB VRAM occupied per GPU;
- roughly 85–100% reported GPU utilization;
- worker priority `nice=19`;
- CPU consumption remains small relative to 384 cores.

At the time this handoff was written, both nodes had completed training and
returned to eight active hold workers.

### 5.2 HiroWorld

A 50-seed × 20,000-step experiment was run for 11 conditions, including:

- extrinsic;
- novelty;
- surprise;
- suffering;
- difficulty;
- full Hiro;
- five Hiro ablations.

Compact summaries are checked in under:

```text
results/main_v2_summary/
```

Representative v2 findings:

- surprise agent Noisy-TV ratio: about 64%;
- suffering agent treadmill ratio: about 61%;
- additive-effort Hiro treadmill ratio: about 48%;
- full Hiro challenge ratio: about 94%;
- full Hiro Noisy-TV and treadmill ratios: each below 1%;
- removing the safety constraint produced substantial accumulated damage;
- extrinsic-only achieved higher short-run external return but lower skill.

These results validate the benchmark implementation and predicted failure
modes. They do not by themselves establish an LLM RL result.

### 5.3 Qwen3.5 inference and preference prototype

Qwen3.5-9B was successfully loaded from the mounted checkpoint and used for
generation. A synthetic trajectory-pair generator and forced-choice evaluator
were implemented.

The first bare-letter forced-choice scoring method showed a strong `A` format
prior and is not reliable. `evaluate_zero_shot.py` was changed to score fuller
answer suffixes, but the preference experiment has not yet been completed or
validated on a full test set.

### 5.4 Real Qwen3.5-9B LoRA-GRPO

An end-to-end one-step smoke test succeeded:

- Qwen3.5-9B load;
- LoRA injection;
- grouped generation;
- deterministic answer reward;
- GRPO/DAPO-style loss;
- backward pass and optimizer step;
- checkpoint save;
- automatic hold restoration.

Two distributed eight-GPU pilot pairs were then run.

#### Pilot v1

Runs:

```text
outcome_pilot
hiro_pilot
```

Configuration:

- 20 optimizer steps;
- 96-token completions;
- checkpoints every five steps.

Saved checkpoints include steps 5, 10, 15, 20 and `final`.

The run revealed that Qwen3.5's default thinking trace was almost always
truncated at 96 tokens, causing many zero outcome rewards. This run is an
engineering record, not a valid comparison.

#### Pilot v2

Runs:

```text
outcome_pilot_v2
hiro_pilot_v2
```

Configuration:

- 10 optimizer steps;
- 384-token completions;
- checkpoints every two steps;
- eight GPUs per condition.

Saved checkpoints include:

```text
checkpoint-4
checkpoint-6
checkpoint-8
checkpoint-10
final
```

Outcome v2 finished in about 210 seconds. Hiro v2 finished in about 228
seconds. The last logged mean rewards were approximately 0.79 and 0.68,
respectively.

However, 384 tokens still produced high truncation ratios, commonly 62–94%.
These runs prove that the distributed training and checkpoint paths work; they
are not sufficient to claim one reward is better.

## 6. Critical methodological issues found

Do not simply scale the current prototype to a long run.

### 6.1 Synthetic arithmetic is only a smoke test

`generate_math_curriculum.py` creates simple arithmetic/algebra. It is useful
for deterministic pipeline testing but is not representative enough for the
main result.

The main training data should be a realistic public verifiable-reasoning mix,
preferably including:

- DAPO-Math-17K or an equivalent hard-math RL set;
- OpenR1-Math and/or NuminaMath;
- GSM8K for easier natural-language problems;
- strict decontamination against MATH-500, AIME, GSM8K test, and other held-out
  benchmarks.

Direct Hugging Face access from the nodes timed out during testing. Possible
solutions:

1. locate an internal Tencent mirror or shared copy;
2. download public files locally and upload them to the persistent Ceph path;
3. use an accessible model/dataset proxy;
4. record dataset revision, hash, license, and preprocessing steps.

### 6.2 Prompt-level Hiro terms can vanish under GRPO centering

If every completion for one prompt receives the same challenge or
learning-progress bonus, group-relative centering removes that bonus:

```text
advantage_i = reward_i - group_mean(reward)
```

Therefore the main Hiro mechanism should be implemented as a **prompt/task
curriculum sampler**, not merely as a constant added to every completion in a
group.

Recommended split:

- prompt-level Hiro score controls which tasks/difficulty buckets are sampled;
- completion-level reward contains correctness, format validity, verifier
  consistency, non-repetition, and soft overlong shaping.

### 6.3 Rank-local mutable reward state is wrong for distributed training

The current `RewardState` object keeps history in each process. In an
eight-process run, each rank has its own unsynchronized state. This makes the
online learning-progress estimate inconsistent.

Before a real Hiro run, replace it with one of:

- a shared curriculum state updated by rank 0 and broadcast periodically;
- offline epoch-level competence estimates;
- a sampler/controller process;
- all-reduced per-bucket success statistics.

### 6.4 GRPO-family implementation choices matter

The final study should explicitly compare or align with:

- original GRPO/DeepSeekMath group-relative optimization;
- DeepSeek-R1 rule-based verifiable reward;
- Dr.GRPO corrections for question/length bias;
- DAPO dynamic sampling, token-level loss, Clip-Higher, and soft overlong
  punishment.

Current TRL settings used `loss_type="dapo"`, but that alone is not a complete
DAPO reproduction. Dynamic prompt filtering/sampling and curriculum logic are
still missing.

### 6.5 Rollout length and overlong handling

Qwen3.5 thinks by default and did not reliably finish simple problems within
384 tokens. Do not use a hard short cutoff without accounting for it.

Needed:

- inspect realistic completion-length distributions;
- choose a larger maximum, likely 768–2048 depending on the dataset;
- use soft overlong punishment rather than an abrupt reward cliff;
- report clipped ratio and terminated length;
- evaluate accuracy under a fixed inference token budget.

### 6.6 Evaluation is implemented but has not yet run remotely

A first held-out evaluator now exists in `llm_rl/evaluate_checkpoints.py`. It:

- loads the base model or saved LoRA checkpoints;
- runs fixed prompts and deterministic per-sample seeds;
- uses the shared symbolic/numeric answer verifier;
- reports empirical pass@1 and pass@k;
- stratifies by source and difficulty;
- measures response length, truncation, format validity, and verifier status;
- discovers every `checkpoint-*` directory plus `final`.

It still needs to be executed against the base model and all four pilot runs on
the remote nodes. Realistic held-out datasets and explicit reward-hacking audit
sets are also still required.

### 6.7 Robust answer verification is now available

`llm_rl/math_verifier.py` supports:

- explicit `<answer>...</answer>` extraction;
- balanced `\boxed{...}` extraction;
- integer, decimal, fraction, and simple symbolic equivalence;
- common LaTeX fractions and square roots;
- explicit `correct`, `incorrect`, `no_answer`, `invalid`, and `ambiguous`
  outcomes;
- rejection of unsupported identifiers and implicit number scavenging from
  reasoning text.

The GRPO reward now uses this shared verifier. Deterministic unit tests are in
`tests/`.

### 6.8 Real-data preparation and synchronized curriculum are implemented locally

The local checkout now includes:

- `scripts/download_real_math_data.sh`, which downloads pinned MATH-500 and
  GSM8K test data and retrieves the first 17,000 DAPO-Math-17K records;
- `llm_rl/prepare_real_math_data.py`, which normalizes all sources to the common
  prompt/answer schema, removes exact duplicate training prompts, and performs
  exact canonical-text decontamination against held-out prompts;
- `data/manifests/real_math_manifest.json`, recording revisions, licenses,
  SHA-256 hashes, row counts, and filtering statistics;
- `llm_rl/curriculum.py`, implementing bucket-level attempts, pass rates,
  previous-window rates, signed learning progress, zero-gradient group counts,
  coverage, checkpoint serialization, and weighted sampling;
- `llm_rl/curriculum_trainer.py`, integrating the dynamic sampler with TRL's
  repeated-prompt grouping;
- distributed reward-statistic synchronization using
  `torch.distributed.all_gather_object`;
- exact curriculum-state saving in every checkpoint and restoration on resume.

Current prepared counts:

- 17,000 DAPO input records;
- 14,882 unique training prompts after removing 2,118 duplicates;
- five deterministic proxy difficulty buckets with counts 223, 4,380, 6,135,
  2,960, and 1,184;
- 500 MATH-500 and 1,319 GSM8K held-out problems.

All 14,882 DAPO reference answers pass the current verifier when wrapped in the
required answer format. MATH-500 still contains structured answer types
(tuples, sets, text labels, base notation, and other LaTeX) that require a
dataset-complete verifier before MATH-500 accuracy is publication-grade.

At the end of the latest local session, both DevCloud SSH endpoints accepted a
TCP connection but closed it before SSH key exchange. Therefore the new code,
data, and evaluator have not yet been uploaded or run on the H20 nodes.

## 7. Recommended next actions

Execute these in order.

### Step 1: freeze current artifacts

Verify both nodes still have eight `gpu_hold` processes and copy/checksum the
four pilot run directories if a second persistent backup location is
available.

### Step 2: obtain realistic data

An initial pinned DAPO-Math-17K + MATH-500 + GSM8K bundle and manifest now
exists locally. Next:

- restore remote access and upload/regenerate the data on persistent storage;
- add semantic/near-duplicate decontamination, not only exact canonical hashes;
- consider adding OpenR1-Math/NuminaMath only after licensing and overlap
  auditing;
- manually audit a stratified sample and all verifier-invalid references.

### Step 3: run and extend robust answer verification

The initial verifier and unit tests are implemented. Before realistic-data
training:

- run it over each acquired dataset and manually inspect all invalid/ambiguous
  classes;
- add dataset-specific answer formats only through regression tests;
- keep explicit extraction so prompt/intermediate-number leakage is not
  accepted.

### Step 4: implement paper-aligned baselines

At minimum:

1. standard GRPO + uniform sampling;
2. Dr.GRPO-style normalization + uniform sampling;
3. DAPO-style loss + dynamic sampling;
4. DAPO + Hiro curriculum;
5. DAPO + challenge-only;
6. DAPO + learning-progress-only;
7. length/effort reward hacking baseline.

### Step 5: implement synchronized Hiro curriculum

The initial synchronized implementation is complete locally and maintains:

- attempt count;
- recent pass rate;
- previous-window pass rate;
- positive/negative learning progress;
- zero-gradient group rate;
- sample coverage.

These statistics produce prompt sampling probabilities and are saved inside
every checkpoint. DAPO now has five deterministic proxy difficulty buckets,
but they must be validated against base-model pass rates and replaced or
recalibrated if they are not monotonic. The implementation still needs an
eight-GPU integration test and deterministic resume test on the actual TRL
stack.

### Step 6: execute evaluation before long training

The evaluator is implemented. Run the base model and pilot checkpoints on the
same held-out suite. This gives a baseline and catches any broken verifier
before spending more GPU-hours.

### Step 7: short calibration, then long runs

Suggested progression:

- 5-step integration test;
- 20-step calibration;
- 100–200-step pilot;
- inspect reward/accuracy divergence;
- only then launch multi-seed longer runs.

Always save frequent resumable checkpoints to persistent storage.

## 8. Useful commands

### Check hold

```bash
cd /root/Hiro_Research
./scripts/check_remote_hold.sh
```

### Generate smoke-test data

```bash
python3 -m llm_rl.generate_math_curriculum
```

### Run one node-wide GRPO job

```bash
cd /root/Hiro_Research
NPROC=8 ./scripts/launch_distributed_grpo.sh \
  outcome example_run \
  --max-steps 20 \
  --save-steps 5 \
  --max-completion-length 384 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --num-generations 4
```

### Resume

`train_grpo.py` accepts:

```text
--resume-from-checkpoint /persistent/path/checkpoint-N
```

For the future synchronized curriculum implementation, resume must also restore
the curriculum-controller state.

### Evaluate base model and all checkpoints in one run

```bash
cd /root/Hiro_Research
./scripts/run_math_eval.sh \
  --data "${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl/outcome_pilot_v2/math_curriculum.jsonl" \
  --checkpoint-root "${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl/outcome_pilot_v2/checkpoints" \
  --output "${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl/outcome_pilot_v2/evaluation" \
  --max-new-tokens 1024 \
  --temperature 0
```

Use a separate invocation without `--checkpoint-root` for the base model. Use
`--samples K --temperature 0.8` for empirical pass@k sampling.

## 9. Security and repository hygiene

- SSH passwords and local askpass helpers are not stored in this repository.
- Raw model checkpoints remain in persistent remote storage, not Git.
- Generated JSONL data and raw per-seed simulation CSV files are ignored.
- Compact aggregate CSV summaries are committed.

## 10. Current bottom line

The project has:

- a working controlled benchmark;
- predicted reward-hacking failure modes;
- working Qwen3.5-9B inference;
- working one-node eight-GPU LoRA-GRPO;
- persistent resumable checkpoints;
- two completed outcome/Hiro pilot pairs.

It does **not yet** have:

- realistic primary training data;
- a synchronized history-aware Hiro curriculum;
- a faithful dynamic-sampling DAPO implementation;
- remote results from the new symbolic verifier/evaluation harness;
- multi-seed evidence that Hiro improves real-model RL.

Those missing items are the next milestone and should be completed before
claiming a satisfactory scientific result.
