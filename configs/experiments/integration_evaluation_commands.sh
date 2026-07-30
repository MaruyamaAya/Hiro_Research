./scripts/evaluate_run.sh "${HIRO_PERSIST_ROOT:-${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl}/integration_g_u_seed42" "${HIRO_TRAIN_DATA:?set persistent prepared DAPO JSONL}" validation
./scripts/evaluate_run.sh "${HIRO_PERSIST_ROOT:-${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl}/integration_dr_u_seed42" "${HIRO_TRAIN_DATA:?set persistent prepared DAPO JSONL}" validation
./scripts/evaluate_run.sh "${HIRO_PERSIST_ROOT:-${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl}/integration_dapo_u_seed42" "${HIRO_TRAIN_DATA:?set persistent prepared DAPO JSONL}" validation
./scripts/evaluate_run.sh "${HIRO_PERSIST_ROOT:-${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl}/integration_dapo_histdyn_seed42" "${HIRO_TRAIN_DATA:?set persistent prepared DAPO JSONL}" validation
./scripts/evaluate_run.sh "${HIRO_PERSIST_ROOT:-${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl}/integration_dapo_hiro_seed42" "${HIRO_TRAIN_DATA:?set persistent prepared DAPO JSONL}" validation
./scripts/evaluate_run.sh "${HIRO_PERSIST_ROOT:-${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl}/integration_dapo_challenge_seed42" "${HIRO_TRAIN_DATA:?set persistent prepared DAPO JSONL}" validation
./scripts/evaluate_run.sh "${HIRO_PERSIST_ROOT:-${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl}/integration_dapo_progress_seed42" "${HIRO_TRAIN_DATA:?set persistent prepared DAPO JSONL}" validation
./scripts/evaluate_run.sh "${HIRO_PERSIST_ROOT:-${TAIJI_BASIC_OUTPUT_PATH}/hiro_rl}/integration_dapo_effort_seed42" "${HIRO_TRAIN_DATA:?set persistent prepared DAPO JSONL}" validation
