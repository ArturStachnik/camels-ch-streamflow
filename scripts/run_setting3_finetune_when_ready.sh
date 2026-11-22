#!/usr/bin/env bash
# For each setting-2 fold in turn: wait until its final model has test results, then write
# the fine-tuning configs of every finished fold and run the fine-tuning jobs not yet done.
PY=${PY:-python}
cd "$(dirname "$0")/.."
for k in "$@"; do
  until compgen -G "runs/setting2_fold${k}_*/test/model_epoch030/test_results.p" >/dev/null || compgen -G "runs/setting2_fold${k}_*/continue_training_from_epoch*/test/model_epoch030/test_results.p" >/dev/null; do sleep 120; done
  $PY scripts/make_setting3_configs.py finetune 2>&1 | tail -1
  bash scripts/run_setting3.sh finetune
done
