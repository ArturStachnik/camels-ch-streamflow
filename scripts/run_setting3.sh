#!/usr/bin/env bash
# Train and evaluate the setting-3 runs of one kind (scratch or finetune), one after another.
# A run whose directory already holds test results is skipped.
NH=${NH:-nh-run}
cd "$(dirname "$0")/.."
kind=$1
for cfg in configs/setting3/${kind}_*.yml; do
  name=$(basename $cfg .yml)
  if ls -d runs/setting3/${name}_*/test/model_epoch*/test_results.p >/dev/null 2>&1; then continue; fi
  if [ "$kind" = finetune ]; then mode=finetune; else mode=train; fi
  $NH $mode --config-file $cfg >> runs/setting3/stdout_${kind}.log 2>&1 || { echo "$name train failed"; continue; }
  D=$(ls -d runs/setting3/${name}_* | tail -1)
  $NH evaluate --run-dir $D --period test >> runs/setting3/stdout_${kind}.log 2>&1 || echo "$name eval failed"
done
echo "$kind done $(date +%H:%M); runs with results: $(ls runs/setting3/${kind}_*/test/model_epoch*/test_results.p | wc -l)"
