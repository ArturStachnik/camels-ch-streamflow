#!/usr/bin/env bash
# Train and evaluate the five setting-2 folds, at most three at a time on one GPU.
# A fold whose first run stopped before the end is resumed from its last saved epoch with
# NeuralHydrology's continue_training (weights and optimizer state); the resumed part is
# written to a subfolder continue_training_from_epochNNN of the original run.
# Usage: bash scripts/run_setting2.sh [fold ...]   (default: all five)
# Set NH to the nh-run executable of the environment if it is not the default below.
NH=${NH:-/c/Users/Koza/anaconda3/envs/camelsch/Scripts/nh-run.exe}
cd "$(dirname "$0")/.."
final_dir() {  # folder holding the weights of epoch 30, if any
  local hit
  hit=$(ls runs/setting2_fold${1}_*/model_epoch030.pt runs/setting2_fold${1}_*/continue_training_from_epoch*/model_epoch030.pt 2>/dev/null | tail -1)
  [ -n "$hit" ] && dirname "$hit"
}
run_fold() {
  k=$1
  log=runs/stdout_setting2_fold$k.log
  if [ -z "$(final_dir $k)" ]; then
    D=$(ls -d runs/setting2_fold${k}_* 2>/dev/null | tail -1)
    if [ -n "$D" ] && ls $D/model_epoch*.pt >/dev/null 2>&1; then
      echo "fold $k: resuming $D $(date +%H:%M)"
      # continue_training trains a full set of epochs counted from the resume point. It was
      # stopped by hand once the weights of epoch 30 existed; only those weights are evaluated.
      $NH continue_training --run-dir $D >> $log 2>&1 || { echo "fold $k resume stopped"; }
    else
      $NH train --config-file configs/setting2_fold$k.yml >> $log 2>&1 || { echo "fold $k train failed"; return 1; }
    fi
  fi
  F=$(final_dir $k)
  [ -f $F/test/model_epoch030/test_results.p ] || $NH evaluate --run-dir $F --period test --epoch 30 >> $log 2>&1
  echo "fold $k done: $F $(date +%H:%M)"
}
pids=()
for k in ${@:-0 1 2 3 4}; do
  while [ $(jobs -rp | wc -l) -ge 3 ]; do wait -n; done
  run_fold $k &
  sleep 20
done
wait
echo "all folds done $(date +%H:%M)"
