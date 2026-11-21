"""Run verify_metrics.py on every evaluated model of settings 1 to 3 and collect the outputs.

This script computes no metric. It calls verify_metrics.py once per evaluated unit (a fold,
a single-basin run, a gradient-boosting model) with the basin list that unit must contain,
and concatenates the per-basin metrics, benchmarks and check outcomes into one folder per
group. A unit fails if any hard check fails. The alarm on the median NSE across basins is
meaningless for single-basin runs and is recorded but not counted for setting 3.

Usage: python verification/verify_all.py [setting2] [gbm] [setting3]
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SINGLE_BASIN_NOT_APPLICABLE = {"alarm_median_nse_above_0.90"}


def fold_run(k):
    """Folder with the final (epoch 30) weights of setting-2 fold k: the run itself, or the
    continue_training subfolder if the run was interrupted and resumed."""
    hits = sorted(ROOT.glob(f"runs/setting2_fold{k}_*/model_epoch030.pt")) +         sorted(ROOT.glob(f"runs/setting2_fold{k}_*/continue_training_from_epoch*/model_epoch030.pt"))
    return hits[-1].parent if hits else None


def latest(pattern):
    runs = sorted(ROOT.glob(pattern))
    return runs[-1] if runs else None


def verify(run, basin_file, epoch, out):
    cmd = [PY, "verification/verify_metrics.py", "--run-dirs", run.relative_to(ROOT).as_posix(), "--period", "test",
           "--epoch", str(epoch), "--basin-file", basin_file, "--sim-clip", "zero", "--out", out.relative_to(ROOT).as_posix()]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (out / "stdout.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
    return p.returncode


def collect(units, out, single_basin=False, keep_unit_folders=True):
    per_basin, bench, checks = [], [], []
    for name, folder, code in units:
        rep = json.loads((folder / "verification_report.json").read_text())
        pb = pd.read_csv(folder / "per_basin_metrics.csv", dtype={"basin": str})
        pb = pb[pb.member != "ensemble"].assign(unit=name)
        per_basin.append(pb)
        bench.append(pd.read_csv(folder / "benchmarks.csv", dtype={"basin": str}).assign(unit=name))
        for c in rep["checks"]:
            na = single_basin and c["id"] in SINGLE_BASIN_NOT_APPLICABLE
            checks.append(dict(unit=name, check=c["id"], status=c["status"], hard=c["hard"], alarm=c["alarm"],
                               counts=not na, description=c["description"]))
        if not keep_unit_folders:
            shutil.rmtree(folder)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(per_basin).to_csv(out / "per_basin_metrics.csv", index=False, lineterminator="\n")
    pd.concat(bench).to_csv(out / "benchmarks.csv", index=False, lineterminator="\n")
    ch = pd.DataFrame(checks)
    ch.to_csv(out / "checks.csv", index=False, lineterminator="\n")
    failed = ch[(ch.status == "fail") & ch.hard & ch.counts]
    summary = dict(n_units=len(units), n_checks=int(len(ch)), n_hard_failures=int(len(failed)),
                   n_alarms=int(((ch.status == "fail") & ch.alarm & ch.counts).sum()),
                   n_not_applicable_alarms_triggered=int(((ch.status == "fail") & ~ch.counts).sum()),
                   n_findings=int((ch.status == "finding").sum()), failed=failed[["unit", "check"]].to_dict("records"),
                   overall="pass" if failed.empty else "fail")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    print(out.relative_to(ROOT).as_posix(), json.dumps({k: v for k, v in summary.items() if k != "failed"}), summary["failed"][:10])
    return summary


def main():
    todo = sys.argv[1:] or ["setting2", "gbm", "setting3"]
    if "setting2" in todo:
        base, units = ROOT / "results/verification_setting2", []
        for k in range(5):
            run, out = fold_run(k), base / f"fold{k}"
            out.mkdir(parents=True, exist_ok=True)
            units.append((f"fold{k}", out, verify(run, f"splits/setting2/fold{k}_test.txt", 30, out)))
        collect(units, base / "combined")
    if "gbm" in todo:
        base, units = ROOT / "results/verification_gbm", []
        jobs = [("setting1", "runs/gbm_setting1", "splits/basins_all.txt")] + \
               [(f"setting2_fold{k}", f"runs/gbm_setting2_fold{k}", f"splits/setting2/fold{k}_test.txt") for k in range(5)]
        for name, run, basin_file in jobs:
            out = base / name
            out.mkdir(parents=True, exist_ok=True)
            units.append((name, out, verify(ROOT / run, basin_file, 1, out)))
        collect(units[:1], base / "setting1_combined")
        collect(units[1:], base / "setting2_combined")
    if "setting3" in todo:
        base, units = ROOT / "results/verification_setting3", []
        names = sorted({p.name.rsplit("_", 2)[0] for p in (ROOT / "runs/setting3").glob("*_*_*y_*") if p.is_dir()})
        for name in names:
            run = latest(f"runs/setting3/{name}_*")
            epoch = int(next(run.glob("test/model_epoch*")).name[-3:])
            basin = name.split("_")[1]
            out = base / "units" / name
            out.mkdir(parents=True, exist_ok=True)
            units.append((name, out, verify(run, f"splits/setting3/basin_{basin}.txt", epoch, out)))
        collect(units, base, single_basin=True, keep_unit_folders=False)
        shutil.rmtree(base / "units", ignore_errors=True)


if __name__ == "__main__":
    main()
