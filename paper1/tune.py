"""R1 tuning on inner validation only (E1). The outer test folds are never scored.

    python paper1/tune.py --set LR=3e-4                    # one setting, 5 folds of repeat 0
    python paper1/tune.py --set LR=1e-4 --set PATIENCE=10
    python paper1/tune.py --synthetic --folds 1            # smoke test, no real data

For each of the first --folds outer folds, every network is trained on that fold's training people with
the usual inner split. Only the inner-validation curve is kept: best MAE, the epoch it came from, and
the epochs run. The outer test people of the fold are passed as an empty set, so no test prediction is
ever made. Results go to results/tune_<rung>_<data>_<timestamp>/ {tune.log, tune.json, curves.json}.

PREREGISTRATION.md section 6 allows R1 tuning of learning rate, epochs and patience, and feature and
target scaling. Keys outside that list need --deviation, and the run is labelled a deviation.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import data
import models
import protocol as P
import run as R

PREREGISTERED = {"LR": float, "MAX_EPOCHS": int, "PATIENCE": int}
DEVIATION = {"WEIGHT_DECAY": float}       # dropout and LSTM width need model changes first


def parse_sets(items, allow_deviation):
    out = {}
    for item in items:
        key, _, val = item.partition("=")
        if key in PREREGISTERED:
            out[key] = PREREGISTERED[key](val)
        elif key in DEVIATION and allow_deviation:
            out[key] = DEVIATION[key](val)
        elif key in DEVIATION:
            sys.exit(f"{key} is not on the pre-registered tuning list (section 6). Add --deviation to run it anyway.")
        else:
            sys.exit(f"{key} cannot be tuned here. Allowed: {sorted(PREREGISTERED)} (+ {sorted(DEVIATION)} with --deviation)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="R1", choices=list(C.RUNGS))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--deviation", action="store_true", help="allow keys outside the pre-registered tuning list")
    ap.add_argument("--folds", type=int, default=5, help="first N outer folds of repeat 0")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    overrides = parse_sets(args.set, args.deviation)
    for k, v in overrides.items():
        setattr(C, k, v)                     # protocol.fit_predict reads these at call time

    if args.synthetic:
        X, y, pid, _ = data.make_synthetic()
    else:
        R.check_preregistration()
        R.check_clean_tree()
        X, y, pid, _ = data.load_chunks()
    people = np.unique(pid)
    folds = list(P.outer_folds(people, n_repeats=1))[:args.folds]
    build = models.builders(args.rung)

    tag = "synthetic" if args.synthetic else "edaic"
    label = "_".join(f"{k}{v}" for k, v in overrides.items()) or "default"
    run_id = f"tune_{args.rung}_{tag}_{label}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(C.RESULTS_DIR, run_id)
    os.makedirs(out_dir)
    sys.stdout = R.Tee(os.path.join(out_dir, "tune.log"))
    deviation = any(k in DEVIATION for k in overrides)
    print(f"{run_id} | commit {R.code_version()[:10]} | {len(people)} participants | folds {len(folds)} | "
          f"{'DEVIATION' if deviation else 'pre-registered levers only'}")
    print("settings: " + ", ".join(f"{k}={getattr(C, k)}" for k in ("LR", "WEIGHT_DECAY", "MAX_EPOCHS", "PATIENCE")))
    print("inner validation only; outer test folds are not scored\n")

    rows, curves, t0 = {k: [] for k in build}, {k: [] for k in build}, time.time()
    empty = np.array([], dtype=people.dtype)
    for run, (rep, tr_people, _) in enumerate(folds):
        for name, mk in build.items():
            hist = []
            _, _, epochs, _ = P.fit_predict(mk, X, y, pid, tr_people, empty, seed=C.SEED + run, history=hist)
            val = [h["val_mae"] for h in hist]
            best = int(np.argmin(val))
            rows[name].append({"best_val_mae": float(val[best]), "best_epoch": best, "epochs_run": epochs,
                               "val_epoch0": float(val[0])})
            curves[name].append(hist)
        print(f"  fold {run + 1}/{len(folds)}  " + "  ".join(f"{k}={rows[k][-1]['best_val_mae']:.2f}@{rows[k][-1]['best_epoch']}"
                                                      for k in build) + f"  [{time.time() - t0:.0f}s]", flush=True)

    print(f"\n{'model':<12}{'best val MAE':>14}{'best epoch':>12}{'epoch>0':>9}")
    summary = {}
    for k, r in rows.items():
        m = float(np.mean([x["best_val_mae"] for x in r]))
        e = float(np.mean([x["best_epoch"] for x in r]))
        later = sum(x["best_epoch"] > 0 for x in r)
        summary[k] = {"mean_best_val_mae": m, "mean_best_epoch": e, "folds_best_after_epoch0": later}
        print(f"{k:<12}{m:>14.3f}{e:>12.1f}{later:>6}/{len(r)}")
    overall = float(np.mean([s["mean_best_val_mae"] for s in summary.values()]))
    print(f"\nall networks: mean best inner-validation MAE {overall:.3f}  [{time.time() - t0:.0f}s]")

    json.dump({"run_id": run_id, "git_commit": R.code_version(), "overrides": overrides, "deviation": deviation,
               "folds": len(folds), "summary": summary, "overall_mean_best_val_mae": overall, "per_fold": rows},
              open(os.path.join(out_dir, "tune.json"), "w"), indent=1)
    json.dump(curves, open(os.path.join(out_dir, "curves.json"), "w"))


if __name__ == "__main__":
    main()
