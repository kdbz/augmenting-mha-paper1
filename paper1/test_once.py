"""E5: score the official test split once, after the code freeze (PREREGISTRATION.md §2, §4a).

    git tag paper1-freeze                 # freeze: tag the commit this script runs from
    python paper1/test_once.py            # retrain on train + dev, score the 56 test people, once
    python paper1/test_once.py --synthetic   # smoke test on synthetic data; never touches E-DAIC

Gates (real data): the pre-registration is committed, paper1/ is clean, the tag `paper1-freeze` points at HEAD,
and no earlier test run exists in results/. The run is reported whatever it shows.

Models: the median, mean and ridge baselines, and early and late F3 on every rung (R1-R5). The paper's verdict
is read from F3 on the highest rung (§5); the other rungs are reported alongside. Each network is trained once
(seed C.SEED) on all train + dev participants with the usual inner split for early stopping.

The pre-registration defines no test-split statistic. A paired bootstrap over the 56 test people (10,000
resamples) is reported as a descriptive interval only.
"""
import argparse
import csv
import glob
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

TAG = "paper1-freeze"


def check_freeze():
    head = R.git("rev-parse", "HEAD")
    tagged = R.git("rev-list", "-n", "1", TAG)
    if not tagged:
        sys.exit(f"No tag '{TAG}'. Freeze first: git tag {TAG}")
    if tagged != head:
        sys.exit(f"Tag '{TAG}' points at {tagged[:10]}, not HEAD {head[:10]}. The test runs only from the frozen commit.")
    if glob.glob(os.path.join(C.RESULTS_DIR, "TEST_edaic_*")):
        sys.exit("A test run already exists. The official test split is scored once (PREREGISTRATION.md §4a).")


def bootstrap(err_a, err_b, n=10000, seed=C.SEED):
    """Paired bootstrap over people of mean(|err_b|) - mean(|err_a|); positive = a better. Descriptive only."""
    rng = np.random.default_rng(seed)
    d = np.abs(err_b) - np.abs(err_a)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        X, y, pid, _ = data.make_synthetic()
        people = np.unique(pid)
        train_people, test_people = people[:160], people[160:]
    else:
        R.check_preregistration()
        R.check_clean_tree()
        check_freeze()
        X, y, pid, split = data.load_chunks(splits=C.CV_SPLITS + ("test",))
        train_people = np.unique(pid[np.isin(split, C.CV_SPLITS)])
        test_people = np.unique(pid[split == "test"])
        assert not set(train_people) & set(test_people)

    tag = "synthetic" if args.synthetic else "edaic"
    run_id = f"TEST_{tag}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(C.RESULTS_DIR, run_id)
    os.makedirs(out_dir)
    sys.stdout = R.Tee(os.path.join(out_dir, "run.log"))
    print(f"{run_id} | commit {R.code_version()[:10]} | train {len(train_people)} people | test {len(test_people)} people | device {P.DEVICE}")

    y_tr, y_te = P.person_labels(y, pid, train_people), P.person_labels(y, pid, test_people)
    preds = {"median_predictor": np.full(len(y_te), np.median(y_tr), np.float32),
             "mean_predictor": np.full(len(y_te), y_tr.mean(), np.float32),
             "functionals_ridge": P.ridge_predict(X, y, pid, train_people, test_people)}
    epochs, t0 = {}, time.time()
    for rung in C.RUNGS:
        for name, mk in models.builders(rung, ("F3",), singles=False).items():
            key = f"{rung}_{name}"
            preds[key], _, epochs[key], _ = P.fit_predict(mk, X, y, pid, train_people, test_people, seed=C.SEED)
            print(f"  {key:<12} trained, {epochs[key]} epochs  [{time.time() - t0:.0f}s]", flush=True)

    scores = {k: P.score(v, y_te) for k, v in preds.items()}
    print(f"\n{'model':<19}{'MAE':>8}{'RMSE':>8}{'R2':>9}")
    for k, sc in scores.items():
        print(f"{k:<19}{sc['mae']:8.3f}{sc['rmse']:8.3f}{sc['r2']:+9.3f}")

    err = {k: v - y_te for k, v in preds.items()}
    comps = {}
    print("\nDescriptive paired bootstrap over test people (positive = first model better), 95% interval:")
    for rung in C.RUNGS:
        pairs = [(f"{rung}_early_F3", f"{rung}_late_F3"), (f"{rung}_early_F3", "median_predictor"), (f"{rung}_late_F3", "median_predictor")]
        for a, b in pairs:
            d, ci = bootstrap(err[a], err[b])
            comps[f"{a}_vs_{b}"] = {"delta_mae": d, "boot95": ci}
            print(f"  {a} vs {b}: ΔMAE {d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    top = list(C.RUNGS)[-1]
    print(f"\nVerdict model (§5): F3 on {top}. Early {scores[f'{top}_early_F3']['mae']:.3f}, late {scores[f'{top}_late_F3']['mae']:.3f}, "
          f"median predictor {scores['median_predictor']['mae']:.3f} (MAE on {len(test_people)} test people).")

    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:   # E-DAIC-derived: git-ignored
        w = csv.writer(f); w.writerow(["model", "participant", "y_true", "y_pred"])
        for k, v in preds.items():
            w.writerows([k, int(p), float(t), round(float(q), 4)] for p, t, q in zip(test_people, y_te, v))
    json.dump({"run_id": run_id, "git_commit": R.code_version(), "freeze_tag": None if args.synthetic else TAG,
               "n_train": int(len(train_people)), "n_test": int(len(test_people)), "seed": C.SEED,
               "scores": scores, "epochs": epochs, "descriptive_bootstrap": comps,
               "environment": R.environment(), "wall_seconds": round(time.time() - t0)},
              open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
