"""Post hoc real-data positive control: can the frozen pipeline learn anything from the E-DAIC features?

    python paper1/positive_control.py                    # R1 and R5, F3 arms, 1 repeat of 5 folds
    python paper1/positive_control.py --rungs R5 --repeats 2

The target is the corpus gender field (female = 1, male = 0) instead of PHQ-8. Everything else is
the frozen protocol, called unchanged: protocol.fit_predict (same folds, inner split, scaler, L1
loss, optimiser, early stopping), models.builders (same capacity-matched arms) and
protocol.ridge_predict. Only train + dev participants are loaded; the official test split is never
read and no PHQ-8 model is fitted.

This analysis was not in PREREGISTRATION.md. It was added on 13 October 2026, after the results,
in response to review, and must be reported as post hoc. It writes aggregate scores only to
results/CONTROL_gender_<timestamp>/ {run.log, results.json}: no per-participant output.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import data
import models
import protocol as P
from run import Tee, code_version, environment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="R1,R5")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    X, _, pid, split = data.load_chunks()
    assert "test" not in set(split), "the official test split must not be loaded"
    lab = data.load_labels().set_index("Participant_ID")["Gender"]
    people = np.unique(pid)
    g_people = np.array([1.0 if lab.loc[int(p)] == "female" else 0.0 for p in people], np.float32)
    g_true = dict(zip(people, g_people))
    y = np.array([g_true[p] for p in pid], np.float32)          # one label per participant, on every chunk

    run_id = f"CONTROL_gender_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(C.RESULTS_DIR, run_id)
    os.makedirs(out_dir)
    sys.stdout = Tee(os.path.join(out_dir, "run.log"))
    print(f"{run_id} | commit {code_version()[:10]} (script itself uncommitted) | {len(people)} participants | "
          f"{len(pid)} chunks | female {int(g_people.sum())}, male {int((1 - g_people).sum())} | POST HOC")

    t0 = time.time()
    res, folds = {}, list(P.outer_folds(people, n_repeats=args.repeats))

    def record(name, rep, test_people, pred):
        true = np.array([g_true[p] for p in test_people])
        r = res.setdefault(name, {"auc": [], "accuracy": [], "mae": [], "pooled": {}})
        r["auc"].append(float(roc_auc_score(true, pred)))
        r["accuracy"].append(float(((pred >= 0.5) == (true == 1)).mean()))
        r["mae"].append(P.score(pred, true)["mae"])
        r["pooled"].setdefault(rep, ([], []))
        r["pooled"][rep][0].extend(true.tolist()); r["pooled"][rep][1].extend(np.asarray(pred, float).tolist())

    for rep, train_people, test_people in folds:
        const = np.full(len(test_people), float(np.median(P.person_labels(y, pid, train_people))), np.float32)
        true = np.array([g_true[p] for p in test_people])
        r = res.setdefault("median_predictor", {"auc": [], "accuracy": [], "mae": [], "pooled": {}})
        r["auc"].append(0.5); r["accuracy"].append(float(((const >= 0.5) == (true == 1)).mean()))
        r["mae"].append(P.score(const, true)["mae"])
        record("functionals_ridge", rep, test_people, P.ridge_predict(X, y, pid, train_people, test_people))

    epochs = {}
    for rung in args.rungs.split(","):
        build = models.builders(rung, ("F3",), singles=False)
        for name, make_model in build.items():
            key = f"{rung}_{name}"
            for i, (rep, train_people, test_people) in enumerate(folds):
                pred, _, n_epochs, _ = P.fit_predict(make_model, X, y, pid, train_people, test_people, C.SEED + i)
                record(key, rep, test_people, pred)
                epochs.setdefault(key, []).append(int(n_epochs))
                print(f"  {key:16s} fold {i + 1}/{len(folds)}  AUC {res[key]['auc'][-1]:.3f}  "
                      f"acc {res[key]['accuracy'][-1]:.3f}  MAE {res[key]['mae'][-1]:.3f}  epochs {n_epochs}", flush=True)

    print(f"\n{'model':18s} {'AUC mean':>9s} {'AUC pooled':>11s} {'accuracy':>9s} {'MAE':>7s}")
    summary = {}
    for name, r in res.items():
        pooled = [float(roc_auc_score(t, p)) for t, p in r["pooled"].values()] if r["pooled"] else [0.5]
        summary[name] = {"auc_folds": r["auc"], "auc_pooled_per_repeat": pooled, "accuracy_folds": r["accuracy"],
                         "mae_folds": r["mae"], "epochs": epochs.get(name)}
        print(f"{name:18s} {np.mean(r['auc']):9.3f} {np.mean(pooled):11.3f} {np.mean(r['accuracy']):9.3f} "
              f"{np.mean(r['mae']):7.3f}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"run_id": run_id, "post_hoc": True, "target": "gender (female = 1)", "git_commit": code_version(),
                   "n_participants": int(len(people)), "n_female": int(g_people.sum()), "n_chunks": int(len(pid)),
                   "repeats": args.repeats, "n_folds": C.N_FOLDS, "rungs": args.rungs.split(","), "seed": C.SEED,
                   "environment": environment(), "wall_seconds": int(time.time() - t0), "results": summary}, f, indent=1)
    print(f"\n{int(time.time() - t0)} s -> {out_dir}")


if __name__ == "__main__":
    main()
