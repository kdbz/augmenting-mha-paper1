"""Run one rung of the model ladder under the full protocol.

    python paper1/run.py --synthetic --repeats 1         # positive control, no real data
    python paper1/run.py --rung R1 --diagnose            # one fold, real data
    python paper1/run.py --rung R1                       # full 5 x 10 CV on train + dev, whole feature ladder
    python paper1/run.py --rung R3 --features F3         # optional rungs: primary feature set only

Every run writes one folder, results/<rung>_<data>_<timestamp>/:
    run.log           everything printed to the console
    results.json      settings, provenance, capacity, per-fit metrics, pooled R2, tests  (aggregate only)
    curves.json       per-epoch training and inner-validation MAE for every fit
    predictions.csv   one row per participant per fit                      (E-DAIC-derived: not public)
    chunk_predictions.npz  every test-chunk prediction of every fit        (E-DAIC-derived: not public)
    checkpoints/      best-epoch weights for every fit                     (E-DAIC-derived: not public)
    config.py         copy of the settings the run used

Real-data runs refuse to start unless PREREGISTRATION.md and all paper1/ code are committed.
The official test split is never loaded here.
"""
import argparse
import csv
import shutil
import hashlib
import platform
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import data
import models
import protocol as P
import stats


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=C.HERE, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_version():
    """The commit this run's code came from. On Kaggle there is no .git, so the SNAPSHOT file
    written next to the code at upload time (`git rev-parse HEAD > paper1/SNAPSHOT`) stands in."""
    if C.IS_KAGGLE:
        snap = os.path.join(C.HERE, "SNAPSHOT")
        if not os.path.exists(snap):
            sys.exit("No paper1/SNAPSHOT. Write the commit hash into it before uploading to Kaggle.")
        return open(snap).read().strip()
    return git("rev-parse", "HEAD")


def check_clean_tree():
    """Real-data results must trace to one commit: refuse if any tracked code in paper1/ has changed."""
    if C.IS_KAGGLE:
        return
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=no", "--", ".").splitlines()
             if "/results/" not in l and "/cache/" not in l]
    if dirty:
        sys.exit("Uncommitted changes in paper1/. Commit them before any real-data run:\n  " + "\n  ".join(dirty))


def environment():
    import sklearn, torch
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "sklearn": sklearn.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
            "host": "kaggle" if C.IS_KAGGLE else "local"}


def check_preregistration():
    if C.IS_KAGGLE:
        return  # Kaggle runs a snapshot of the frozen commit
    rel = os.path.relpath(C.PREREG_FILE, C.HERE)   # git runs with cwd=C.HERE
    if not git("log", "--oneline", "--", rel):
        sys.exit("PREREGISTRATION.md is not committed. Commit it before any real-data run.")
    if git("status", "--porcelain", "--", rel):
        sys.exit("PREREGISTRATION.md has uncommitted changes. Commit them before any real-data run.")


class Tee:
    """Send console output to the terminal and to run.log."""

    def __init__(self, path):
        self.f, self.out = open(path, "w"), sys.stdout

    def write(self, s):
        self.out.write(s); self.f.write(s)

    def flush(self):
        self.out.flush(); self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="R1", choices=list(C.RUNGS))
    ap.add_argument("--features", default="all", help="'all' (singles + F2 + F3) or a comma list, e.g. F3")
    ap.add_argument("--synthetic", action="store_true", help="positive-control data, no E-DAIC")
    ap.add_argument("--diagnose", action="store_true", help="one fold only")
    ap.add_argument("--repeats", type=int, default=C.N_REPEATS)
    args = ap.parse_args()

    if args.synthetic:
        X, y, pid, _ = data.make_synthetic()
    else:
        check_preregistration()
        check_clean_tree()
        X, y, pid, _ = data.load_chunks()
    people = np.unique(pid)
    y_people = P.person_labels(y, pid, people)
    y_true = {int(p): float(v) for p, v in zip(people, y_people)}

    sets = tuple(C.FEATURE_SETS) if args.features == "all" else tuple(args.features.split(","))
    build = models.builders(args.rung, sets, singles=args.features == "all")
    caps = models.capacities(args.rung)

    tag = "synthetic" if args.synthetic else "edaic"
    run_id = f"{args.rung}_{tag}{'_diagnose' if args.diagnose else ''}_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(C.RESULTS_DIR, run_id)
    os.makedirs(os.path.join(out_dir, "checkpoints"))
    shutil.copy(os.path.join(C.HERE, "config.py"), out_dir)
    sys.stdout = Tee(os.path.join(out_dir, "run.log"))

    print(f"{run_id} | commit {code_version()[:10]} | {len(people)} participants | {len(pid)} chunks | "
          f"{X['au'].shape[1]} steps | device {P.DEVICE}")
    print(f"PHQ-8 mean {y_people.mean():.2f}  median {np.median(y_people):.2f}  SD {y_people.std(ddof=1):.2f}")
    for fs, c in caps.items():
        print(f"capacity {fs}: late {c['late']:,} vs early (h={c['early_hidden']}) {c['early']:,}  gap {c['gap']:.1%}")
    print()

    names = ["median_predictor", "mean_predictor", "functionals_ridge"] + list(build)
    res = {k: {"mae": [], "rmse": [], "r2": [], "epochs": []} for k in names}
    preds = {k: [] for k in names}
    sens = {k: {"mae": [], "rmse": [], "r2": []} for k in build}    # median-of-chunks sensitivity analysis
    sens_preds = {k: [] for k in build}
    chunk_out = {}
    curves, folds_log, sizes, t0 = {k: [] for k in build}, [], [], time.time()
    folds = list(P.outer_folds(people, n_repeats=args.repeats))
    if args.diagnose:
        folds = folds[:1]

    pred_csv = open(os.path.join(out_dir, "predictions.csv"), "w", newline="")
    writer = csv.writer(pred_csv)
    writer.writerow(["run", "repeat", "model", "participant", "y_true", "y_pred"])

    for run, (rep, tr_people, te_people) in enumerate(folds):
        assert not set(tr_people) & set(te_people), "participant in both train and test"
        sizes.append((len(tr_people), len(te_people)))
        folds_log.append((rep, [int(p) for p in te_people]))
        y_te, y_tr = P.person_labels(y, pid, te_people), P.person_labels(y, pid, tr_people)

        def record(name, pred, epochs=None):
            for k, v in P.score(pred, y_te).items():
                res[name][k].append(v)
            res[name]["epochs"].append(epochs)
            preds[name].append([float(v) for v in pred])
            writer.writerows([run, rep, name, int(p), float(t), round(float(v), 4)]
                             for p, t, v in zip(te_people, y_te, pred))

        record("median_predictor", np.full(len(y_te), np.median(y_tr), np.float32))
        record("mean_predictor", np.full(len(y_te), y_tr.mean(), np.float32))
        record("functionals_ridge", P.ridge_predict(X, y, pid, tr_people, te_people))
        for name, mk in build.items():
            hist = []
            pred, state, epochs, chunk_pred = P.fit_predict(mk, X, y, pid, tr_people, te_people, seed=C.SEED + run, history=hist)
            record(name, pred, epochs)
            if chunk_pred is not None:   # R3 predicts per person: no chunk predictions, no median-of-chunks
                test_idx = np.where(np.isin(pid, te_people))[0]
                pred_med = P.aggregate(chunk_pred, pid[test_idx], te_people, "median")
                for k, v in P.score(pred_med, y_te).items():
                    sens[name][k].append(v)
                sens_preds[name].append([float(v) for v in pred_med])
                chunk_out[f"{name}_run{run:02d}"] = chunk_pred.astype(np.float32)
                chunk_out[f"idx_run{run:02d}"] = test_idx.astype(np.int32)
            curves[name].append(hist)
            torch.save(state, os.path.join(out_dir, "checkpoints", f"{name}_run{run:02d}.pt"))
        pred_csv.flush()
        print(f"  run {run + 1:2d}/{len(folds)}  " + "  ".join(f"{k}={res[k]['mae'][-1]:.2f}" for k in names)
              + f"  [{time.time() - t0:.0f}s]", flush=True)
    pred_csv.close()

    for k in names:
        res[k]["r2_pooled"] = P.pooled_r2(preds[k], folds_log, y_true)
    for k in build:
        sens[k]["r2_pooled"] = P.pooled_r2(sens_preds[k], folds_log, y_true)
    np.savez_compressed(os.path.join(out_dir, "chunk_predictions.npz"), **chunk_out)

    subgroups = {}
    if not args.synthetic and len(folds) > 1:   # descriptive gender breakdown, no tests (PREREGISTRATION.md §5)
        lab = data.load_labels()
        gender = {int(p): g for p, g in zip(lab.Participant_ID, lab.Gender)}
        print("\nSubgroups by gender (descriptive only, no tests): pooled MAE | pooled R2, mean over repeats")
        for k in ["median_predictor", "functionals_ridge", f"early_{C.PRIMARY}", f"late_{C.PRIMARY}"]:
            if k in res:
                subgroups[k] = P.subgroup_scores(preds[k], folds_log, y_true, gender)
                print(f"  {k:<19}" + "   ".join(f"{g} (n={v['n']}): {np.mean(v['mae_pooled']):.3f} | {np.mean(v['r2_pooled']):+.3f}"
                                              for g, v in subgroups[k].items() if v["mae_pooled"]))

    sd = lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    print(f"\n{'model':<19}{'MAE':<17}{'RMSE':<17}{'R2 (fold)':<17}R2 (pooled)")
    for k in names:
        m, r, q, qp = (np.array(res[k][x], float) for x in ("mae", "rmse", "r2", "r2_pooled"))
        pooled = f"{qp.mean():+.3f} ± {sd(qp):.3f}" if len(qp) else "n/a (incomplete repeat)"
        print(f"{k:<19}{m.mean():6.3f} ± {sd(m):<8.3f}{r.mean():6.3f} ± {sd(r):<8.3f}{q.mean():+.3f} ± {sd(q):<8.3f}{pooled}")

    tests = {}
    if len(folds) > 1:
        n_tr, n_te = np.mean([s[0] for s in sizes]), np.mean([s[1] for s in sizes])
        pairs = [(f"early_{C.PRIMARY}", f"late_{C.PRIMARY}", "primary")]
        pairs += [(f"early_{fs}", f"late_{fs}", "secondary") for fs in sets if fs != C.PRIMARY]
        pairs += [(a, "median_predictor", "baseline") for a in [f"{arm}_{fs}" for fs in sets for arm in ("early", "late")]
                  + ["functionals_ridge", "au_only", "gaze_only", "pose_only"]]
        if args.features == "all":   # ablation: does adding a channel help, within each arm?
            for arm in ("early", "late"):
                pairs += [(f"{arm}_F2", "au_only", "ablation"), (f"{arm}_F3", f"{arm}_F2", "ablation")]
        print("\nOnly [primary] is confirmatory; every other test is uncorrected and read descriptively (§5).")
        for a, b, role in pairs:
            if a in res and b in res:
                t = stats.compare(res, a, b, n_tr, n_te)
                t["role"], t["meaningful"] = role, bool(t["p"] < 0.05 and abs(t["delta_mae"]) >= C.SMALLEST_MEANINGFUL)
                if role == "baseline":   # signal check: p < .05 and pooled R2 > 0, no size threshold (§5)
                    r2p = res[a]["r2_pooled"]
                    t["meaningful"] = bool(t["p"] < 0.05 and t["delta_mae"] > 0 and r2p and np.mean(r2p) > 0)
                tests[f"{a}_vs_{b}"] = t
                print(f"[{role}] {a} vs {b}: ΔMAE {t['delta_mae']:+.3f} [95% CI {t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]"
                      f"  t({t['n_runs'] - 1}) = {t['t']:.2f}, p = {t['p']:.4f}  ({a} better in {t['a_wins']}/{t['n_runs']})"
                      f"{('  BEATS MEDIAN' if role == 'baseline' else '  MEANINGFUL') if t['meaningful'] else ''}")

    if len(folds) > 1:   # sensitivity: same comparisons with median-of-chunks aggregation
        print("\nSensitivity analysis: median of chunk predictions instead of the mean"
              + ("" if any(sens[k]["mae"] for k in build) else " (not applicable: this rung predicts per person)"))
        sres = dict(res, **{k: sens[k] for k in build})
        for key, t0_ in list(tests.items()):
            a, b = key.split("_vs_")
            if a in build and (b in build or b == "median_predictor") and sens[a]["mae"] and (b not in build or sens[b]["mae"]):
                t = stats.compare(sres, a, b, n_tr, n_te)
                t["role"] = "sensitivity:" + t0_["role"]
                if t0_["role"] == "baseline":
                    r2p = sens[a]["r2_pooled"]
                    t["meaningful"] = bool(t["p"] < 0.05 and t["delta_mae"] > 0 and r2p and np.mean(r2p) > 0)
                else:
                    t["meaningful"] = bool(t["p"] < 0.05 and abs(t["delta_mae"]) >= C.SMALLEST_MEANINGFUL)
                tests[key + "@median"] = t
                agree = "same verdict" if t["meaningful"] == t0_["meaningful"] else "VERDICT CHANGES"
                print(f"  {a} vs {b}: ΔMAE {t['delta_mae']:+.3f} [95% CI {t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}]"
                      f"  p = {t['p']:.4f}  ({agree})")

    provenance = {"environment": environment(), "seed": C.SEED, "wall_seconds": round(time.time() - t0),
                  "prereg_sha256": sha256(C.PREREG_FILE)}
    if not args.synthetic:
        cv_labels = data.load_labels().query("official_split in @C.CV_SPLITS")
        provenance.update({
            "cache_sha256": sha256(C.CHUNK_CACHE),
            "label_sha256": {s: sha256(os.path.join(C.LABEL_DIR, f"{s}_split.csv")) for s in ("train", "dev", "test")},
            "excluded_participants": sorted(set(int(p) for p in cv_labels.Participant_ID) - set(y_true))})
    settings = {k: getattr(C, k) for k in dir(C) if k.isupper() and isinstance(getattr(C, k), (int, float, str, tuple, dict, set))}
    settings = json.loads(json.dumps(settings, default=list))
    json.dump({"run_id": run_id, "rung": args.rung, "features": sets, "data": tag, "git_commit": code_version(),
               "n_participants": int(len(people)), "n_chunks": int(len(pid)), "runs": len(folds),
               "capacity": caps, "settings": settings, "provenance": provenance,
               "results": res, "sensitivity_median": sens, "subgroups": subgroups, "tests": tests}, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    json.dump(curves, open(os.path.join(out_dir, "curves.json"), "w"))
    print("\nwrote", out_dir)


if __name__ == "__main__":
    main()
