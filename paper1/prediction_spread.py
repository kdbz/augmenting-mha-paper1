"""Post hoc check behind the spread columns of Table 4 (added 5 Nov 2026; no refitting, no test-set data).
Reads the private cross-validation predictions.csv files; prints only aggregates.
Usage: python3 prediction_spread.py /path/to/augmenting-mh-assessment/paper1/results"""
import collections, csv, statistics as st, sys
RUNS = {"R1": "R1_edaic_20261007-123028", "R2": "R2_edaic_20261007-131857", "R3": "R3_edaic_20261007-141338",
        "R4": "R4_edaic_20261007-145409", "R5": "R5_edaic_20261007-154336"}
for rung, run in RUNS.items():
    folds = collections.defaultdict(list)
    for row in csv.DictReader(open(f"{sys.argv[1]}/{run}/predictions.csv")):
        if row["model"] in ("early_F3", "late_F3"):
            folds[row["model"], row["repeat"], row["run"]].append((float(row["y_pred"]), float(row["y_true"])))
    for model in ("early_F3", "late_F3"):
        sd = [st.stdev(p for p, _ in v) for k, v in folds.items() if k[0] == model]
        true = [st.stdev(t for _, t in v) for k, v in folds.items() if k[0] == model]
        print(f"{rung} {model}: folds {len(sd)}, median SD {st.median(sd):.2f}, range {min(sd):.2f}-{max(sd):.2f}, "
              f"folds SD<0.5: {sum(s < .5 for s in sd)}, observed SD median {st.median(true):.2f}")
