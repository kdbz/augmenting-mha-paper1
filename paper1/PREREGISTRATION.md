# Pre-registration: fusion level for visual PHQ-8 estimation on E-DAIC

> **DRAFT. Fill in every [bracket] in your own words, then commit.**
> `run.py` will not touch real data until this file is committed and unmodified.
> The commit timestamp is the pre-registration. Anything changed after it is reported as a deviation.

**Authors:** Wallace Muchiri, Joseph Orero, Stephen Kimani
**Date committed:** (the git commit date)

## 0. Prior data contact

A preliminary analysis on E-DAIC (17 Sep 2026, earlier pipeline) found no early-vs-late difference,
and neither arm beat a mean predictor. This pre-registration was written after that result. It fixes
the protocol, model ladder and decision rules before any run of the `paper1/` code on real data.

## 1. Question

Does joint (early) encoding of pose, gaze and facial action units predict PHQ-8 severity better
than per-channel (late) encoding of the same channels, at matched model capacity?

## 2. Data

- E-DAIC, OpenFace 2.1.0 pose (6), gaze (8) and AU (35) features. Frames kept if `success == 1` and
  `confidence >= 0.90`. Participant 627 excluded (frame count inconsistent with its audio).
- 700-frame chunks, 70% stride, every 5th frame (140 steps). Each chunk takes its participant's PHQ-8 score.
- Cross-validation on the official train + dev splits (218 participants after exclusions).
- The official test split (56) is held out. It is used once, after the code freeze, to evaluate
  the frozen models retrained on train + dev.

## 3. Task and metrics

- Task: PHQ-8 Regression.
- Primary error metric: participant-level MAE, where chunk predictions are averaged per participant. RMSE is reported alongside.
- Primary goodness-of-fit metric: R², following Danylenko & Unold (2026, pp. 6 and 14, Table 4 item 3). MAE and RMSE have no upper bound and do not show how much of the variation in PHQ-8 a model explains, so a low MAE can come from the narrow range of the target rather than from the model. R² in [0, 1] is the share of variance explained; a negative R² means the model fits worse than predicting the mean. Their own audio model had an MAE comparable to the published literature and a negative R².
- R² is reported two ways: the mean over the 50 test folds, and the pooled out-of-fold R² of each repeat (all 218 predictions of one repeat scored together, giving 10 values). The pooled value is the one interpreted, because folds of about 44 participants make per-fold R² unstable.
- Error-distribution plots (predicted vs observed PHQ-8 per participant) are reported for the primary comparison (Table 4 item 8).

## 4. Protocol

- 5-fold cross-validation over participants, repeated 10 times with a fresh shuffle each repeat (50 fits per model).
- 15% of each fold's training participants are held out for early stopping only. The scaler is fitted on training participants only.
- Identical training for every model: Adam, lr 1e-3, weight decay 1e-4, batch 128, L1 loss, up to 40 epochs, patience 6.

## 4a. Reproducibility

- Real-data runs refuse to start unless every tracked file in `paper1/` is committed. Each results file records the commit, the SHA-256 of the chunk cache and the label files, the excluded participants, the software versions and the GPU.
- Seeds are fixed per fit (`SEED + run`) and cuDNN runs in deterministic mode. Early and late arms see the same folds and the same seeds.
- Every run writes one folder, `results/<rung>_<data>_<timestamp>/`, with: `run.log` (full console output), `results.json` (settings, provenance, parameter counts, per-fit MAE / RMSE / R² / epochs, pooled R², all tests), `curves.json` (per-epoch training and inner-validation MAE for every fit), `predictions.csv` (every participant-level prediction of every fit), `checkpoints/` (best-epoch weights of every fit) and a copy of `config.py` (Table 4 item 11).
- Dependencies are pinned in `requirements.txt`. Code, `run.log`, `results.json` and `curves.json` are released publicly at submission. `predictions.csv` and `checkpoints/` contain E-DAIC-derived labels and models, so they are kept out of the public repository and shared only with people who hold an E-DAIC agreement. E-DAIC itself is available on request from USC ICT.
- The official test split is scored once, after the code freeze, and that run is reported whatever it shows.

## 5. Comparisons

- **Primary:** early vs late fusion on all three channels (F3, §7), compared on every rung that is run.
- **Secondary:** early vs late fusion on AU + gaze (F2), on R1 and R2.
- **Baselines:** median predictor (the stricter one for MAE), mean predictor, and session statistics with ridge regression.
- **References:** each channel on its own (AU, gaze, pose), using the same encoder, on R1 and R2.
- **Test:** Nadeau–Bengio corrected resampled *t*-test, α = .05, plus a 95% CI on the paired MAE difference built from the same corrected variance, so the CI excludes 0 exactly when p < .05.
- **Multiplicity:** only the primary comparison (early vs late on F3) is confirmatory. All other
  tests (secondary, baseline, ablation and sensitivity) are reported without correction for
  multiple comparisons and are interpreted descriptively.
- **Subgroups (descriptive):** for the median predictor, the ridge baseline and both F3 arms, pooled
  MAE and R² are reported separately for women and men (self-reported gender in the E-DAIC labels).
  No test is run on the subgroups and no conclusion is drawn from them; they are reported for
  transparency (TRIPOD+AI).
- **Sensitivity analysis (aggregation):** participant predictions are the mean of their chunk
  predictions. Every model-vs-model and model-vs-median comparison is repeated with the median of
  the chunk predictions instead, from the same fits, to check that no single outlying chunk drives
  the result. The mean remains primary; any change of verdict under the median is reported.
- **Smallest meaningful difference:** 1.0 PHQ-8 points, because the severity bands are 5 points
  wide and a smaller difference moves almost no one between bands. Differences that are
  statistically detectable but below 1.0 are reported as "detectable, not meaningful".
- **Decision rule.** "A > B" means corrected *t* p < .05 **and** the mean MAE difference ≥ 1.0.
  "A ≈ B" means anything else. This rule applies to model-vs-model comparisons (fusion and ablation).
- **Beating the median.** A model "beats the median" when the corrected *t* gives p < .05 in its
  favour **and** its mean pooled R² is above 0. No size threshold applies: this comparison asks
  whether the model carries any signal, not whether it is meaningfully better than another model.
  The median is the best constant predictor for MAE and the mean is the best for squared error, so
  a model must beat both.
  The conclusion is drawn from F3 on the highest rung run. Lower rungs and the feature ladder are
  reported in full, and any disagreement with that conclusion is stated in the paper.

## 6. Model ladder

Rungs are climbed in this order. Climbing can stop at any rung, but no rung is added, skipped or
reordered because of a result. Every rung that is run is reported.

| Rung | Change | Status |
|---|---|---|
| R1 | LSTM encoder, last hidden state | committed |
| R2 | + attention pooling over frames | committed |
| R3 | + attention over chunks, replacing the mean | optional |
| R4 | bidirectional LSTM | optional |
| R5 | + multi-head self-attention before pooling | optional |

Only LSTM-based encoders are used; the encoder is held constant so that the arms differ only in
fusion level. Underfitting is checked by the logged R1 tuning, the attention rungs and the
non-neural session-statistics ridge baseline. Other encoder families are out of scope and are
named as a limitation.

R3–R5 are optional and run in this order, on F3 only. Climbing stops at the code freeze (Fri 16 Oct 2026) or at
the first rung whose 50-fit run would not finish by then, whichever comes first. Results never
decide when climbing stops.

Both arms receive each change, and the arms are re-matched to within 2% of parameters on every
rung. The late arm never has attention across channels.

**Tuning allowed on R1 before the main runs:** learning rate, epochs and patience, feature and
target scaling. It is chosen on inner validation only, applied identically to both arms and to
every later rung, and logged.

## 7. Feature ladder (ablation study)

The feature ladder is the study's ablation (Table 4 item 9). Channels are added in this order, fixed before any real-data run: AU first (the largest channel,
35 columns), then gaze, then pose. The order is not changed because of a result.

| Step | Channels | Fusion | Run on |
|---|---|---|---|
| F1 | AU | none | R1, R2 |
| — | gaze alone, pose alone (references) | none | R1, R2 |
| F2 | AU + gaze | early vs late | R1, R2 |
| F3 | AU + gaze + pose | early vs late (primary) | R1, R2, R3–R5 |

Every step on a rung uses that rung's frozen settings. The ablation is reported as secondary,
with the same test and decision rule, within each arm: F2 vs AU alone (does adding gaze help?)
and F3 vs F2 (does adding pose help?). It shows whether each added channel lowers MAE and raises
R², and at which step early and late fusion start to differ.
Committed configurations: 14 (R1, R2). Optional: 6 (R3–R5). Each is fitted 50 times.

## 8. What we will conclude

| Outcome | Conclusion |
|---|---|
| Early ≈ late, both beat the median | At this N, fusion level does not measurably change PHQ-8 prediction from these channels; separate extraction costs nothing we can detect. |
| Early > late, both beat the median | Joint encoding recovers signal that per-channel encoding discards; separate extraction has a measurable cost. |
| Late > early, both beat the median | Per-channel encoding outperforms joint encoding at matched capacity; the field's default is defensible. |
| Only one arm beats the median | That arm is reported as the only one carrying signal; the fusion-level comparison is reported but read as secondary. |
| Neither beats the median | Consistent with Danylenko & Unold (2026): under leakage control, visual-channel models on E-DAIC do not beat a trivial predictor, and fusion level is not the operative question. |

## Amendments before data contact

| Date | Change | Before any real-data run of this code? |
|---|---|---|
| 2026-10-06 | §5: multiplicity statement and descriptive gender subgroups added | Yes |
