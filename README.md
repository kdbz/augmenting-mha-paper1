# Early versus late fusion of visual features for estimating depressive symptom severity

Code, protocol and aggregate results for the paper *Early versus late fusion of visual features for estimating depressive symptom severity: a pre-specified evaluation on E-DAIC* (under review).

The study compares two ways of combining facial action units, gaze and head pose from interview video to estimate Patient Health Questionnaire-8 (PHQ-8) scores in the Extended Distress Analysis Interview Corpus (E-DAIC):

- **early fusion:** the three channels are joined before a shared encoder;
- **late fusion:** each channel has its own encoder, and the outputs are joined afterwards.

The two designs are capacity-matched on five recurrent architectures (R1 to R5). They are evaluated with participant-level five-fold cross-validation repeated ten times (n = 218), plus one evaluation on the official test partition (n = 56). Both are compared against two constant predictors: the training median and the training mean.

**Result:** neither design beat both constant predictors, in cross-validation or on test, so the two designs could not be ranked.

This repository is a snapshot without the development history. The analysis code is exactly as frozen on 7 October 2026 (commit `e25ff75`), before the official test partition was scored. The only additions after the freeze are the result files of three later runs (tuning, the single official test evaluation, and a post hoc positive control) and the script for that control.

## Contents

| Path | What it is |
|---|---|
| `paper1/PREREGISTRATION.md` | The protocol, committed on 6 October 2026 before any real-data run |
| `paper1/config.py` | Every setting (paths, rungs, training, folds, seeds) |
| `paper1/data.py` | Builds the chunk cache from the OpenFace feature files; generates the synthetic positive-control data |
| `paper1/models.py` | Early, late and single-channel models for each rung, with capacity matching |
| `paper1/protocol.py` | Participant-level repeated cross-validation, inner early stopping, baselines |
| `paper1/stats.py` | Corrected resampled *t* test (Nadeau and Bengio) and bootstrap confidence intervals |
| `paper1/run.py` | Entry point; writes `results/<run>/results.json`, stamped with the git commit |
| `paper1/test_protocol.py` | Leakage and protocol tests, on synthetic data only |
| `paper1/tune.py` | Tuning on inner-validation folds only; outer test folds are never scored |
| `paper1/test_once.py` | The single official test evaluation: retrains on train + development, scores the 56 test participants once |
| `paper1/positive_control.py` | Post hoc real-data positive control (gender as the target), aggregate output only |
| `paper1/results/` | Aggregate results, learning curves and logs for every run reported in the paper |
| `requirements.txt` | Python environment |

The result files hold aggregates only, such as means across folds and repeats and subgroup summaries. They contain no participant-level labels or predictions.

## Data

E-DAIC is not included and cannot be redistributed. It is available on request from the University of Southern California Institute for Creative Technologies, under its End-User License Agreement (https://dcapswoz.ict.usc.edu/).

After access is granted, place the corpus so that the following paths exist:

```
wwwedaic/labels2019/{train,dev,test}_split.csv, Detailed_PHQ8_Labels.csv
wwwedaic/data/<ID>_P/features/   (OpenFace 2.1.0 outputs supplied with the corpus)
```

Put `wwwedaic/` next to `paper1/`, or change `DATA_ROOT` in `paper1/config.py`. On Kaggle, the code reads from `/kaggle/input/edaic`.

## Running

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python paper1/test_protocol.py                 # leakage tests, no data needed
.venv/bin/python paper1/run.py --synthetic --repeats 1   # synthetic positive control, about 20 s
.venv/bin/python paper1/data.py                          # build the chunk cache from E-DAIC
.venv/bin/python paper1/run.py --rung R1                 # full 5 x 10 cross-validation for one rung
```

Rungs are `R1` to `R5`. `--features F3` restricts a run to the three-channel comparison. A GPU is recommended for R3 to R5; the paper's runs used Kaggle (2 × T4).

## Citation

If you use this code, please cite the paper. Full citation details will be added on publication.

## Licence

MIT. The code only; the licence does not cover E-DAIC.
