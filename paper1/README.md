# paper1 — fused visual features for PHQ-8 severity

Early vs late fusion of OpenFace pose, gaze and AUs on E-DAIC, climbed rung by rung (R1 LSTM → R2 attention → …).
Plan: `~/Documents/mbuku/paper-1-visual/implementation-plan.md`.

| File | Role |
|---|---|
| `config.py` | every setting; Kaggle paths behind `IS_KAGGLE` |
| `data.py` | chunk cache from the OpenFace CSVs; synthetic positive-control data |
| `models.py` | late / early / single-channel arms, capacity matching per rung |
| `protocol.py` | participant-level repeated CV, inner early stopping, baselines |
| `stats.py` | Nadeau–Bengio corrected *t*, bootstrap CI |
| `run.py` | entry point; writes `results/*.json` stamped with the git commit |
| `test_protocol.py` | leakage tests (synthetic only) |
| `PREREGISTRATION.md` | must be committed before any real-data run |

```bash
.venv/bin/python paper1/test_protocol.py                  # 7 leakage tests
.venv/bin/python paper1/run.py --synthetic --repeats 1    # positive control, ~20 s
.venv/bin/python paper1/data.py                           # rebuild cache (paper1/cache/, gitignored)
.venv/bin/python paper1/run.py --rung R1 --diagnose       # one fold + learning curves (needs pre-registration)
.venv/bin/python paper1/run.py --rung R1                  # full 5 x 10 CV
```
