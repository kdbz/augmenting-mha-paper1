"""Leakage and protocol tests. They run on synthetic data only, so they are safe before pre-registration.

    python paper1/test_protocol.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import data
import models
import protocol as P
import stats


def test_no_participant_in_two_folds():
    people = np.arange(200)
    for _, tr, te in P.outer_folds(people, n_repeats=3):
        assert not set(tr) & set(te)
        fit, val = P.inner_split(tr, seed=1)
        assert not set(fit) & set(val) and not set(val) & set(te)
        assert set(fit) | set(val) == set(tr)


def test_every_participant_tested_once_per_repeat():
    people = np.arange(200)
    folds = list(P.outer_folds(people, n_repeats=2))
    for rep in range(2):
        tested = np.concatenate([te for r, _, te in folds if r == rep])
        assert sorted(tested) == list(people)


def test_folds_differ_across_repeats():
    """The GroupKFold bug of 17 Sep: every repeat produced identical folds."""
    people = np.arange(200)
    folds = list(P.outer_folds(people, n_repeats=2))
    first = [set(te) for r, _, te in folds if r == 0]
    second = [set(te) for r, _, te in folds if r == 1]
    assert first != second


def test_scaler_uses_training_chunks_only():
    X, y, pid, _ = data.make_synthetic(n_people=40, chunks=2, steps=5)
    idx = np.where(pid < 20)[0]
    X_shifted = {c: X[c].copy() for c in C.CHANNELS}
    X_shifted["au"][pid >= 20] += 1000.0              # change only the non-training participants
    a, b = P.Scaler(X, idx).st["au"][0], P.Scaler(X_shifted, idx).st["au"][0]
    assert np.allclose(a, b)


def test_official_test_split_never_in_cv_set():
    assert "test" not in C.CV_SPLITS
    if os.path.exists(C.CHUNK_CACHE):
        _, _, _, split = data.load_chunks()
        assert "test" not in set(split)


def test_capacity_matched_on_every_rung():
    for rung, spec in C.RUNGS.items():
        for fs, ch in C.FEATURE_SETS.items():
            _, late, early, gap = models.match_capacity(spec["pooling"], ch, spec.get("aggregation", "mean"), spec.get("bidirectional", False), spec.get("heads", 0))
            assert gap <= C.CAPACITY_TOL, (rung, fs, late, early)


def test_feature_sets_use_only_their_channels():
    """F2 must not see pose: changing pose leaves both F2 arms' output unchanged."""
    torch.manual_seed(0)
    b = {c: torch.randn(4, 10, d) for c, d in C.CHANNELS.items()}
    b2 = dict(b, pose=torch.randn(4, 10, C.CHANNELS["pose"]))
    for rung, spec in C.RUNGS.items():
        counts = [1, 3] if spec.get("aggregation") == "attention" else None     # person-level rungs: 4 chunks = 2 people
        for name in ("early_F2", "late_F2"):
            m = models.builders(rung)[name]().eval()
            assert torch.allclose(m(b, counts), m(b2, counts)), (rung, name)


def test_pooled_r2_needs_every_participant():
    y_true = {1: 0.0, 2: 10.0, 3: 20.0, 4: 5.0}
    folds = [(0, [1, 2]), (0, [3, 4]), (1, [1, 3])]
    preds = [[0.0, 10.0], [20.0, 5.0], [1.0, 1.0]]
    assert P.pooled_r2(preds, folds, y_true) == [1.0]     # repeat 1 is incomplete, so it is skipped


def test_late_arm_has_no_cross_channel_path():
    """Changing one channel's input must move only that channel's embedding in the late arm."""
    for rung, spec in C.RUNGS.items():                       # attention stays inside each channel on every rung
        torch.manual_seed(0)
        m = models.LateFusion(pooling=spec["pooling"], bidirectional=spec.get("bidirectional", False), heads=spec.get("heads", 0)).eval()
        b = {c: torch.randn(4, 10, d) for c, d in C.CHANNELS.items()}
        emb = {c: m.encoders[c](b[c]) for c in C.CHANNELS}
        b2 = dict(b, au=torch.randn(4, 10, C.CHANNELS["au"]))
        emb2 = {c: m.encoders[c](b2[c]) for c in C.CHANNELS}
        assert torch.allclose(emb["pose"], emb2["pose"]) and torch.allclose(emb["gaze"], emb2["gaze"]), rung
        assert not torch.allclose(emb["au"], emb2["au"]), rung


def test_ci_agrees_with_corrected_t():
    """The CI and the test use the same variance: the CI excludes 0 exactly when p < .05."""
    rng = np.random.default_rng(0)
    for shift in np.linspace(0, 1.5, 16):
        d = rng.normal(shift, 1.0, 50)
        _, p = stats.nadeau_bengio(d, 174, 44)
        lo, hi = stats.corrected_ci(d, 174, 44)
        assert (p < 0.05) == (lo > 0 or hi < 0), (shift, p, lo, hi)


def test_same_seed_same_predictions():
    """A fit is reproducible: same data, same fold, same seed gives the same predictions."""
    X, y, pid, _ = data.make_synthetic(n_people=40, chunks=3, steps=10)
    people = np.unique(pid)
    tr, te = people[:30], people[30:]
    a, _, _, _ = P.fit_predict(models.builders("R1")["early_F3"], X, y, pid, tr, te, seed=7)
    b, _, _, _ = P.fit_predict(models.builders("R1")["early_F3"], X, y, pid, tr, te, seed=7)
    assert np.allclose(a, b, atol=1e-5), np.abs(a - b).max()


def test_median_aggregation_resists_one_bad_chunk():
    pid = np.array([1, 1, 1, 1, 1, 2, 2, 2])
    pred = np.array([10, 10, 10, 10, 60, 5, 6, 7], np.float32)   # one wild chunk for participant 1
    assert np.allclose(P.aggregate(pred, pid, [1, 2], "median"), [10, 6])
    assert np.allclose(P.aggregate(pred, pid, [1, 2]), [20, 6])


def test_subgroup_scores_split_by_group():
    y_true = {1: 0.0, 2: 10.0, 3: 4.0, 4: 8.0}
    folds = [(0, [1, 2]), (0, [3, 4])]
    preds = [[0.0, 10.0], [5.0, 7.0]]
    out = P.subgroup_scores(preds, folds, y_true, {1: "a", 2: "a", 3: "b", 4: "b"})
    assert out["a"]["n"] == 2 and out["a"]["mae_pooled"] == [0.0]
    assert out["b"]["mae_pooled"] == [1.0]


def test_tuning_fit_makes_no_test_prediction():
    """tune.py passes an empty test set: inner validation runs, nothing outside the training people is predicted."""
    X, y, pid, _ = data.make_synthetic(n_people=40, chunks=3, steps=10)
    people = np.unique(pid)
    hist = []
    pred, _, _, chunk_pred = P.fit_predict(models.builders("R1")["late_F3"], X, y, pid, people[:30],
                                           np.array([], dtype=people.dtype), seed=0, history=hist)
    assert len(pred) == 0 and len(chunk_pred) == 0 and len(hist) > 0


def test_frame_attention_is_a_weighted_average():
    """R2: weights are positive, sum to 1 over frames, and the pooled vector lies inside the range of the frames."""
    torch.manual_seed(0)
    enc = models.Encoder(C.CHANNELS["au"], 16, "attention").eval()
    x = torch.randn(3, 12, C.CHANNELS["au"])
    out, _ = enc.lstm(x)
    w = enc.frame_weights(out)
    assert w.shape == (3, 12) and (w > 0).all() and torch.allclose(w.sum(1), torch.ones(3))
    pooled = enc(x)
    assert (pooled <= out.max(1).values + 1e-6).all() and (pooled >= out.min(1).values - 1e-6).all()


def test_r2_adds_parameters_and_stays_matched():
    """R2 differs from R1 only by the attention scorer; capacity is re-matched for it."""
    for fs, ch in C.FEATURE_SETS.items():
        h1, late1, _, _ = models.match_capacity("last", ch)
        h2, late2, early2, gap2 = models.match_capacity("attention", ch)
        per_branch = 32 * 16 + 16 + 16 + 1                   # Linear(32,16) + Linear(16,1)
        assert late2 - late1 == per_branch * len(ch), fs
        assert gap2 <= C.CAPACITY_TOL, fs


def test_r3_padding_changes_nothing():
    """A person's prediction is the same alone or batched with someone who has more chunks (padding is masked)."""
    torch.manual_seed(0)
    for rung in ("R3", "R4", "R5"):
        for name in ("late_F3", "early_F3"):
            m = models.builders(rung, ("F3",), singles=False)[name]().eval()
            b = {c: torch.randn(7, 10, d) for c, d in C.CHANNELS.items()}     # person A: 2 chunks, person B: 5 chunks
            both = m(b, [2, 5])
            alone = m({c: v[:2] for c, v in b.items()}, [2])
            assert torch.allclose(both[0], alone[0], atol=1e-6), (rung, name)


def test_r3_chunk_attention_ignores_padding():
    pool = models.ChunkPool(16)
    e = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, False, False, False], [True] * 5])
    w = pool.weights(e, mask)
    assert torch.allclose(w.sum(1), torch.ones(2)) and (w[0, 2:] == 0).all() and (w[0, :2] > 0).all()


def test_r3_late_arm_keeps_channels_apart():
    """At R3 each channel pools its own chunks: changing AU moves only the AU part of the person embedding."""
    for rung in ("R3", "R4", "R5"):
        torch.manual_seed(0)
        m = models.builders(rung, ("F3",), singles=False)["late_F3"]().eval()
        b = {c: torch.randn(5, 10, d) for c, d in C.CHANNELS.items()}
        b2 = dict(b, au=torch.randn(5, 10, C.CHANNELS["au"]))
        h = m.encoders["au"].out_dim
        e1, e2 = m.embed(b, [2, 3]), m.embed(b2, [2, 3])
        order = list(m.channels)
        for c in order:
            part = slice(order.index(c) * h, (order.index(c) + 1) * h)
            same = torch.allclose(e1[:, part], e2[:, part])
            assert same == (c != "au"), (rung, c)


def test_r3_fit_predicts_one_value_per_person():
    X, y, pid, _ = data.make_synthetic(n_people=40, chunks=3, steps=10)
    people = np.unique(pid)
    pred, _, _, chunk_pred = P.fit_predict(models.builders("R3")["early_F3"], X, y, pid, people[:30], people[30:], seed=0)
    assert pred.shape == (10,) and chunk_pred is None


def test_r4_reads_both_directions():
    """R4: the encoder is bidirectional, so a change in the last frame moves the summary of the first frames too."""
    torch.manual_seed(0)
    enc = models.Encoder(C.CHANNELS["au"], 8, "attention", bidirectional=True).eval()
    x = torch.randn(1, 12, C.CHANNELS["au"])
    x2 = x.clone(); x2[0, -1] += 5.0
    o1, _ = enc.lstm(x); o2, _ = enc.lstm(x2)
    assert enc.out_dim == 16 and o1.shape[-1] == 16
    assert not torch.allclose(o1[0, 0, 8:], o2[0, 0, 8:])       # backward half of step 1 sees the last frame
    assert torch.allclose(o1[0, 0, :8], o2[0, 0, :8])           # forward half of step 1 does not


def test_r5_self_attention_stays_inside_a_window():
    """R5: frames attend to frames of the same window only; another window's content cannot leak in."""
    torch.manual_seed(0)
    enc = models.Encoder(C.CHANNELS["au"], 8, "attention", bidirectional=True, heads=4).eval()
    x = torch.randn(3, 12, C.CHANNELS["au"])
    x2 = x.clone(); x2[2] = torch.randn(12, C.CHANNELS["au"])
    assert torch.allclose(enc(x)[:2], enc(x2)[:2]) and not torch.allclose(enc(x)[2], enc(x2)[2])


def test_r5_widths_divide_by_heads():
    for fs, ch in C.FEATURE_SETS.items():
        hd = C.RUNGS["R5"]["heads"]
        h, _, _, gap = models.match_capacity("attention", ch, "attention", True, hd)
        assert (2 * h) % hd == 0 and gap <= C.CAPACITY_TOL, fs


def test_r5_checkpointing_changes_no_numbers():
    """Gradient checkpointing (R5, training) gives the same outputs and gradients as a plain pass."""
    torch.manual_seed(0)
    enc = models.Encoder(C.CHANNELS["au"], 8, "attention", bidirectional=True, heads=2).train()
    x = torch.randn(10, 12, C.CHANNELS["au"])
    a = enc.encode(x, slice_size=3); a.sum().backward()
    g1 = [p.grad.clone() for p in enc.parameters()]
    enc.zero_grad()
    b = enc(x); b.sum().backward()
    assert torch.allclose(a, b, atol=1e-6)
    assert all(torch.allclose(u, p.grad, atol=1e-5) for u, p in zip(g1, enc.parameters()))


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"  pass  {name}")
    print(f"{len(tests)} tests passed")
