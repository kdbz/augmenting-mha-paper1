"""Leakage-controlled evaluation: every split, scaler and stopping decision is made at participant level.

Outer loop  - repeated K-fold over participants (fresh shuffle per repeat).
Inner split - a share of the training participants held out for early stopping only.
Scaling     - fitted on the training participants of that fold, never on validation or test.
Aggregation - chunk predictions are averaged to one prediction per participant before scoring.
"""
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import config as C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True   # same seed, same fold -> same numbers on the same machine
torch.backends.cudnn.benchmark = False


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------------------------------------ splits

def outer_folds(people, n_folds=C.N_FOLDS, n_repeats=C.N_REPEATS, seed=C.SEED):
    """Yield (repeat, train_people, test_people). Assignments differ across repeats."""
    for rep in range(n_repeats):
        kf = KFold(n_folds, shuffle=True, random_state=seed + rep)
        for tr, te in kf.split(people):
            yield rep, people[tr], people[te]


def inner_split(train_people, seed, share=C.INNER_VAL):
    rng = np.random.default_rng(seed)
    val = rng.choice(train_people, max(2, int(round(share * len(train_people)))), replace=False)
    return np.setdiff1d(train_people, val), val


def person_labels(y, pid, people):
    return np.array([y[pid == p][0] for p in people], np.float32)


# ------------------------------------------------------------------------------------------------ metrics

def score(pred, true):
    e = pred - true
    denom = float(np.sum((true - true.mean()) ** 2))
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": float(1 - np.sum(e ** 2) / denom) if denom > 0 else float("nan")}


def aggregate(chunk_pred, chunk_pid, people, how="mean"):
    """One prediction per participant from their chunks. Mean is primary; median is the
    pre-registered sensitivity analysis (PREREGISTRATION.md §5)."""
    f = np.median if how == "median" else np.mean
    return np.array([f(chunk_pred[chunk_pid == p]) for p in people], np.float32)


# ------------------------------------------------------------------------------------------------ deep models

class Scaler:
    """Per-channel feature standardisation from training chunks only."""

    def __init__(self, X, idx):
        self.st = {c: (X[c][idx].mean((0, 1)), X[c][idx].std((0, 1)) + 1e-8) for c in C.CHANNELS}

    def batch(self, X, idx):
        return {c: torch.from_numpy((X[c][idx] - self.st[c][0]) / self.st[c][1]).to(DEVICE) for c in C.CHANNELS}


def predict(model, scaler, X, idx):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            out.append(model(scaler.batch(X, idx[i:i + 512])).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def pooled_r2(preds, folds, y_true):
    """Out-of-fold R2 per repeat: each repeat predicts every participant once, so its five folds are
    pooled and scored against the full label variance. Steadier than averaging R2 over small folds."""
    out = {}
    for (rep, people), pred in zip(folds, preds):
        out.setdefault(rep, ({}, ))[0].update(dict(zip(people, pred)))
    r2 = []
    for rep, (d,) in sorted(out.items()):
        if len(d) == len(y_true):
            p = np.array([d[k] for k in y_true], float); t = np.array(list(y_true.values()), float)
            r2.append(score(p, t)["r2"])
    return r2


def subgroup_scores(preds, folds, y_true, group):
    """Descriptive only (PREREGISTRATION.md §5): pooled MAE and R2 per repeat within each subgroup.
    `group` maps participant -> label. No test is run on these numbers."""
    by_rep = {}
    for (rep, people), pred in zip(folds, preds):
        by_rep.setdefault(rep, {}).update(dict(zip(people, pred)))
    out = {}
    for g in sorted(set(group.values())):
        members = [p for p in y_true if group.get(p) == g]
        mae, r2 = [], []
        for rep, d in sorted(by_rep.items()):
            if all(p in d for p in members):
                sc = score(np.array([d[p] for p in members], float), np.array([y_true[p] for p in members], float))
                mae.append(sc["mae"]); r2.append(sc["r2"])
        out[g] = {"n": len(members), "mae_pooled": mae, "r2_pooled": r2}
    return out


def person_batch(scaler, X, pid, people, chunks_of):
    """All chunks of `people`, grouped person by person, and how many each has (for models.by_person)."""
    idx = np.concatenate([chunks_of[p] for p in people])
    return scaler.batch(X, idx), [len(chunks_of[p]) for p in people]


def predict_people(model, scaler, X, pid, people, chunks_of, size=C.PERSON_BATCH):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(people), size):
            b, counts = person_batch(scaler, X, pid, people[i:i + size], chunks_of)
            out.append(model(b, counts).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def fit_predict_people(make_model, X, y, pid, train_people, test_people, seed, history=None):
    """R3: the model sees all of a person's chunks at once and predicts one score per person.
    Same split, scaler, median start, loss, optimiser and early stopping as fit_predict; the batch is
    C.PERSON_BATCH people instead of C.BATCH chunks. There are no chunk predictions, so the fourth
    return value is None and the median-of-chunks sensitivity analysis does not apply."""
    seed_everything(seed)
    fit_people, val_people = inner_split(train_people, seed)
    chunks_of = {p: np.where(pid == p)[0] for p in np.concatenate([train_people, test_people])}
    scaler = Scaler(X, np.where(np.isin(pid, fit_people))[0])
    model = make_model().to(DEVICE)
    y_fit = person_labels(y, pid, fit_people)
    with torch.no_grad():
        model.head.bias.fill_(float(np.median(y_fit)))
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    loss_fn = nn.L1Loss()
    y_val = person_labels(y, pid, val_people)
    label = dict(zip(fit_people, y_fit))
    rng = np.random.default_rng(seed)

    best, best_state, stale = np.inf, None, 0
    for epoch in range(C.MAX_EPOCHS):
        model.train(); order = rng.permutation(fit_people); losses = []
        for i in range(0, len(order), C.PERSON_BATCH):
            ppl = order[i:i + C.PERSON_BATCH]
            b, counts = person_batch(scaler, X, pid, ppl, chunks_of)
            opt.zero_grad()
            loss = loss_fn(model(b, counts), torch.tensor([label[p] for p in ppl], dtype=torch.float32, device=DEVICE))
            loss.backward(); opt.step(); losses.append(loss.item())
        val_mae = score(predict_people(model, scaler, X, pid, val_people, chunks_of), y_val)["mae"]
        if history is not None:
            history.append({"epoch": epoch, "train_person_mae": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best - 1e-4:
            best, best_state, stale = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= C.PATIENCE:
                break
    model.load_state_dict(best_state)
    return predict_people(model, scaler, X, pid, test_people, chunks_of), best_state, epoch + 1, None


def fit_predict(make_model, X, y, pid, train_people, test_people, seed, history=None):
    """Train with early stopping on inner-validation participants.
    Returns (per-participant mean predictions, best state_dict, epochs run, test-chunk predictions).
    Test-chunk predictions are in the order of the test chunks, so other aggregations can be computed.
    Person-level models (R3) are routed to fit_predict_people."""
    if getattr(make_model(), "aggregation", "mean") == "attention":
        return fit_predict_people(make_model, X, y, pid, train_people, test_people, seed, history)
    seed_everything(seed)
    fit_people, val_people = inner_split(train_people, seed)
    fit_idx = np.where(np.isin(pid, fit_people))[0]
    val_idx = np.where(np.isin(pid, val_people))[0]
    test_idx = np.where(np.isin(pid, test_people))[0]

    scaler = Scaler(X, fit_idx)
    model = make_model().to(DEVICE)
    with torch.no_grad():                                  # start from the median, not from zero
        model.head.bias.fill_(float(np.median(person_labels(y, pid, fit_people))))
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    loss_fn = nn.L1Loss()
    y_val = person_labels(y, pid, val_people)
    rng = np.random.default_rng(seed)

    best, best_state, stale = np.inf, None, 0
    for epoch in range(C.MAX_EPOCHS):
        model.train(); order = rng.permutation(fit_idx); losses = []
        for i in range(0, len(order), C.BATCH):
            b = order[i:i + C.BATCH]
            opt.zero_grad()
            loss = loss_fn(model(scaler.batch(X, b)), torch.from_numpy(y[b]).to(DEVICE))
            loss.backward(); opt.step(); losses.append(loss.item())
        val_mae = score(aggregate(predict(model, scaler, X, val_idx), pid[val_idx], val_people), y_val)["mae"]
        if history is not None:
            history.append({"epoch": epoch, "train_chunk_mae": float(np.mean(losses)), "val_mae": val_mae})
        if val_mae < best - 1e-4:
            best, best_state, stale = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= C.PATIENCE:
                break
    model.load_state_dict(best_state)
    chunk_pred = predict(model, scaler, X, test_idx)
    return aggregate(chunk_pred, pid[test_idx], test_people), best_state, epoch + 1, chunk_pred


# ------------------------------------------------------------------------------------------------ baselines

def functionals(X, pid, people):
    """Session statistics per feature: mean, SD, 10th/50th/90th percentile over all frames."""
    rows = []
    for p in people:
        feats = np.concatenate([X[c][pid == p].reshape(-1, X[c].shape[2]) for c in C.CHANNELS], 1)
        rows.append(np.concatenate([feats.mean(0), feats.std(0),
                                    *np.percentile(feats, [10, 50, 90], axis=0)]))
    return np.array(rows, np.float32)


def ridge_predict(X, y, pid, train_people, test_people):
    sc = StandardScaler().fit(functionals(X, pid, train_people))
    m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(
        sc.transform(functionals(X, pid, train_people)), person_labels(y, pid, train_people))
    return m.predict(sc.transform(functionals(X, pid, test_people))).astype(np.float32)
