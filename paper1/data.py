"""E-DAIC OpenFace features as overlapping chunks, one PHQ-8 label per participant.

    python paper1/data.py            # build the chunk cache (once; ~15 min locally)

Each chunk carries its participant ID so predictions can be aggregated back to one number per
participant, and so every split can be made at participant level.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def load_labels():
    frames = []
    for name in ("train", "dev", "test"):
        df = pd.read_csv(os.path.join(C.LABEL_DIR, f"{name}_split.csv"))
        df.columns = [c.strip() for c in df.columns]
        df["official_split"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()   # some entries carry a trailing space
    return df[["Participant_ID", "PHQ_Score", "official_split", "Gender"]].dropna()


def build_cache():
    lab = load_labels()
    au_cols = None
    out = {c: [] for c in C.CHANNELS}
    pid, y, split, dropped = [], [], [], []

    for _, row in lab.iterrows():
        p = int(row.Participant_ID)
        if p in C.BROKEN:
            dropped.append((p, "broken interview")); continue
        f = os.path.join(C.DATA_ROOT, "data", f"{p}_P", "features", f"{p}_OpenFace2.1.0_Pose_gaze_AUs.csv")
        if not os.path.exists(f):
            dropped.append((p, "no OpenFace file")); continue
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        if au_cols is None:
            au_cols = [c for c in df.columns if c.startswith("AU")]
        d = df.loc[(df["success"] == 1) & (df["confidence"] >= C.CONF_MIN)]
        if len(d) < C.CHUNK:
            dropped.append((p, f"only {len(d)} valid frames")); continue

        cols = {"pose": C.POSE, "gaze": C.GAZE, "au": au_cols}
        arr = {c: d[cols[c]].to_numpy(np.float32) for c in C.CHANNELS}
        for s in range(0, len(d) - C.CHUNK + 1, C.STRIDE):
            sl = slice(s, s + C.CHUNK, C.SUBSAMPLE)
            for c in C.CHANNELS:
                out[c].append(arr[c][sl])
            pid.append(p); y.append(float(row.PHQ_Score)); split.append(row.official_split)
        print(f"  {p}  valid frames {len(d):6d}", flush=True)

    os.makedirs(C.CACHE_DIR, exist_ok=True)
    np.savez_compressed(C.CHUNK_CACHE, **{c: np.stack(out[c]) for c in C.CHANNELS},
                        pid=np.array(pid), y=np.array(y, np.float32), split=np.array(split),
                        au_cols=np.array(au_cols))
    for p, why in dropped:
        print(f"  dropped {p}: {why}")
    print("cache ->", C.CHUNK_CACHE)


def load_chunks(splits=C.CV_SPLITS):
    """Chunks for the requested official splits. The CV set never includes 'test'."""
    z = np.load(C.CHUNK_CACHE, allow_pickle=True)
    keep = np.isin(z["split"], splits)
    X = {c: z[c][keep].astype(np.float32) for c in C.CHANNELS}
    assert all(X[c].shape[2] == d for c, d in C.CHANNELS.items()), "channel widths differ from config"
    return X, z["y"][keep].astype(np.float32), z["pid"][keep], z["split"][keep]


def make_synthetic(n_people=200, chunks=6, steps=40, seed=C.SEED):
    """Positive control: PHQ-8 built from a real gaze x AU interaction (ported from exp/dry_run.py).

    Early fusion can form the product of two channels; late fusion's linear head cannot. If the
    pipeline stops recovering that gap, the pipeline is broken. This is not a simulation of E-DAIC.
    """
    rng = np.random.default_rng(seed)
    traits = {c: rng.standard_normal(n_people).astype(np.float32) for c in C.CHANNELS}
    y_person = (10 + 1.2 * traits["gaze"] + 0.8 * traits["au"]
                + 3.0 * traits["gaze"] * traits["au"] + 1.5 * traits["pose"] * traits["gaze"]
                + rng.normal(0, 1.5, n_people))
    y_person = np.clip(y_person, 0, 24).astype(np.float32)

    pid = np.repeat(np.arange(n_people), chunks)
    X = {c: (rng.standard_normal((len(pid), steps, d)) + traits[c][pid, None, None]).astype(np.float32)
         for c, d in C.CHANNELS.items()}
    return X, y_person[pid], pid, np.full(len(pid), "train")


if __name__ == "__main__":
    build_cache()
