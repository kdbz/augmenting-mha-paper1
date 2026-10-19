"""Paper 1 configuration: fused visual features for PHQ-8 severity on E-DAIC.

Every setting the experiment depends on lives here, so the pre-registration can point at one file.
Paths sit behind IS_KAGGLE in the same way as visual/config.py.
"""
import os

# ==============================================================================================================================
# 1. PATHS
# ==============================================================================================================================
IS_KAGGLE = os.path.exists("/kaggle/input")
HERE = os.path.dirname(os.path.abspath(__file__))

if IS_KAGGLE:
    DATA_ROOT = "/kaggle/input/edaic"                 # private dataset: labels2019/ + data/{ID}_P/features/
    CACHE_DIR = "/kaggle/working/cache"
    RESULTS_DIR = "/kaggle/working/results"
else:
    DATA_ROOT = os.path.join(HERE, "..", "wwwedaic")
    CACHE_DIR = os.path.join(HERE, "cache")
    RESULTS_DIR = os.path.join(HERE, "results")

LABEL_DIR = os.path.join(DATA_ROOT, "labels2019")
CHUNK_CACHE = os.path.join(CACHE_DIR, "edaic_chunks.npz")
PREREG_FILE = os.path.join(HERE, "PREREGISTRATION.md")

# ==============================================================================================================================
# 2. DATA
# ==============================================================================================================================
POSE = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
GAZE = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z",
        "gaze_angle_x", "gaze_angle_y"]
# AU columns (17 intensity *_r + 18 presence *_c) are read from the file header
CHANNELS = {"pose": 6, "gaze": 8, "au": 35}

CHUNK = 700                  # frames, ~23 s at 30 fps (visual/config.py: _chunk_max_size)
STRIDE = int(CHUNK * 0.70)   # overlapping chunks (visual/config.py: _chunk_stride)
SUBSAMPLE = 5                # every 5th frame -> 140 steps per chunk, ~6 fps
CONF_MIN = 0.90              # OpenFace frame kept only if success == 1 and confidence >= this
BROKEN = {627}               # visual/config.py: frame count inconsistent with the audio

# ==============================================================================================================================
# 3. PROTOCOL
# ==============================================================================================================================
SEED = 0
N_FOLDS = 5
N_REPEATS = 10               # 5 x 10 = 50 fits per model
INNER_VAL = 0.15             # share of training participants held out for early stopping
CV_SPLITS = ("train", "dev") # the official test split is never part of cross-validation
SMALLEST_MEANINGFUL = 1.0    # PHQ-8 points; a difference must also clear this to count (PREREGISTRATION.md §5)

# ==============================================================================================================================
# 4. TRAINING (identical for every arm on every rung)
# ==============================================================================================================================
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
PATIENCE = 6                 # epochs without inner-validation improvement before stopping
LOSS = "l1"                  # MAE is the primary metric

# ==============================================================================================================================
# 5. MODEL LADDER (order fixed in PREREGISTRATION.md)
# ==============================================================================================================================
LATE_HIDDEN = 32             # per-channel LSTM width in the late arm; the early arm is matched to it
CAPACITY_TOL = 0.02          # early and late parameter counts must agree within 2%
# Feature ladder = ablation study (PREREGISTRATION.md §7). Single channels are always run as references.
FEATURE_SETS = {"F2": ("au", "gaze"), "F3": ("au", "gaze", "pose")}
PRIMARY = "F3"
RUNGS = {
    "R1": {"pooling": "last"},        # LSTM, last hidden state
    "R2": {"pooling": "attention"},   # + frame attention pooling (B1, 7 Oct 2026)
}
