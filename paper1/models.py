"""The two arms of the fusion comparison, built rung by rung.

Late fusion: one LSTM per channel -> concatenate the channel embeddings -> linear head.
Early fusion: one LSTM over the frame-wise concatenation of all channels -> linear head.
Both arms take a channel subset, so the feature ladder (F2 = AU + gaze, F3 = all three) reuses them.

The arms differ in exactly one thing: where the channels are joined. On every rung the early arm's
width is chosen so the two arms have the same number of trainable parameters (within
config.CAPACITY_TOL). The late arm has no path between channels before the head.
"""
import torch
import torch.nn as nn

import config as C


class Encoder(nn.Module):
    """LSTM over one chunk; `pooling` turns the frame sequence into one vector (rung-dependent).

    last       R1: the final hidden state.
    attention  R2: learned frame attention, ported from visual/FAURegressionBiLSTM.py (frame_attention):
               score each step with hidden -> hidden//2 -> tanh -> 1, softmax over the steps, weighted sum.
    """

    def __init__(self, in_dim, hidden, pooling="last", bidirectional=False, heads=0):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=bidirectional)
        self.pooling = pooling
        self.out_dim = hidden * (2 if bidirectional else 1)       # R4: forward and backward states side by side
        self.heads = heads
        if heads:                                                  # R5: self-attention across the frames of one window
            if self.out_dim % heads:
                raise ValueError(f"width {self.out_dim} is not divisible by {heads} heads")
            self.mhsa = nn.MultiheadAttention(self.out_dim, heads, batch_first=True)
            self.norm = nn.LayerNorm(self.out_dim)
        if pooling == "attention":
            self.attn = nn.Sequential(nn.Linear(self.out_dim, self.out_dim // 2), nn.Tanh(), nn.Linear(self.out_dim // 2, 1))
        elif pooling != "last":
            raise NotImplementedError(f"pooling '{pooling}' belongs to a later rung")
        if bidirectional and pooling == "last":
            raise ValueError("a bidirectional encoder needs attention pooling: h[-1] would keep only the backward pass")

    def encode(self, x, slice_size=256):
        """Window embeddings for many windows at once. R5 only, while training: run the encoder in slices
        with gradient checkpointing, so self-attention fits in GPU memory. Same arithmetic, recomputed in
        the backward pass, so results are unchanged; only slower."""
        if not (self.heads and self.training and torch.is_grad_enabled()):
            return self(x)
        from torch.utils.checkpoint import checkpoint
        return torch.cat([checkpoint(self, x[i:i + slice_size], use_reentrant=False)
                          for i in range(0, x.shape[0], slice_size)])

    def frame_weights(self, out):               # out: (batch, steps, hidden) -> (batch, steps), rows sum to 1
        return torch.softmax(self.attn(out).squeeze(-1), dim=1)

    def forward(self, x):                       # x: (batch, steps, in_dim)
        out, (h, _) = self.lstm(x)
        if self.pooling == "last":
            return h[-1]                        # (batch, hidden)
        if self.heads:                          # residual + layer norm, as in visual/FAURegressionBiLSTM.py
            a, _ = self.mhsa(out, out, out, need_weights=False)
            out = self.norm(out + a)
        return (out * self.frame_weights(out).unsqueeze(-1)).sum(1)   # (batch, hidden)


ALL = tuple(C.CHANNELS)


class ChunkPool(nn.Module):
    """R3: learned attention over one person's chunks, replacing the mean of chunk predictions.

    Same scorer shape as frame attention (hidden -> hidden//2 -> tanh -> 1), softmax over that person's real
    chunks only; padding gets weight 0.
    """

    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))

    def weights(self, e, mask):                 # e: (people, max_chunks, hidden), mask: (people, max_chunks) bool
        s = self.attn(e).squeeze(-1).masked_fill(~mask, float("-inf"))
        return torch.softmax(s, dim=1)

    def forward(self, e, mask):
        return (e * self.weights(e, mask).unsqueeze(-1)).sum(1)     # (people, hidden)


def by_person(e, counts):
    """(chunks, hidden) in person order + chunks per person -> padded (people, max_chunks, hidden) and its mask."""
    parts = torch.split(e, counts)
    padded = nn.utils.rnn.pad_sequence(parts, batch_first=True)
    mask = torch.arange(padded.shape[1], device=e.device)[None, :] < torch.tensor(counts, device=e.device)[:, None]
    return padded, mask


class LateFusion(nn.Module):
    """aggregation='mean': one prediction per chunk (R1, R2), averaged outside the model.
    aggregation='attention': one prediction per person (R3); each channel pools its own chunks, so the
    channels still meet only in the head."""

    def __init__(self, hidden=C.LATE_HIDDEN, pooling="last", channels=ALL, aggregation="mean", bidirectional=False, heads=0):
        super().__init__()
        self.channels, self.aggregation = tuple(channels), aggregation
        self.encoders = nn.ModuleDict({c: Encoder(C.CHANNELS[c], hidden, pooling, bidirectional, heads) for c in self.channels})
        d = self.encoders[self.channels[0]].out_dim
        if aggregation == "attention":
            self.pools = nn.ModuleDict({c: ChunkPool(d) for c in self.channels})
        self.head = nn.Linear(d * len(self.channels), 1)

    def embed(self, b, counts=None):
        if self.aggregation == "mean":
            return torch.cat([self.encoders[c](b[c]) for c in self.channels], 1)
        return torch.cat([self.pools[c](*by_person(self.encoders[c].encode(b[c]), counts)) for c in self.channels], 1)

    def forward(self, b, counts=None):
        return self.head(self.embed(b, counts)).squeeze(1)


class EarlyFusion(nn.Module):
    def __init__(self, hidden, pooling="last", channels=ALL, aggregation="mean", bidirectional=False, heads=0):
        super().__init__()
        self.channels, self.aggregation = tuple(channels), aggregation
        self.encoder = Encoder(sum(C.CHANNELS[c] for c in self.channels), hidden, pooling, bidirectional, heads)
        if aggregation == "attention":
            self.pool = ChunkPool(self.encoder.out_dim)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, b, counts=None):
        x = torch.cat([b[c] for c in self.channels], 2)
        e = self.encoder.encode(x) if self.aggregation == "attention" else self.encoder(x)
        if self.aggregation == "attention":
            e = self.pool(*by_person(e, counts))
        return self.head(e).squeeze(1)


class SingleChannel(nn.Module):
    """Reference: one channel alone, same encoder width as a late-arm branch."""

    def __init__(self, channel, hidden=C.LATE_HIDDEN, pooling="last", aggregation="mean", bidirectional=False, heads=0):
        super().__init__()
        self.channel, self.aggregation = channel, aggregation
        self.encoder = Encoder(C.CHANNELS[channel], hidden, pooling, bidirectional, heads)
        if aggregation == "attention":
            self.pool = ChunkPool(self.encoder.out_dim)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, b, counts=None):
        e = self.encoder.encode(b[self.channel]) if self.aggregation == "attention" else self.encoder(b[self.channel])
        if self.aggregation == "attention":
            e = self.pool(*by_person(e, counts))
        return self.head(e).squeeze(1)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_capacity(pooling="last", channels=ALL, aggregation="mean", bidirectional=False, heads=0):
    """Early-arm width whose parameter count is closest to the late arm's, for one channel subset.
    With self-attention (R5) only widths whose output divides by the number of heads are candidates."""
    target = n_params(LateFusion(pooling=pooling, channels=channels, aggregation=aggregation, bidirectional=bidirectional, heads=heads))
    mult = 2 if bidirectional else 1
    cand = [h for h in range(8, 512) if not heads or (h * mult) % heads == 0]
    hidden = min(cand, key=lambda h: abs(n_params(EarlyFusion(h, pooling, channels, aggregation, bidirectional, heads)) - target))
    got = n_params(EarlyFusion(hidden, pooling, channels, aggregation, bidirectional, heads))
    gap = abs(got - target) / target
    assert gap <= C.CAPACITY_TOL, f"capacity gap {gap:.1%} exceeds {C.CAPACITY_TOL:.0%}"
    return hidden, target, got, gap


def builders(rung, feature_sets=tuple(C.FEATURE_SETS), singles=True):
    """Model constructors for one rung. Names: au_only, gaze_only, pose_only, early_F2, late_F2, ..."""
    spec = C.RUNGS[rung]
    pooling, agg, bi, hd = spec["pooling"], spec.get("aggregation", "mean"), spec.get("bidirectional", False), spec.get("heads", 0)
    out = {}
    if singles:
        for c in ("au", "gaze", "pose"):                 # F1 = AU alone; gaze and pose are references
            out[f"{c}_only"] = (lambda c=c: SingleChannel(c, pooling=pooling, aggregation=agg, bidirectional=bi, heads=hd))
    for fs in feature_sets:
        ch = C.FEATURE_SETS[fs]
        eh, *_ = match_capacity(pooling, ch, agg, bi, hd)
        out[f"late_{fs}"] = (lambda ch=ch: LateFusion(pooling=pooling, channels=ch, aggregation=agg, bidirectional=bi, heads=hd))
        out[f"early_{fs}"] = (lambda ch=ch, eh=eh: EarlyFusion(eh, pooling, ch, agg, bi, hd))
    return out


def capacities(rung):
    spec = C.RUNGS[rung]
    pooling, agg, bi, hd = spec["pooling"], spec.get("aggregation", "mean"), spec.get("bidirectional", False), spec.get("heads", 0)
    return {fs: dict(zip(("early_hidden", "late", "early", "gap"), match_capacity(pooling, ch, agg, bi, hd)))
            for fs, ch in C.FEATURE_SETS.items()}
