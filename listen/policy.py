"""Language-conditioned visuomotor policy: CNN + FiLM from a frozen sentence encoder, action chunks.

The same recipe for every condition. Only the training data changes.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import MAX_STEP

CHUNK = 5
LANG_DIM = 384


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


_cache: dict[str, np.ndarray] = {}


def embed(texts: list[str]) -> np.ndarray:
    """Frozen MiniLM sentence embeddings, cached per string."""
    missing = [t for t in set(texts) if t not in _cache]
    if missing:
        for t, e in zip(missing, _encoder().encode(missing, normalize_embeddings=True)):
            _cache[t] = e.astype(np.float32)
    return np.stack([_cache[t] for t in texts])


class FiLMBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, cout)
        self.film = nn.Linear(LANG_DIM, 2 * cout)

    def forward(self, x, lang):
        h = self.norm(self.conv(x))
        gamma, beta = self.film(lang).chunk(2, -1)
        return F.relu(h * (1 + gamma[..., None, None]) + beta[..., None, None])


class Policy(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        chans = [3, width, 2 * width, 4 * width, 4 * width]
        self.blocks = nn.ModuleList(FiLMBlock(a, b) for a, b in zip(chans[:-1], chans[1:]))
        self.lang_proj = nn.Sequential(nn.Linear(LANG_DIM, 128), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(4 * width * 4 * 4 + 4 + 128, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, CHUNK * 4),
        )

    def forward(self, image, proprio, lang):
        """image uint8 (B, 64, 64, 3) -> normalized action chunk (B, CHUNK, 4)."""
        x = image.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        for blk in self.blocks:
            x = blk(x, lang)
        p = proprio * torch.tensor([4.0, 4.0, 4.0, 10.0], device=proprio.device)
        h = torch.cat([x.flatten(1), p, self.lang_proj(lang)], 1)
        return self.head(h).view(-1, CHUNK, 4)


class TemporalEnsemble:
    """ACT-style execution: every step predicts a chunk of the next CHUNK actions; the action
    actually executed is a weighted average of every prediction made for *this* step.

    Without it the policy freezes at the block: "arrived" and "start closing" look identical,
    so the per-step gripper label is a coin flip and regression parks it at ~ -0.5 forever.
    Earlier chunks already predicted the close, and the ensemble lets them vote."""

    def __init__(self, decay: float = 0.1):
        self.decay = decay
        self.chunks: list[np.ndarray] = []

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        self.chunks = (self.chunks + [chunk])[-CHUNK:]
        preds = np.stack([c[len(self.chunks) - 1 - i] for i, c in enumerate(self.chunks)])  # oldest first
        w = np.exp(-self.decay * np.arange(len(preds)))
        return (w[:, None] * preds).sum(0) / w.sum()


def normalize_actions(a: np.ndarray) -> np.ndarray:
    """Deltas to [-1, 1] by MAX_STEP; the gripper is already +-1."""
    out = a.copy()
    out[..., :3] /= MAX_STEP
    return out


def denormalize_action(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[..., :3] *= MAX_STEP
    return out


def random_shift(img: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-style augmentation: pad by replication and crop back at a random offset per image."""
    b, h, w, c = img.shape
    x = F.pad(img.permute(0, 3, 1, 2).float(), (pad,) * 4, mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (b,), device=img.device)
    oy = torch.randint(0, 2 * pad + 1, (b,), device=img.device)
    rows = (oy[:, None] + torch.arange(h, device=img.device)[None])[:, :, None].expand(b, h, w)
    cols = (ox[:, None] + torch.arange(w, device=img.device)[None])[:, None, :].expand(b, h, w)
    batch = torch.arange(b, device=img.device)[:, None, None].expand(b, h, w)
    return x[batch, :, rows, cols].to(torch.uint8)  # (b, h, w, c)
