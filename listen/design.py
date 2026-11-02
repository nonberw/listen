"""The experimental design, fixed up front: scenes, default targets, contrastive sets, instructions.

Everything is derived from DESIGN_SEED so every condition sees exactly the same world.
"""

from __future__ import annotations

import numpy as np

from .env import COLORS, Scene

DESIGN_SEED = 2026
N_TRAIN_SCENES = 40
N_HELDOUT_SCENES = 10
DEMOS_PER_SCENE = 30
FRACTIONS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
EVAL_SCENES = tuple(range(20, 40))  # never contrastive for any f <= 0.5

TRAIN_TEMPLATES = (
    "pick up the {c} block",
    "grab the {c} block",
    "lift the {c} cube",
    "get the {c} one",
    "pick the {c} block up",
)
HELDOUT_TEMPLATES = (
    "could you raise the {c} block",
    "take hold of the {c} cube",
    "{c} block, pick it up",
)


def _spots(rng) -> np.ndarray:
    while True:
        p = rng.uniform([0.32, -0.17], [0.50, 0.17], (3, 2))
        if min(np.linalg.norm(p[i] - p[j]) for i in range(3) for j in range(i)) > 0.09:
            return p


def make_scenes(n: int, rng) -> list[Scene]:
    names = list(COLORS)
    return [Scene(tuple(rng.choice(names, 3, replace=False)), _spots(rng)) for _ in range(n)]


class Design:
    def __init__(self):
        rng = np.random.default_rng(DESIGN_SEED)
        self.train_scenes = make_scenes(N_TRAIN_SCENES, rng)
        self.heldout_scenes = make_scenes(N_HELDOUT_SCENES, rng)

        # Default target per scene, balanced so every color is the default equally often:
        # a global "always go for red" prior can't explain anything.
        counts = dict.fromkeys(COLORS, 0)
        self.default_target = []
        for s in self.train_scenes:
            order = sorted(range(3), key=lambda i: (counts[s.colors[i]], rng.random()))
            self.default_target.append(order[0])
            counts[s.colors[order[0]]] += 1
        self.default_color_counts = counts

        # Nested contrastive sets: f=0.05 is a subset of f=0.1, and so on.
        # Scenes 20-39 stay non-contrastive for every f <= 0.5, so they are the evaluation set.
        self.contrastive_order = [int(i) for i in rng.permutation(20)]

    def contrastive(self, f: float) -> set[int]:
        if f >= 1.0:
            return set(range(N_TRAIN_SCENES))
        return set(self.contrastive_order[: round(N_TRAIN_SCENES * f)])

    def demo_plan(self, f: float) -> list[tuple[int, int, int]]:
        """(scene, target, count) triples. Same total for every f."""
        contrastive = self.contrastive(f)
        plan = []
        for s in range(N_TRAIN_SCENES):
            if s in contrastive:
                plan += [(s, t, DEMOS_PER_SCENE // 3) for t in range(3)]
            else:
                plan.append((s, self.default_target[s], DEMOS_PER_SCENE))
        return plan

    @staticmethod
    def instruction(color: str, rng, heldout: bool = False) -> str:
        templates = HELDOUT_TEMPLATES if heldout else TRAIN_TEMPLATES
        return templates[rng.integers(len(templates))].format(c=color)
