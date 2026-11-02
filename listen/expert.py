"""Scripted expert with privileged block positions: hover, descend, close, lift.

It is reactive (each action is computed from the current state), so under DART noise the
recorded label is always the correct action *from the perturbed state*.
"""

from __future__ import annotations

import numpy as np

from .env import BLOCK_HALF, MAX_STEP, PickEnv

HOVER = 0.10
GRASP_Z = BLOCK_HALF - 0.004
LIFT_Z = 0.16


class Expert:
    def __init__(self, target: int):
        self.target = target
        self.phase = "hover"
        self.closed_steps = 0

    def act(self, env: PickEnv) -> np.ndarray:
        ee = env.ee_target
        block = env.block_pos(self.target)
        if self.phase == "hover":
            goal = np.array([block[0], block[1], HOVER])
            if np.linalg.norm(goal - ee) < 0.006:
                self.phase = "descend"
            return self._toward(ee, goal, grip=-1)
        if self.phase == "descend":
            goal = np.array([block[0], block[1], GRASP_Z])
            if np.linalg.norm(goal - ee) < 0.004:
                self.phase = "close"
            return self._toward(ee, goal, grip=-1)
        if self.phase == "close":
            self.closed_steps += 1
            if self.closed_steps >= 4:
                self.phase = "lift"
            return np.array([0.0, 0.0, 0.0, 1.0])
        goal = np.array([ee[0], ee[1], LIFT_Z])
        return self._toward(ee, goal, grip=1)

    @staticmethod
    def _toward(ee, goal, grip):
        d = goal - ee
        n = np.linalg.norm(d)
        if n > MAX_STEP:
            d *= MAX_STEP / n
        return np.array([*d, float(grip)])
