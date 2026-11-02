"""Language-conditioned pick-up: three colored blocks, one instruction, one camera.

The arm and its low-level control come from ARMageddon/YOINK. The policy acts at 10 Hz in
end-effector space (move by dx, dy, dz; open/close), and each action is executed by six 60 Hz
IK + gravity-compensated motor ticks.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from armageddon import default_arm, solve
from armageddon.kinematics import quat_from_matrix
from armageddon.mjcf import ARM_MATERIALS, arm_actuators, arm_xml
from yoink.motion import GRIPPER_CLOSED, GRIPPER_OPEN, Controller, gripper_down

COLORS = {
    "red": (0.88, 0.12, 0.12),
    "green": (0.12, 0.72, 0.22),
    "blue": (0.15, 0.3, 0.95),
    "yellow": (0.95, 0.78, 0.08),
}
BLOCK_HALF = 0.02
IMG = 64
POLICY_HZ = 10
TICKS_PER_STEP = 6           # 60 Hz control ticks per policy step
MAX_STEP = 0.03              # meters of end-effector motion per policy step
WORK_LOW = np.array([0.22, -0.26, 0.015])
WORK_HIGH = np.array([0.60, 0.26, 0.25])
HOME_EE = np.array([0.36, 0.0, 0.18])
COMMIT_RADIUS = 0.03         # gripper closing within this xy distance of a block = committed to it
LIFT_HEIGHT = 0.07


def _v(a) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(a, dtype=float).ravel())


def build_xml() -> str:
    chain = default_arm()
    blocks = "\n".join(
        f"""    <body name="block{i}" pos="{0.35 + 0.08 * i} 0 {BLOCK_HALF}">
      <freejoint/>
      <geom name="block{i}" type="box" size="{BLOCK_HALF} {BLOCK_HALF} {BLOCK_HALF}" rgba="1 1 1 1" mass="0.05"
            contype="1" conaffinity="1" condim="4" friction="1.2 0.02 0.002" solref="0.005 1"/>
    </body>"""
        for i in range(3)
    )
    return f"""<mujoco model="listen">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.4 0.4 0.42" diffuse="0.5 0.5 0.5" specular="0.05 0.05 0.05"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.19 0.24" rgb2="0.16 0.15 0.2" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="8 8"/>
{ARM_MATERIALS}
  </asset>
  <default>
    <joint damping="3" armature="0.01"/>
    <geom contype="1" conaffinity="0"/>
  </default>
  <worldbody>
    <light pos="0.5 0 1.5" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="floor" type="plane" size="2 2 0.1" material="floor" contype="0" conaffinity="1"/>
    <geom name="mat" type="box" pos="0.41 0 0.0005" size="0.17 0.24 0.0005" rgba="0.55 0.55 0.58 1" contype="0" conaffinity="0"/>
    <camera name="policy" pos="0.95 0 0.62" xyaxes="0 1 0 -0.62 0 0.78" fovy="48"/>
{arm_xml(chain, gripper="parallel")}
{blocks}
  </worldbody>
  <actuator>
{arm_actuators(chain, gripper="parallel")}
  </actuator>
</mujoco>
"""


@dataclass
class Scene:
    """A scene template: which three colors, at which nominal (x, y) spots."""
    colors: tuple[str, str, str]
    spots: np.ndarray  # (3, 2)


@dataclass
class StepInfo:
    committed: int | None   # index of the first block the gripper closed on
    lifted: int | None      # index of a block above LIFT_HEIGHT


class PickEnv:
    def __init__(self, render: bool = True):
        self.chain = default_arm()
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.ctrl = Controller(self.model, self.data, self.chain)
        self.renderer = mujoco.Renderer(self.model, IMG, IMG) if render else None
        self.block_adr = [self.model.jnt_qposadr[self.model.body(f"block{i}").jntadr[0]] for i in range(3)]
        self.R = gripper_down(0.0)
        self.ee_target = HOME_EE.copy()
        self.committed: int | None = None
        self.was_closed = False

    # --------------------------------------------------------------- reset --

    def reset(self, scene: Scene, rng: np.random.Generator) -> dict:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for i, color in enumerate(scene.colors):
            m.geom(f"block{i}").rgba[:3] = COLORS[color]
            xy = scene.spots[i] + rng.uniform(-0.02, 0.02, 2)
            yaw = rng.uniform(-0.26, 0.26)
            a = self.block_adr[i]
            d.qpos[a:a + 3] = (xy[0], xy[1], BLOCK_HALF)
            d.qpos[a + 3:a + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        start = HOME_EE + rng.uniform([-0.03, -0.05, -0.02], [0.03, 0.05, 0.02])
        q = solve(self.chain, start, self.R, q0=np.array([0.0, 0.4, 1.2, 0.0, 1.0, 0.0]), restarts=5,
                  rng=np.random.default_rng(0)).q
        d.qpos[:6] = q
        d.ctrl[:6] = q
        d.qpos[6:8] = GRIPPER_OPEN
        mujoco.mj_forward(m, d)
        self.ctrl.q_des = q.copy()
        self.ctrl.grip = GRIPPER_OPEN
        self.ee_target = start
        self.committed, self.was_closed = None, False
        self.ctrl.wait(0.2)  # let the blocks settle
        return self.observe()

    # ---------------------------------------------------------------- step --

    def step(self, action: np.ndarray) -> tuple[dict, StepInfo]:
        """action = (dx, dy, dz, grip); grip > 0 closes the gripper."""
        delta = np.clip(action[:3], -MAX_STEP, MAX_STEP)
        start = self.ee_target.copy()
        self.ee_target = np.clip(start + delta, WORK_LOW, WORK_HIGH)
        closing = action[3] > 0
        self.ctrl.grip = GRIPPER_CLOSED if closing else GRIPPER_OPEN
        for k in range(1, TICKS_PER_STEP + 1):
            p = start + (self.ee_target - start) * k / TICKS_PER_STEP
            self.ctrl.q_des = solve(self.chain, p, self.R, q0=self.ctrl.q_des, max_iters=15).q
            self.ctrl.tick()
        if closing and not self.was_closed and self.committed is None:
            self.committed = self._block_under_gripper()
        self.was_closed = closing
        return self.observe(), StepInfo(self.committed, self._lifted())

    def _block_under_gripper(self) -> int | None:
        ee = self.data.site("ee").xpos
        if ee[2] > 0.07:
            return None
        dist = [np.linalg.norm(self.block_pos(i)[:2] - ee[:2]) for i in range(3)]
        i = int(np.argmin(dist))
        return i if dist[i] < COMMIT_RADIUS else None

    def _lifted(self) -> int | None:
        for i in range(3):
            if self.block_pos(i)[2] > LIFT_HEIGHT:
                return i
        return None

    def block_pos(self, i: int) -> np.ndarray:
        return self.data.body(f"block{i}").xpos.copy()

    def ee_pos(self) -> np.ndarray:
        return self.data.site("ee").xpos.copy()

    def observe(self) -> dict:
        obs = {"proprio": np.concatenate([self.ee_pos(), [self.data.qpos[6] + self.data.qpos[7]]]).astype(np.float32)}
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera="policy")
            obs["image"] = self.renderer.render().copy()
        return obs

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


__all__ = ["PickEnv", "Scene", "StepInfo", "COLORS", "IMG", "quat_from_matrix"]
