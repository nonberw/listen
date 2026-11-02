"""Demonstration pool: for every training scene and every block in it, DEMOS_PER_SCENE expert demos.

Each experimental condition then *samples* its dataset from this pool, so conditions differ only
in which (scene, target) pairs they include, never in how the demos were produced.
"""

from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .design import DEMOS_PER_SCENE, Design
from .env import PickEnv
from .expert import Expert

MAX_STEPS = 40
DART_SIGMA = 0.004


def _collect(args):
    scene_idx, target, n, seed = args
    design = Design()
    scene = design.train_scenes[scene_idx]
    rng = np.random.default_rng(seed)
    env = PickEnv()
    demos = []
    while len(demos) < n:
        obs = env.reset(scene, rng)
        ex = Expert(target)
        imgs, props, acts = [], [], []
        for _ in range(MAX_STEPS):
            a = ex.act(env)
            imgs.append(obs["image"])
            props.append(obs["proprio"])
            acts.append(a)
            noisy = a.copy()
            noisy[:3] += rng.normal(0, DART_SIGMA, 3)
            obs, info = env.step(noisy)
            if info.lifted is not None:
                break
        if info.lifted == target and info.committed == target:  # keep successful demos only
            demos.append((np.array(imgs), np.array(props), np.array(acts, np.float32)))
    env.close()
    return scene_idx, target, demos


def build_pool(out: Path, workers: int = 8):
    design = Design()
    jobs = [(s, t, DEMOS_PER_SCENE, 1000 * s + t) for s in range(len(design.train_scenes)) for t in range(3)]
    out.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool:
        for k, (s, t, demos) in enumerate(pool.imap_unordered(_collect, jobs)):
            np.savez_compressed(out / f"s{s:02d}_t{t}.npz",
                                images=np.concatenate([d[0] for d in demos]),
                                proprio=np.concatenate([d[1] for d in demos]),
                                actions=np.concatenate([d[2] for d in demos]),
                                lengths=np.array([len(d[2]) for d in demos]))
            print(f"\r{k + 1}/{len(jobs)} (scene, target) pairs collected", end="", flush=True)
    print()


def load_demos(pool_dir: Path, scene: int, target: int):
    z = np.load(pool_dir / f"s{scene:02d}_t{target}.npz")
    ends = np.cumsum(z["lengths"])
    starts = ends - z["lengths"]
    return [(z["images"][a:b], z["proprio"][a:b], z["actions"][a:b]) for a, b in zip(starts, ends)]
