"""Roll out a trained policy on the pre-registered evaluation sets. Writes one record per episode.

    python scripts/evaluate.py runs/f0.25_s0

Sets (fixed episode seeds, identical for every condition):
  counterfactual  300 eps: eval scenes (20-39), ask for a block the scene was never demonstrated with
  nominal         150 eps: eval scenes, ask for the scene's trained target
  heldout_scene    90 eps: 10 never-seen layouts x 3 blocks
  paraphrase_cf   120 eps: like counterfactual, but with paraphrase templates never seen in training
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from listen.design import EVAL_SCENES, Design  # noqa: E402

MAX_STEPS = 40
WORKERS = 8


def episode_specs():
    d = Design()
    specs = []
    cf_pairs = [(s, t) for s in EVAL_SCENES for t in range(3) if t != d.default_target[s]]
    for k in range(300):
        s, t = cf_pairs[k % len(cf_pairs)]
        specs.append(("counterfactual", "train", s, t, k, False))
    for k in range(150):
        s = EVAL_SCENES[k % len(EVAL_SCENES)]
        specs.append(("nominal", "train", s, d.default_target[s], 1000 + k, False))
    for k in range(90):
        specs.append(("heldout_scene", "heldout", k // 9 % 10, k % 3, 2000 + k, False))
    for k in range(120):
        s, t = cf_pairs[k % len(cf_pairs)]
        specs.append(("paraphrase_cf", "train", s, t, 3000 + k, True))
    return specs


_worker = {}


def _init(run_dir):
    import torch

    from listen.env import PickEnv
    from listen.policy import Policy

    torch.set_num_threads(1)
    policy = Policy()
    policy.load_state_dict(torch.load(Path(run_dir) / "policy.pt", map_location="cpu"))
    policy.eval()
    _worker.update(policy=policy, env=PickEnv(), design=Design())


def _run(spec):
    import torch

    from listen.policy import TemporalEnsemble, denormalize_action, embed

    kind, scene_set, s, t, seed, heldout_text = spec
    env, policy, design = _worker["env"], _worker["policy"], _worker["design"]
    scene = (design.train_scenes if scene_set == "train" else design.heldout_scenes)[s]
    rng = np.random.default_rng(seed)
    obs = env.reset(scene, rng)
    text = design.instruction(scene.colors[t], rng, heldout=heldout_text)
    lang = torch.as_tensor(embed([text]))
    info = None
    ensemble = TemporalEnsemble()
    for step in range(MAX_STEPS):
        with torch.no_grad():
            chunk = policy(torch.as_tensor(obs["image"])[None], torch.as_tensor(obs["proprio"])[None], lang)[0].numpy()
        obs, info = env.step(denormalize_action(ensemble(chunk)))
        if info.lifted is not None:
            break
    color = lambda i: None if i is None else scene.colors[i]  # noqa: E731
    default = design.default_target[s] if scene_set == "train" else None
    return {"set": kind, "scene_set": scene_set, "scene": s, "instruction": text,
            "target": scene.colors[t], "default": color(default),
            "committed": color(info.committed), "lifted": color(info.lifted), "steps": step + 1}


def evaluate(run_dir: Path):
    specs = episode_specs()
    with Pool(WORKERS, initializer=_init, initargs=(str(run_dir),)) as pool:
        records = pool.map(_run, specs, chunksize=4)
    (run_dir / "episodes.json").write_text(json.dumps(records))
    return records


def summarize(records):
    out = {}
    for kind in ("counterfactual", "nominal", "heldout_scene", "paraphrase_cf"):
        r = [x for x in records if x["set"] == kind]
        out[kind] = {
            "n": len(r),
            "grounding": np.mean([x["committed"] == x["target"] for x in r]),
            "default": np.mean([x["default"] is not None and x["committed"] == x["default"] for x in r]),
            "lift_success": np.mean([x["lifted"] == x["target"] for x in r]),
            "no_commit": np.mean([x["committed"] is None for x in r]),
        }
    return out


if __name__ == "__main__":
    run = Path(sys.argv[1])
    recs = evaluate(run)
    for k, v in summarize(recs).items():
        print(f"{k:15s} n={v['n']:3d} grounding {v['grounding']:.1%}  default {v['default']:.1%}  "
              f"lift {v['lift_success']:.1%}  no-commit {v['no_commit']:.1%}")
