"""What does the policy do when it commits to nothing? Logs the gripper path in counterfactual episodes.

    python scripts/freeze_analysis.py runs/f0_s0 runs/f0.5_s0

For each no-commit episode: which block the gripper ended up closest to (in xy), how close,
and how low it got. Uses the first 120 counterfactual specs of the pre-registered set.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from listen.design import Design  # noqa: E402
from listen.env import PickEnv  # noqa: E402
from listen.policy import Policy, TemporalEnsemble, denormalize_action, embed  # noqa: E402
from scripts.evaluate import MAX_STEPS, episode_specs  # noqa: E402


def analyze(run: Path, n: int = 120):
    design, env = Design(), PickEnv()
    pol = Policy()
    pol.load_state_dict(torch.load(run / "policy.pt", map_location="cpu"))
    pol.eval()
    rows = []
    for kind, _, s, t, seed, _ in [x for x in episode_specs() if x[0] == "counterfactual"][:n]:
        scene = design.train_scenes[s]
        rng = np.random.default_rng(seed)
        obs = env.reset(scene, rng)
        lang = torch.as_tensor(embed([design.instruction(scene.colors[t], rng)]))
        ens, path = TemporalEnsemble(), []
        for _ in range(MAX_STEPS):
            with torch.no_grad():
                ch = pol(torch.as_tensor(obs["image"])[None], torch.as_tensor(obs["proprio"])[None], lang)[0].numpy()
            obs, info = env.step(denormalize_action(ens(ch)))
            path.append(env.ee_pos())
            if info.lifted is not None:
                break
        if info.committed is not None:
            continue
        path = np.array(path)
        blocks = np.array([env.block_pos(i)[:2] for i in range(3)])
        final = path[-1]
        d = np.linalg.norm(blocks - final[:2], axis=1)
        nearest = int(np.argmin(d))
        role = "instructed" if nearest == t else ("default" if nearest == design.default_target[s] else "other")
        rows.append({"nearest": role, "xy_dist_cm": float(100 * d[nearest]), "min_z_cm": float(100 * path[:, 2].min()),
                     "final_z_cm": float(100 * final[2]),
                     "dist_to_instructed_cm": float(100 * d[t]), "dist_to_default_cm": float(100 * d[design.default_target[s]])})
    env.close()
    return rows


if __name__ == "__main__":
    out = {}
    for run in map(Path, sys.argv[1:]):
        rows = analyze(run)
        roles = {r: np.mean([x["nearest"] == r for x in rows]) for r in ("instructed", "default", "other")}
        print(f"{run.name}: {len(rows)} no-commit episodes out of 120")
        print("  gripper ends up nearest to:", {k: f"{v:.0%}" for k, v in roles.items()})
        print(f"  median xy distance to that block: {np.median([x['xy_dist_cm'] for x in rows]):.1f} cm, "
              f"median final height: {np.median([x['final_z_cm'] for x in rows]):.1f} cm, "
              f"median lowest height: {np.median([x['min_z_cm'] for x in rows]):.1f} cm")
        out[run.name] = rows
    (ROOT / "results" / "freeze_analysis.json").write_text(json.dumps(out, indent=1))
