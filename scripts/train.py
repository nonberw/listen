"""Train one policy for one condition (contrastive fraction f) and one seed.

    python scripts/train.py --f 0.25 --seed 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from listen.data import load_demos  # noqa: E402
from listen.design import Design  # noqa: E402
from listen.env import COLORS  # noqa: E402
from listen.policy import CHUNK, Policy, embed, normalize_actions, random_shift  # noqa: E402


def build_dataset(f: float, seed: int, regime: str = "main"):
    design = Design()
    rng = np.random.default_rng(10_000 + seed)
    imgs, props, chunks, langs = [], [], [], []
    plan = design.demo_plan(f) if regime == "main" else design.few_scene_plan(f)
    pool_dir = ROOT / "data" / ("pool" if regime == "main" else "pool_few")
    for scene, target, count in plan:
        pool = load_demos(pool_dir, scene, target)
        color = design.train_scenes[scene].colors[target]
        for k in rng.choice(len(pool), count, replace=False):
            im, pr, ac = pool[k]
            ac = normalize_actions(ac)
            T = len(ac)
            idx = np.minimum(np.arange(T)[:, None] + np.arange(CHUNK)[None], T - 1)  # pad with the last action
            text = design.instruction(color, rng)
            imgs.append(im)
            props.append(pr)
            chunks.append(ac[idx])
            langs.append(np.repeat(embed([text]), T, 0))
    return (np.concatenate(imgs), np.concatenate(props), np.concatenate(chunks).astype(np.float32),
            np.concatenate(langs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default=None)
    ap.add_argument("--regime", choices=["main", "few"], default="main")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    prefix = "" if args.regime == "main" else "few_"
    out = Path(args.out or ROOT / "runs" / f"{prefix}f{args.f:g}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    imgs, props, chunks, langs = build_dataset(args.f, args.seed, args.regime)
    imgs, props, chunks, langs = (torch.as_tensor(x, device=dev) for x in (imgs, props, chunks, langs))
    n = len(imgs)
    print(f"dataset: {n} frames, built in {time.perf_counter() - t0:.0f} s", flush=True)

    policy = Policy().to(dev)
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    losses = []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch,), device=dev)
        pred = policy(random_shift(imgs[idx]), props[idx], langs[idx])
        loss = F.mse_loss(pred, chunks[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 1000 == 0:
            print(f"step {step:6d} | loss {np.mean(losses[-1000:]):.4f} | {time.perf_counter() - t0:.0f} s", flush=True)

    torch.save(policy.state_dict(), out / "policy.pt")
    (out / "train.json").write_text(json.dumps({
        "f": args.f, "seed": args.seed, "regime": args.regime, "frames": n, "steps": args.steps,
        "final_loss": float(np.mean(losses[-500:])), "seconds": time.perf_counter() - t0,
        "colors": list(COLORS),
    }, indent=2))


if __name__ == "__main__":
    main()
