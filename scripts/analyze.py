"""Pre-registered analysis. Reads runs/f*_s*/episodes.json, writes results/summary.json + figures.

    python scripts/analyze.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from listen.design import FRACTIONS  # noqa: E402

RNG = np.random.default_rng(0)
B = 10_000


def load() -> pd.DataFrame:
    rows = []
    for run in sorted((ROOT / "runs").glob("f*_s*")):
        m = re.fullmatch(r"f([\d.]+)_s(\d+)", run.name)
        ep = run / "episodes.json"
        if not m or not ep.exists():
            continue
        for r in json.loads(ep.read_text()):
            r.update(f=float(m[1]), seed=int(m[2]))
            rows.append(r)
    df = pd.DataFrame(rows)
    df["grounded"] = (df["committed"] == df["target"]).astype(int)
    df["defaulted"] = (df["default"].notna() & (df["committed"] == df["default"])).astype(int)
    df["lift"] = (df["lifted"] == df["target"]).astype(int)
    return df


def boot_ci(df, col):
    """Two-level bootstrap: resample training runs, then episodes within each run."""
    groups = [g[col].to_numpy() for _, g in df.groupby("seed")]
    stats = np.empty(B)
    for b in range(B):
        picks = RNG.integers(len(groups), size=len(groups))
        stats[b] = np.mean(np.concatenate([RNG.choice(groups[i], len(groups[i])) for i in picks]))
    return float(df[col].mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def one_sided_vs_zero(df, f, col="grounded"):
    """P(mean_f <= mean_0) under a run-level bootstrap of both conditions."""
    a = [g[col].mean() for _, g in df[df.f == 0.0].groupby("seed")]
    b = [g[col].mean() for _, g in df[df.f == f].groupby("seed")]
    diffs = np.array([np.mean(RNG.choice(b, len(b))) - np.mean(RNG.choice(a, len(a))) for _ in range(B)])
    return float(np.mean(diffs <= 0)), float(np.mean(b) - np.mean(a))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[k] = running
    return out


def main():
    df = load()
    if df.empty:
        print("no runs yet")
        return
    summary = {"conditions": {}, "tests": {}}
    for f in FRACTIONS:
        d = df[df.f == f]
        if d.empty:
            continue
        c = {"runs": int(d.seed.nunique())}
        for kind in ("counterfactual", "nominal", "heldout_scene", "paraphrase_cf"):
            k = d[d.set == kind]
            c[kind] = {m: boot_ci(k, m) for m in ("grounded", "defaulted", "lift")}
            c[kind]["per_seed"] = {int(s): float(g.grounded.mean()) for s, g in k.groupby("seed")}
        summary["conditions"][str(f)] = c

    cf = df[df.set == "counterfactual"]
    # H3/H2: one-sided comparisons of each f in (0, 1) against f = 0, Holm-corrected.
    raw = {}
    for f in FRACTIONS[1:-1]:
        if (cf.f == f).any() and (cf.f == 0).any():
            p, diff = one_sided_vs_zero(cf, f)
            raw[str(f)] = p
            summary["tests"][f"cf_grounding_f{f}_vs_0"] = {"diff": diff, "p_one_sided": p}
    for k, p in holm(raw).items():
        summary["tests"][f"cf_grounding_f{k}_vs_0"]["p_holm"] = p

    # Logistic regression with categorical f and a fixed seed effect (as pre-registered).
    sub = cf[cf.f < 1.0]
    if sub.f.nunique() > 1:
        model = smf.glm("grounded ~ C(f) + C(seed)", data=sub, family=sm.families.Binomial()).fit()
        summary["glm"] = {name: {"coef": float(model.params[name]), "p": float(model.pvalues[name])}
                          for name in model.params.index}

    # f80: smallest f with mean counterfactual grounding >= 80%.
    f80 = [f for f in FRACTIONS if str(f) in summary["conditions"]
           and summary["conditions"][str(f)]["counterfactual"]["grounded"][0] >= 0.8]
    summary["f80"] = min(f80) if f80 else None

    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(summary)
    print(json.dumps({f: {k: round(v["grounded"][0], 3) for k, v in c.items() if isinstance(v, dict) and "grounded" in v}
                      for f, c in summary["conditions"].items()}, indent=1))
    print("tests:", json.dumps(summary["tests"], indent=1))
    print("f80:", summary["f80"])


def plot(summary):
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    fs = [float(f) for f in summary["conditions"]]
    x = np.arange(len(fs))
    labels = [f"{f:g}" for f in fs]
    for kind, color, label in (("counterfactual", "#d62728", "counterfactual instruction (unseen in this scene)"),
                               ("paraphrase_cf", "#ff9896", "counterfactual + unseen paraphrase"),
                               ("heldout_scene", "#1f77b4", "held-out scene layouts"),
                               ("nominal", "#2ca02c", "nominal (trained instruction)")):
        m = [summary["conditions"][str(f)][kind]["grounded"] for f in fs]
        y = np.array([v[0] for v in m])
        err = np.array([[v[0] - v[1], v[2] - v[0]] for v in m]).T
        axes[0].errorbar(x, 100 * y, yerr=100 * err, marker="o", capsize=3, color=color, label=label, lw=1.8)
        if kind == "counterfactual":
            for f in fs:
                for s, v in summary["conditions"][str(f)][kind]["per_seed"].items():
                    axes[0].scatter(fs.index(f) + 0.08, 100 * v, s=10, color=color, alpha=0.35, zorder=3)
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("fraction of contrastive scenes in training (f)")
    axes[0].set_ylabel("goes for the instructed block (%)")
    axes[0].set_ylim(-3, 103)
    axes[0].axhline(100 / 3, color="gray", ls=":", lw=1)
    axes[0].text(0.02, 100 / 3 + 1.5, "chance (1 of 3)", color="gray", fontsize=7)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].set_title("Grounding vs. training-data structure", fontsize=10)

    d = [summary["conditions"][str(f)]["counterfactual"]["defaulted"][0] for f in fs]
    g = [summary["conditions"][str(f)]["counterfactual"]["grounded"][0] for f in fs]
    other = [max(0.0, 1 - a - b) for a, b in zip(d, g)]
    axes[1].bar(x, 100 * np.array(g), color="#d62728", label="instructed block")
    axes[1].bar(x, 100 * np.array(d), bottom=100 * np.array(g), color="#7f7f7f", label="scene's trained block (shortcut)")
    axes[1].bar(x, 100 * np.array(other), bottom=100 * (np.array(g) + np.array(d)), color="#c7c7c7",
                label="other block / none")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("f")
    axes[1].set_ylabel("share of counterfactual episodes (%)")
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].set_title("What the policy does when asked for something new", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "grounding.png")
    fig.savefig(ROOT / "paper" / "grounding.pdf")


if __name__ == "__main__":
    main()
