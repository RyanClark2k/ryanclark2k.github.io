"""Generate three charts for the blog posts.

Run with the circuits venv (has matplotlib):
    /Volumes/git/rlvr/circuits/.venv/bin/python make_charts.py

Produces:
  grpo-synthesis-vs-oss.png   — Synthesis arm vs Public OSS, mean reward over training
  boolmin-training-curve.png  — BoolMin 250-step trajectory (reward + correctness)
  rna-training-curve.png      — RNA 150-step trajectory (reward + perfect-rate)
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# Match the blog's visual palette.
BG = "#fdfcf9"
INK = "#1f1f1f"
MUTED = "#6b6b6b"
RULE = "#e3ddd0"
ACCENT = "#b34c2c"
GOOD = "#2c7a4d"
COOL = "#4a6fa5"
BAD = "#c8442a"

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": RULE,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.labelsize": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "savefig.facecolor": BG,
    "savefig.edgecolor": "none",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

OUTDIR = Path(__file__).parent


# ----------------------------------------------------------------------------
# Chart 1: GRPO Synthesis vs Public OSS
# Single source of truth: synthetic-codebases/grpo-experiment/results/eval_summary.json
# ----------------------------------------------------------------------------
GRPO_EVAL_SUMMARY = Path(
    "/Volumes/git/synthetic-codebases/grpo-experiment/results/eval_summary.json"
)

def make_grpo_chart():
    with open(GRPO_EVAL_SUMMARY) as f:
        summary = json.load(f)
    syn_arm = summary["arms"]["synthesis"]
    oss_arm = summary["arms"]["public_oss"]
    steps      = [c["step"]        for c in syn_arm["checkpoints"]]
    synthesis  = [c["mean_reward"] for c in syn_arm["checkpoints"]]
    public_oss = [c["mean_reward"] for c in oss_arm["checkpoints"]]
    syn_baseline = syn_arm["untrained_baseline_mean_reward"]
    oss_baseline = oss_arm["untrained_baseline_mean_reward"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    # Faint ±0.025 band around the Public OSS baseline to show the "oscillating
    # within a band" point visually.
    ax.axhspan(oss_baseline - 0.025, oss_baseline + 0.025, alpha=0.08, color=COOL, zorder=0)

    # Baselines as dashed lines.
    ax.axhline(syn_baseline, color=ACCENT, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
    ax.axhline(oss_baseline, color=COOL, linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)

    # Main series.
    ax.plot(steps, synthesis, color=ACCENT, marker="o", markersize=7,
            linewidth=2.3, label="Synthesis arm  (+8.9pp final)", zorder=5)
    ax.plot(steps, public_oss, color=COOL, marker="s", markersize=6,
            linewidth=2.0, label="Public OSS arm  (−1.3pp final)", zorder=4)

    # Endpoint annotations.
    ax.annotate("0.729\n(peak)", xy=(50, 0.729), xytext=(52, 0.748),
                color=ACCENT, fontsize=10, fontweight="bold",
                ha="left", va="center")
    ax.annotate("0.826", xy=(50, 0.826), xytext=(52, 0.826),
                color=COOL, fontsize=10, fontweight="bold",
                ha="left", va="center")

    ax.set_xlim(-3, 62)
    ax.set_ylim(0.55, 0.92)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(["0\n(untrained)", "10", "20", "30", "40", "final"])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean reward")
    ax.set_title("GRPO mean reward: matched-budget Synthesis vs Public OSS",
                 loc="left", pad=14)
    ax.legend(loc="lower right", fontsize=10)

    fig.text(0.01, -0.02,
             "Same model (Qwen3-30B-A3B), same hyperparameters, same step budget.  Shaded band = ±0.025 around Public OSS baseline.",
             color=MUTED, fontsize=9, style="italic")

    fig.savefig(OUTDIR / "grpo-synthesis-vs-oss.png", dpi=150,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote grpo-synthesis-vs-oss.png")


# ----------------------------------------------------------------------------
# Chart 2: BoolMin 250-step training curve.
# trial (50 steps) + trial_250 (200 steps resumed) = 250 cumulative steps.
# ----------------------------------------------------------------------------
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def make_boolmin_chart():
    trial = load_jsonl("/Volumes/git/rlvr/circuits/runs/trial/latest/train_log.jsonl")
    trial_250 = load_jsonl("/Volumes/git/rlvr/circuits/runs/trial_250/latest/train_log.jsonl")

    steps_a = [s["step"] for s in trial]
    reward_a = [s["reward_mean"] for s in trial]
    correct_a = [s["correct_rate"] for s in trial]

    # trial_250 step 0 = cumulative step 50.
    steps_b = [s["step"] + 50 for s in trial_250]
    reward_b = [s["reward_mean"] for s in trial_250]
    correct_b = [s["correct_rate"] for s in trial_250]

    steps_all = steps_a + steps_b
    reward_all = reward_a + reward_b
    correct_all = correct_a + correct_b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)

    ax1.plot(steps_all, reward_all, color=ACCENT, linewidth=1.2, alpha=0.35)
    # Add a simple 10-step rolling mean for readability.
    win = 10
    smoothed = [sum(reward_all[max(0, i - win + 1): i + 1]) / min(win, i + 1)
                for i in range(len(reward_all))]
    ax1.plot(steps_all, smoothed, color=ACCENT, linewidth=2.2,
             label="reward_mean (10-step avg)")
    ax1.axvline(50, color=MUTED, linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.text(50, 0.05, "  resume from\n  trial checkpoint",
             color=MUTED, fontsize=8.5, transform=ax1.get_xaxis_transform(),
             va="bottom", ha="left")
    ax1.set_xlabel("Training step (cumulative)")
    ax1.set_ylabel("Mean reward")
    ax1.set_title("Mean reward", loc="left", pad=10)
    ax1.set_ylim(0, 1.15)
    ax1.set_xlim(-5, 260)

    ax2.plot(steps_all, correct_all, color=GOOD, linewidth=1.2, alpha=0.35)
    smoothed_c = [sum(correct_all[max(0, i - win + 1): i + 1]) / min(win, i + 1)
                  for i in range(len(correct_all))]
    ax2.plot(steps_all, smoothed_c, color=GOOD, linewidth=2.2,
             label="correct_rate (10-step avg)")
    ax2.axvline(50, color=MUTED, linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel("Training step (cumulative)")
    ax2.set_ylabel("Correct rate")
    ax2.set_title("In-domain correctness", loc="left", pad=10)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("BoolMin-RLVR: 250-step in-domain training trajectory (seed=2 / 10)",
                 fontsize=12.5, fontweight="bold", x=0.02, y=1.02, ha="left")

    fig.text(0.01, -0.04,
             "Easy-tier-only training. Step 0–49: trial (LR 2e-5, batch 8, KL 0.01). Step 50–249: trial_250 resumed from trial's final checkpoint.",
             color=MUTED, fontsize=9, style="italic")

    fig.savefig(OUTDIR / "boolmin-training-curve.png", dpi=150,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote boolmin-training-curve.png")


# ----------------------------------------------------------------------------
# Chart 3: RNA 150-step training curve.
# trial (50 steps) + trial_100 (100 steps resumed) = 150 cumulative steps.
# ----------------------------------------------------------------------------
def make_rna_chart():
    trial = load_jsonl("/Volumes/git/rlvr/rna/runs/trial/2026-04-07_08-05-54/train_log.jsonl")
    trial_100 = load_jsonl("/Volumes/git/rlvr/rna/runs/trial_100/2026-04-07_23-05-17/train_log.jsonl")

    steps_a = [s["step"] for s in trial]
    reward_a = [s["reward_mean"] for s in trial]
    perfect_a = [s["perfect_rate"] for s in trial]

    steps_b = [s["step"] + 50 for s in trial_100]
    reward_b = [s["reward_mean"] for s in trial_100]
    perfect_b = [s["perfect_rate"] for s in trial_100]

    steps_all = steps_a + steps_b
    reward_all = reward_a + reward_b
    perfect_all = perfect_a + perfect_b

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)

    win = 10

    ax1.plot(steps_all, reward_all, color=ACCENT, linewidth=1.2, alpha=0.35)
    smoothed = [sum(reward_all[max(0, i - win + 1): i + 1]) / min(win, i + 1)
                for i in range(len(reward_all))]
    ax1.plot(steps_all, smoothed, color=ACCENT, linewidth=2.2,
             label="reward_mean (10-step avg)")
    ax1.axvline(50, color=MUTED, linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.text(50, 0.05, "  resume from\n  trial checkpoint",
             color=MUTED, fontsize=8.5, transform=ax1.get_xaxis_transform(),
             va="bottom", ha="left")
    ax1.set_xlabel("Training step (cumulative)")
    ax1.set_ylabel("Mean reward")
    ax1.set_title("Mean reward", loc="left", pad=10)
    ax1.set_ylim(0, 1.15)
    ax1.set_xlim(-3, 155)

    ax2.plot(steps_all, perfect_all, color=GOOD, linewidth=1.2, alpha=0.35)
    smoothed_p = [sum(perfect_all[max(0, i - win + 1): i + 1]) / min(win, i + 1)
                  for i in range(len(perfect_all))]
    ax2.plot(steps_all, smoothed_p, color=GOOD, linewidth=2.2,
             label="perfect_rate (10-step avg)")
    ax2.axvline(50, color=MUTED, linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel("Training step (cumulative)")
    ax2.set_ylabel("Perfect-fold rate")
    ax2.set_title("In-domain perfect-fold rate", loc="left", pad=10)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("RNAmin-RLVR: 150-step in-domain training trajectory (seed=2 / 20)",
                 fontsize=12.5, fontweight="bold", x=0.02, y=1.02, ha="left")

    fig.text(0.01, -0.04,
             "Easy-tier-only training. ViennaRNA verifier. Step 0–49: trial (50 steps). Step 50–149: trial_100 resumed from trial's final checkpoint.",
             color=MUTED, fontsize=9, style="italic")

    fig.savefig(OUTDIR / "rna-training-curve.png", dpi=150,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote rna-training-curve.png")


if __name__ == "__main__":
    make_grpo_chart()
    make_boolmin_chart()
    make_rna_chart()
