"""Regenerate two report figures on CPU (no model needed), from cached data:
  - training_curves.pdf : from the 10-epoch run's saved trainer_state.json
  - method_comparison.pdf: 2-row (2x3) per-artist panels, readable fonts,
                           replacing the unreadable 1x5 wide strip.
Run: uv run python regen_report_figures.py
"""
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIG = HERE.parent / "report" / "figures"
RES = HERE / "results"
ARTISTS = ["Gojira", "Tool", "Death", "Mastodon", "Opeth"]

# ---------------- training curves (10-epoch) ----------------
state = json.load(open(HERE / "artifacts/classifier_e10/checkpoint-620/trainer_state.json"))
lh = state["log_history"]
# aggregate per-step train loss to per-epoch mean (matches 03_classifier.ipynb)
per_epoch = {}
for e in lh:
    if "loss" in e:
        per_epoch.setdefault(round(e["epoch"]), []).append(e["loss"])
tr_ep = sorted(per_epoch)
tr_loss = [mean(per_epoch[e]) for e in tr_ep]
ev = [(e["epoch"], e["eval_loss"], e["eval_accuracy"]) for e in lh if "eval_loss" in e]
ev_ep = [x[0] for x in ev]; ev_loss = [x[1] for x in ev]; ev_acc = [x[2] for x in ev]
best_ep = 5  # best checkpoint (acc 0.873)

plt.rcParams.update({"font.size": 17})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
ax1.plot(tr_ep, tr_loss, "o-", lw=2.4, ms=8, label="Train loss", color="#1565c0")
ax1.plot(ev_ep, ev_loss, "s-", lw=2.4, ms=8, label="Validation loss", color="#e65100")
ax1.axvline(best_ep, ls="--", color="gray", lw=1.2, alpha=.7)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss", fontweight="bold")
ax1.legend(fontsize=15); ax1.grid(alpha=.3)
ax2.plot(ev_ep, ev_acc, "o-", lw=2.4, ms=8, color="#2e7d32")
ax2.axvline(best_ep, ls="--", color="gray", lw=1.2, alpha=.7)
ax2.annotate(f"best: {max(ev_acc):.3f}\n(epoch {best_ep})", xy=(best_ep, max(ev_acc)),
             xytext=(6.2, 0.72), fontsize=15,
             arrowprops=dict(arrowstyle="->", color="gray"))
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation accuracy")
ax2.set_title("Accuracy", fontweight="bold"); ax2.set_ylim(0.45, 0.95); ax2.grid(alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "training_curves.pdf", bbox_inches="tight")
print("wrote training_curves.pdf (10-epoch)")

# ---------------- method comparison (2x3 panels) ----------------
def stats(path, artist):
    p = RES / path
    if not p.exists():
        return None, None
    col = json.load(open(p)).get("df", {}).get(artist)
    if not col:
        return None, None
    return mean(col), pstdev(col)

# Two groups: baselines (one panel, full 0-1 range) and adapter ablations
# (one panel, y-zoomed to expose LoRA/DoRA/SW detail). Grouped bars: x = artist.
BASELINES = [
    ("Zero",      "#9e9e9e", "baselines/{a}/zero_shot.json"),
    ("Few",       "#607d8b", "baselines/{a}/few_shot.json"),
    ("Zero (it)", "#00897b", "baselines/{a}/zero_shot_it.json"),
    ("Few (it)",  "#8e24aa", "baselines/{a}/few_shot_it.json"),
]
ADAPTERS = [
    ("LoRA", "#1565c0", "adapters/{a}/lora_r8.json"),
    ("DoRA", "#e65100", "adapters/{a}/dora_r8.json"),
    ("SW",   "#2e7d32", "adapters/{a}/lora_r8_sw.json"),
]

def grouped(ax, methods, title, ylim, zoom_note=False):
    x = np.arange(len(ARTISTS))
    w = 0.8 / len(methods)
    for j, (lab, col, tmpl) in enumerate(methods):
        means = [stats(tmpl.format(a=a.lower()), a)[0] or 0 for a in ARTISTS]
        stds = [stats(tmpl.format(a=a.lower()), a)[1] or 0 for a in ARTISTS]
        off = (j - (len(methods) - 1) / 2) * w
        ax.bar(x + off, means, w, yerr=stds, capsize=3, label=lab,
               color=col, edgecolor="black", linewidth=.5)
    ax.set_xticks(x); ax.set_xticklabels(ARTISTS, fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylim(*ylim); ax.axhline(1.0, color="gray", ls="--", lw=.6, alpha=.5)
    ax.set_ylabel("Target-artist attribution")
    ax.grid(axis="y", alpha=.3); ax.legend(fontsize=12, ncol=2 if len(methods) > 2 else 1)
    if zoom_note:
        ax.text(0.015, 0.03, "y-axis zoomed", transform=ax.transAxes,
                fontsize=10, color="gray", style="italic")

plt.rcParams.update({"font.size": 14})
fig, (axb, axa) = plt.subplots(1, 2, figsize=(15, 5.4))
grouped(axb, BASELINES, "Baselines", (0, 1.12))
grouped(axa, ADAPTERS, "Adapters (LoRA / DoRA / SW)", (0.45, 1.06), zoom_note=True)
fig.tight_layout()
fig.savefig(FIG / "method_comparison.pdf", bbox_inches="tight", dpi=300)
print("wrote method_comparison.pdf (baselines | adapters-zoomed)")

# ---------------- rank ablation (grouped bar, zoomed) ----------------
RANK_ARTISTS = ["Gojira", "Tool", "Mastodon"]
RANKS = [("r=4", "#42a5f5"), ("r=8", "#1565c0"), ("r=16", "#0d2f6b")]
plt.rcParams.update({"font.size": 15})
figr, axr = plt.subplots(figsize=(8, 5))
x = np.arange(len(RANK_ARTISTS)); w = 0.8 / len(RANKS)
for j, (lab, col) in enumerate(RANKS):
    r = lab.split("=")[1]
    means = [stats(f"adapters/{a.lower()}/lora_r{r}.json", a)[0] or 0 for a in RANK_ARTISTS]
    stds = [stats(f"adapters/{a.lower()}/lora_r{r}.json", a)[1] or 0 for a in RANK_ARTISTS]
    off = (j - (len(RANKS) - 1) / 2) * w
    axr.bar(x + off, means, w, yerr=stds, capsize=4, label=lab,
            color=col, edgecolor="black", linewidth=.5)
axr.set_xticks(x); axr.set_xticklabels(RANK_ARTISTS)
axr.set_ylim(0.55, 1.05); axr.axhline(1.0, color="gray", ls="--", lw=.6, alpha=.5)
axr.set_ylabel("Target-artist attribution")
axr.set_title("LoRA rank ablation", fontsize=16, fontweight="bold")
axr.grid(axis="y", alpha=.3); axr.legend(title="Rank", fontsize=13, title_fontsize=13)
axr.text(0.015, 0.03, "y-axis zoomed", transform=axr.transAxes,
         fontsize=10, color="gray", style="italic")
figr.tight_layout()
figr.savefig(FIG / "rank_ablation.pdf", bbox_inches="tight", dpi=300)
print("wrote rank_ablation.pdf")
