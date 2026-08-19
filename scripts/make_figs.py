"""Reproduce paper figures from locked numbers. Writes paper/figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CLEAN = "#5C6B73"
DIE = "#C0392B"
RESTORE = "#1A7A4C"
NOISE = "#3D5A80"
PAD = "#2A9D8F"
STUFF = "#E76F51"
HEAVY = "#9B2226"
LIGHT = "#F4F1EE"
INK = "#1B1B1B"
MUTED = "#555555"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)
    print(f"wrote {stem}")


def fig1_write_store_read() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.35))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(-0.15, 4.55)
    ax.axis("off")

    def box(x, y, w, h, fc, title, sub):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor=fc, edgecolor=INK, linewidth=0.9,
            )
        )
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=8, color=MUTED)

    by, bh = 1.85, 1.55
    box(0.35, by, 3.3, bh, "#D8E2DC", "Write", r"$B_t,\;\Delta_t$ into $h$")
    box(4.55, by, 3.3, bh, "#F4D6D6", "Store", r"L17 $h$ holds the bind")
    box(8.75, by, 3.3, bh, "#D6EADF", "Read", r"$C_t$ at the query")
    for x0, x1 in ((3.65, 4.55), (7.85, 8.75)):
        ax.annotate("", xy=(x1, by + bh / 2), xytext=(x0, by + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.25))

    toks = ["K", "V", "K", "V*", "...", "Q"]
    cols = [LIGHT, LIGHT, LIGHT, "#F4A261", LIGHT, "#7DCEC4"]
    gap = 0.72
    x0 = 4.15
    for i, (t, c) in enumerate(zip(toks, cols)):
        x = x0 + i * gap
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.42), 0.58, 0.72,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=c, edgecolor=INK, lw=0.65,
            )
        )
        ax.text(x + 0.29, 0.78, t, ha="center", va="center",
                fontsize=9, fontweight="bold")

    vstar = x0 + 3 * gap + 0.29
    q = x0 + 5 * gap + 0.29
    ax.annotate("", xy=(6.2, by), xytext=(vstar, 1.18),
                arrowprops=dict(arrowstyle="-|>", color=DIE, lw=1.15,
                                connectionstyle="arc3,rad=0.0"))
    ax.annotate("", xy=(10.4, by), xytext=(q, 1.18),
                arrowprops=dict(arrowstyle="-|>", color=RESTORE, lw=1.15,
                                connectionstyle="arc3,rad=0.0"))
    ax.text(vstar, 0.14, "value", ha="center", va="top", fontsize=8, color=DIE)
    ax.text(q, 0.14, "query", ha="center", va="top", fontsize=8, color=RESTORE)

    ax.text(6.2, 4.32, "The bind lives in recurrent state, not the residual bus",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(6.2, 3.95, "Copy L17 $h$ restores AR. Copy the residual at the same sites does not.",
            ha="center", fontsize=8, color=MUTED)
    save(fig, "fig1_write_store_read")


def fig2_layer_sweep() -> None:
    layers = list(range(24))
    acc = [
        0.938, 0.961, 0.969, 0.961, 0.961, 0.961, 0.961, 0.961,
        0.961, 0.961, 0.961, 0.961, 0.969, 0.961, 0.961, 0.953,
        0.969, 0.297, 0.953, 0.961, 0.953, 0.961, 0.961, 0.961,
    ]
    colors = [DIE if i == 17 else CLEAN for i in layers]
    fig, ax = plt.subplots(figsize=(7.4, 3.15))
    ax.bar(layers, acc, color=colors, width=0.78, edgecolor="none", zorder=3)
    ax.axhline(0.961, color=RESTORE, ls="--", lw=1.0, zorder=2)
    ax.set_xlabel("Layer (zero $h$ on write, one layer at a time)")
    ax.set_ylabel("AR accuracy")
    ax.set_ylim(0, 1.18)
    ax.set_xlim(-0.7, 23.7)
    ax.set_xticks([0, 8, 17, 23])
    ax.text(17, 0.52, "0.297", ha="center", va="center",
            fontsize=8, color=DIE, fontweight="bold")
    ax.text(0.2, 0.99, "clean 0.961", ha="left", va="bottom",
            fontsize=8, color=RESTORE)
    ax.set_title("Only layer 17 write-wipe collapses associative recall (n=128)")
    save(fig, "fig2_layer_sweep")


def fig3_causal() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.7), sharey=True)

    ax = axes[0]
    names = ["clean", "noise", "wipe", "resid.", "L17 $h$"]
    vals = [0.961, 0.961, 0.242, 0.242, 0.961]
    cols = [CLEAN, NOISE, DIE, DIE, RESTORE]
    ax.bar(range(len(vals)), vals, color=cols, width=0.62, zorder=3)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names)
    ax.set_ylabel("AR accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title("A. Residual miss")

    ax = axes[1]
    names = ["value", "write", "query", "L16", "L18"]
    vals = [0.969, 0.961, 0.953, 0.258, 0.242]
    cols = [RESTORE, RESTORE, RESTORE, DIE, DIE]
    ax.bar(range(len(vals)), vals, color=cols, width=0.62, zorder=3)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names)
    ax.set_title("B. Restore site / layer")
    ax.axhline(0.242, color=DIE, ls=":", lw=0.8, zorder=1)

    ax = axes[2]
    labs = ["clean", "wipe", "noise", "L17 $h$"]
    short = [0.961, 0.039, 0.961, 0.945]
    mid = [0.844, 0.039, 0.836, np.nan]
    cols_c = [CLEAN, DIE, NOISE, RESTORE]
    x = np.array([0.0, 1.35])
    w = 0.20
    offsets = np.array([-0.39, -0.13, 0.13, 0.39])
    for i, (lab, c) in enumerate(zip(labs, cols_c)):
        ys = [short[i], mid[i]]
        xs = x + offsets[i]
        plotted = [y if not np.isnan(y) else 0 for y in ys]
        mask = [not np.isnan(y) for y in ys]
        ax.bar([xs[j] for j in range(2) if mask[j]],
               [plotted[j] for j in range(2) if mask[j]],
               w * 0.92, color=c, label=lab, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["ATR-short", "ATR-mid"])
    ax.set_title("C. ATR")
    ax.set_xlim(-0.7, 2.05)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24),
              ncol=2, fontsize=7.5, columnspacing=1.2, handlelength=1.2)

    fig.suptitle("Causal interventions on $h$ vs residual (n=128)", fontsize=10, y=0.98)
    fig.subplots_adjust(bottom=0.28, top=0.86, wspace=0.22)
    fig.savefig(OUT / "fig3_causal.pdf")
    fig.savefig(OUT / "fig3_causal.png")
    plt.close(fig)
    print("wrote fig3_causal")


def fig4_probes() -> None:
    sites = ["first\n(t=0)", "value token\n(on the bus)", "last write\n(no letter)", "query L23\n(readout)"]
    residual = [0.031, 1.000, 0.073, 0.979]
    h = [0.021, 0.875, 0.104, 0.885]
    x = np.arange(len(sites))
    w = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 3.45))
    ax.bar(x - w / 2, residual, w, color=NOISE, label="residual", zorder=3)
    ax.bar(x + w / 2, h, w, color=DIE, label=r"$h$ (best view)", zorder=3)
    ax.axhline(1 / 52, color="#888", ls="--", lw=0.9, label="chance (52-way)")
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("linear probe test accuracy")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.55),
              borderaxespad=0.0)
    ax.set_title("Store is not readout (n=384, 52-way, not v1)")
    fig.subplots_adjust(right=0.78)
    fig.savefig(OUT / "fig4_probes_over_t.pdf")
    fig.savefig(OUT / "fig4_probes_over_t.png")
    plt.close(fig)
    print("wrote fig4_probes_over_t")


def fig5_pad_vs_stuff() -> None:
    seeds = ["seed 1", "seed 2", "seed 3"]
    pad = [0.219, 0.078, 0.266]
    light = [-0.594, -0.641, -0.578]
    heavy = [-0.469, -0.484, -0.344]
    x = np.arange(len(seeds))
    w = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axhline(0, color="#222", lw=0.7, zorder=2)
    ax.bar(x - w, pad, w, color=PAD, label="OOD pad", zorder=3)
    ax.bar(x, light, w, color=STUFF, label="stuffing, light", zorder=3)
    ax.bar(x + w, heavy, w, color=HEAVY, label="stuffing, heavy", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylabel("accuracy change")
    ax.set_ylim(-0.72, 0.38)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title("Two failures: pad recovery (sign) vs stuffing collapse (n=64)")
    fig.subplots_adjust(right=0.72)
    fig.savefig(OUT / "fig5_pad_vs_stuff.pdf")
    fig.savefig(OUT / "fig5_pad_vs_stuff.png")
    plt.close(fig)
    print("wrote fig5_pad_vs_stuff")


if __name__ == "__main__":
    fig1_write_store_read()
    fig2_layer_sweep()
    fig3_causal()
    fig4_probes()
    fig5_pad_vs_stuff()
    print("done")
