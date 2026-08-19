"""Reproduce all paper figures from the locked numbers in logs/findings.md.

Does not re-run the model. Writes figs/*.pdf and figs/*.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs"
OUT.mkdir(parents=True, exist_ok=True)

CLEAN = "#5C6B73"
DIE = "#C0392B"
RESTORE = "#1A7A4C"
NOISE = "#3D5A80"
PAD = "#2A9D8F"
STUFF = "#E76F51"
ACCENT = "#264653"
LIGHT = "#EDF2F4"

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
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)
    print(f"wrote figs/{stem}.pdf")


def fig1_write_store_read() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, fc, text, sub=""):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=fc, edgecolor="#1B1B1B", linewidth=0.8,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), text,
                ha="center", va="center", fontsize=10, fontweight="bold", color="#1B1B1B")
        if sub:
            ax.text(x + w / 2, y + 0.32, sub, ha="center", va="center", fontsize=7.5, color="#333")

    box(0.3, 1.3, 3.2, 1.7, "#D8E2DC", "Write", r"$B_t,\;\Delta_t$ into $h$")
    box(4.4, 1.3, 3.2, 1.7, "#F4D6D6", "Store", r"L17 $h$ holds the bind")
    box(8.5, 1.3, 3.2, 1.7, "#D6EADF", "Read", r"$C_t$ at the query")

    for x0, x1 in ((3.5, 4.4), (7.6, 8.5)):
        ax.annotate("", xy=(x1, 2.15), xytext=(x0, 2.15),
                    arrowprops=dict(arrowstyle="-|>", color="#1B1B1B", lw=1.2))

    toks = ["K", "V", "K", "V*", "…", "Q"]
    cols = [LIGHT, LIGHT, LIGHT, "#F4A261", LIGHT, "#2A9D8F"]
    for i, (t, c) in enumerate(zip(toks, cols)):
        x = 0.45 + i * 0.52
        ax.add_patch(FancyBboxPatch((x, 0.25), 0.46, 0.7, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="#1B1B1B", lw=0.6))
        ax.text(x + 0.23, 0.6, t, ha="center", va="center", fontsize=8, fontweight="bold")

    ax.annotate("", xy=(6.0, 1.3), xytext=(2.5, 0.95),
                arrowprops=dict(arrowstyle="-|>", color=DIE, lw=1.0,
                                connectionstyle="arc3,rad=0.15"))
    ax.text(3.7, 0.95, "value token", fontsize=7, color=DIE, ha="center")
    ax.annotate("", xy=(10.1, 1.3), xytext=(3.55, 0.95),
                arrowprops=dict(arrowstyle="-|>", color=RESTORE, lw=1.0,
                                connectionstyle="arc3,rad=-0.12"))

    ax.text(6.0, 3.85, "Associative recall: the bind lives in recurrent state, not the residual bus",
            ha="center", fontsize=9, fontweight="bold")
    ax.text(6.0, 3.45, "Copying L17 $h$ restores accuracy. Copying the residual at the same sites does not.",
            ha="center", fontsize=7.5, color="#333")
    save(fig, "fig1_write_store_read")


def fig2_layer_sweep() -> None:
    layers = list(range(24))
    acc = [
        0.938, 0.961, 0.969, 0.961, 0.961, 0.961, 0.961, 0.961,
        0.961, 0.961, 0.961, 0.961, 0.969, 0.961, 0.961, 0.953,
        0.969, 0.297, 0.953, 0.961, 0.953, 0.961, 0.961, 0.961,
    ]
    colors = [DIE if i == 17 else CLEAN for i in layers]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.bar(layers, acc, color=colors, width=0.78, edgecolor="none")
    ax.axhline(0.961, color=RESTORE, ls="--", lw=0.9, label="clean 0.961")
    ax.set_xlabel("Layer (zero $h$ on write tokens, one layer at a time)")
    ax.set_ylabel("AR accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(i) if i in (0, 8, 16, 17, 18, 23) else "" for i in layers])
    ax.text(17, 0.36, "L17\n0.297", ha="center", va="bottom", fontsize=8, color=DIE, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Only layer 17 write-wipe collapses associative recall (n=128)")
    save(fig, "fig2_layer_sweep")


def fig3_causal() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.15), sharey=True)

    # A: wipe vs residual vs restore
    ax = axes[0]
    names = ["clean", "residual\nnoise\n(write)", "late $h$\nwipe", "restore\nresidual", "restore\nL17 $h$"]
    vals = [0.961, 0.961, 0.242, 0.242, 0.961]
    cols = [CLEAN, NOISE, DIE, DIE, RESTORE]
    ax.bar(range(len(vals)), vals, color=cols, width=0.72)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names)
    ax.set_ylabel("AR accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("A. Store ≠ residual")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=7)

    # B: when / which layer
    ax = axes[1]
    names = ["$h$@value", "$h$@last\nwrite", "$h$@query", "L16 $h$", "L18 $h$"]
    vals = [0.969, 0.961, 0.953, 0.258, 0.242]
    cols = [RESTORE, RESTORE, RESTORE, DIE, DIE]
    ax.bar(range(len(vals)), vals, color=cols, width=0.72)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names)
    ax.set_title("B. Time and neighbors")
    ax.axhline(0.242, color=DIE, ls=":", lw=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=7)

    # C: ATR
    ax = axes[2]
    x = np.arange(2)
    w = 0.22
    clean = [0.961, 0.844]
    wipe = [0.039, 0.039]
    res_w = [0.961, 0.836]
    h_rest = [0.945, np.nan]  # ATR-mid restore not run
    ax.bar(x - 1.5 * w, clean, w, color=CLEAN, label="clean")
    ax.bar(x - 0.5 * w, wipe, w, color=DIE, label="late $h$ wipe")
    ax.bar(x + 0.5 * w, res_w, w, color=NOISE, label="residual noise (write)")
    ax.bar(x[0] + 1.5 * w, 0.945, w, color=RESTORE, label="restore L17 $h$")
    ax.set_xticks(x)
    ax.set_xticklabels(["ATR-short", "ATR-mid"])
    ax.set_title("C. ATR (same checkpoint)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.5)
    ax.set_ylim(0, 1.08)

    fig.suptitle("Causal interventions on $h$ vs residual (n=128; paper checkpoint)", fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_causal")


def fig4_probes() -> None:
    # best test acc by site × feature (any layer), from probes_over_t.md
    sites = ["first\n(t=0)", "value token", "last write\n(end of list)", "query\n(L23)"]
    residual = [0.031, 1.000, 0.073, 0.979]
    h = [0.021, 0.875, 0.104, 0.885]
    x = np.arange(len(sites))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.bar(x - w / 2, residual, w, color=NOISE, label="residual")
    ax.bar(x + w / 2, h, w, color=DIE, label=r"$h$ (best view)")
    ax.axhline(1 / 52, color="#888", ls="--", lw=0.8, label="chance (52-way)")
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("linear probe test accuracy")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Store ≠ readout: last-write has no linear value code (n=384, not v1)")
    ax.text(1, 1.04, "the token itself", ha="center", fontsize=7, color="#555")
    ax.text(2, 0.18, "no linear\nstore", ha="center", fontsize=7, color=DIE)
    ax.text(3, 1.02, "readout", ha="center", fontsize=7, color=NOISE)
    save(fig, "fig4_probes_over_t")


def fig5_pad_vs_stuff() -> None:
    seeds = ["seed 1", "seed 2", "seed 3"]
    pad = [0.219, 0.078, 0.266]
    light = [-0.594, -0.641, -0.578]
    heavy = [-0.469, -0.484, -0.344]
    x = np.arange(len(seeds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.axhline(0, color="#222", lw=0.7)
    ax.bar(x - w, pad, w, color=PAD, label=r"OOD pad  ($\Delta\leftarrow 0$ junk)")
    ax.bar(x, light, w, color=STUFF, label="stuffing, light")
    ax.bar(x + w, heavy, w, color="#9B2226", label="stuffing, heavy")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylabel(r"accuracy change after $\Delta\leftarrow 0$ on filler+pad")
    ax.set_ylim(-0.75, 0.40)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Two failures: pad recovery (sign) vs stuffing collapse (n=64)")
    ax.text(1.0, 0.32, "do not lock +0.22; seed 2 is +0.08", ha="center", fontsize=7.5, color="#444")
    save(fig, "fig5_pad_vs_stuff")


if __name__ == "__main__":
    fig1_write_store_read()
    fig2_layer_sweep()
    fig3_causal()
    fig4_probes()
    fig5_pad_vs_stuff()
    print("done", OUT)
