"""Publication figures for the workshop paper. Locked numbers only. No model run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# PyTorch-adjacent palette: charcoal + flame orange for the die condition.
INK = "#1C1917"
MUTED = "#57534E"
CLEAN = "#44403C"
DIE = "#EE4C2C"
RESTORE = "#15803D"
NOISE = "#1D4ED8"
PAD = "#0F766E"
STUFF = "#9F1239"
LIGHT = "#F5F5F4"
GRID = "#E7E5E4"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)
    print(f"wrote {stem}")


def _bars(ax, xs, ys, colors, width=0.72):
    ax.bar(xs, ys, color=colors, width=width, edgecolor="none", zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for x, y in zip(xs, ys):
        if np.isnan(y):
            continue
        ax.text(x, y + 0.028, f"{y:.3f}", ha="center", va="bottom", fontsize=7, color=INK)


def fig1_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 2.85))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 4.35)
    ax.axis("off")

    def box(x, y, w, h, fc, title, sub):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.03,rounding_size=0.14",
                facecolor=fc, edgecolor=INK, linewidth=0.9,
            )
        )
        ax.text(x + w / 2, y + h / 2 + 0.22, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + 0.38, sub, ha="center", va="center", fontsize=8, color=MUTED)

    box(0.25, 1.35, 3.35, 1.85, "#FFE4DC", "Write", r"$B_t,\;\Delta_t$ into $h$")
    box(4.4, 1.35, 3.35, 1.85, "#FEE2E2", "Store", r"layer-17 $h$ holds the bind")
    box(8.55, 1.35, 3.35, 1.85, "#DCFCE7", "Read", r"$C_t$ at the query")
    for x0, x1 in ((3.6, 4.4), (7.75, 8.55)):
        ax.annotate("", xy=(x1, 2.25), xytext=(x0, 2.25),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.35))

    toks = ["K", "V", "K", "V*", "...", "Q"]
    cols = [LIGHT, LIGHT, LIGHT, "#FDBA74", LIGHT, "#5EEAD4"]
    for i, (t, c) in enumerate(zip(toks, cols)):
        x = 0.4 + i * 0.55
        ax.add_patch(FancyBboxPatch((x, 0.22), 0.48, 0.72,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=c, edgecolor=INK, lw=0.65))
        ax.text(x + 0.24, 0.58, t, ha="center", va="center", fontsize=8.5, fontweight="bold")

    ax.annotate("", xy=(6.05, 1.35), xytext=(2.55, 0.94),
                arrowprops=dict(arrowstyle="-|>", color=DIE, lw=1.05,
                                connectionstyle="arc3,rad=0.16"))
    ax.text(3.55, 0.98, "value token", fontsize=7, color=DIE, ha="center")
    ax.text(6.1, 4.12, "The bind lives in recurrent state, not the residual bus",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(6.1, 3.72, "Copying L17 $h$ restores AR. Copying the residual at the same sites does not.",
            ha="center", fontsize=8, color=MUTED)
    save(fig, "fig1_write_store_read")


def fig2_layer_sweep() -> None:
    layers = list(range(24))
    acc = [
        0.938, 0.961, 0.969, 0.961, 0.961, 0.961, 0.961, 0.961,
        0.961, 0.961, 0.961, 0.961, 0.969, 0.961, 0.961, 0.953,
        0.969, 0.297, 0.953, 0.961, 0.953, 0.961, 0.961, 0.961,
    ]
    colors = [DIE if i == 17 else "#A8A29E" for i in layers]
    fig, ax = plt.subplots(figsize=(7.2, 2.95))
    ax.bar(layers, acc, color=colors, width=0.82, edgecolor="none", zorder=3)
    ax.axhline(0.961, color=RESTORE, ls="--", lw=1.0, label="clean 0.961", zorder=2)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Layer (zero $h$ on write tokens, one layer at a time)")
    ax.set_ylabel("AR accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(i) if i in (0, 8, 16, 17, 18, 23) else "" for i in layers])
    ax.text(17, 0.34, "L17\n0.297", ha="center", va="bottom", fontsize=8,
            color=DIE, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Only layer 17 write-wipe collapses associative recall (n=128)")
    save(fig, "fig2_layer_sweep")


def fig3_causal() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.25), sharey=True)

    ax = axes[0]
    names = ["clean", "residual\nnoise", "late $h$\nwipe", "restore\nresidual", "restore\nL17 $h$"]
    vals = [0.961, 0.961, 0.242, 0.242, 0.961]
    cols = [CLEAN, NOISE, DIE, DIE, RESTORE]
    _bars(ax, range(len(vals)), vals, cols)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names)
    ax.set_ylabel("AR accuracy")
    ax.set_ylim(0, 1.14)
    ax.set_title("A. Store is not residual")

    ax = axes[1]
    names = ["$h$@value", "$h$@last\nwrite", "$h$@query", "L16 $h$", "L18 $h$"]
    vals = [0.969, 0.961, 0.953, 0.258, 0.242]
    cols = [RESTORE, RESTORE, RESTORE, DIE, DIE]
    _bars(ax, range(len(vals)), vals, cols)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names)
    ax.axhline(0.242, color=DIE, ls=":", lw=0.9, zorder=1)
    ax.set_title("B. Time and neighbors")

    ax = axes[2]
    x = np.arange(2)
    w = 0.2
    ax.bar(x - 1.5 * w, [0.961, 0.844], w, color=CLEAN, label="clean", zorder=3)
    ax.bar(x - 0.5 * w, [0.039, 0.039], w, color=DIE, label="late $h$ wipe", zorder=3)
    ax.bar(x + 0.5 * w, [0.961, 0.836], w, color=NOISE, label="residual noise", zorder=3)
    ax.bar(x[0] + 1.5 * w, 0.945, w, color=RESTORE, label="restore L17 $h$", zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(["ATR-short", "ATR-mid"])
    ax.set_title("C. ATR, same checkpoint")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    ax.set_ylim(0, 1.14)

    fig.suptitle("Causal interventions on $h$ versus residual (n=128)", fontsize=10.5, y=1.03)
    fig.tight_layout()
    save(fig, "fig3_causal")


def fig4_probes() -> None:
    sites = ["first\n(t=0)", "value token", "last write\n(end of list)", "query\n(L23)"]
    residual = [0.031, 1.000, 0.073, 0.979]
    h = [0.021, 0.875, 0.104, 0.885]
    x = np.arange(len(sites))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar(x - w / 2, residual, w, color=NOISE, label="residual", zorder=3)
    ax.bar(x + w / 2, h, w, color=DIE, label=r"$h$ (best view)", zorder=3)
    ax.axhline(1 / 52, color=MUTED, ls="--", lw=0.9, label="chance (52-way)")
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(sites)
    ax.set_ylabel("linear probe test accuracy")
    ax.set_ylim(0, 1.16)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Store is not readout: last-write has no linear value code")
    ax.text(1, 1.05, "the token itself", ha="center", fontsize=7, color=MUTED)
    ax.text(2, 0.16, "no linear store", ha="center", fontsize=7.5, color=DIE, fontweight="bold")
    ax.text(3, 1.03, "readout", ha="center", fontsize=7.5, color=NOISE, fontweight="bold")
    save(fig, "fig4_probes_over_t")


def fig5_failures() -> None:
    seeds = ["seed 1", "seed 2", "seed 3"]
    pad = [0.219, 0.078, 0.266]
    light = [-0.594, -0.641, -0.578]
    heavy = [-0.469, -0.484, -0.344]
    x = np.arange(len(seeds))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.5, 3.25))
    ax.axhline(0, color=INK, lw=0.8)
    ax.bar(x - w, pad, w, color=PAD, label=r"OOD pad ($\Delta\leftarrow 0$ junk)", zorder=3)
    ax.bar(x, light, w, color="#FB7185", label="stuffing, light", zorder=3)
    ax.bar(x + w, heavy, w, color=STUFF, label="stuffing, heavy", zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_ylabel(r"accuracy change after $\Delta\leftarrow 0$ on filler+pad")
    ax.set_ylim(-0.78, 0.42)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Two failures: pad recovery (sign) versus stuffing collapse")
    ax.text(1.0, 0.33, "do not lock +0.22; seed 2 is +0.08", ha="center", fontsize=7.5, color=MUTED)
    save(fig, "fig5_pad_vs_stuff")


def fig6_svd() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.05), sharey=True)
    ax = axes[0]
    ks = ["1", "2", "4", "8"]
    vals = [0.961, 0.914, 0.672, 0.297]
    cols = [CLEAN, CLEAN, "#F97316", DIE]
    _bars(ax, range(4), vals, cols)
    ax.set_xticks(range(4))
    ax.set_xticklabels(ks)
    ax.set_xlabel("zero top-$k$ singular values")
    ax.set_ylabel("AR accuracy")
    ax.set_ylim(0, 1.14)
    ax.set_title("Dropping rank")
    ax.axhline(0.297, color=DIE, ls=":", lw=0.8)

    ax = axes[1]
    ks = ["1", "4", "8"]
    vals = [0.312, 0.875, 0.961]
    cols = [DIE, "#F97316", RESTORE]
    _bars(ax, range(3), vals, cols)
    ax.set_xticks(range(3))
    ax.set_xticklabels(ks)
    ax.set_xlabel("keep top-$k$ only")
    ax.set_title("Keeping rank")
    fig.suptitle("SVD of L17 $h$ at last write: distributed, not rank-1 (n=128)", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_svd")


def fig7_emergence() -> None:
    steps = [0, 50, 100, 150, 200, 400, 600, 800]
    clean = [0.000, 0.961, 0.914, 0.945, 0.938, 0.945, 0.953, 0.961]
    wipe = [np.nan, 0.352, 0.297, 0.367, 0.367, 0.359, 0.320, 0.297]
    fig, ax = plt.subplots(figsize=(6.8, 3.15))
    ax.plot(steps, clean, "o-", color=RESTORE, lw=1.8, ms=6, label="clean AR")
    ax.plot(steps[1:], wipe[1:], "s-", color=DIE, lw=1.8, ms=6, label="L17 write-wipe")
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("finetune step")
    ax.set_ylabel("AR accuracy")
    ax.set_ylim(-0.05, 1.08)
    ax.legend(frameon=False, loc="center right")
    ax.set_title("L17 store is present by step 50 (pretrained AR is 0%)")
    ax.annotate("already storing", xy=(50, 0.352), xytext=(180, 0.18),
                fontsize=7.5, color=DIE,
                arrowprops=dict(arrowstyle="-|>", color=DIE, lw=0.9))
    save(fig, "fig7_emergence")


def fig8_maps() -> None:
    labels = ["queried\nvalue", "other\nvalue", "key", "query", "other"]
    mass = [0.111, 0.333, 0.444, 0.111, 0.000]
    cols = [NOISE, "#93C5FD", DIE, PAD, "#A8A29E"]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    _bars(ax, range(5), mass, cols, width=0.7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean softmax mass")
    ax.set_ylim(0, 0.58)
    ax.set_title(r"Control map $\mathrm{mean}_E(h_t \cdot C_{\mathrm{query}})$ lights keys, not values")
    save(fig, "fig8_control_maps")


if __name__ == "__main__":
    fig1_schematic()
    fig2_layer_sweep()
    fig3_causal()
    fig4_probes()
    fig5_failures()
    fig6_svd()
    fig7_emergence()
    fig8_maps()
    print("done", OUT)
