#!/usr/bin/env python3
"""Figures for the coactivation-block-structure report (P1/P2/P3).

All numbers are transcribed from the job logs recorded in the labtool
journal cards:
  P1  20260725-000051-coact-llama2-p1-stats
  P2  20260725-004032-coact-llama2-p2-blocks
  P3  20260725-033520-coact-llama2-p3-prep / 20260725-034614-coact-llama2-p3-ppl
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.family": "Helvetica Neue",
    "axes.unicode_minus": False,
    "figure.dpi": 170,
    "savefig.dpi": 170,
    "axes.edgecolor": "#9a998f",
    "axes.labelcolor": "#1a1a19",
    "text.color": "#1a1a19",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

OUT = os.path.dirname(os.path.abspath(__file__))
LAYERS = [0, 8, 16, 24, 31]
XL = [f"Layer {l}" for l in LAYERS]


def finish(fig, ax, path):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def dashed(color, label, ls="--"):
    return Line2D([0], [0], color=color, ls=ls, lw=1.4, label=label)


# ---------------------------------------------------------------- Fig 1
def fig1():
    data = {16: [6.14, 6.34, 6.01, 6.34, 3.92],
            32: [7.87, 8.17, 7.88, 8.21, 5.19],
            64: [9.13, 9.37, 9.22, 9.43, 6.62]}
    colors = {16: "#85B7EB", 32: "#378ADD", 64: "#185FA5"}
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    x = range(len(LAYERS))
    w = 0.26
    handles = []
    for k, g in enumerate([16, 32, 64]):
        pos = [i + (k - 1) * w for i in x]
        b = ax.bar(pos, data[g], w * 0.92, label=f"group size g = {g} tokens",
                   color=colors[g])
        ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
        handles.append(b)
    ax.axhline(9.96, color="#A32D2D", ls="--", lw=1.4)
    ax.axhline(1.0, color="#3B6D11", ls="--", lw=1.4)
    ax.set_xticks(list(x), XL)
    ax.set_ylim(0, 14.6)
    ax.set_ylabel("Union inflation  U(g)  =  E[ neurons used by a group ] / K_eff")
    ax.set_xlabel("Transformer layer (0 = closest to input, 31 = last layer)")
    ax.set_title("Figure 1.  Union inflation of group-shared masks (LLaMA-2-7B, s = 0.9)",
                 fontsize=11.5, loc="left", pad=12)
    leg = list(handles) + [
        dashed("#A32D2D", "saturation bound d / K_eff = 9.96 (union covers all neurons, i.e. equivalent to dense)"),
        dashed("#3B6D11", "lower bound 1.0 (all tokens select identical neurons, i.e. sharing is free)")]
    ax.legend(handles=leg, frameon=False, ncol=2, fontsize=8.5,
              loc="upper left", handlelength=1.6, columnspacing=1.4)
    finish(fig, ax, f"{OUT}/fig1_union_inflation.png")


# ---------------------------------------------------------------- Fig 2
def fig2():
    clus = [0.359, 0.252, 0.269, 0.283, 0.569]
    rand = [0.167, 0.167, 0.167, 0.168, 0.168]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    x = range(len(LAYERS))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], clus, w * 0.92,
                label="PPMI-clustered blocks", color="#2a78d6")
    b2 = ax.bar([i + w / 2 for i in x], rand, w * 0.92,
                label="random balanced blocks (control)", color="#888780")
    ax.bar_label(b1, fmt="%.3f", fontsize=8.5, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8.5, padding=2)
    ax.set_xticks(list(x),
                  [f"{n}\n({c / r:.2f}×)" for n, c, r in zip(XL, clus, rand)])
    ax.set_ylim(0, 0.80)
    ax.set_ylabel("Coverage  =  (selected neurons inside top-m blocks) / K_t")
    ax.set_xlabel("Transformer layer   (parenthesis: clustered ÷ random ratio)")
    ax.set_title("Figure 2.  P2 — per-token coverage of the within-budget blocks (B = 64, s = 0.9)",
                 fontsize=11.5, loc="left", pad=12)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    finish(fig, ax, f"{OUT}/fig2_p2_coverage.png")


# ---------------------------------------------------------------- Fig 3
def fig3():
    ratios = {64: [2.18, 1.36, 1.44, 1.54, 4.12],
              128: [2.07, 1.32, 1.38, 1.47, 3.91],
              256: [1.91, 1.25, 1.31, 1.41, 3.77]}
    colors = {64: "#9FE1CB", 128: "#1D9E75", 256: "#0F6E56"}
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    x = range(len(LAYERS))
    w = 0.26
    handles = []
    for k, B in enumerate([64, 128, 256]):
        pos = [i + (k - 1) * w for i in x]
        b = ax.bar(pos, ratios[B], w * 0.92, label=f"block size B = {B}",
                   color=colors[B])
        ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
        handles.append(b)
    ax.axhline(1.3, color="#A32D2D", ls="--", lw=1.4)
    ax.axhline(1.0, color="#6b6a66", ls=":", lw=1.2)
    ax.set_xticks(list(x), XL)
    ax.set_ylim(0, 6.6)
    ax.set_ylabel("Within-block mass, clustered ÷ random  (ratio)")
    ax.set_xlabel("Transformer layer")
    ax.set_title("Figure 3.  P2 — co-activation mass captured inside blocks, relative to random partitions",
                 fontsize=11.5, loc="left", pad=12)
    leg = list(handles) + [
        dashed("#A32D2D", "pre-registered rejection threshold 1.3×"),
        dashed("#6b6a66", "1.0× = identical to random (no structure)", ls=":")]
    ax.legend(handles=leg, frameon=False, ncol=3, fontsize=8.5,
              loc="upper left", handlelength=1.6, columnspacing=1.4)
    finish(fig, ax, f"{OUT}/fig3_p2_mass_ratio.png")


# ---------------------------------------------------------------- Fig 4
def fig4():
    labels = ["B=64\ng=16", "B=64\ng=64", "B=256\ng=16", "B=256\ng=64"]
    clus = [4539.17, 12032.11, 6950.12, 7776.23]
    rand = [9673.61, 11941.22, 14763.07, 23918.58]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = range(len(labels))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], clus, w * 0.92,
                label="PPMI-clustered blocks", color="#2a78d6")
    b2 = ax.bar([i + w / 2 for i in x], rand, w * 0.92,
                label="random blocks (control)", color="#888780")
    ax.bar_label(b1, fontsize=8.5, padding=2, labels=[f"{v:,.0f}" for v in clus])
    ax.bar_label(b2, fontsize=8.5, padding=2, labels=[f"{v:,.0f}" for v in rand])
    ax.set_yscale("log")
    ax.set_ylim(4, 400000)
    ax.axhline(5.4738, color="#3B6D11", ls="--", lw=1.4)
    ax.axhline(8.1096, color="#BA7517", ls="--", lw=1.4)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("WikiText-2 perplexity  (lower is better, log scale)")
    ax.set_xlabel("Configuration — block size B × group size g")
    ax.set_title("Figure 4.  P3 — oracle perplexity of group-shared block masks (s = 0.9, all 32 layers)",
                 fontsize=11.5, loc="left", pad=12)
    leg = [b1, b2,
           dashed("#3B6D11", "dense baseline 5.4738"),
           dashed("#BA7517", "per-token Top-K anchor 8.1096 (same budget, free per-token selection)")]
    ax.legend(handles=leg, frameon=False, ncol=2, fontsize=8.5,
              loc="upper left", handlelength=1.6, columnspacing=1.4)
    finish(fig, ax, f"{OUT}/fig4_p3_ppl.png")


# ---------------------------------------------------------------- Fig 7
def fig7():
    # Token-pair selection overlap vs sparsity (mean over 32 layers).
    # Source: job 20260724-173030-larosa-llama2-topk-overlap
    # (journal card in topic larosa-intermediate-sparsity).
    s = [0.5, 0.7, 0.9]
    series = {
        "adjacent tokens (distance 1)":   ([0.575, 0.434, 0.316], "#185FA5", "o"),
        "distance 16 (same 16-token group)": ([0.531, 0.359, 0.208], "#378ADD", "s"),
        "distance 64 (same 64-token group)": ([0.527, 0.352, 0.197], "#85B7EB", "^"),
        "random token pair":              ([0.524, 0.347, 0.187], "#888780", "D"),
    }
    chance = [0.501, 0.301, 0.100]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for name, (ys, c, mk) in series.items():
        ax.plot(s, ys, marker=mk, color=c, lw=2, ms=6, label=name)
    ax.plot(s, chance, ls="--", color="#A32D2D", lw=1.6,
            label="chance = K/d (expected overlap of two independent random selections)")
    for xi, yi in zip(s, chance):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8.5, color="#A32D2D")
    for xi, yi in zip(s, series["adjacent tokens (distance 1)"][0]):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color="#185FA5")
    ax.set_xticks(s, [f"s = {v}" for v in s])
    ax.set_ylim(0, 0.72)
    ax.set_xlim(0.46, 0.94)
    ax.set_ylabel("Overlap  =  (neurons selected by both tokens) / K")
    ax.set_xlabel("Sparsity level s  (fraction of neurons zeroed per token)")
    ax.set_title("Figure 7.  How much do two tokens agree on their neurons? (LLaMA-2-7B, mean over 32 layers)",
                 fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    finish(fig, ax, f"{OUT}/fig7_overlap_vs_sparsity.png")


# ------------------------------------------------- Figs 5-6 (qualitative)
# Derived from job 20260804-224920-coact-selection-slice (layer 16, first
# 256 tokens of one WikiText-2 sequence, s = 0.5/0.7/0.9). The small derived
# arrays are stored next to this script so the figures can be regenerated
# without the 24 MB dump.
import numpy as np
from matplotlib.colors import ListedColormap

SPARSITIES = ["0.5", "0.7", "0.9"]
CHANCE = {"0.5": 0.50, "0.7": 0.30, "0.9": 0.10}


def _data():
    return np.load(f"{OUT}/fig_data_selection.npz")


def fig5():
    z = _data()
    cmap = ListedColormap(["#f4f3ee", "#185FA5", "#eb6834"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, s in zip(axes, SPARSITIES):
        r = z[f"raster_{s}"]
        ax.imshow(r, cmap=cmap, vmin=0, vmax=2, aspect="auto",
                  interpolation="nearest")
        shared, new = int((r == 1).sum()), int((r == 2).sum())
        ax.set_title(f"s = {s}   (K = {int(z[f'keff_{s}'])} of 11,008 neurons)\n"
                     f"kept from previous token {shared / (shared + new):.0%}"
                     f"   ·   newly selected {new / (shared + new):.0%}",
                     fontsize=10, pad=8)
        ax.set_xlabel("neuron index (window of 320 consecutive neurons)")
        ax.grid(False)
    axes[0].set_ylabel("token position (24 consecutive tokens)")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#185FA5"),
               plt.Rectangle((0, 0), 1, 1, color="#eb6834"),
               plt.Rectangle((0, 0), 1, 1, color="#f4f3ee", ec="#c3c2b7")]
    fig.legend(handles,
               ["also selected by the previous token",
                "selected by this token only (new)",
                "not selected"],
               frameon=False, fontsize=9, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Figure 5.  Which neurons each token selects (LLaMA-2-7B, layer 16) — "
                 "higher sparsity leaves less agreement with the neighbouring token",
                 fontsize=11.5, x=0.008, ha="left", y=1.04)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.savefig(f"{OUT}/fig5_selection_raster.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", f"{OUT}/fig5_selection_raster.png")


def fig6():
    z = _data()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, s in zip(axes, SPARSITIES):
        o = z[f"overlap_{s}"]
        im = ax.imshow(o, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        off = o[~np.eye(o.shape[0], dtype=bool)]
        adj = np.mean([o[i, i + 1] for i in range(o.shape[0] - 1)])
        ax.set_title(f"s = {s}   ·   chance {CHANCE[s]:.2f}\n"
                     f"adjacent {adj:.2f}   ·   all pairs {off.mean():.2f}",
                     fontsize=10, pad=8)
        ax.set_xlabel("token position")
        ax.grid(False)
        if ax is axes[0]:
            ax.set_ylabel("token position")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("overlap between the two tokens' neuron sets  (1 = identical)",
                 fontsize=9)
    fig.suptitle("Figure 6.  Pairwise overlap among 48 consecutive tokens (layer 16) — "
                 "the matrix uniformly fades as sparsity rises",
                 fontsize=11.5, x=0.008, ha="left", y=1.0)
    fig.savefig(f"{OUT}/fig6_overlap_matrix.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", f"{OUT}/fig6_overlap_matrix.png")


fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
