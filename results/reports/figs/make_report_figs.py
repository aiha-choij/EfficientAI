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
    "font.family": "Apple SD Gothic Neo",
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

OUT = "/Users/choijungwook/Workspace/EfficientAI/results/reports/figs"
os.makedirs(OUT, exist_ok=True)
LAYERS = [0, 8, 16, 24, 31]
XL = [f"layer {l}" for l in LAYERS]


def finish(fig, ax, path):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def dashed(color, label):
    return Line2D([0], [0], color=color, ls="--", lw=1.4, label=label)


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
        b = ax.bar(pos, data[g], w * 0.92, label=f"g={g} 토큰", color=colors[g])
        ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
        handles.append(b)
    ax.axhline(9.96, color="#A32D2D", ls="--", lw=1.4)
    ax.axhline(1.0, color="#3B6D11", ls="--", lw=1.4)
    ax.set_xticks(list(x), XL)
    ax.set_ylim(0, 14.6)
    ax.set_ylabel("union tax  =  E[ |∪ S_t| ] / K_eff     (배)")
    ax.set_xlabel("층 (layer index) — 0은 입력에 가까운 층, 31은 마지막 층")
    ax.set_title("그림 1. Union tax — 토큰 그룹이 mask를 공유할 때의 예산 팽창 (LLaMA2-7B, s=0.9)",
                 fontsize=11.5, loc="left", pad=12)
    leg = list(handles) + [
        dashed("#A32D2D", "포화 상한 9.96× = d/K_eff (합집합이 전체 뉴런을 덮음 → dense와 동일)"),
        dashed("#3B6D11", "하한 1.0× (모든 토큰이 같은 뉴런 선택 → 공유가 공짜)")]
    ax.legend(handles=leg, frameon=False, ncol=2, fontsize=8.5,
              loc="upper left", handlelength=1.6, columnspacing=1.4)
    finish(fig, ax, f"{OUT}/fig1_union_tax.png")


# ---------------------------------------------------------------- Fig 2
def fig2():
    clus = [0.359, 0.252, 0.269, 0.283, 0.569]
    rand = [0.167, 0.167, 0.167, 0.168, 0.168]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    x = range(len(LAYERS))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], clus, w * 0.92,
                label="PPMI 클러스터 블록", color="#2a78d6")
    b2 = ax.bar([i + w / 2 for i in x], rand, w * 0.92,
                label="무작위 균형 블록 (통제군)", color="#888780")
    ax.bar_label(b1, fmt="%.3f", fontsize=8.5, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8.5, padding=2)
    ax.set_xticks(list(x), [f"{n}\n({c / r:.2f}×)" for n, c, r in zip(XL, clus, rand)])
    ax.set_ylim(0, 0.80)
    ax.set_ylabel("coverage  =  상위 m개 블록에 든 선택 뉴런 수 / K_t     (0 – 1)")
    ax.set_xlabel("층        (괄호 안 = 클러스터 ÷ 무작위 배율)")
    ax.set_title("그림 2. P2 — 예산 안의 블록이 토큰의 실제 선택을 얼마나 덮는가 (B=64, s=0.9)",
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
        b = ax.bar(pos, ratios[B], w * 0.92, label=f"B={B}", color=colors[B])
        ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
        handles.append(b)
    ax.axhline(1.3, color="#A32D2D", ls="--", lw=1.4)
    ax.axhline(1.0, color="#6b6a66", ls=":", lw=1.2)
    ax.set_xticks(list(x), XL)
    ax.set_ylim(0, 6.6)
    ax.set_ylabel("배율  =  클러스터 블록의 within-block mass ÷ 무작위 블록의 값")
    ax.set_xlabel("층")
    ax.set_title("그림 3. P2 — 블록 내부에 모인 동시활성 질량 (무작위 대비 배율)",
                 fontsize=11.5, loc="left", pad=12)
    leg = list(handles) + [
        dashed("#A32D2D", "사전 등록 문턱 1.3× (미달이면 치환 축 즉시 기각)"),
        Line2D([0], [0], color="#6b6a66", ls=":", lw=1.2,
               label="1.0× = 무작위와 동일 (구조 없음)")]
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
                label="PPMI 클러스터 블록", color="#2a78d6")
    b2 = ax.bar([i + w / 2 for i in x], rand, w * 0.92,
                label="무작위 블록 (통제군)", color="#888780")
    ax.bar_label(b1, fontsize=8.5, padding=2, labels=[f"{v:,.0f}" for v in clus])
    ax.bar_label(b2, fontsize=8.5, padding=2, labels=[f"{v:,.0f}" for v in rand])
    ax.set_yscale("log")
    ax.set_ylim(4, 400000)
    ax.axhline(5.4738, color="#3B6D11", ls="--", lw=1.4)
    ax.axhline(8.1096, color="#BA7517", ls="--", lw=1.4)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("wikitext-2 PPL   (낮을수록 좋음, 로그 스케일)")
    ax.set_xlabel("설정 — 블록 크기 B × 그룹 크기 g")
    ax.set_title("그림 4. P3 — 블록 mask oracle PPL (s=0.9, 32개 층 전부 적용)",
                 fontsize=11.5, loc="left", pad=12)
    leg = [b1, b2,
           dashed("#3B6D11", "dense 5.4738"),
           dashed("#BA7517", "per-token top-K 앵커 8.1096 (같은 예산, 토큰별 자유 선택)")]
    ax.legend(handles=leg, frameon=False, ncol=2, fontsize=8.5,
              loc="upper left", handlelength=1.6, columnspacing=1.4)
    finish(fig, ax, f"{OUT}/fig4_p3_ppl.png")


fig1(); fig2(); fig3(); fig4()
