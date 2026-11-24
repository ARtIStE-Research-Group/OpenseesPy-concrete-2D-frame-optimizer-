"""Figure of the ARCOPY vs SAP2000 comparison for the six frames (run compare_frames.py first)."""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CMP = HERE
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
C_ARC, C_SAP = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF, "pdf.fonttype": 42})
FR = ["F4-5B", "F6-5B", "F8-5B", "F4-7B", "F6-7B", "F8-7B"]
_ps = os.path.join(CMP, "pairs_static.csv")            # 25 MB: kept in the repository as pairs_static.zip
P = pd.read_csv(_ps if os.path.exists(_ps) else os.path.join(CMP, "pairs_static.zip"),
                usecols=["frame", "quantity", "diff_steps"])
D = pd.read_csv(os.path.join(CMP, "drifts.csv"))
M = pd.read_csv(os.path.join(CMP, "modal.csv"))
R = pd.read_csv(os.path.join(CMP, "pairs_rsa.csv"))
S = json.load(open(os.path.join(CMP, "summary.json")))


def minus(s):
    return s.replace("-", "−")


def style3d(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor(SURF); a.pane.set_edgecolor(GRID)
        a._axinfo["grid"]["color"] = GRID; a._axinfo["grid"]["linewidth"] = 0.5


fig = plt.figure(figsize=(7.2, 6.6))
# (a) 3D storey drift profiles under the wind-leading ULS combination
ax = fig.add_axes([0.0, 0.47, 0.52, 0.50], projection="3d"); style3d(ax)
for j, f in enumerate(FR):
    d = D[(D.frame == f) & (D.case == "ULS_W+_leading")].sort_values("storey")
    st = np.concatenate([[0], d.storey.values])
    xa = np.concatenate([[0], d.arcopy_pct.values]); xs = np.concatenate([[0], d.sap2000_pct.values])
    ax.plot(xa[1:], np.full(len(d), j), d.storey.values, color=C_ARC, lw=1.5)
    ax.scatter(xs[1:], np.full(len(d), j), d.storey.values, s=11, facecolor="white", edgecolor=C_SAP, linewidth=0.9,
               depthshade=False)
    ax.plot([0, 0], [j, j], [1, d.storey.max()], color=GRID, lw=0.8)
ax.set_yticks(range(len(FR))); ax.set_yticklabels(FR, fontsize=6.0, ha="left", va="center")
ax.set_zticks([1, 2, 4, 6, 8]); ax.set_xlabel("storey drift (%)", labelpad=-7, fontsize=6.8)
ax.set_zlabel("storey", labelpad=-8, fontsize=6.8)
ax.tick_params(axis="x", pad=-3, labelsize=6.3); ax.tick_params(axis="y", pad=-4); ax.tick_params(axis="z", pad=-2, labelsize=6.3)
ax.set_box_aspect((1.0, 1.5, 0.9)); ax.view_init(elev=22, azim=-58)
fig.text(0.01, 0.965, "(a) Storey drifts, ULS with leading wind W+", fontsize=7.8, color=INK)
fig.text(0.04, 0.943, "ARCOPY: lines; SAP2000: circles", fontsize=6.2, color=INK2)
# (b) 3D periods of the first four modes
ax = fig.add_axes([0.50, 0.47, 0.50, 0.50], projection="3d"); style3d(ax)
for j, f in enumerate(FR):
    m = M[(M.frame == f) & (M["mode"] <= 4)]
    for _, r in m.iterrows():
        ax.plot([r["mode"], r["mode"]], [j, j], [0, r.T_arcopy], color=C_ARC, lw=3.2, solid_capstyle="butt")
    ax.scatter(m["mode"], np.full(len(m), j), m.T_sap2000, s=16, facecolor="white", edgecolor=C_SAP, linewidth=1.0,
               depthshade=False, zorder=10)
ax.set_yticks(range(len(FR))); ax.set_yticklabels(FR, fontsize=6.0, ha="left", va="center")
ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("mode", labelpad=-8, fontsize=6.8)
ax.set_zlabel("period T (s)", labelpad=-6, fontsize=6.8)
ax.tick_params(axis="x", pad=-3, labelsize=6.3); ax.tick_params(axis="y", pad=-4); ax.tick_params(axis="z", pad=-1, labelsize=6.3)
ax.set_box_aspect((0.9, 1.5, 0.9)); ax.view_init(elev=22, azim=-52)
fig.text(0.53, 0.965, "(b) Periods of the first four modes (cracked stiffness)", fontsize=7.8, color=INK)
fig.text(0.56, 0.943, minus(f"ARCOPY: bars; SAP2000: circles; largest difference {S['T_max_diff_pct']:.4f}%, "
                            f"MAC ≥ {np.floor(S['MAC_min'] * 1e6) / 1e6:.6f}"), fontsize=6.2, color=INK2)
# (c) static differences per frame
ax = fig.add_axes([0.085, 0.07, 0.40, 0.32])
ax.axhspan(-0.5, 0.5, color=C_SAP, alpha=0.07, lw=0)
for v in (-0.5, 0.5):
    ax.axhline(v, color=C_SAP, lw=0.9)
rng = np.random.default_rng(5)
for i, f in enumerate(FR):
    d = P[P.frame == f].diff_steps.values
    sub = rng.choice(d, size=min(2500, d.size), replace=False)
    ax.scatter(i + rng.uniform(-0.33, 0.33, sub.size), sub, s=1.6, color=C_ARC, alpha=0.3, lw=0, rasterized=True)
    b, sd = d.mean(), d.std(ddof=1)
    ax.plot([i - 0.4, i + 0.4], [b, b], color=INK, lw=1.1)
    for v in (b - 1.96 * sd, b + 1.96 * sd):
        ax.plot([i - 0.4, i + 0.4], [v, v], color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.text(i, -0.80, minus(f"n = {d.size:,}\nSD {sd:.3f}"), fontsize=5.4, color=INK2, ha="center", va="bottom")
ax.set_xticks(range(len(FR))); ax.set_xticklabels(FR, fontsize=6.3); ax.set_xlim(-0.6, len(FR) - 0.4)
ax.set_ylim(-0.82, 0.75); ax.grid(axis="x", visible=False); ax.tick_params(labelsize=6.3)
ax.set_ylabel("ARCOPY − SAP2000\n(units of the last printed digit)", fontsize=6.8)
ax.legend(handles=[Line2D([], [], color=INK, lw=1.1, label="bias"),
                   Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 2)), label="bias ± 1.96 SD"),
                   Line2D([], [], color=C_SAP, lw=0.9, label="± 0.5 digit (rounding)")],
          loc="upper left", fontsize=5.6, ncol=3, bbox_to_anchor=(0.0, 1.03), columnspacing=1.0, handlelength=1.8)
ax.set_title(f"(c) Static analysis: all {S['n_static']:,} values", loc="left", fontsize=7.8, x=-0.19)
# (d) response-spectrum differences per frame
ax = fig.add_axes([0.61, 0.11, 0.37, 0.28])
ax.axhline(0, color=AXIS, lw=1.0)
for i, f in enumerate(FR):
    d = R[R.frame == f]
    big = d[d.sap2000 >= 5.0].diff_pct.values
    small = d[d.sap2000 < 5.0].diff_pct.values
    ax.scatter(i - 0.15 + rng.uniform(-0.12, 0.12, big.size), big, s=3, color=C_ARC, alpha=0.5, lw=0)
    ax.scatter(i + 0.2 + rng.uniform(-0.1, 0.1, small.size), small, s=3, color=MUTED, alpha=0.6, lw=0)
    ax.text(i, -0.137, f"n = {len(d)}", fontsize=5.4, color=INK2, ha="center", va="bottom")
ax.set_xticks(range(len(FR))); ax.set_xticklabels(FR, fontsize=6.3); ax.set_xlim(-0.6, len(FR) - 0.4)
ax.set_ylim(-0.14, 0.14); ax.grid(axis="x", visible=False); ax.tick_params(labelsize=6.3)
ax.set_ylabel("ARCOPY − SAP2000 (% of SAP2000)", fontsize=6.8)
ax.legend(handles=[Line2D([], [], color=C_ARC, marker="o", ls="none", ms=3, label="values ≥ 5 kN, kNm or mm"),
                   Line2D([], [], color=MUTED, marker="o", ls="none", ms=3, label="smaller values (rounding of the printed digits)")],
          loc="upper center", fontsize=5.6, bbox_to_anchor=(0.5, -0.1), ncol=1)
ax.set_title(f"(d) Response spectrum (CQC): all {S['n_rsa']:,} values", loc="left", fontsize=7.8, x=-0.2)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"Fig_six_frames_verification.{ext}"), dpi=300, bbox_inches="tight")
big = R[R.sap2000 >= 5.0].diff_pct.abs().max(); print("RSA max % for values >= 5:", big)
