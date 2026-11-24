"""Figures of the ARCOPY vs SAP2000 comparison for the two-storey portal (run compare_portal_sap.py first).

Fig_P_verification      the figure for the paper:
                        (a) 3D: bending moments of the first-floor beams in load cases and combinations,
                            ARCOPY curves and SAP2000 stations
                        (b) 3D: the two modes (cracked stiffness), periods, mass ratios and MAC
                        (c) static agreement: all differences in units of the last digit printed by SAP2000
                        (d) response spectrum: differences of every value, for the SAP2000 model(s)
Fig_P2_static_agreement supplementary: Bland-Altman plots per family of quantities and the distribution
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.lines import Line2D                                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize       # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
C_ARC, C_SAP, C_3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "lines.linewidth": 1.6, "pdf.fonttype": 42})

P = pd.read_csv(os.path.join(HERE, "pairs_static.csv"))
SUM = json.load(open(os.path.join(HERE, "summary.json")))
NODES = {1: (0, 0), 2: (5, 0), 3: (10, 0), 4: (0, 4), 5: (5, 4), 6: (10, 4), 7: (0, 8), 8: (5, 8), 9: (10, 8)}
ELEM = {1: (1, 4), 2: (2, 5), 3: (3, 6), 4: (4, 7), 5: (5, 8), 6: (6, 9), 7: (4, 5), 8: (5, 6), 9: (7, 8), 10: (8, 9)}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def minus(s):
    return s.replace("-", "−")


def style3d(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor(SURF)
        a.pane.set_edgecolor(GRID)
        a._axinfo["grid"]["color"] = GRID
        a._axinfo["grid"]["linewidth"] = 0.5


def frame3d(ax, y=0.0, z0=0.0):
    for ni, nj in ELEM.values():
        (x1, z1), (x2, z2) = NODES[ni], NODES[nj]
        ax.plot([x1, x2], [y, y], [z1 + z0, z2 + z0], color=INK2, lw=1.3, zorder=1)
    for n in (1, 2, 3):
        x, z = NODES[n]
        ax.plot([x - 0.35, x + 0.35], [y, y], [z + z0, z + z0], color=INK2, lw=2.2)


FREE = [4, 5, 6, 7, 8, 9]
SE = SUM.get("seismic", {})
TAGS = [t for t in ("same_mass", "sap_mass") if t in SE]
LAB = {"same_mass": "SAP2000 with ARCOPY's mass", "sap_mass": "SAP2000 mass from elements and loads"}
COLT = {"same_mass": C_ARC, "sap_mass": C_SAP}
FAMC = {"N, V": C_ARC, "M": "#1c5cab", "displ., react.": C_3}
FAMQ = {"N, V": ["N", "V"], "M": ["M"], "displ., react.": ["ux", "uy", "rz", "Rx", "Ry", "Mz"]}


def draw_waterfall(ax):
    style3d(ax)
    rows = [("SW", "SW"), ("G", "G"), ("live01", "live01"), ("live02", "live02"), ("live03", "live03"),
            ("live04", "live04"), ("W+", "W+"), ("CROWD max", "CROWD max"), ("CROWD min", "CROWD min"),
            ("SLS_char_W+ max", "SLS-c W+ max"), ("SLS_char_W+ min", "SLS-c W+ min"),
            ("ULS_Q_leading_W+ max", "ULS-Q W+ max"), ("ULS_Q_leading_W+ min", "ULS-Q W+ min")]
    for j, (case, _) in enumerate(rows):
        d = P[(P.case == case) & (P.quantity == "M")]
        for eid, x0 in ((7, 0.0), (8, 5.0)):
            e = d[d["item"] == f"E{eid}"].sort_values("station")
            xs, za, zs = x0 + e.x_m.values, e.arcopy.values, e.sap2000.values
            verts = [list(zip(xs, np.full_like(xs, j), np.zeros_like(xs))) +
                     list(zip(xs[::-1], np.full_like(xs, j), za[::-1]))]
            ax.add_collection3d(Poly3DCollection(verts, facecolor=C_ARC, alpha=0.12, edgecolor="none"))
            ax.plot(xs, np.full_like(xs, j), za, color=C_ARC, lw=1.3)
            ax.scatter(xs, np.full_like(xs, j), zs, s=6, facecolor="white", edgecolor=C_SAP, linewidth=0.75,
                       depthshade=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([l for _, l in rows], fontsize=5.4, va="center", ha="left")
    ax.set_xticks([0, 5, 10])
    ax.tick_params(axis="x", pad=-3, labelsize=6.3)
    ax.tick_params(axis="y", pad=-4)
    ax.tick_params(axis="z", pad=-1, labelsize=6.3)
    ax.set_xlabel("x along beams 7 and 8 (m)", labelpad=-7, fontsize=6.8)
    ax.set_zlabel("M (kNm)", labelpad=-6, fontsize=6.8)
    ax.set_box_aspect((1.0, 1.6, 0.8))
    ax.view_init(elev=24, azim=-60)
    d = P[(P.case == "ULS_Q_leading_W+ min") & (P.quantity == "M") & (P["item"] == "E7")]
    r = d.loc[d.sap2000.idxmin()]
    ax.text(5.3, len(rows) - 1, r.sap2000 - 6, minus(f"{r.arcopy:.4f}\n{r.sap2000:.4f}"), fontsize=5.8,
            color=INK, ha="left", va="top")


def draw_modes(ax, tag):
    style3d(ax)
    md = pd.read_csv(os.path.join(HERE, f"modal_{tag}.csv"))
    shapes = json.load(open(os.path.join(HERE, f"shapes_{tag}.json")))
    info = []
    for m, ymode in ((1, 0.0), (2, 1.0)):
        sa = np.array(shapes[str(m)]["arcopy"], float).reshape(-1, 2)
        ss = np.array(shapes[str(m)]["sap2000"], float).reshape(-1, 2)
        ss = ss * (np.dot(sa.ravel(), ss.ravel()) / np.dot(ss.ravel(), ss.ravel()))   # same scale and sign
        sc = 1.8 / np.max(np.abs(sa[:, 0]))
        frame3d(ax, y=ymode)
        Ua = {n: (0.0, 0.0) for n in (1, 2, 3)}
        Us = dict(Ua)
        for i, n in enumerate(FREE):
            Ua[n], Us[n] = tuple(sa[i] * sc), tuple(ss[i] * sc)
        for ni, nj in ELEM.values():
            (x1, z1), (x2, z2) = NODES[ni], NODES[nj]
            ax.plot([x1 + Ua[ni][0], x2 + Ua[nj][0]], [ymode, ymode], [z1 + Ua[ni][1], z2 + Ua[nj][1]],
                    color=C_ARC, lw=1.7)
        for n in FREE:
            x, z = NODES[n]
            ax.scatter([x + Us[n][0]], [ymode], [z + Us[n][1]], s=18, facecolor="white", edgecolor=C_SAP,
                       linewidth=1.1, depthshade=False)
        r = md[md["mode"] == m].iloc[0]
        info.append(f"Mode {m}:  T = {r.T_arcopy_s:.4f} / {r.T_sap2000_s:.4f} s,  mass ratio "
                    f"{100 * r.mass_arcopy:.1f} / {100 * r.mass_sap2000:.1f}%,  MAC = {r.MAC:.6f}")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["mode 1", "mode 2"], fontsize=6.0)
    ax.set_xticks([0, 5, 10])
    ax.set_zticks([0, 4, 8])
    ax.set_zlim(0, 9)
    ax.tick_params(pad=-3, labelsize=6.3)
    ax.set_xlabel("x (m)", labelpad=-8, fontsize=6.8)
    ax.set_zlabel("z (m)", labelpad=-8, fontsize=6.8)
    ax.set_box_aspect((12, 13, 9))
    ax.view_init(elev=20, azim=-52)
    return info


def draw_static_stats(ax):
    ax.axhspan(-0.5, 0.5, color=C_SAP, alpha=0.07, lw=0)
    for v in (-0.5, 0.5):
        ax.axhline(v, color=C_SAP, lw=0.9)
    xs = 0
    rng = np.random.default_rng(3)
    ticks = []
    for i, (fam, qs) in enumerate(FAMQ.items()):
        d = P[P.quantity.isin(qs)].diff_steps.values
        x = i + rng.uniform(-0.33, 0.33, d.size)
        ax.scatter(x, d, s=2.0, color=FAMC[fam], alpha=0.35, lw=0, rasterized=True)
        b, sd = d.mean(), d.std(ddof=1)
        ax.plot([i - 0.4, i + 0.4], [b, b], color=INK, lw=1.1)
        for v in (b - 1.96 * sd, b + 1.96 * sd):
            ax.plot([i - 0.4, i + 0.4], [v, v], color=INK2, lw=0.9, ls=(0, (3, 2)))
        ax.text(i, -0.78, minus(f"n = {d.size:,}\nbias {b:+.3f}\nSD {sd:.3f}"), fontsize=5.8, color=INK2,
                ha="center", va="bottom")
        ticks.append(fam)
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks, fontsize=6.8)
    ax.set_xlim(-0.6, len(ticks) - 0.4)
    ax.set_ylim(-0.8, 0.75)
    ax.set_ylabel("ARCOPY \u2212 SAP2000\n(units of the last printed digit)", fontsize=6.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=6.3)
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.1, label="bias"),
                       Line2D([], [], color=INK2, lw=0.9, ls=(0, (3, 2)), label="bias \u00b1 1.96 SD"),
                       Line2D([], [], color=C_SAP, lw=0.9, label="\u00b1 0.5 digit (rounding)")],
              loc="upper left", fontsize=5.6, ncol=3, bbox_to_anchor=(0.0, 1.03), columnspacing=1.0,
              handlelength=1.8)


def draw_rsa(ax):
    order = [("base shear", "base", "Base shear"), ("ux", "joint", "Displacements"), ("V", None, "Shear V"),
             ("M", None, "Moment M"), ("N", "column", "N columns"), ("N", "beam", "N beams")]
    w = 0.34
    lim = 0.5
    for k_, tag in enumerate(TAGS):
        R = pd.read_csv(os.path.join(HERE, f"pairs_rsa_{tag}.csv"))
        for i, (q, et, _) in enumerate(order):
            d = R[(R.quantity == q) & ((R.etype == et) if et else True)]
            y = i + (k_ - (len(TAGS) - 1) / 2) * w
            v = d.diff_pct.values
            inside = np.abs(v) <= lim
            if inside.any():
                ax.plot([v[inside].min(), v[inside].max()], [y, y], color=COLT[tag], lw=4.5, alpha=0.3,
                        solid_capstyle="round")
                ax.scatter(v[inside], np.full(inside.sum(), y), s=8, color=COLT[tag], lw=0, zorder=3)
            if (~inside).any():
                ax.text(lim - 0.02, y, minus(f"off scale: {v.min():+.0f}% to {v.max():+.0f}%"), fontsize=5.6,
                        color=COLT[tag], ha="right", va="center")
            elif tag == TAGS[0] or len(TAGS) == 1:
                ax.text(lim - 0.02, y, minus(f"max |e| {np.abs(v).max():.3f}%"), fontsize=5.6, color=INK2,
                        ha="right", va="center")
            else:
                ax.text(lim - 0.02, y, minus(f"mean {v.mean():+.3f}%"), fontsize=5.6, color=INK2, ha="right",
                        va="center")
    ax.axvline(0, color=AXIS, lw=1.0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o[2] for o in order], fontsize=6.5)
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("ARCOPY \u2212 SAP2000 (% of SAP2000)", fontsize=6.8)
    ax.tick_params(labelsize=6.3)
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=COLT[t], marker="o", ls="none", ms=3.5, label=LAB[t]) for t in TAGS],
              loc="upper center", fontsize=5.8, bbox_to_anchor=(0.4, -0.2), ncol=1)


# ---------------------------------------------------------------- paper figure
fig = plt.figure(figsize=(7.2, 6.6))
ax = fig.add_axes([0.0, 0.47, 0.52, 0.50], projection="3d")
draw_waterfall(ax)
fig.text(0.01, 0.965, "(a) Bending moments of beams 7 and 8", fontsize=7.8, color=INK)
fig.text(0.04, 0.943, "ARCOPY: lines; SAP2000: circles; label: ARCOPY / SAP2000 (kNm)", fontsize=6.2, color=INK2)
if TAGS:
    ax = fig.add_axes([0.50, 0.47, 0.50, 0.50], projection="3d")
    info = draw_modes(ax, TAGS[0])
    fig.text(0.53, 0.448, "\n".join(info), fontsize=6.2, color=INK, va="bottom", linespacing=1.5)
    fig.text(0.53, 0.965, "(b) Modes with cracked stiffness", fontsize=7.8, color=INK)
    fig.text(0.56, 0.943, "values: ARCOPY / SAP2000" + ("" if TAGS[0] == "same_mass"
             else " (SAP2000 mass from elements and loads)"), fontsize=6.2, color=INK2)
ax = fig.add_axes([0.085, 0.06, 0.37, 0.33])
draw_static_stats(ax)
ax.set_title(f"(c) Static analysis: all {SUM['static_pairs']:,} values", loc="left", fontsize=7.8, x=-0.2)
if TAGS:
    ax = fig.add_axes([0.62, 0.10, 0.36, 0.29])
    draw_rsa(ax)
    ax.set_title("(d) Response spectrum (CQC): all values", loc="left", fontsize=7.8, x=-0.3)
save(fig, "Fig_P_verification")

# ---------------------------------------------------------------- Fig P2
groups = [("Member forces N, V (kN)", ["N", "V"], "SAP2000 value (kN)"),
          ("Bending moments M (kNm)", ["M"], "SAP2000 value (kNm)"),
          ("Displacements, reactions", ["ux", "uy", "rz", "Rx", "Ry", "Mz"], "SAP2000 value (mm, mrad, kN, kNm)")]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 1, 0.55]}, sharey=True)
rng = np.random.default_rng(7)
for ax, (title, qs, xl) in zip(axs[:3], groups):
    d = P[P.quantity.isin(qs)]
    ax.axhspan(-0.5, 0.5, color=C_SAP, alpha=0.08, lw=0)
    ax.scatter(d.sap2000, d.diff_steps, s=3, color=C_ARC, alpha=0.35, lw=0, rasterized=True)
    b, sd = d.diff_steps.mean(), d.diff_steps.std()
    ax.axhline(b, color=INK, lw=1.0)
    for v in (b - 1.96 * sd, b + 1.96 * sd):
        ax.axhline(v, color=INK2, lw=0.9, ls=(0, (4, 2)))
    for v in (-0.5, 0.5):
        ax.axhline(v, color=C_SAP, lw=0.9)
    ax.set_title(title, loc="left", fontsize=8)
    ax.set_xlabel(xl, fontsize=7.5)
    ax.text(0.02, 0.97, minus(f"n = {len(d):,}\nbias {b:+.3f}, SD {sd:.3f}\nLoA {b - 1.96 * sd:+.2f} to "
                              f"{b + 1.96 * sd:+.2f}"),
            transform=ax.transAxes, fontsize=6.3, color=INK2, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    ax.set_ylim(-0.85, 0.85)
axs[0].set_ylabel("ARCOPY − SAP2000\n(units of the last printed digit)")
ax = axs[3]
bins = np.linspace(-0.5, 0.5, 21)
ax.hist(P.diff_steps, bins=bins, orientation="horizontal", density=True, color=C_ARC, alpha=0.55,
        edgecolor="white", linewidth=0.5)
ax.plot([1.0, 1.0], [-0.5, 0.5], color=C_SAP, lw=1.2)
ax.text(1.05, 0.58, "uniform\n(rounding)", fontsize=6.3, color=INK2, ha="center")
ax.set_xlabel("density", fontsize=7.5)
ax.set_title("All", loc="left", fontsize=8)
ax.text(0.05, 0.03, f"SD {P.diff_steps.std():.3f}\n(rounding: 0.289)", transform=ax.transAxes, fontsize=6.3,
        color=INK2)
fig.legend(handles=[Line2D([], [], color=INK, lw=1.0, label="bias"),
                    Line2D([], [], color=INK2, lw=0.9, ls=(0, (4, 2)), label="95% limits of agreement"),
                    Line2D([], [], color=C_SAP, lw=0.9, label="± half of the last printed digit")],
           loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), fontsize=7)
fig.tight_layout(w_pad=0.6)
save(fig, "Fig_P2_static_agreement")

print("figures in", OUT)
