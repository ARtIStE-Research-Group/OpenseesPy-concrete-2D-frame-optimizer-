"""Draws model.png: geometry, numbering and load cases of the test RC portal."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X, Y = [0.0, 5.0, 10.0], [0.0, 4.0, 8.0]
INK, MUTED, BLUE, RED, GREEN = "#1A202C", "#718096", "#2A6F97", "#C44536", "#2F855A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def frame(ax):
    for x in X:
        ax.plot([x, x], [0, 8], color=INK, lw=2.4, zorder=2)
    for y in Y[1:]:
        ax.plot([0, 10], [y, y], color=INK, lw=2.4, zorder=2)
    for x in X:
        ax.add_patch(plt.Rectangle((x - 0.45, -0.28), 0.9, 0.28, color="#4A5568", zorder=1))
        for k in range(6):
            ax.plot([x - 0.45 + 0.18 * k, x - 0.27 + 0.18 * k], [-0.28, -0.55], color="#4A5568", lw=0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def tag(ax, x, y, text, fc="white", ec="#A0AEC0", color=INK, size=9):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=ec, lw=0.8))


def arrow(ax, x0, y0, x1, y1, color, lw=1.2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=color, lw=lw,
                                 zorder=4))


fig = plt.figure(figsize=(13.5, 7.0))
a = fig.add_axes([0.01, 0.17, 0.47, 0.75])
b = fig.add_axes([0.51, 0.17, 0.48, 0.75])

# (a) geometry and numbering
frame(a)
k = 1
for y in Y:
    for x in X:
        a.plot(x, y, "o", ms=6, mfc="white", mec=INK, zorder=5)
        a.text(x + 0.18, y + 0.18, str(k), fontsize=8, color=MUTED, zorder=5)
        k += 1
cols = {1: (0, 2), 2: (5, 2), 3: (10, 2), 4: (0, 6), 5: (5, 6), 6: (10, 6)}
for e, (x, y) in cols.items():
    tag(a, x, y, f"{e}")
beams = {7: (2.5, 4, "Beam 1"), 8: (7.5, 4, "Beam 2"), 9: (2.5, 8, "Beam 3"), 10: (7.5, 8, "Beam 4")}
for e, (x, y, nm) in beams.items():
    tag(a, x, y, f"{e}")
    a.text(x, y - 0.55, nm, ha="center", fontsize=8.5, color=MUTED)
for xa, xb in ((0, 5), (5, 10)):
    a.annotate("", (xa, -1.05), (xb, -1.05), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
    a.text((xa + xb) / 2, -1.35, "5.00 m", ha="center", va="top", fontsize=9, color=MUTED)
for ya, yb in ((0, 4), (4, 8)):
    a.annotate("", (-1.1, ya), (-1.1, yb), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
    a.text(-1.3, (ya + yb) / 2, "4.00 m", ha="right", va="center", fontsize=9, color=MUTED, rotation=90)
a.set_xlim(-2.2, 11.6)
a.set_ylim(-1.9, 9.3)
a.set_title("(a) Geometry and numbering", loc="left", fontsize=12, color=INK)
fig.text(0.02, 0.025, "Columns 1–6: 300 × 300, 8Φ20, Φ8/150 (3 legs)\nBeams 7–10: b 500 × h 300, 4+4Φ20, "
         "Φ6/150 (4 legs)\nC25/30, B450C, E = 30 GPa; node numbers in grey, element IDs in boxes",
         fontsize=9, color=MUTED, va="bottom")

# (b) load cases
frame(b)
for (xa, xb, y, g, q, lc) in ((0, 5, 4, 24, 8, "live01"), (5, 10, 4, 24, 8, "live02"),
                               (0, 5, 8, 20, 6, "live03"), (5, 10, 8, 20, 6, "live04")):
    for xx in [xa + 0.5 * i for i in range(11)]:
        arrow(b, xx, y + 0.7, xx, y + 0.06, BLUE, lw=0.9)
    b.plot([xa, xb], [y + 0.7, y + 0.7], color=BLUE, lw=0.9)
    b.text((xa + xb) / 2, y + 0.85, f"G = {g} kN/m\n{lc} = {q} kN/m", ha="center", va="bottom", fontsize=8.8,
           color=BLUE, linespacing=1.25)
for (y0, y1, w) in ((0.0, 4.0, 1.4), (4.0, 8.0, 2.8)):   # W+: left exterior columns, per storey
    L = 0.9 * w / 2.8
    ys = [y0 + 0.4 + 0.8 * i for i in range(5)]
    for yy in ys:
        arrow(b, -L - 0.1, yy, -0.08, yy, GREEN, lw=0.9)
    b.plot([-L - 0.1] * 2, [ys[0], ys[-1]], color=GREEN, lw=0.9)
    b.text(-1.15, (y0 + y1) / 2, f"{w} kN/m", ha="right", va="center", fontsize=8.8, color=GREEN)
b.text(-1.15, 8.75, "W+ (wind_01)", ha="right", va="center", fontsize=9, color=GREEN)
b.text(10.35, 7.6, "W− (wind_02):\nmirrored, on the\nright columns,\nto the left\n"
       "storey 2: 2.8 kN/m\nstorey 1: 1.4 kN/m", fontsize=8.5, color=GREEN, va="top")
b.set_xlim(-3.4, 13.4)
b.set_ylim(-1.0, 10.2)
b.set_title("(b) Load cases", loc="left", fontsize=12, color=INK)
fig.text(0.52, 0.025, "Self-weight SW from the sections (2500 kg/m³)\nCROWD = Range Add (live01–04);  "
         "CROWD_SEISM = Linear Add (live01–04)\nWIND = Envelope (W+; W−);  seismic mass = SW + G + 0.3 × CROWD_SEISM",
         fontsize=9, color=MUTED, va="bottom")
fig.savefig(os.path.join(HERE, "model.png"), dpi=170, facecolor="white")
print("model.png written")
