"""
Common visual style for all ARCOPY figures (tuned for the Spyder Plots pane).

apply_style() is called once by main.py. The drawing helpers below are used
by every frame-type plot so that structure, supports, walls and colours look
the same in all figures.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D


# ============================================================
# PALETTE
# ============================================================
COLORS = {
    "column": "#1D3557",
    "beam": "#457B9D",
    "brace": "#8D99AE",
    "wall_fill": "#E9C46A",
    "wall_edge": "#B08A2E",
    "node": "#1D1D1D",
    "support": "#3A3A3A",
    "ground": "#6B6B6B",
    "load": "#C1121F",
    "load_h": "#D17A00",
    "undeformed": "#B8BEC6",
    "deformed": "#E76F51",
    "text": "#222222",
    "muted": "#6C757D",
    "grid": "#E3E6EA",
    "pos": "#2A6F97",   # positive diagram ordinates
    "neg": "#C44536",   # negative diagram ordinates
    "N": "#2A6F97",
    "V": "#2A9D8F",
    "M": "#7B2CBF",
    "highlight": "#E63946",
    "ok": "#2A9D8F",
}

SERIES = ["#1D3557", "#E76F51", "#2A9D8F", "#7B2CBF", "#F4A261",
          "#457B9D", "#C1121F", "#6A994E", "#8D99AE", "#BC6C25"]

# utilisation bands (same thresholds as constants.BAND_*)
UTIL_BANDS = [
    (0.70, "#2A9D8F", "η ≤ 0.70"),
    (0.85, "#E9C46A", "0.70 < η ≤ 0.85"),
    (1.00, "#F4A261", "0.85 < η ≤ 1.00"),
    (np.inf, "#D62828", "η > 1.00"),
]

DRIFT_BANDS = [
    (0.005, "#D8F3DC", "≤ 0.5%"),
    (0.010, "#FFF3B0", "0.5–1.0%"),
    (0.015, "#FFD8A8", "1.0–1.5%"),
    (np.inf, "#FFC9C9", "> 1.5%"),
]


def util_color(eta):
    try:
        e = float(eta)
    except Exception:
        return "#BBBBBB"
    if not np.isfinite(e):
        return "#BBBBBB"
    for lim, col, _ in UTIL_BANDS:
        if e <= lim:
            return col
    return UTIL_BANDS[-1][1]


def drift_band_color(ratio):
    r = abs(float(ratio))
    for lim, col, _ in DRIFT_BANDS:
        if r <= lim:
            return col
    return DRIFT_BANDS[-1][1]


# ============================================================
# GLOBAL STYLE
# ============================================================
def apply_style():
    mpl.rcParams.update({
        "figure.figsize": (10.0, 6.5),
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Calibri", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10.5,
        "axes.labelcolor": COLORS["text"],
        "axes.edgecolor": "#9AA0A6",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "text.color": COLORS["text"],
        "figure.titlesize": 14,
        "figure.titleweight": "semibold",
        "figure.max_open_warning": 0,
    })
    # Spyder inline backend: keep crisp figures if available
    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats("retina")
    except Exception:
        pass


def subtitle(ax, text):
    """Small grey line under the axes title (call after set_title)."""
    t_left = ax.get_title(loc="left")
    t_center = ax.get_title(loc="center")
    if t_left:
        ax.set_title(t_left, loc="left", pad=20)
    elif t_center:
        ax.set_title("", loc="center")
        ax.set_title(t_center, loc="left", pad=20)
    ax.text(0.0, 1.005, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color=COLORS["muted"])


# ============================================================
# FRAME DRAWING HELPERS
# ============================================================
def frame_extent(nodes):
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    Lref = max(xmax - xmin, ymax - ymin, 1.0)
    return xmin, xmax, ymin, ymax, Lref


def draw_supports(ax, nodes, size=None, zorder=6):
    """Fixed supports (hatched ground block) at all base nodes (y = 0)."""
    xmin, xmax, ymin, ymax, Lref = frame_extent(nodes)
    s = size if size is not None else 0.025 * Lref
    base = [p for p in nodes.values() if abs(p[1] - ymin) < 1e-9]
    for (x, y) in base:
        ax.add_patch(Rectangle((x - 1.4 * s, y - 0.55 * s), 2.8 * s, 0.55 * s,
                               fc=COLORS["support"], ec="none", zorder=zorder))
        for k in range(6):
            x0 = x - 1.4 * s + k * 0.56 * s
            ax.plot([x0, x0 - 0.45 * s], [y - 0.55 * s, y - 1.0 * s],
                    color=COLORS["ground"], lw=0.8, zorder=zorder)
    if base:
        ax.plot([xmin - 0.06 * Lref, xmax + 0.06 * Lref], [ymin - 1.0 * s] * 2,
                color=COLORS["ground"], lw=0.6, alpha=0.6, zorder=1)


def draw_walls(ax, nodes, geom, walls, *, highlight=False, zorder=1, label=True):
    if not walls or geom is None:
        return
    from framekit.geometry import node_id_at
    fc = COLORS["wall_fill"]
    ec = COLORS["highlight"] if highlight else COLORS["wall_edge"]
    for w in walls:
        ids = [node_id_at(geom, w.floor_bot, w.bay_left - 1),
               node_id_at(geom, w.floor_bot, w.bay_right - 1),
               node_id_at(geom, w.floor_top, w.bay_right - 1),
               node_id_at(geom, w.floor_top, w.bay_left - 1)]
        pts = [nodes[n] for n in ids]
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.2,
                             alpha=0.45, hatch="////", zorder=zorder))
        if label:
            cx = np.mean([p[0] for p in pts]); cy = np.mean([p[1] for p in pts])
            ax.text(cx, cy, "WALL", ha="center", va="center", fontsize=8,
                    color=COLORS["wall_edge"], fontweight="bold", zorder=zorder + 1,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))


def draw_frame(ax, nodes, elements, *, lw_column=3.2, lw_beam=2.6, alpha=1.0,
               color=None, zorder=3, element_colors=None, element_lw=None):
    """Members as solid lines; columns/beams in different colours."""
    for eid, (ni, nj, etype) in elements.items():
        x1, y1 = nodes[ni]
        x2, y2 = nodes[nj]
        if element_colors is not None and eid in element_colors:
            col = element_colors[eid]
        elif color is not None:
            col = color
        else:
            col = COLORS["column"] if etype == "column" else COLORS["beam"]
        lw = (element_lw or {}).get(eid, lw_column if etype == "column" else lw_beam)
        ax.plot([x1, x2], [y1, y2], color=col, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=zorder)


def draw_nodes(ax, nodes, *, size=4.0, zorder=7):
    xy = np.array(list(nodes.values()), dtype=float)
    ax.plot(xy[:, 0], xy[:, 1], "o", ms=size, mfc="white",
            mec=COLORS["node"], mew=1.1, zorder=zorder, ls="none")


def label_elements(ax, nodes, elements, *, fontsize=7.5, zorder=8):
    for eid, (ni, nj, etype) in elements.items():
        x1, y1 = nodes[ni]
        x2, y2 = nodes[nj]
        col = COLORS["column"] if etype == "column" else COLORS["beam"]
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(eid), fontsize=fontsize,
                ha="center", va="center", color=col, zorder=zorder,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=col, lw=0.8))


def label_nodes(ax, nodes, *, fontsize=7, zorder=8):
    _, _, _, _, Lref = frame_extent(nodes)
    d = 0.012 * Lref
    for nid, (x, y) in nodes.items():
        ax.text(x - d, y + d, str(nid), fontsize=fontsize, ha="right", va="bottom",
                color=COLORS["muted"], zorder=zorder)


def draw_dimensions(ax, nodes, geom=None, *, fontsize=8, side="left"):
    """Bay lengths below the frame and storey heights on the left."""
    xmin, xmax, ymin, ymax, Lref = frame_extent(nodes)
    xs = sorted({round(p[0], 9) for p in nodes.values()})
    ys = sorted({round(p[1], 9) for p in nodes.values()})
    yd = ymin - 0.09 * Lref
    xd = xmin - 0.07 * Lref if side == "left" else xmax + 0.07 * Lref
    kw = dict(color=COLORS["muted"], lw=0.7)
    for a, b in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(a, yd), xytext=(b, yd), arrowprops=dict(arrowstyle="<->", **kw))
        ax.text((a + b) / 2, yd - 0.012 * Lref, f"{b - a:.2f} m", ha="center", va="top",
                fontsize=fontsize, color=COLORS["muted"])
    for a, b in zip(ys[:-1], ys[1:]):
        ax.annotate("", xy=(xd, a), xytext=(xd, b), arrowprops=dict(arrowstyle="<->", **kw))
        ax.text(xd - 0.012 * Lref if side == "left" else xd + 0.012 * Lref, (a + b) / 2,
                f"{b - a:.2f} m", ha="right" if side == "left" else "left", va="center",
                fontsize=fontsize, color=COLORS["muted"], rotation=90)


def finish_frame_axes(ax, nodes, *, pad=0.12, pad_right=None, pad_left=None,
                      pad_bottom=None, pad_top=None, show_axes=False):
    xmin, xmax, ymin, ymax, Lref = frame_extent(nodes)
    pl = pad if pad_left is None else pad_left
    pr = pad if pad_right is None else pad_right
    pb = pad if pad_bottom is None else pad_bottom
    pt = pad if pad_top is None else pad_top
    ax.set_xlim(xmin - pl * Lref, xmax + pr * Lref)
    ax.set_ylim(ymin - pb * Lref, ymax + pt * Lref)
    ax.set_aspect("equal", adjustable="box")
    if not show_axes:
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")


def frame_legend(ax, *, walls=False, extra=None, loc="upper left", ncol=1):
    h = [Line2D([], [], color=COLORS["column"], lw=3.2, label="Column"),
         Line2D([], [], color=COLORS["beam"], lw=2.6, label="Beam")]
    if walls:
        from matplotlib.patches import Patch
        h.append(Patch(fc=COLORS["wall_fill"], ec=COLORS["wall_edge"], alpha=0.5,
                       hatch="////", label="Shear wall"))
    if extra:
        h += list(extra)
    ax.legend(handles=h, loc=loc, ncol=ncol, bbox_to_anchor=(1.0, 1.0) if loc == "upper left" else None)


def band_legend_handles(bands, title_fmt="{}"):
    from matplotlib.patches import Patch
    return [Patch(fc=c, ec="#777777", lw=0.5, label=title_fmt.format(lbl)) for _, c, lbl in bands]


# ============================================================
# UTILISATION MAP (shared by EC2 checks and retrofit plots)
# ============================================================
def plot_utilization_frame(nodes, elements, eta_map, title="Member utilization", *,
                           geom=None, walls=None, highlight_ids=None, info=None,
                           value_label="η"):
    """Members coloured by utilisation band with η labels."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(11, 7.5))
    draw_walls(ax, nodes, geom, walls, label=False)
    _, _, _, _, Lref = frame_extent(nodes)
    highlight_ids = set(int(e) for e in (highlight_ids or []))

    for eid, (ni, nj, etype) in elements.items():
        x1, y1 = nodes[ni]
        x2, y2 = nodes[nj]
        eta = eta_map.get(eid, eta_map.get(int(eid), np.nan))
        col = util_color(eta)
        if int(eid) in highlight_ids:
            ax.plot([x1, x2], [y1, y2], color="#222222", lw=10, alpha=0.18,
                    solid_capstyle="butt", zorder=2)
        ax.plot([x1, x2], [y1, y2], color=col, lw=6.5, solid_capstyle="butt", zorder=3)
        txt = f"{eid}\n{value_label}={eta:.2f}" if np.isfinite(eta) else f"{eid}\n–"
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        off = 0.0 if etype == "column" else 0.0
        ax.text(xm + off, ym, txt, ha="center", va="center", fontsize=7.5, zorder=6,
                fontweight="bold" if (np.isfinite(eta) and eta > 1.0) else "normal",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=col, lw=1.0, alpha=0.95))

    draw_nodes(ax, nodes, size=3.5)
    draw_supports(ax, nodes)
    finish_frame_axes(ax, nodes, pad_left=0.06, pad_right=0.06, pad_bottom=0.08, pad_top=0.06)

    etas = np.array([v for v in eta_map.values() if v is not None and np.isfinite(v)], dtype=float)
    handles = band_legend_handles(UTIL_BANDS)
    if highlight_ids:
        handles.append(Line2D([], [], color="#222222", lw=8, alpha=0.18, label="Retrofitted member"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), fontsize=8.5)
    ax.set_title(title)
    # summary printed in the console (a long text line above the frame would
    # shrink the drawing)
    if etas.size:
        n_fail = int(np.sum(etas > 1.0))
        print(f"{title}: max {value_label} = {etas.max():.2f}, members with {value_label} > 1: "
              f"{n_fail} of {etas.size}")
        if info:
            print(f"  ({info})")
    plt.show()
    return fig, ax


def plot_table(df_rows, headers, title, *, eta_cols=(), max_rows=40, col_widths=None):
    """Clean table figure; columns listed in eta_cols are coloured by band."""
    rows = list(df_rows)[:max_rows]
    n = len(rows)
    fig_h = 0.9 + 0.32 * (n + 1)
    fig, ax = plt.subplots(figsize=(min(1.25 * len(headers) + 2, 15), fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, loc="upper center", cellLoc="center",
                   colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.25)
    eta_idx = [headers.index(c) for c in eta_cols if c in headers]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(1.2)
        if r == 0:
            cell.set_facecolor(COLORS["column"])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F4F6F8" if r % 2 else "white")
            if c in eta_idx:
                try:
                    v = float(rows[r - 1][c])
                    cell.set_facecolor(util_color(v))
                    if v > 1.0:
                        cell.set_text_props(weight="bold")
                except Exception:
                    pass
    ax.set_title(title)
    plt.show()
    return fig, ax


# ============================================================
# AUTOSAVE: every figure shown with plt.show() is also saved as PNG
# ============================================================
_ORIG_SHOW = None
_AUTOSAVE = {"folder": None, "n": 0}


def enable_figure_autosave(folder):
    """Save every figure displayed with plt.show() into `folder` (PNG, 200 dpi)."""
    global _ORIG_SHOW
    import os
    os.makedirs(folder, exist_ok=True)
    _AUTOSAVE["folder"] = folder
    _AUTOSAVE["n"] = len([f for f in os.listdir(folder) if f.lower().endswith(".png")])
    if _ORIG_SHOW is None:
        _ORIG_SHOW = plt.show

    def _show(*args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            if getattr(fig, "_arcopy_saved", False):
                continue
            title = ""
            try:
                if fig._suptitle is not None:
                    title = fig._suptitle.get_text()
                if not title and fig.axes:
                    ax = fig.axes[0]
                    title = ax.get_title(loc="left") or ax.get_title()
            except Exception:
                pass
            slug = "".join(ch if ch.isalnum() else "_" for ch in title)[:60].strip("_") or "figure"
            _AUTOSAVE["n"] += 1
            try:
                fig.savefig(os.path.join(_AUTOSAVE["folder"], f"{_AUTOSAVE['n']:03d}_{slug}.png"),
                            dpi=200, bbox_inches="tight", facecolor="white")
            except Exception as exc:
                print(f"⚠ figure not saved: {exc}")
            fig._arcopy_saved = True
        return _ORIG_SHOW(*args, **kwargs)

    plt.show = _show
