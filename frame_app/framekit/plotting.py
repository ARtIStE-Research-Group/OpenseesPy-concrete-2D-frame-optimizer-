import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from framekit.plot_style import (
    COLORS, draw_frame, draw_supports, draw_walls, draw_nodes, label_elements,
    label_nodes, draw_dimensions, finish_frame_axes, frame_extent, subtitle,
)


def _arrow(ax, tail, tip, color, lw=1.1, ms=9, zorder=9):
    ax.annotate("", xy=tip, xytext=tail, zorder=zorder,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=ms, shrinkA=0, shrinkB=0))


def _tag(ax, x, y, text, color, ha="center", va="bottom", fontsize=7.5, zorder=10):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color, zorder=zorder,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, lw=0.6, alpha=0.92))


def _is_horizontal_direction(direction):
    return str(direction).strip().lower() not in {"vertical", "y", "global_y", "vertical_y"}


def plot_structure(
    nodes,
    elements,
    geom=None,
    walls=None,
    loads=None,
    title="Structure",
    *,
    load_arrow_scale=0.07,
    load_head_width_scale=0.005,
    load_head_length_scale=0.008,
    load_offset_scale=0.0,
    n_load_arrows=5,
    load_label_fontsize=7.5,
    stick_horizontal_loads_to_columns=True,
    show_ids=None,
    show_dimensions=True,
):
    """Plot frame geometry, supports, shear walls and (optionally) the loads
    of one load case.

    Distributed loads are drawn as load blocks whose height is proportional
    to the intensity (max height = load_arrow_scale * frame size); arrows
    point in the true load direction and always end on the loaded member.
    """
    xmin, xmax, ymin, ymax, Lref = frame_extent(nodes)
    has_loads = bool(loads)
    if show_ids is None:
        show_ids = True

    fig, ax = plt.subplots(figsize=(11, 7.5))

    draw_walls(ax, nodes, geom, walls)
    draw_frame(ax, nodes, elements, alpha=(0.9 if has_loads else 1.0))
    draw_supports(ax, nodes)
    draw_nodes(ax, nodes)
    if show_ids:
        label_elements(ax, nodes, elements, fontsize=(6.8 if has_loads else 7.5))
        if not has_loads:
            label_nodes(ax, nodes)
    if show_dimensions:
        left_h = any(
            _is_horizontal_direction(getattr(L, "direction", "")) and getattr(L, "load_type", "") != "point"
            and getattr(L, "element", None) in elements
            and abs(nodes[elements[L.element][0]][0] - xmin) < 1e-9
            for L in (loads or [])
        )
        draw_dimensions(ax, nodes, geom, side=("right" if left_h else "left"))

    pad_l = pad_r = 0.14
    pad_t = 0.10
    legend_extra = []

    if has_loads:
        vals = []
        for L in loads:
            if getattr(L, "load_type", None) == "triangular":
                vals += [abs(L.w1 or 0.0), abs(L.w2 or 0.0)]
            elif getattr(L, "load_type", None) == "uniform":
                vals.append(abs(getattr(L, "value", 0.0)))
        wmax = max(vals) if vals else 1.0
        wmax = max(wmax, 1e-9)
        pvals = [abs(getattr(L, "value", 0.0)) for L in loads if getattr(L, "load_type", None) == "point"]
        pmax = max(pvals) if pvals else 1.0
        pmax = max(pmax, 1e-9)
        hmax = float(load_arrow_scale) * Lref
        n_arr = max(2, int(n_load_arrows))

        def h_of(w):
            return hmax * abs(w) / wmax

        # Combine several loads acting on the same element/direction so that
        # each member shows one block with the resultant intensity.
        agg = {}
        points = []
        for L in loads:
            lt = getattr(L, "load_type", None)
            d = "h" if _is_horizontal_direction(getattr(L, "direction", "")) else "v"
            if lt == "point" and getattr(L, "node", None) is not None:
                points.append(L)
                continue
            if getattr(L, "element", None) is None or L.element not in elements:
                continue
            if lt == "uniform":
                w1 = w2 = float(L.value)
            elif lt == "triangular":
                w1, w2 = float(L.w1 or 0.0), float(L.w2 or 0.0)
            else:
                continue
            a = agg.setdefault((L.element, d), [0.0, 0.0])
            a[0] += w1
            a[1] += w2

        used_v = used_h = False
        for (eid, d), (w1, w2) in agg.items():
            if abs(w1) < 1e-12 and abs(w2) < 1e-12:
                continue
            ni, nj, etype = elements[eid]
            x1, y1 = nodes[ni]
            x2, y2 = nodes[nj]
            Le = np.hypot(x2 - x1, y2 - y1)
            if Le < 1e-12:
                continue
            col = COLORS["load"] if d == "v" else COLORS["load_h"]
            ts = np.linspace(0.04, 0.96, n_arr)
            tt = np.linspace(0.0, 1.0, 30)
            ww = (1 - tt) * w1 + tt * w2
            uniform = abs(w1 - w2) < 1e-9
            label = f"{w1:.2f} kN/m" if uniform else f"{w1:.2f} → {w2:.2f} kN/m"
            member_horizontal = abs(y2 - y1) < 1e-9 * max(Le, 1.0)

            if d == "v" and member_horizontal:
                used_v = True
                xx = x1 + tt * (x2 - x1)
                top = y1 + np.array([h_of(w) for w in ww])
                ax.add_patch(Polygon(np.column_stack([np.r_[xx, xx[::-1]], np.r_[np.full_like(xx, y1), top[::-1]]]),
                                     closed=True, fc=col, ec="none", alpha=0.10, zorder=4))
                ax.plot(xx, top, color=col, lw=0.9, zorder=5)
                for t in ts:
                    w = (1 - t) * w1 + t * w2
                    x = x1 + t * (x2 - x1)
                    h = h_of(w)
                    if h < 1e-9:
                        continue
                    if w < 0:   # downward: from top of block to the beam
                        _arrow(ax, (x, y1 + h), (x, y1), col)
                    else:       # upward: from the beam upwards
                        _arrow(ax, (x, y1), (x, y1 + h), col)
                _tag(ax, 0.5 * (x1 + x2), y1 + max(h_of(w1), h_of(w2)) + 0.008 * Lref, label, col,
                     fontsize=load_label_fontsize)
                pad_t = max(pad_t, (y1 + hmax - ymax) / Lref + 0.08)

            elif d == "h" and not member_horizontal:
                used_h = True
                yy = y1 + tt * (y2 - y1)
                sgn_ref = np.sign(w1 + w2) if abs(w1 + w2) > 1e-12 else 1.0
                # block on the windward side of the column
                xs_out = x1 - sgn_ref * np.array([h_of(w) for w in ww])
                ax.add_patch(Polygon(np.column_stack([np.r_[np.full_like(yy, x1), xs_out[::-1]], np.r_[yy, yy[::-1]]]),
                                     closed=True, fc=col, ec="none", alpha=0.12, zorder=4))
                ax.plot(xs_out, yy, color=col, lw=0.9, zorder=5)
                for t in ts:
                    w = (1 - t) * w1 + t * w2
                    y = y1 + t * (y2 - y1)
                    h = h_of(w)
                    if h < 1e-9:
                        continue
                    s = np.sign(w) if abs(w) > 1e-12 else 1.0
                    _arrow(ax, (x1 - s * h, y), (x1, y), col)
                xl = x1 - sgn_ref * (max(h_of(w1), h_of(w2)) + 0.01 * Lref)
                _tag(ax, xl, 0.5 * (y1 + y2), label, col, ha=("right" if sgn_ref > 0 else "left"),
                     va="center", fontsize=load_label_fontsize)
                if sgn_ref > 0:
                    pad_l = max(pad_l, (xmin - (x1 - hmax)) / Lref + 0.22)
                else:
                    pad_r = max(pad_r, ((x1 + hmax) - xmax) / Lref + 0.22)

            else:
                # Load parallel to the member (e.g. self-weight of a column):
                # small arrows drawn beside the member.
                if d == "v":
                    used_v = True
                else:
                    used_h = True
                c, s_ = (x2 - x1) / Le, (y2 - y1) / Le
                off = 0.022 * Lref
                px, py = s_, -c   # right-hand side of the member
                a_len = 0.035 * Lref
                for t in np.linspace(0.2, 0.8, 3):
                    w = (1 - t) * w1 + t * w2
                    bx = x1 + t * (x2 - x1) + off * px
                    by = y1 + t * (y2 - y1) + off * py
                    if d == "v":
                        dv = (0.0, -a_len if w < 0 else a_len)
                    else:
                        dv = (-a_len if w < 0 else a_len, 0.0)
                    _arrow(ax, (bx - dv[0] / 2, by - dv[1] / 2), (bx + dv[0] / 2, by + dv[1] / 2), col, lw=0.9, ms=7)
                _tag(ax, 0.5 * (x1 + x2) + 2.1 * off * px, 0.5 * (y1 + y2) + 2.1 * off * py, label, col,
                     ha="left" if px >= 0 else "right", va="center", fontsize=load_label_fontsize - 0.5)

        for L in points:
            x, y = nodes[L.node]
            val = float(getattr(L, "value", 0.0))
            if abs(val) < 1e-12:
                continue
            ln = 0.05 * Lref + hmax * abs(val) / pmax * 0.6
            s = np.sign(val)
            if _is_horizontal_direction(getattr(L, "direction", "")):
                used_h = True
                col = COLORS["load_h"]
                _arrow(ax, (x - s * ln, y), (x, y), col, lw=1.6, ms=11)
                _tag(ax, x - s * ln, y + 0.01 * Lref, f"{val:.2f} kN", col, fontsize=load_label_fontsize)
            else:
                used_v = True
                col = COLORS["load"]
                _arrow(ax, (x, y - s * ln), (x, y), col, lw=1.6, ms=11)
                _tag(ax, x + 0.01 * Lref, y - s * ln, f"{val:.2f} kN", col, ha="left", va="center",
                     fontsize=load_label_fontsize)

        if used_v:
            legend_extra.append(Line2D([], [], color=COLORS["load"], lw=1.2, marker=">", ms=5, label="Vertical load"))
        if used_h:
            legend_extra.append(Line2D([], [], color=COLORS["load_h"], lw=1.2, marker=">", ms=5, label="Horizontal load"))

    finish_frame_axes(ax, nodes, pad_left=pad_l, pad_right=pad_r, pad_top=pad_t, pad_bottom=0.16)

    handles = [Line2D([], [], color=COLORS["column"], lw=3.2, label="Column"),
               Line2D([], [], color=COLORS["beam"], lw=2.6, label="Beam")]
    if walls:
        from matplotlib.patches import Patch
        handles.append(Patch(fc=COLORS["wall_fill"], ec=COLORS["wall_edge"], alpha=0.5, hatch="////", label="Shear wall"))
    handles += legend_extra
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    ax.set_title(title)
    n_el = len(elements)
    n_col = sum(1 for e in elements.values() if e[2] == "column")
    info = f"{len(nodes)} nodes · {n_col} columns · {n_el - n_col} beams"
    if walls:
        info += f" · {len(walls)} wall panel(s)"
    subtitle(ax, info)
    plt.show()
