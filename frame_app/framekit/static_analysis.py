import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openseespy.opensees as ops

from framekit.geometry import node_id_at


# ============================================================
# NODE DISPLACEMENTS
# ============================================================
def get_node_disps(nodes):
    U = {}
    for nid in nodes:
        try:
            rz = ops.nodeDisp(nid, 3)
        except Exception:
            rz = 0.0

        U[nid] = (
            float(ops.nodeDisp(nid, 1)),
            float(ops.nodeDisp(nid, 2)),
            float(rz)
        )
    return U


# ============================================================
# STORY DRIFTS
# ============================================================
def compute_story_drifts(geom, nodes, U):
    nfp = geom.num_bays + 1
    drifts = []

    for st in range(1, geom.num_floors + 1):
        story_pairs = []
        story_drifts_ratio = []

        for ix in range(nfp):
            n_top = node_id_at(geom, st, ix)
            n_bot = node_id_at(geom, st - 1, ix)

            u_bot = float(U[n_bot][0])
            u_top = float(U[n_top][0])

            du = u_top - u_bot
            drift_abs = abs(du)
            drift_ratio = drift_abs / float(geom.floor_height)
            cumulative_disp_abs = abs(u_top)

            story_pairs.append({
                "story": st,
                "grid_x": ix,
                "node_bot": n_bot,
                "node_top": n_top,
                "u_bot_m": u_bot,
                "u_top_m": u_top,
                "delta_u_m": du,
                "drift_abs_m": drift_abs,
                "drift_ratio": drift_ratio,
                "cum_disp_top_m": cumulative_disp_abs
            })
            story_drifts_ratio.append(drift_ratio)

        idx_max = int(np.argmax(story_drifts_ratio)) if story_drifts_ratio else 0
        ctrl = story_pairs[idx_max]

        drifts.append({
            "story": st,
            "max_drift_m": ctrl["drift_abs_m"],
            "max_drift_ratio": ctrl["drift_ratio"],
            "max_cum_disp_m": ctrl["cum_disp_top_m"],
            "control_grid_x": ctrl["grid_x"],
            "control_node_bot": ctrl["node_bot"],
            "control_node_top": ctrl["node_top"],
            "all_positions": story_pairs
        })

    return drifts


# ============================================================
# DRIFT TABLES
# ============================================================
def drifts_to_dataframe(drifts, combo_name=None):
    rows = []
    for d in drifts:
        rows.append({
            "Case": combo_name if combo_name is not None else "",
            "Combination": combo_name if combo_name is not None else "",
            "Story": d["story"],
            "MaxDrift_m": d["max_drift_m"],
            "MaxDrift_mm": 1000.0 * d["max_drift_m"],
            "MaxDriftRatio": d["max_drift_ratio"],
            "MaxDriftPercent": 100.0 * d["max_drift_ratio"],
            "MaxCumDisp_m": d.get("max_cum_disp_m", np.nan),
            "MaxCumDisp_mm": 1000.0 * d.get("max_cum_disp_m", np.nan),
            "ControlGridX": d["control_grid_x"],
            "ControlBottomNode": d["control_node_bot"],
            "ControlTopNode": d["control_node_top"]
        })
    return pd.DataFrame(rows)


# ============================================================
# DRIFT COLOR BANDS
# ============================================================
def drift_color(drift_ratio):
    r = abs(drift_ratio)
    if r <= 0.005:
        return (0.75, 0.95, 0.75, 1.0)
    if r <= 0.010:
        return (1.00, 1.00, 0.70, 1.0)
    if r <= 0.015:
        return (1.00, 0.85, 0.60, 1.0)
    return (1.00, 0.70, 0.70, 1.0)


# ============================================================
# CURVED DEFORMED SHAPE + DRIFTS + CUMULATIVE DISPLACEMENTS
# ============================================================
def plot_deformed_shape(
    geom,
    nodes,
    elements,
    scale=None,
    npts=25,
    title="Deformed Shape + Drifts",
    show_story_drift=True,
    show_cumulative_disp=True,
    walls=None,
):
    from framekit.plot_style import (
        COLORS, DRIFT_BANDS, drift_band_color, draw_supports, draw_walls,
        finish_frame_axes, subtitle, band_legend_handles,
    )
    from matplotlib.lines import Line2D

    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    Lref = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)

    U = get_node_disps(nodes)
    umax = max([abs(u) for nid in U for u in (U[nid][0], U[nid][1])] + [1e-12])

    if scale is None:
        scale = 0.06 * Lref / umax

    drifts = compute_story_drifts(geom, nodes, U)

    fig, ax = plt.subplots(figsize=(11, 7.5))

    x_min, x_max = min(xs), max(xs)

    # Storey drift bands
    for d in drifts:
        st = d["story"]
        y0 = (st - 1) * geom.floor_height
        ax.add_patch(plt.Rectangle((x_min, y0), x_max - x_min, geom.floor_height,
                                   fc=drift_band_color(d["max_drift_ratio"]), ec="white",
                                   lw=1.5, alpha=0.75, zorder=0))

    draw_walls(ax, nodes, geom, walls, label=False)

    # Undeformed structure
    for eid, (ni, nj, _) in elements.items():
        x1, y1 = nodes[ni]
        x2, y2 = nodes[nj]
        ax.plot([x1, x2], [y1, y2], color=COLORS["undeformed"], lw=1.2, ls=(0, (4, 3)), zorder=2)

    # Curved deformed members using beam shape functions
    for eid, (ni, nj, etype) in elements.items():
        x1, y1 = nodes[ni]
        x2, y2 = nodes[nj]
        u1, v1, th1 = U[ni]
        u2, v2, th2 = U[nj]
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1e-12:
            continue
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        ui, vi = c * u1 + s * v1, -s * u1 + c * v1
        uj, vj = c * u2 + s * v2, -s * u2 + c * v2

        xi = np.linspace(0.0, 1.0, npts)
        xloc = xi * L
        u_ax = (1.0 - xi) * ui + xi * uj
        N1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
        N2 = L * (xi - 2.0 * xi**2 + xi**3)
        N3 = 3.0 * xi**2 - 2.0 * xi**3
        N4 = L * (-xi**2 + xi**3)
        v_tr = N1 * vi + N2 * th1 + N3 * vj + N4 * th2
        xg = x1 + c * xloc
        yg = y1 + s * xloc
        ax.plot(xg + scale * (c * u_ax - s * v_tr), yg + scale * (s * u_ax + c * v_tr),
                color=COLORS["deformed"], lw=2.6 if etype == "column" else 2.2, zorder=4,
                solid_capstyle="round")

    # deformed nodes
    for nid, (x, y) in nodes.items():
        ax.plot(x + scale * U[nid][0], y + scale * U[nid][1], "o", ms=3.5, mfc="white",
                mec=COLORS["deformed"], mew=1.0, zorder=5)

    draw_supports(ax, nodes)

    # Storey labels on the right
    x_text = x_max + 0.09 * Lref
    for d in drifts:
        st = d["story"]
        y_top = st * geom.floor_height
        drift_mm = 1000.0 * d["max_drift_m"]
        cum_mm = 1000.0 * d.get("max_cum_disp_m", 0.0)
        parts = []
        if show_story_drift:
            parts.append(f"Δ = {drift_mm:.2f} mm  ({100 * d['max_drift_ratio']:.3f}%)")
        if show_cumulative_disp:
            parts.append(f"u = {cum_mm:.2f} mm")
        ax.annotate("\n".join(parts), xy=(x_max + scale * U[max(nodes, key=lambda n: (abs(nodes[n][1] - y_top) < 1e-9, nodes[n][0]))][0], y_top),
                    xytext=(x_text, y_top), va="center", ha="left", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", color=COLORS["muted"], lw=0.6),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))
        ax.text(x_min - 0.03 * Lref, y_top - 0.5 * geom.floor_height, f"S{st}", ha="right", va="center",
                fontsize=9, color=COLORS["muted"], fontweight="bold")

    finish_frame_axes(ax, nodes, pad_left=0.08, pad_right=0.38, pad_bottom=0.08, pad_top=0.08)

    handles = [Line2D([], [], color=COLORS["undeformed"], lw=1.2, ls=(0, (4, 3)), label="Undeformed"),
               Line2D([], [], color=COLORS["deformed"], lw=2.4, label="Deformed")]
    handles += band_legend_handles(DRIFT_BANDS, "Drift {}")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=8.5)

    ax.set_title(title)
    subtitle(ax, f"Displacements magnified × {scale:,.0f}   ·   max |u| = {1000 * umax:.2f} mm")
    plt.show()
