"""
Compact professional retrofit report (Word + Excel).

Only the selected (best) strategy is described in detail; the other
alternatives appear in one ranking table and in the comparison charts.
All N/V/M diagrams are computed station by station from OpenSees for the
final retrofitted structure and all ULS and seismic combinations
(envelopes), never from constant demand ordinates.
"""
import os
import math
import datetime as _dt
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from matplotlib.lines import Line2D

from framekit.plot_style import (
    COLORS, UTIL_BANDS, util_color, draw_frame, draw_supports, draw_nodes, draw_walls,
    label_elements, draw_dimensions, finish_frame_axes, band_legend_handles, frame_extent,
)


# ============================================================
# small helpers
# ============================================================
def _num(v, default=np.nan):
    try:
        v = float(v)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _eta_map(df):
    if df is None or getattr(df, "empty", True):
        return {}
    col_e = "Element" if "Element" in df.columns else "ElementID"
    return {int(r[col_e]): _num(r.get("eta")) for _, r in df.iterrows()}


def _fmt(v, nd=2):
    v = _num(v)
    if not np.isfinite(v):
        return "–"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{nd}f}"


class _Figs:
    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.items = []

    def save(self, fig, name, caption):
        path = os.path.join(self.folder, f"{len(self.items) + 1:02d}_{name}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        self.items.append((path, caption))
        return path


# ============================================================
# analysis helpers for the report
# ============================================================
def _final_combos(ctx, element_sections, walls, wall_tw_mm):
    """Combinations of a configuration: original + added self-weight (+ seismic)."""
    from framekit.retrofit_optimizer import _combos_with_extra_self_weight, _seismic_eval
    base_walls = {(int(w.bay_left), int(w.floor_top)) for w in (ctx["base_walls"] or [])}
    new_walls = [w for w in (walls or []) if (int(w.bay_left), int(w.floor_top)) not in base_walls]
    c = _combos_with_extra_self_weight(
        ctx["combos"], elements=ctx["elements"], column_sec=ctx["column_sec"], beam_sec=ctx["beam_sec"],
        element_sections=element_sections, concrete_density_kgm3=ctx["concrete_density"],
        nodes=ctx["nodes"], geom=ctx["geom"], new_walls=new_walls, wall_tw_mm=wall_tw_mm,
        wall_density_kgm3=ctx["wall_density"])
    names = [n for n in ctx["names"] if n in c]
    ev = None
    if ctx.get("seismic_settings"):
        ev = _seismic_eval(ctx["seismic_settings"], nodes=ctx["nodes"], elements=ctx["elements"], geom=ctx["geom"],
                           combos=c, combo_names=names, Ec_Pa=ctx["Ec_Pa"], Es_Pa=ctx["Es_Pa"],
                           column_sec=ctx["column_sec"], beam_sec=ctx["beam_sec"], walls=walls,
                           wall_tw_mm=wall_tw_mm, wall_E_Pa=ctx["wall_E_Pa"], element_sections=element_sections,
                           ops_build_model=ctx["ops_build_model"])
        if ev is not None:
            c, names = ev["combos"], ev["combo_names"]
    return c, names, ev


def _static_drift_profile(ctx, combos, names, walls, wall_tw_mm, element_sections):
    """Max storey drift ratio per storey over all combinations (static)."""
    import openseespy.opensees as ops
    from framekit.geometry import node_id_at
    geom = ctx["geom"]
    out = np.zeros(geom.num_floors)
    for cn in names:
        if str(cn).startswith("SEIS"):
            continue
        ctx["ops_build_model"](ctx["nodes"], ctx["elements"], ctx["Ec_Pa"], ctx["Es_Pa"], ctx["column_sec"],
                               ctx["beam_sec"], geom=geom, walls=walls, wall_tw_mm=wall_tw_mm,
                               wall_E_Pa=ctx["wall_E_Pa"], element_sections=element_sections)
        ctx["ops_apply_combo_loads"](ctx["nodes"], ctx["elements"], combos[cn]["loads"])
        if ctx["ops_run_static"]() != 0:
            continue
        u = [np.mean([ops.nodeDisp(node_id_at(geom, s, ix), 1) for ix in range(geom.num_bays + 1)])
             for s in range(geom.num_floors + 1)]
        d = np.abs(np.diff(u)) / geom.floor_height
        out = np.maximum(out, d)
    return out


def _nvm_envelopes(ctx, combos, names, walls, wall_tw_mm, element_sections, npts=41):
    """Station-by-station N/V/M envelopes over the strength combinations."""
    from framekit.diagrams import _nvm_arrays_for_element
    env = {}
    for cn in names:
        if not str(cn).startswith(("ULS", "SEIS")):
            continue
        ctx["ops_build_model"](ctx["nodes"], ctx["elements"], ctx["Ec_Pa"], ctx["Es_Pa"], ctx["column_sec"],
                               ctx["beam_sec"], geom=ctx["geom"], walls=walls, wall_tw_mm=wall_tw_mm,
                               wall_E_Pa=ctx["wall_E_Pa"], element_sections=element_sections)
        ctx["ops_apply_combo_loads"](ctx["nodes"], ctx["elements"], combos[cn]["loads"])
        if ctx["ops_run_static"]() != 0:
            continue
        for eid in ctx["elements"]:
            r = _nvm_arrays_for_element(eid, ctx["elements"], combos[cn]["loads"], ctx["nodes"], npts=npts)
            if r is None:
                continue
            xg, yg, N, V, M, L, c, s = r
            e = env.setdefault(eid, {"x": np.linspace(0, L, npts), "geo": (xg, yg, c, s), "L": L,
                                     "Nmax": np.full(npts, -np.inf), "Nmin": np.full(npts, np.inf),
                                     "Vmax": np.full(npts, -np.inf), "Vmin": np.full(npts, np.inf),
                                     "Mmax": np.full(npts, -np.inf), "Mmin": np.full(npts, np.inf)})
            for k, arr in (("N", N), ("V", V), ("M", M)):
                a = np.asarray(arr) / 1000.0
                e[k + "max"] = np.maximum(e[k + "max"], a)
                e[k + "min"] = np.minimum(e[k + "min"], a)
    return env


def _modes(ctx, combos, walls, wall_tw_mm, element_sections, n_modes=3):
    from framekit.dynamics import ops_assign_floor_masses
    from framekit.ec8_seismic import floor_masses_from_combos, modal_x_properties
    kf = float((ctx.get("seismic_settings") or {}).get("stiffness_factor", 1.0))
    masses = floor_masses_from_combos(ctx["geom"], ctx["nodes"], ctx["elements"], combos,
                                      (ctx.get("seismic_settings") or {}).get("extra_mass_floor_kg", 0.0))
    ctx["ops_build_model"](ctx["nodes"], ctx["elements"], ctx["Ec_Pa"], ctx["Es_Pa"], ctx["column_sec"],
                           ctx["beam_sec"], geom=ctx["geom"], walls=walls, wall_tw_mm=wall_tw_mm,
                           wall_E_Pa=ctx["wall_E_Pa"], element_sections=element_sections, stiffness_factor=kf)
    ops_assign_floor_masses(ctx["nodes"], ctx["geom"], masses)
    return modal_x_properties(ctx["geom"], max(n_modes, 3))


# ============================================================
# figures
# ============================================================
def _fig_structure(F, ctx):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    draw_walls(ax, ctx["nodes"], ctx["geom"], ctx["base_walls"])
    draw_frame(ax, ctx["nodes"], ctx["elements"])
    draw_supports(ax, ctx["nodes"])
    draw_nodes(ax, ctx["nodes"])
    label_elements(ax, ctx["nodes"], ctx["elements"], fontsize=7)
    draw_dimensions(ax, ctx["nodes"], ctx["geom"])
    finish_frame_axes(ax, ctx["nodes"], pad=0.12, pad_bottom=0.16)
    ax.legend(handles=[Line2D([], [], color=COLORS["column"], lw=3, label="Column"),
                       Line2D([], [], color=COLORS["beam"], lw=2.6, label="Beam")]
              + ([Patch(fc=COLORS["wall_fill"], ec=COLORS["wall_edge"], hatch="////", label="Existing wall")]
                 if ctx["base_walls"] else []), loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return F.save(fig, "structure", "Structural model with member numbering and dimensions.")


def _fig_sections(F, ctx):
    from framekit.section_library import _draw_rc_section
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    _draw_rc_section(axes[0], ctx["beam_sec"], "Beam section")
    _draw_rc_section(axes[1], ctx["column_sec"], "Column section")
    return F.save(fig, "sections", "Existing RC sections (dimensions in mm).")


def _draw_util(ax, ctx, eta_map, walls=None, highlight=None, title=""):
    nodes, elements = ctx["nodes"], ctx["elements"]
    draw_walls(ax, nodes, ctx["geom"], walls, label=False)
    hl = set(int(e) for e in (highlight or []))
    for eid, (ni, nj, et) in elements.items():
        x1, y1 = nodes[ni]
        x2, y2 = nodes[nj]
        e = eta_map.get(int(eid), np.nan)
        if int(eid) in hl:
            ax.plot([x1, x2], [y1, y2], color="#222222", lw=9, alpha=0.18, solid_capstyle="butt", zorder=2)
        ax.plot([x1, x2], [y1, y2], color=util_color(e), lw=5, solid_capstyle="butt", zorder=3)
        if np.isfinite(e):
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{e:.2f}", fontsize=6.5, ha="center", va="center", zorder=6,
                    fontweight="bold" if e > 1 else "normal",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=util_color(e), lw=0.7))
    draw_supports(ax, nodes)
    finish_frame_axes(ax, nodes, pad=0.06)
    ax.set_title(title, loc="center", fontsize=11)


def _fig_util_before(F, ctx):
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_util(ax, ctx, _eta_map(ctx["df_worst_before"]), ctx["base_walls"])
    ax.legend(handles=band_legend_handles(UTIL_BANDS), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    return F.save(fig, "utilization_before", "Member utilization η of the existing structure "
                  "(envelope of the ULS and seismic combinations).")


def _fig_eta_ranking(F, ctx):
    eb = _eta_map(ctx["df_worst_before"])
    ef = _eta_map(ctx["df_worst_final"])
    ids = sorted(eb, key=lambda e: -eb[e])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(ids))
    ax.bar(x - 0.2, [eb[e] for e in ids], 0.4, color=[util_color(eb[e]) for e in ids], ec="#555555", lw=0.4,
           label="Before retrofit")
    if ef:
        ax.bar(x + 0.2, [ef.get(e, np.nan) for e in ids], 0.4, color=COLORS["column"], alpha=0.85,
               label="After retrofit")
    ax.axhline(1.0, color=COLORS["highlight"], lw=1.2, ls="--", label="η = 1")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in ids], fontsize=7, rotation=90)
    ax.set_xlabel("Member (sorted by utilization before retrofit)")
    ax.set_ylabel("Utilization η")
    ax.legend(loc="upper right")
    return F.save(fig, "eta_ranking", "Member utilization before and after the selected retrofit, sorted by the "
                  "initial utilization.")


def _fig_drifts(F, ctx, static_before, static_after, seis_before, seis_after):
    geom = ctx["geom"]
    st = np.arange(1, geom.num_floors + 1)
    ncol = 2 if seis_before is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 5.4), squeeze=False)
    ax = axes[0][0]
    ax.plot(100 * static_before, st, "o-", color=COLORS["highlight"], label="Before")
    ax.plot(100 * static_after, st, "s-", color=COLORS["column"], label="After")
    ax.axvline(100 * ctx["drift_limit"], color="#777777", ls="--", lw=1, label=f"Limit {100 * ctx['drift_limit']:.2f}%")
    ax.set_yticks(st)
    ax.set_xlabel("Max interstorey drift ratio [%]")
    ax.set_ylabel("Storey")
    ax.set_title("Static combinations (wind)", loc="center", fontsize=11)
    ax.legend()
    if seis_before is not None:
        ax = axes[0][1]
        lim = float(ctx["seismic_settings"]["dls_limit"])
        ax.plot(100 * seis_before["ν·ds/h [-]"], st, "o-", color=COLORS["highlight"], label="Before")
        if seis_after is not None:
            ax.plot(100 * seis_after["ν·ds/h [-]"], st, "s-", color=COLORS["column"], label="After")
        ax.axvline(100 * lim, color="#777777", ls="--", lw=1, label=f"DLS limit {100 * lim:.2f}%")
        ax.set_yticks(st)
        ax.set_xlabel("ν·ds/h [%]")
        ax.set_title("EC8 damage limitation", loc="center", fontsize=11)
        ax.legend()
    return F.save(fig, "drifts", "Interstorey drift profiles before and after the selected retrofit.")


def _fig_modes(F, ctx, modes_before, modes_after, cfg_before, cfg_after):
    from framekit.dynamics import _draw_mode_on_ax
    rows = [("Before", modes_before, cfg_before), ("After", modes_after, cfg_after)]
    rows = [r for r in rows if r[1]]
    if not rows:
        return None
    nm = 3
    fig, axes = plt.subplots(len(rows), nm, figsize=(3.6 * nm, 3.6 * len(rows)), squeeze=False)
    for i, (lab, modes, cfg) in enumerate(rows):
        top = sorted(modes, key=lambda d: -d["ratio"])[:nm]
        top = sorted(top, key=lambda d: d["mode"])
        _modes(ctx, cfg["combos"], cfg["walls"], cfg["tw"], cfg["sections"], n_modes=max(d["mode"] for d in top))
        for j, md in enumerate(top):
            _draw_mode_on_ax(axes[i][j], ctx["nodes"], ctx["elements"], md["mode"], period=md["T"],
                             title=f"{lab} – mode {md['mode']} (Mx {100 * md['ratio']:.0f}%)")
        for j in range(len(top), nm):
            axes[i][j].axis("off")
    return F.save(fig, "modes", "Main lateral modes before and after retrofit (cracked stiffness), with period "
                  "and effective modal mass in X.")


def _fig_spectrum(F, ctx, T_before, T_after):
    from framekit.ec8_seismic import design_spectrum
    from framekit.response_spectrum import build_ec8_plot_spectrum
    s = ctx["seismic_settings"]
    T, Se = build_ec8_plot_spectrum(s["rsa_params"])
    Sd = design_spectrum(T, s["rsa_params"], s["q"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(T, Se, color=COLORS["column"], lw=2, label="Elastic Se(T)")
    ax.plot(T, Sd, color=COLORS["deformed"], lw=2, label=f"Design Sd(T), q = {s['q']:.2f}")
    for Tm, lab, mk in ((T_before, "T1 before", "o"), (T_after, "T1 after", "s")):
        if Tm is not None and np.isfinite(Tm):
            y = np.interp(Tm, T, Sd)
            ax.plot(Tm, y, mk, ms=8, mfc="white", mec="#222222", mew=1.6)
            ax.annotate(f"{lab} = {Tm:.3f} s", (Tm, y), xytext=(8, 8), textcoords="offset points", fontsize=8.5)
    ax.set_xlim(0, T.max())
    ax.set_ylim(0, max(Se) * 1.1)
    ax.set_xlabel("Period T [s]")
    ax.set_ylabel("Spectral acceleration [m/s²]")
    ax.legend()
    return F.save(fig, "spectrum", "EC8 elastic and design spectra with the fundamental period before and after "
                  "retrofit.")


def _fig_alternatives(F, ctx):
    g = ctx["global_opt_df"]
    if g is None or g.empty:
        return None, None
    top = g.head(5)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(top))
    w = top["Wall_Cost"].astype(float).values if "Wall_Cost" in top else np.zeros(len(top))
    j = top["Jacketing_Cost"].astype(float).values if "Jacketing_Cost" in top else np.zeros(len(top))
    ax.bar(x, w, color=COLORS["wall_fill"], ec="#7A6220", label="Shear walls")
    ax.bar(x, j, bottom=w, color=COLORS["beam"], ec="#1D3557", label="Concrete jacketing")
    for i, (a, b) in enumerate(zip(w, j)):
        ax.text(i, a + b, f"{a + b:,.0f}", ha="center", va="bottom", fontsize=8.5)
    labels = []
    for i, (_, r) in enumerate(top.iterrows()):
        lab = f"#{i + 1}\n{str(r.get('Candidate_ID', ''))[:16]}"
        if str(r.get("Complete_Feasible", "")).upper() != "YES":
            lab += "\n(not feasible)"
        labels.append(lab)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Cost [EUR]")
    ax.set_ylim(0, max(w + j) * 1.15 if len(top) else 1)
    ax.legend(loc="upper left")
    p1 = F.save(fig, "alternatives_cost", "Cost breakdown of the five best retrofit strategies.")

    fig, ax = plt.subplots(figsize=(9, 5))
    du = pd.to_numeric(g.get("Drift_Utilization"), errors="coerce")
    tc = pd.to_numeric(g.get("Total_Retrofit_Cost"), errors="coerce")
    feas = g.get("Complete_Feasible", pd.Series(["NO"] * len(g))).astype(str).str.upper().eq("YES")
    ax.scatter(tc[~feas], du[~feas], s=14, color="#BBBBBB", alpha=0.7, label="Not feasible")
    ax.scatter(tc[feas], du[feas], s=18, color=COLORS["ok"], alpha=0.8, label="Feasible")
    ax.scatter(tc.iloc[:1], du.iloc[:1], s=160, marker="*", color=COLORS["highlight"], ec="#222222", zorder=5,
               label="Selected")
    ax.axhline(1.0, color="#777777", ls="--", lw=1)
    ax.set_xlabel("Total retrofit cost [EUR]")
    ax.set_ylabel("Drift utilization (governing)")
    ax.set_title(f"{len(g)} complete strategies evaluated", loc="center", fontsize=11)
    ax.legend()
    p2 = F.save(fig, "design_space", "Design space of all evaluated strategies: total cost versus governing drift "
                "utilization.")
    return p1, p2


def _fig_layout(F, ctx):
    from framekit.geometry import node_id_at
    fig, ax = plt.subplots(figsize=(10, 6.5))
    base = {(int(w.bay_left), int(w.floor_top)) for w in (ctx["base_walls"] or [])}
    for w in ctx["final_walls"] or []:
        ids = [node_id_at(ctx["geom"], w.floor_bot, w.bay_left - 1), node_id_at(ctx["geom"], w.floor_bot, w.bay_right - 1),
               node_id_at(ctx["geom"], w.floor_top, w.bay_right - 1), node_id_at(ctx["geom"], w.floor_top, w.bay_left - 1)]
        new = (int(w.bay_left), int(w.floor_top)) not in base
        ax.add_patch(Polygon([ctx["nodes"][n] for n in ids], closed=True,
                             fc=COLORS["wall_fill"] if new else "#D6D6D6",
                             ec=COLORS["highlight"] if new else "#6B6B6B", lw=1.8, alpha=0.6,
                             hatch="xx" if new else "////", zorder=1))
    jk = set(int(e) for e in ctx["jacketed_ids"])
    for eid, (ni, nj, et) in ctx["elements"].items():
        x1, y1 = ctx["nodes"][ni]
        x2, y2 = ctx["nodes"][nj]
        if eid in jk:
            ax.plot([x1, x2], [y1, y2], color=COLORS["highlight"], lw=10, alpha=0.25, solid_capstyle="butt")
            ax.plot([x1, x2], [y1, y2], color=COLORS["highlight"], lw=3.5, solid_capstyle="butt")
        else:
            ax.plot([x1, x2], [y1, y2], color="#9AA5B1", lw=2)
    label_elements(ax, ctx["nodes"], ctx["elements"], fontsize=7)
    draw_supports(ax, ctx["nodes"])
    draw_dimensions(ax, ctx["nodes"], ctx["geom"])
    finish_frame_axes(ax, ctx["nodes"], pad=0.12, pad_bottom=0.16)
    ax.legend(handles=[Patch(fc=COLORS["wall_fill"], ec=COLORS["highlight"], hatch="xx", label="New shear wall"),
                       Patch(fc="#D6D6D6", ec="#6B6B6B", hatch="////", label="Existing wall"),
                       Line2D([], [], color=COLORS["highlight"], lw=4, label="Jacketed member")],
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return F.save(fig, "selected_layout", "Selected retrofit strategy: new shear walls and jacketed members.")


def _fig_util_stages(F, ctx):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    _draw_util(axes[0], ctx, _eta_map(ctx["df_worst_before"]), ctx["base_walls"], title="Existing structure")
    _draw_util(axes[1], ctx, _eta_map(ctx["df_worst_after_walls"]), ctx["final_walls"], title="After shear walls")
    _draw_util(axes[2], ctx, _eta_map(ctx["df_worst_final"]), ctx["final_walls"], ctx["jacketed_ids"],
               title="After walls + jacketing")
    fig.legend(handles=band_legend_handles(UTIL_BANDS), loc="lower center", ncol=4, frameon=False)
    return F.save(fig, "utilization_stages", "Member utilization at the three stages of the selected strategy "
                  "(jacketed members shaded).")


def _fig_convergence(F, ctx):
    cj = ctx["cj_opt_df"]
    if cj is None or getattr(cj, "empty", True) or "Iteration" not in cj:
        return None
    it = cj["Iteration"].astype(int)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(it, cj["Max_EC2_Utilization"].astype(float), "o-", color=COLORS["highlight"], label="Max η")
    ax.axhline(1.0, color="#777777", ls="--", lw=1)
    ax.set_xlabel("Jacketing iteration")
    ax.set_ylabel("Max utilization η")
    ax2 = ax.twinx()
    ax2.plot(it, cj["Total_Cost"].astype(float), "s--", color=COLORS["column"], label="Jacketing cost")
    ax2.set_ylabel("Jacketing cost [EUR]")
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_xticks(list(it))
    return F.save(fig, "jacketing_convergence", "Convergence of the iterative jacketing: maximum utilization and "
                  "cost per iteration.")


def _jacket_signature(sec):
    return (round(float(sec.b_mm)), round(float(sec.h_mm)), int(getattr(sec, "jn_top", 0)), int(getattr(sec, "jn_bot", 0)),
            int(getattr(sec, "jn_left", 0)), round(float(getattr(sec, "jphi_mm", 0.0))),
            round(float(sec.stirrup_spacing_mm)))


def _fig_jacket_sections(F, ctx):
    from framekit.section_library import _draw_rc_section
    secs = ctx["jacketed_sections"] or {}
    groups = OrderedDict()
    for eid in sorted(int(e) for e in ctx["jacketed_ids"] if int(e) in secs):
        et = ctx["elements"][eid][2]
        groups.setdefault((et, _jacket_signature(secs[eid])), []).append(eid)
    paths = []
    items = list(groups.items())
    for k in range(0, len(items), 2):
        chunk = items[k:k + 2]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(9.5, 4.6 * len(chunk)), squeeze=False)
        for i, ((et, _sig), eids) in enumerate(chunk):
            base = ctx["column_sec"] if et == "column" else ctx["beam_sec"]
            _draw_rc_section(axes[i][0], base, f"{et.title()} – existing")
            _draw_rc_section(axes[i][1], secs[eids[0]], f"Jacketed: members {', '.join(map(str, eids))}",
                             base_sec=base)
        paths.append(F.save(fig, f"jacket_sections_{k // 2 + 1}",
                            "Jacketed sections (before/after). Members with identical jacket design are grouped."))
    return paths


def _fig_frame_moment(F, ctx, env):
    if not env:
        return None
    from framekit.diagrams import _diagram_offset
    nodes, elements = ctx["nodes"], ctx["elements"]
    _, _, _, _, Lref = frame_extent(nodes)
    mmax = max(max(np.max(np.abs(e["Mmax"])), np.max(np.abs(e["Mmin"]))) for e in env.values())
    sc = 0.12 * Lref / max(mmax, 1e-9)
    fig, ax = plt.subplots(figsize=(11, 7))
    draw_walls(ax, nodes, ctx["geom"], ctx["final_walls"], label=False)
    draw_frame(ax, nodes, elements, color="#5C6770", lw_column=1.8, lw_beam=1.6, zorder=5)
    for eid, e in env.items():
        xg, yg, c, s = e["geo"]
        for arr, colr in ((e["Mmax"], COLORS["pos"]), (e["Mmin"], COLORS["neg"])):
            fx, fy, lx, ly = _diagram_offset(xg, yg, c, s, arr, sc, tension_side=True)
            ax.fill(fx, fy, color=colr, alpha=0.25, lw=0)
            ax.plot(lx, ly, color=colr, lw=0.8)
    draw_supports(ax, nodes)
    finish_frame_axes(ax, nodes, pad=0.14)
    ax.legend(handles=[Patch(fc=COLORS["pos"], alpha=0.4, label="max M (sagging +)"),
                       Patch(fc=COLORS["neg"], alpha=0.4, label="min M (hogging −)")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    ax.set_title(f"Bending moment envelope – max |M| = {mmax:.0f} kNm", loc="center", fontsize=11)
    return F.save(fig, "moment_envelope", "Bending-moment envelope of the retrofitted frame over all ULS and seismic "
                  "combinations, drawn on the tension side (sagging moments below the beam axis).")


def _fig_member_nvm(F, ctx, env, ids):
    paths = []
    for k in range(0, len(ids), 2):
        chunk = ids[k:k + 2]
        fig, axes = plt.subplots(3, len(chunk), figsize=(6.2 * len(chunk), 8), squeeze=False, sharex="col")
        for j, eid in enumerate(chunk):
            e = env[eid]
            et = ctx["elements"][eid][2]
            for i, (q, unit) in enumerate((("N", "kN"), ("V", "kN"), ("M", "kNm"))):
                ax = axes[i][j]
                ax.fill_between(e["x"], e[q + "min"], e[q + "max"], color=COLORS["beam"], alpha=0.18, lw=0)
                ax.plot(e["x"], e[q + "max"], color=COLORS["pos"], lw=1.6, label="max")
                ax.plot(e["x"], e[q + "min"], color=COLORS["neg"], lw=1.6, label="min")
                ax.axhline(0, color="#888888", lw=0.7)
                ax.set_ylabel(f"{q} [{unit}]")
                if q == "M":
                    ax.invert_yaxis()          # moments plotted on the tension side: sagging (+) downward
                for arr in (e[q + "max"], e[q + "min"]):
                    kk = int(np.argmax(np.abs(arr)))
                    ax.annotate(f"{arr[kk]:.0f}", (e["x"][kk], arr[kk]), fontsize=7.5, xytext=(3, 3),
                                textcoords="offset points")
                if i == 0:
                    ax.set_title(f"Member {eid} ({et}, L = {e['L']:.2f} m)", loc="center", fontsize=11)
                    ax.legend(loc="upper right", fontsize=7.5)
            axes[2][j].set_xlabel("Position along member x [m]")
        paths.append(F.save(fig, f"nvm_members_{k // 2 + 1}",
                            "N/V/M envelopes (OpenSees, station by station) of the most utilized members of the "
                            "retrofitted frame. N is positive in tension; M is positive when sagging and is plotted "
                            "downward (tension side)."))
    return paths


def _fig_seismic_storeys(F, ctx, tb, ta):
    if tb is None:
        return None
    st = tb["Story"].values
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    w = 0.38
    axes[0].bar(st - w / 2, tb["DLS utilization"], w, color=COLORS["highlight"], label="Before")
    if ta is not None:
        axes[0].bar(st + w / 2, ta["DLS utilization"], w, color=COLORS["column"], label="After")
    axes[0].axhline(1, color="#777777", ls="--", lw=1)
    axes[0].set_xlabel("Storey")
    axes[0].set_ylabel("DLS utilization")
    axes[0].set_xticks(st)
    axes[0].legend()
    axes[1].bar(st - w / 2, tb["theta"], w, color=COLORS["highlight"], label="Before")
    if ta is not None:
        axes[1].bar(st + w / 2, ta["theta"], w, color=COLORS["column"], label="After")
    axes[1].axhline(0.1, color="#777777", ls="--", lw=1, label="θ = 0.10")
    axes[1].set_xlabel("Storey")
    axes[1].set_ylabel("θ")
    axes[1].set_xticks(st)
    axes[1].legend()
    return F.save(fig, "seismic_storeys", "EC8 storey checks: damage-limitation utilization and interstorey drift "
                  "sensitivity θ, before and after retrofit.")


def _fig_cost(F, row):
    keys = [("Wall_Material_Cost", "Wall materials"), ("Wall_Labor_Cost", "Wall labour"),
            ("Wall_Indirect_Cost", "Wall indirect"), ("Wall_Foundation_Connection_Cost", "Wall foundation"),
            ("Wall_Constructability_Cost", "Wall constructability"), ("Wall_Site_Overhead_Cost", "Wall site overhead"),
            ("Jacketing_Material_Cost", "Jacket materials"), ("Jacketing_Labor_Cost", "Jacket labour"),
            ("Jacketing_Indirect_Cost", "Jacket indirect"), ("Jacketing_Site_Overhead_Cost", "Jacket site overhead")]
    vals = [(lab, _num(row.get(k), 0.0)) for k, lab in keys]
    vals = [(l, v) for l, v in vals if v > 0]
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(vals) + 1.5))
    labs, vs = zip(*vals)
    cols = [COLORS["wall_fill"] if l.startswith("Wall") else COLORS["beam"] for l in labs]
    y = np.arange(len(vs))[::-1]
    ax.barh(y, vs, color=cols, ec="#555555", lw=0.4)
    for yy, v in zip(y, vs):
        ax.text(v, yy, f"  {v:,.0f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.set_xlabel("Cost [EUR]")
    ax.set_xlim(0, max(vs) * 1.2)
    ax.grid(axis="y", visible=False)
    return F.save(fig, "cost_breakdown", "Cost breakdown of the selected strategy.")


# ============================================================
# Word helpers
# ============================================================
def _doc_table(doc, df, title=None, widths=None, font=8):
    from docx.shared import Pt, RGBColor
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    if df is None or df.empty:
        return
    if title:
        p = doc.add_paragraph()
        r = p.add_run(title)
        r.bold = True
        r.font.size = Pt(9.5)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]
        cell.text = str(c)
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(font)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        sh = OxmlElement("w:shd")
        sh.set(qn("w:val"), "clear")
        sh.set(qn("w:fill"), "1D3557")
        cell._tc.get_or_add_tcPr().append(sh)
    for k, (_, row) in enumerate(df.iterrows()):
        cells = t.add_row().cells
        for i, v in enumerate(row.values):
            cells[i].text = v if isinstance(v, str) else _fmt(v)
            for par in cells[i].paragraphs:
                for run in par.runs:
                    run.font.size = Pt(font)
            if k % 2:
                sh = OxmlElement("w:shd")
                sh.set(qn("w:val"), "clear")
                sh.set(qn("w:fill"), "F2F4F7")
                cells[i]._tc.get_or_add_tcPr().append(sh)
    doc.add_paragraph()


def _doc_fig(doc, path, caption, counter, width_in=6.3):
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    if not path or not os.path.exists(path):
        return
    # keep every figure inside one page: limit the height as well as the width
    max_h_in = 7.0
    w_in = float(width_in)
    try:
        from PIL import Image
        with Image.open(path) as im:
            wpx, hpx = im.size
        if hpx > 0 and w_in * hpx / wpx > max_h_in:
            w_in = max_h_in * wpx / hpx
    except Exception:
        pass
    doc.add_picture(path, width=Inches(w_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True   # caption stays with its figure
    counter[0] += 1
    p = doc.add_paragraph()
    r = p.add_run(f"Figure {counter[0]}. ")
    r.bold = True
    r.font.size = Pt(9)
    r2 = p.add_run(caption)
    r2.font.size = Pt(9)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _para(doc, text, size=10):
    from docx.shared import Pt
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(size)
    return p


# ============================================================
# MAIN ENTRY
# ============================================================
def _valid_hist(h):
    return h[pd.to_numeric(h["Total_Cost"], errors="coerce").notna()] if "Total_Cost" in h else h


def export_retrofit_report_v2(**ctx):
    """Build the compact Word report and the full-data Excel workbook.

    Required keys: output_dir, nodes, elements, geom, combos, names, Ec_Pa, Es_Pa, column_sec, beam_sec,
    ops_build_model, ops_apply_combo_loads, ops_run_static, base_walls, wall_tw_mm, wall_E_Pa,
    concrete_density, wall_density, drift_limit, global_opt_df, df_worst_before.
    Optional: seismic_settings, code_info, material, ec2mat, logo_paths, author, supervisor, title, project.
    """
    out = ctx["output_dir"]
    os.makedirs(out, exist_ok=True)
    F = _Figs(os.path.join(out, "figures"))
    g = ctx.get("global_opt_df")
    best = g.iloc[0] if g is not None and not g.empty else None
    cand = best.get("_candidate") if best is not None else None
    ctx["final_walls"] = list(best.get("_walls", ctx["base_walls"]) or []) if best is not None else list(ctx["base_walls"] or [])
    ctx["final_tw"] = float(getattr(cand, "wall_tw_mm", ctx["wall_tw_mm"])) if cand is not None else ctx["wall_tw_mm"]
    ctx["jacketed_sections"] = best.get("_element_sections") if best is not None else None
    ctx["jacketed_ids"] = [int(x) for x in str(best.get("Jacketed_Members", "") if best is not None else "").split(",") if str(x).strip()]
    ctx["df_worst_after_walls"] = best.get("_df_worst_after_walls") if best is not None else None
    ctx["df_worst_final"] = best.get("_df_worst_final") if best is not None else None
    ctx["cj_opt_df"] = best.get("_cj_opt_df") if best is not None else None
    ctx["member_report"] = best.get("_member_report") if best is not None else None

    import openseespy.opensees as ops  # noqa: F401  (model rebuilt below)
    print("Report: re-analysing the existing and the retrofitted structure...")
    c0, n0, ev0 = _final_combos(ctx, None, ctx["base_walls"], ctx["wall_tw_mm"])
    c1, n1, ev1 = _final_combos(ctx, ctx["jacketed_sections"], ctx["final_walls"], ctx["final_tw"])
    # EC2 checks of the existing structure with exactly the same combinations
    # (ULS + EC8 lateral-force seismic action) used for the retrofitted one
    if ctx.get("ec2mat") is not None:
        try:
            from framekit.retrofit_optimizer import run_ec2_member_assessment
            dfb, _, _ = run_ec2_member_assessment(
                nodes=ctx["nodes"], elements=ctx["elements"], geom=ctx["geom"], combos=c0, combo_names=n0,
                Ec_Pa=ctx["Ec_Pa"], Es_Pa=ctx["Es_Pa"], column_sec=ctx["column_sec"], beam_sec=ctx["beam_sec"],
                ec2mat=ctx["ec2mat"], ops_build_model=ctx["ops_build_model"],
                ops_apply_combo_loads=ctx["ops_apply_combo_loads"], ops_run_static=ctx["ops_run_static"],
                walls=ctx["base_walls"], wall_tw_mm=ctx["wall_tw_mm"], wall_E_Pa=ctx["wall_E_Pa"])
            if dfb is not None and not dfb.empty:
                ctx["df_worst_before"] = dfb
        except Exception as exc:
            print(f"⚠ baseline EC2 check for the report failed: {exc}")
    if ctx.get("df_worst_before") is None or getattr(ctx["df_worst_before"], "empty", True):
        ctx["df_worst_before"] = None
    if ctx.get("df_worst_final") is None:
        ctx["df_worst_final"] = ctx.get("df_worst_after_walls")

    static_b = _static_drift_profile(ctx, c0, n0, ctx["base_walls"], ctx["wall_tw_mm"], None)
    static_a = _static_drift_profile(ctx, c1, n1, ctx["final_walls"], ctx["final_tw"], ctx["jacketed_sections"])
    env = _nvm_envelopes(ctx, c1, n1, ctx["final_walls"], ctx["final_tw"], ctx["jacketed_sections"])
    modes_b = modes_a = None
    try:
        modes_b = _modes(ctx, c0, ctx["base_walls"], ctx["wall_tw_mm"], None)
        modes_a = _modes(ctx, c1, ctx["final_walls"], ctx["final_tw"], ctx["jacketed_sections"])
    except Exception as exc:
        print(f"⚠ modal analysis for the report failed: {exc}")

    eta_f = _eta_map(ctx["df_worst_final"])
    crit = sorted(eta_f, key=lambda e: -eta_f[e])[:4] if eta_f else []
    crit = [e for e in crit if e in env]

    # ---------------- figures ----------------
    figs = {}
    figs["structure"] = _fig_structure(F, ctx)
    figs["sections"] = _fig_sections(F, ctx)
    if ctx["df_worst_before"] is not None:
        figs["util_before"] = _fig_util_before(F, ctx)
    if modes_b:
        figs["modes"] = _fig_modes(F, ctx, modes_b, modes_a,
                                   {"combos": c0, "walls": ctx["base_walls"], "tw": ctx["wall_tw_mm"], "sections": None},
                                   {"combos": c1, "walls": ctx["final_walls"], "tw": ctx["final_tw"], "sections": ctx["jacketed_sections"]})
    if ctx.get("seismic_settings"):
        figs["spectrum"] = _fig_spectrum(F, ctx, ev0["T1"] if ev0 else None, ev1["T1"] if ev1 else None)
    figs["alt_cost"], figs["design_space"] = _fig_alternatives(F, ctx)
    sh = ctx.get("smart_history")
    if sh is not None and not getattr(sh, "empty", True):
        try:
            from framekit.smart_search import plot_learning_curve, plot_search_map, best_segments_from_history
            fig, ax = plt.subplots(figsize=(9, 4.6))
            if plot_learning_curve(sh, ax) is not None:
                figs["smart_curve"] = F.save(fig, "smart_learning_curve",
                                             "Smart search learning curve: cost of each trial and best feasible "
                                             "cost found so far (filled = feasible, hollow = not feasible).")
            else:
                plt.close(fig)
            nb, nf = int(ctx["geom"].num_bays), int(ctx["geom"].num_floors)
            fig, ax = plt.subplots(figsize=(1.6 * nb + 3, 0.8 * nf + 2))
            if plot_search_map(sh, ctx["geom"], best_segments_from_history(sh, g), ax) is not None:
                figs["smart_map"] = F.save(fig, "smart_search_map",
                                           "Smart search map: how often a wall was tried in each panel; the "
                                           "selected layout is outlined.")
            else:
                plt.close(fig)
        except Exception as exc:
            print(f"⚠ smart-search figures failed: {exc}")
    figs["layout"] = _fig_layout(F, ctx)
    figs["stages"] = _fig_util_stages(F, ctx)
    if ctx["df_worst_before"] is not None:
        figs["ranking"] = _fig_eta_ranking(F, ctx)
    figs["drifts"] = _fig_drifts(F, ctx, static_b, static_a, ev0["storey_table"] if ev0 else None,
                                 ev1["storey_table"] if ev1 else None)
    figs["seis_storeys"] = _fig_seismic_storeys(F, ctx, ev0["storey_table"] if ev0 else None,
                                                ev1["storey_table"] if ev1 else None)
    figs["convergence"] = _fig_convergence(F, ctx)
    figs["jsections"] = _fig_jacket_sections(F, ctx)
    figs["m_env"] = _fig_frame_moment(F, ctx, env)
    figs["nvm"] = _fig_member_nvm(F, ctx, env, crit)
    figs["cost"] = _fig_cost(F, best) if best is not None else None

    # ---------------- tables ----------------
    eb = _eta_map(ctx["df_worst_before"])
    ea = _eta_map(ctx["df_worst_after_walls"])
    n_fail_b = sum(1 for v in eb.values() if v > 1)
    n_fail_a = sum(1 for v in ea.values() if v > 1)
    n_fail_f = sum(1 for v in eta_f.values() if v > 1)
    seis = ctx.get("seismic_settings")
    summary = [
        ("Structure", f"{ctx['geom'].num_floors} storeys × {ctx['geom'].num_bays} bays, storey height "
                      f"{ctx['geom'].floor_height:.2f} m"),
        ("Code basis", (ctx.get("code_info") or {}).get("code", "–")
         + (f", {ctx['code_info'].get('knowledge_level')} (CF = {ctx['code_info'].get('CF'):.2f})"
            if (ctx.get("code_info") or {}).get("knowledge_level") else "")),
    ]
    if seis:
        summary += [("Seismic action", f"ag = {seis['rsa_params']['ag'] / 9.81:.3f} g, ground {seis['rsa_params']['soil_class']}, "
                                       f"type {seis['rsa_params']['spectrum_type']}, q = {seis['q']:.2f}, EI factor {seis['stiffness_factor']:.2f}")]
    summary += [
        ("Members with η > 1", f"{n_fail_b} before → {n_fail_a} after walls → {n_fail_f} final"),
        ("Max utilization η", f"{_fmt(max(eb.values()) if eb else np.nan)} → {_fmt(max(eta_f.values()) if eta_f else np.nan)}"),
        ("Max static drift ratio", f"{100 * static_b.max():.3f}% → {100 * static_a.max():.3f}% (limit {100 * ctx['drift_limit']:.2f}%)"),
    ]
    if ev0 and ev1:
        summary += [("EC8 DLS utilization", f"{ev0['dls_utilization']:.2f} → {ev1['dls_utilization']:.2f}"),
                    ("Fundamental period T1", f"{ev0['T1']:.3f} s → {ev1['T1']:.3f} s"),
                    ("Base shear Fb (LFM)", f"{ev0['Fb_kN']:.0f} kN → {ev1['Fb_kN']:.0f} kN")]
    if best is not None:
        summary += [("Selected strategy", str(best.get("Description", ""))),
                    ("Jacketed members", f"{len(ctx['jacketed_ids'])} ({', '.join(map(str, ctx['jacketed_ids']))})"),
                    ("Cost walls / jacketing / total [EUR]",
                     f"{_fmt(best.get('Wall_Cost'))} / {_fmt(best.get('Jacketing_Cost'))} / {_fmt(best.get('Total_Retrofit_Cost'))}"),
                    ("Strategy fully feasible", str(best.get("Complete_Feasible", "")))]
    df_summary = pd.DataFrame(summary, columns=["Item", "Value"])

    top_cols = [("Candidate_ID", "Strategy"), ("Num_Wall_Panels", "Wall panels"), ("Jacketing_Num_Members", "Jacketed"),
                ("Wall_Cost", "Walls [EUR]"), ("Jacketing_Cost", "Jackets [EUR]"),
                ("Total_Retrofit_Cost", "Total [EUR]"), ("Drift_Utilization", "Drift util."),
                ("Final_Max_EC2_Utilization", "Max η"), ("Complete_Feasible", "Feasible")]
    df_top = None
    if g is not None and not g.empty:
        df_top = pd.DataFrame({lab: g.head(5)[k].values for k, lab in top_cols if k in g.columns})
        df_top.insert(0, "Rank", range(1, len(df_top) + 1))

    # per-member jacketing table
    rows = []
    secs = ctx["jacketed_sections"] or {}
    mr = ctx["member_report"]
    mrd = {int(r["Element"]): r for _, r in mr.iterrows()} if mr is not None and not getattr(mr, "empty", True) else {}
    for eid in ctx["jacketed_ids"]:
        s = secs.get(eid)
        if s is None:
            continue
        r = mrd.get(eid, {})
        bars = f"{int(s.jn_top) + int(s.jn_bot) + int(s.jn_left) + int(s.jn_right)}Ø{s.jphi_mm:.0f}" if s.jphi_mm else "–"
        rows.append({"Member": eid, "Type": ctx["elements"][eid][2], "Jacket t [mm]": _num(s.jacket_t_mm),
                     "Section [mm]": f"{s.b_mm:.0f}×{s.h_mm:.0f}", "Jacket bars": bars,
                     "Stirrups": f"Ø{s.stirrup_phi_mm:.0f}/{s.stirrup_spacing_mm:.0f}",
                     "Governing mode": str(r.get("Problem_Before", "")) if len(r) else "",
                     "η before": eb.get(eid, np.nan), "η after walls": ea.get(eid, np.nan), "η final": eta_f.get(eid, np.nan),
                     "Bar spacing": str(getattr(s, "jacket_check", {}).get("status", ""))})
    df_jk = pd.DataFrame(rows)

    df_walls = pd.DataFrame([{"Bay": int(w.bay_left), "Storey": int(w.floor_top),
                              "Thickness [mm]": _num(getattr(w, "tw_mm", None) or ctx["final_tw"]),
                              "Status": "existing" if (int(w.bay_left), int(w.floor_top)) in
                              {(int(b.bay_left), int(b.floor_top)) for b in (ctx["base_walls"] or [])} else "new"}
                             for w in ctx["final_walls"]])

    # ---------------- Word ----------------
    docx_path = None
    try:
        import docx
        from docx.shared import Pt, Inches, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        doc = docx.Document()
        st = doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(10)
        for sec_ in doc.sections:
            sec_.left_margin = sec_.right_margin = Inches(0.9)
        cnt = [0]

        # ---------------- cover page (with table of contents) ----------------
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.enum.text import WD_BREAK

        def _center(par):
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_after = Pt(2)
            return par

        def _line(text, size=10.5, bold=False, italic=False, color=None, space_after=2):
            par = _center(doc.add_paragraph())
            run = par.add_run(text)
            run.bold, run.italic = bold, italic
            run.font.size = Pt(size)
            if color:
                run.font.color.rgb = RGBColor(*color)
            par.paragraph_format.space_after = Pt(space_after)
            return par

        polito = ctx.get("logo_polito")
        artiste = ctx.get("logo_artiste")
        arcopy = ctx.get("logo_arcopy")
        if polito and os.path.exists(polito):
            _center(doc.add_paragraph()).add_run().add_picture(polito, height=Inches(0.95))
        if arcopy and os.path.exists(arcopy):
            _center(doc.add_paragraph()).add_run().add_picture(arcopy, width=Inches(3.6))
        _line(ctx.get("title", "Seismic Assessment and Retrofit Optimization Report"), 19, bold=True,
              color=(0x1D, 0x35, 0x57), space_after=4)
        _line(ctx.get("subtitle", "AI and Structural Engineering"), 11.5, italic=True, space_after=6)
        if ctx.get("student"):
            _line(f"Student: {ctx['student']}", 10.5, bold=True)
        if ctx.get("role"):
            _line(f"Role: {ctx['role']}", 10)
        if ctx.get("supervisor"):
            _line(f"Supervisor: {ctx['supervisor']}", 10)
        _line(f"Date: {_dt.date.today().isoformat()}    ·    Run: {ctx.get('run_name', '')}", 10, space_after=8)

        # ARTISTE logo closes the cover page
        doc.add_paragraph().paragraph_format.space_after = Pt(18)
        if artiste and os.path.exists(artiste):
            _center(doc.add_paragraph()).add_run().add_picture(artiste, width=Inches(2.6))
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

        # table of contents (Word field, filled when the document is opened)
        hp = doc.add_paragraph()
        hr = hp.add_run("Contents")
        hr.bold = True
        hr.font.size = Pt(12)
        hr.font.color.rgb = RGBColor(0x1D, 0x35, 0x57)
        toc_titles = ["1. Summary", "2. Structure and input data", "3. Assessment of the existing structure",
                      "4. Retrofit alternatives"] + (["    4.1 Smart search (Bayesian optimization)"] if (figs.get("smart_curve") or figs.get("smart_map")) else []) + ["5. Selected retrofit strategy",
                      "    5.1 Verification after retrofit", "    5.2 Internal forces of the retrofitted frame",
                      "    5.3 Cost", "6. Conclusions and limitations"]
        tp = doc.add_paragraph()
        r_begin = tp.add_run()
        fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), "begin"); r_begin._r.append(fc)
        r_instr = tp.add_run()
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve")
        it.text = 'TOC \\o "1-2" \\h \\z \\u'
        r_instr._r.append(it)
        r_sep = tp.add_run()
        fs = OxmlElement("w:fldChar"); fs.set(qn("w:fldCharType"), "separate"); r_sep._r.append(fs)
        for i, ttl in enumerate(toc_titles):
            rr = tp.add_run(ttl)
            rr.font.size = Pt(10)
            if i < len(toc_titles) - 1:
                rr.add_break()
        r_end = tp.add_run()
        fe = OxmlElement("w:fldChar"); fe.set(qn("w:fldCharType"), "end"); r_end._r.append(fe)
        # ask Word to refresh the table of contents (page numbers) when the file is opened
        upd = OxmlElement("w:updateFields"); upd.set(qn("w:val"), "true")
        doc.settings.element.append(upd)

        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

        doc.add_heading("1. Summary", level=1)
        txt = (f"The existing frame was verified with Eurocode 2 under the ULS"
               + (" and EC8 seismic" if seis else "") + f" combinations: {n_fail_b} of {len(eb)} members exceed "
               f"their resistance. The optimizer evaluated {len(g) if g is not None else 0} complete retrofit "
               f"strategies (wall layout plus the jacketing required after it). ")
        if best is not None:
            txt += (f"The selected strategy, {best.get('Candidate_ID', '')}, adds {int(_num(best.get('Num_Wall_Panels'), 0))} "
                    f"wall panel(s) and jackets {len(ctx['jacketed_ids'])} members, for a total estimated cost of "
                    f"{_fmt(best.get('Total_Retrofit_Cost'))} EUR. After retrofit the maximum utilization is "
                    f"{_fmt(max(eta_f.values()) if eta_f else np.nan)}.")
        _para(doc, txt)
        _doc_table(doc, df_summary, "Key results", font=8.5)

        doc.add_heading("2. Structure and input data", level=1)
        g0 = ctx["geom"]
        _para(doc, f"This section describes the analysed structure. The model is a planar reinforced-concrete frame "
                   f"with {g0.num_floors} storeys and {g0.num_bays} bays (storey height {g0.floor_height:.2f} m), "
                   f"analysed in OpenSees with elastic beam-column elements fixed at the base. Existing infill "
                   f"shear walls, if any, are modelled as equivalent diagonal braces. Figure {cnt[0] + 1} shows the "
                   f"member numbering used in all tables, and Figure {cnt[0] + 2} the existing column and beam "
                   f"sections with their reinforcement. The code basis is "
                   f"{(ctx.get('code_info') or {}).get('code', 'Eurocode 8')}"
                   + (f", with knowledge level {ctx['code_info'].get('knowledge_level')} and confidence factor "
                      f"CF = {ctx['code_info'].get('CF'):.2f} applied to the material strengths"
                      if (ctx.get("code_info") or {}).get("knowledge_level") else "") + ".")
        _doc_fig(doc, figs["structure"], dict(F.items)[figs["structure"]], cnt)
        _doc_fig(doc, figs["sections"], dict(F.items)[figs["sections"]], cnt, 5.6)

        doc.add_heading("3. Assessment of the existing structure", level=1)
        _para(doc, "This section checks the existing building before any intervention. Every member is verified "
                   "with Eurocode 2 for bending with axial force and for shear, under all ULS combinations"
                   + (" and the EC8 seismic combinations (lateral force method with the design spectrum)" if seis else "")
                   + ". The utilization η is the ratio between the design action and the resistance, taken as the "
                   "envelope of all combinations; η > 1 means the member does not satisfy the code. "
                   f"In the existing frame {n_fail_b} of {len(eb)} members have η > 1 (maximum "
                   f"{_fmt(max(eb.values()) if eb else np.nan)}). The modal analysis and the response spectrum "
                   "below give the dynamic properties used for the seismic action.")
        for key in ("util_before", "modes", "spectrum"):
            if figs.get(key):
                _doc_fig(doc, figs[key], dict(F.items)[figs[key]], cnt)

        doc.add_heading("4. Retrofit alternatives", level=1)
        _para(doc, "Each alternative is a complete strategy: a shear-wall layout plus the concrete jacketing "
                   "required after it. Strategies are ranked by total cost among those satisfying all checks. "
                   "Only the selected strategy is described in detail; the full ranking is in the Excel file.")
        _doc_table(doc, df_top, "Five best strategies", font=8)
        for key in ("alt_cost", "design_space"):
            if figs.get(key):
                _doc_fig(doc, figs[key], dict(F.items)[figs[key]], cnt, 5.8)
        if figs.get("smart_curve") or figs.get("smart_map"):
            doc.add_heading("4.1 Smart search (Bayesian optimization)", level=2)
            _sh = _valid_hist(sh)
            n_tr, n_un = len(_sh), int((~_sh["Cached"].astype(bool)).sum()) if "Cached" in _sh else len(_sh)
            _u = _sh[~_sh["Cached"].astype(bool)] if "Cached" in _sh else _sh
            _col = "Analysis" if "Analysis" in _u else "Trial"
            first = _u[_u["Feasible"].eq("YES")][_col].min() if "Feasible" in _u else np.nan
            best_tr = _u.loc[pd.to_numeric(_u["Total_Cost"]).where(_u["Feasible"].eq("YES")).idxmin(), _col] \
                if _u["Feasible"].eq("YES").any() else np.nan
            _para(doc, f"Wall layouts were searched with a Tree-structured Parzen Estimator (Optuna). The first "
                       f"trials are physics-guided (walls at the failing storeys, full-height walls) and a few are "
                       f"random; afterwards every new layout is proposed from a probabilistic model of the good and "
                       f"bad layouts found so far, with drift, member utilization and jacketing limits as "
                       f"constraints. {n_tr} trials were run ({n_un} unique strategies analysed). The first feasible "
                       f"strategy was found at analysis {_fmt(first, 0)} and the selected one at analysis {_fmt(best_tr, 0)}.")
            for key in ("smart_curve", "smart_map"):
                if figs.get(key):
                    _doc_fig(doc, figs[key], dict(F.items)[figs[key]], cnt, 5.8)

        doc.add_heading("5. Selected retrofit strategy", level=1)
        if best is not None:
            _para(doc, f"Strategy {best.get('Candidate_ID', '')}: {best.get('Description', '')}. "
                       f"{len(ctx['jacketed_ids'])} members are strengthened with RC jackets. Existing bars stay in "
                       f"the core; each jacket has its own cage whose bars respect EC2 8.2(2) minimum clear spacing "
                       f"and, for columns, the EC8 maximum spacing of 200 mm between restrained bars.")
        _doc_fig(doc, figs["layout"], dict(F.items)[figs["layout"]], cnt)
        _doc_table(doc, df_walls, "Shear-wall panels", font=8)
        _doc_table(doc, df_jk, "Jacketing design per member", font=7.5)
        for p in figs["jsections"] or []:
            _doc_fig(doc, p, dict(F.items)[p], cnt, 5.4)
        doc.add_heading("5.1 Verification after retrofit", level=2)
        _para(doc, "This part repeats the checks of Section 3 on the retrofitted structure, with exactly the same "
                   "combinations, so the two states can be compared directly. The utilization maps show the "
                   "three stages of the strategy: existing frame, after the new shear walls, and after the "
                   "jacketing. The ranking chart compares η of every member before and after, the drift profiles "
                   "compare the interstorey drifts with their limits, and the convergence plot shows how the "
                   f"iterative jacketing reduced the maximum utilization. After retrofit {n_fail_f} member(s) "
                   f"have η > 1 and the maximum utilization is {_fmt(max(eta_f.values()) if eta_f else np.nan)}.")
        for key in ("stages", "ranking", "drifts", "seis_storeys", "convergence"):
            if figs.get(key):
                _doc_fig(doc, figs[key], dict(F.items)[figs[key]], cnt)
        doc.add_heading("5.2 Internal forces of the retrofitted frame", level=2)
        _para(doc, "Internal forces are recovered from OpenSees at 41 stations along every member for all ULS and "
                   "seismic combinations. The envelopes below show the maximum and minimum value at each station; "
                   "distributed loads give the parabolic moment and linear shear in the beams, and the self-weight "
                   "gives the linear axial force in the columns.")
        if figs.get("m_env"):
            _doc_fig(doc, figs["m_env"], dict(F.items)[figs["m_env"]], cnt)
        for p in figs["nvm"] or []:
            _doc_fig(doc, p, dict(F.items)[p], cnt)
        doc.add_heading("5.3 Cost", level=2)
        if best is not None:
            _para(doc, "The total cost of the selected strategy is the sum of the shear-wall works and the "
                       "concrete jacketing. For both, the optimizer computes materials and labour from the quantities "
                       "of the design, then adds indirect costs and site overheads (and, for the walls, the "
                       "foundation and connection works), using the unit costs given at the start of the "
                       "optimization. "
                       f"Walls: {_fmt(best.get('Wall_Cost'))} EUR; jacketing: {_fmt(best.get('Jacketing_Cost'))} EUR; "
                       f"total: {_fmt(best.get('Total_Retrofit_Cost'))} EUR. The chart below shows the breakdown "
                       "by cost item.")
        if figs.get("cost"):
            _doc_fig(doc, figs["cost"], dict(F.items)[figs["cost"]], cnt, 5.8)

        doc.add_heading("6. Conclusions and limitations", level=1)
        concl = []
        if best is not None:
            concl.append(f"The selected strategy {'satisfies' if str(best.get('Complete_Feasible', '')).upper() == 'YES' else 'does NOT fully satisfy'} "
                         f"all drift, seismic and member checks at an estimated cost of {_fmt(best.get('Total_Retrofit_Cost'))} EUR.")
        concl.append("The assessment is linear elastic and planar. Shear walls are idealized as equivalent braces and are "
                     "not verified themselves; wall foundations and connections must be designed separately. "
                     "The selected solution should be confirmed with a nonlinear analysis before design.")
        for c in concl:
            _para(doc, c)
        docx_path = os.path.join(out, "retrofit_report.docx")
        doc.save(docx_path)
    except ImportError:
        print("⚠ python-docx not installed: Word report skipped.")

    # ---------------- Excel ----------------
    xlsx_path = os.path.join(out, "retrofit_report_data.xlsx")
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        engine = "openpyxl"

    def _clean(df):
        if df is None or getattr(df, "empty", True):
            return None
        return df.drop(columns=[c for c in df.columns if str(c).startswith("_")], errors="ignore")

    nvm_rows = []
    for eid, e in env.items():
        for i, x in enumerate(e["x"]):
            nvm_rows.append({"Element": eid, "x_m": x, "N_max_kN": e["Nmax"][i], "N_min_kN": e["Nmin"][i],
                             "V_max_kN": e["Vmax"][i], "V_min_kN": e["Vmin"][i],
                             "M_max_kNm": e["Mmax"][i], "M_min_kNm": e["Mmin"][i]})
    with pd.ExcelWriter(xlsx_path, engine=engine) as w:
        for name, df in (("Summary", df_summary), ("Top5", df_top), ("All_strategies", _clean(g)),
                         ("Walls", df_walls), ("Jacketing", df_jk), ("Jacketing_iterations", _clean(ctx["cj_opt_df"])),
                         ("EC2_before", _clean(ctx["df_worst_before"])),
                         ("EC2_after_walls", _clean(ctx["df_worst_after_walls"])),
                         ("EC2_final", _clean(ctx["df_worst_final"])),
                         ("Seismic_storeys_before", ev0["storey_table"] if ev0 else None),
                         ("Seismic_storeys_after", ev1["storey_table"] if ev1 else None),
                         ("NVM_envelopes_final", pd.DataFrame(nvm_rows)),
                         ("Smart_search_history", _clean(ctx.get("smart_history")))):
            if df is not None and not getattr(df, "empty", True):
                df.to_excel(w, sheet_name=name[:31], index=False)
    print(f"✔ Report: {docx_path}\n✔ Report data: {xlsx_path}\n✔ Report figures: {F.folder}")
    return {"docx": docx_path, "xlsx": xlsx_path, "figures": [p for p, _ in F.items]}
