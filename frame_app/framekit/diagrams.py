import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from framekit.opensees_model import ops_ele_end_forces_local, _elem_cos_sin
from framekit.units import _kNpm_to_Npm
from framekit.member_sections import npts_for


def _nvm_arrays_for_element(eid, elements, loads_combo, nodes, npts=60):
    ni, nj, etype = elements[eid]
    npts = npts_for(etype, npts)       # user-defined number of sections (beams / columns), if set
    x1, y1 = nodes[ni]
    x2, y2 = nodes[nj]
    L = math.hypot(x2 - x1, y2 - y1)
    if L < 1e-12:
        return None

    _, c, s = _elem_cos_sin(nodes, elements, eid)
    x_loc = np.linspace(0.0, L, npts)

    # OpenSees local end forces
    N1r, V1r, M1r, N2r, V2r, M2r = ops_ele_end_forces_local(eid)
    # Internal forces on the positive cut face of segment [0, x]:
    # N tension-positive, V = dM/dx, M sagging-positive (bottom fibre in
    # tension for a beam drawn left -> right).
    N1 = -N1r
    V1 = V1r
    M1 = -M1r

    N = np.full(npts, N1)
    V = np.full(npts, V1)
    M = M1 + V1 * x_loc

    # distributed load contribution
    wU = 0.0
    nU = 0.0          # axial (local x) uniform load
    n1t = 0.0
    n2t = 0.0
    w1t = 0.0
    w2t = 0.0
    hasU = False
    hasT = False

    for Ld in loads_combo:
        if Ld.element != eid:
            continue

        if Ld.load_type == "uniform":
            qx_g = _kNpm_to_Npm(Ld.value) if Ld.direction == "horizontal" else 0.0
            qy_g = _kNpm_to_Npm(Ld.value) if Ld.direction == "vertical" else 0.0
            wU += -s * qx_g + c * qy_g
            nU += c * qx_g + s * qy_g
            hasU = True

        elif Ld.load_type == "triangular":
            w1g = _kNpm_to_Npm(Ld.w1 if Ld.w1 is not None else 0.0)
            w2g = _kNpm_to_Npm(Ld.w2 if Ld.w2 is not None else 0.0)

            qx1_g = w1g if Ld.direction == "horizontal" else 0.0
            qy1_g = w1g if Ld.direction == "vertical" else 0.0
            qx2_g = w2g if Ld.direction == "horizontal" else 0.0
            qy2_g = w2g if Ld.direction == "vertical" else 0.0

            w1t += -s * qx1_g + c * qy1_g
            w2t += -s * qx2_g + c * qy2_g
            n1t += c * qx1_g + s * qy1_g
            n2t += c * qx2_g + s * qy2_g
            hasT = True

    # axial equilibrium of [0, x]: N(x) = N1 - ∫ n dx  (n = load along local x)
    if hasU:
        N = N - nU * x_loc
    if hasT:
        N = N - (n1t * x_loc + 0.5 * (n2t - n1t) * x_loc ** 2 / L)

    if hasU:
        V = V + wU * x_loc
        M = M + 0.5 * wU * x_loc ** 2

    if hasT:
        V = V + (w1t * x_loc + 0.5 * (w2t - w1t) * x_loc ** 2 / L)
        M = M + (0.5 * w1t * x_loc ** 2 + (w2t - w1t) * x_loc ** 3 / (6.0 * L))

    xg = x1 + c * x_loc
    yg = y1 + s * x_loc

    return xg, yg, N, V, M, L, c, s


def _label_offset(c, s, v, tension_side, dist):
    """Small shift of a label away from the member, on the side where the diagram is drawn."""
    px, py = -s, c
    if tension_side:
        px, py = -px, -py
    sg = 1.0 if v >= 0 else -1.0
    return sg * dist * px, sg * dist * py


def _diagram_offset(xg, yg, c, s, values, scale, tension_side=False):
    """Offset curve of a diagram along the member normal.

    tension_side=True (bending moments): the diagram is drawn on the side of
    the fibre in tension - sagging moments of a beam below the axis, as in the
    usual structural-engineering convention.
    """
    px = -s
    py = c
    if tension_side:
        px, py = -px, -py

    xs_top = xg + scale * values * px
    ys_top = yg + scale * values * py

    fill_xs = np.concatenate([xg, xs_top[::-1]])
    fill_ys = np.concatenate([yg, ys_top[::-1]])

    return fill_xs, fill_ys, xs_top, ys_top



# ============================================================
# REPORT N/V/M DATA EXPORT
# ============================================================
def _first_col_report(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _worst_combo_map_for_report(df_worst):
    """Return Element -> governing combination from an EC2 worst table."""
    if df_worst is None or getattr(df_worst, "empty", True):
        return {}
    ele_col = _first_col_report(df_worst, ["Element", "ElementID", "Member", "Member ID"])
    combo_col = _first_col_report(df_worst, ["Combination", "Combo", "Case", "Load_Combination"])
    eta_col = _first_col_report(df_worst, ["eta", "Static_EC2_Utilization", "Utilization", "Worst eta", "Max_Eta"])
    if ele_col is None or combo_col is None:
        return {}
    d = df_worst.copy()
    d[ele_col] = pd.to_numeric(d[ele_col], errors="coerce")
    if eta_col is not None:
        d[eta_col] = pd.to_numeric(d[eta_col], errors="coerce")
        d = d.sort_values(eta_col, ascending=False)
    out = {}
    for _, row in d.dropna(subset=[ele_col]).iterrows():
        try:
            eid = int(row[ele_col])
        except Exception:
            continue
        combo = str(row.get(combo_col, "")).strip()
        if combo and combo.lower() != "nan" and eid not in out:
            out[eid] = combo
    return out


def _nvm_result_to_dataframe(result):
    """Convert _nvm_arrays_for_element output to report-ready local station data."""
    if result is None:
        return None
    _xg, _yg, n_arr, v_arr, m_arr, L, _c, _s = result
    npts = len(n_arr)
    return pd.DataFrame({
        "x_m": np.linspace(0.0, float(L), npts),
        "N_kN": np.asarray(n_arr, dtype=float) / 1000.0,
        "V_kN": np.asarray(v_arr, dtype=float) / 1000.0,
        "M_kNm": np.asarray(m_arr, dtype=float) / 1000.0,
    })


def collect_member_force_diagrams_for_report(
    *,
    nodes,
    elements,
    combos,
    combo_names,
    Ec_Pa,
    Es_Pa,
    column_sec,
    beam_sec,
    geom,
    ops_build_model,
    ops_apply_combo_loads,
    ops_run_static,
    stage_specs,
    target_member_ids=None,
    npts=80,
):
    """Collect exact station-by-station N/V/M data for report figures.

    The normal interactive N/V/M plots already compute correct member-force
    diagrams from the current OpenSees element end forces and the applied
    distributed loads. This helper runs the same analysis stage-by-stage and
    stores those same station data so the report does not invent diagrams from
    a single EC2 worst-force ordinate.

    Parameters
    ----------
    stage_specs : dict
        Example::

            {
                "After_Walls": {
                    "df_worst": df_worst_after_sw,
                    "walls": optimized_walls,
                    "wall_tw_mm": optimized_wall_tw_mm,
                    "wall_E_Pa": optimized_wall_E_Pa,
                    "element_sections": None,
                },
                "Final": {
                    "df_worst": df_worst_final,
                    "walls": optimized_walls,
                    "wall_tw_mm": optimized_wall_tw_mm,
                    "wall_E_Pa": optimized_wall_E_Pa,
                    "element_sections": jacketed_element_sections,
                },
            }

    Returns
    -------
    dict
        {element_id: {stage_name: DataFrame(x_m, N_kN, V_kN, M_kNm)}}
    """
    target_set = None
    if target_member_ids is not None:
        target_set = {int(e) for e in target_member_ids if int(e) in elements}

    combo_name_set = set(str(c) for c in (combo_names or []))
    out = {}

    for stage_name, spec in (stage_specs or {}).items():
        if spec is None:
            continue
        df_worst = spec.get("df_worst")
        combo_map = _worst_combo_map_for_report(df_worst)
        if target_set is not None:
            combo_map = {eid: cname for eid, cname in combo_map.items() if eid in target_set}
        if not combo_map:
            continue

        # Group requested members by their governing combination, so each
        # combination is analyzed once per retrofit stage.
        members_by_combo = {}
        for eid, cname in combo_map.items():
            if cname in combos:
                members_by_combo.setdefault(cname, []).append(int(eid))
            elif cname in combo_name_set and cname in combos:
                members_by_combo.setdefault(cname, []).append(int(eid))

        for cname, eids in members_by_combo.items():
            loads_combo = combos[cname]["loads"]
            ids_ok = [e for e in sorted(set(eids)) if e in elements]
            env = nvm_envelope_for_combo(
                nodes, elements, loads_combo, Ec_Pa, Es_Pa, column_sec, beam_sec, geom, ops_build_model,
                ops_apply_combo_loads, ops_run_static, walls=spec.get("walls", None),
                wall_tw_mm=float(spec.get("wall_tw_mm", 200.0)), wall_E_Pa=float(spec.get("wall_E_Pa", 30e9)),
                element_sections=spec.get("element_sections", None), npts=npts, element_ids=ids_ok)
            if env is not None:
                for eid in ids_ok:
                    if eid in env:
                        out.setdefault(int(eid), {})[str(stage_name)] = envelope_to_dataframe(
                            env[eid], getattr(env, "governing", {}).get(eid))
                continue
            ops_build_model(
                nodes,
                elements,
                Ec_Pa,
                Es_Pa,
                column_sec,
                beam_sec,
                geom=geom,
                use_rigid_diaphragm=bool(spec.get("use_rigid_diaphragm", False)),
                walls=spec.get("walls", None),
                wall_tw_mm=float(spec.get("wall_tw_mm", 200.0)),
                wall_E_Pa=float(spec.get("wall_E_Pa", 30e9)),
                element_sections=spec.get("element_sections", None),
            )
            ops_apply_combo_loads(nodes, elements, loads_combo)
            ok = ops_run_static()
            if ok != 0:
                print(f"WARNING: report N/V/M extraction skipped {stage_name} / {cname}: analysis failed.")
                continue

            for eid in sorted(set(eids)):
                if eid not in elements:
                    continue
                data = _nvm_result_to_dataframe(
                    _nvm_arrays_for_element(eid, elements, loads_combo, nodes, npts=npts)
                )
                if data is None or data.empty:
                    continue
                out.setdefault(int(eid), {})[str(stage_name)] = data

    return out

def plot_structure_NVM(
    nodes,
    elements,
    loads_combo,
    combo_name="",
    npts=60,
    scale_N=None,
    scale_V=None,
    scale_M=None,
    diagram_size=0.16,
    label_size=7.5,
    geom=None,
    walls=None,
    env=None,
    value_labels=False,
):
    """N, V, M diagrams of one combination.

    The values are listed in a table beside the frame (member ID, start, end and,
    for M, maximum and minimum); the member IDs are written on the frame.
    value_labels=True writes the values on the diagrams instead (older layout).

    env: envelope of a result-level combination (Range Add / Envelope groups),
    {eid: (xg, yg, (Nmax, Nmin), (Vmax, Vmin), (Mmax, Mmin), L, c, s)};
    both curves are drawn, as in SAP2000: the maximum as a solid line and the
    minimum as a dashed line; the positive part of the maximum and the negative
    part of the minimum are filled (largest sagging / largest hogging). The
    table gives the maximum and the minimum at both ends and along the member.
    """
    from framekit.plot_style import (
        COLORS, draw_frame, draw_supports, draw_nodes, draw_walls,
        finish_frame_axes, subtitle,
    )
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    env_curves = {}
    if env:
        env_curves = {e: (d[2], d[3], d[4]) for e, d in env.items()}   # (hi, lo) of N, V, M
        from framekit.superposition import governing
        gv = getattr(env, "governing", None) or {}
        sets = [("", {e: (d[0], d[1], *(gv.get(e) or (governing(*d[2]), governing(*d[3]), governing(*d[4]))),
                          d[5], d[6], d[7]) for e, d in env.items()})]
    else:
        all_data = {}
        for eid in elements:
            result = _nvm_arrays_for_element(eid, elements, loads_combo, nodes, npts=npts)
            if result is not None:
                all_data[eid] = result
        sets = [("", all_data)]
    all_data = {}
    for _tag, _d in sets:
        for _e, _v in _d.items():
            all_data[(_tag, _e)] = _v

    if not all_data:
        print("❌ No element data available.")
        return

    xs_all = [p[0] for p in nodes.values()]
    ys_all = [p[1] for p in nodes.values()]
    Lref = max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all), 1.0)
    target = diagram_size * Lref

    max_N_kN = max(np.max(np.abs(d[2])) for d in all_data.values()) / 1000.0
    max_V_kN = max(np.max(np.abs(d[3])) for d in all_data.values()) / 1000.0
    max_M_kNm = max(np.max(np.abs(d[4])) for d in all_data.values()) / 1000.0

    if scale_N is None:
        scale_N = target / max(max_N_kN, 1e-9)
    if scale_V is None:
        scale_V = target / max(max_V_kN, 1e-9)
    if scale_M is None:
        scale_M = target / max(max_M_kNm, 1e-9)

    diagrams = [
        ("Axial force N", "kN", 2, scale_N, max_N_kN, ("tension (+)", "compression (−)")),
        ("Shear force V", "kN", 3, scale_V, max_V_kN, ("positive", "negative")),
        ("Bending moment M", "kNm", 4, scale_M, max_M_kNm,
         ("sagging (+), drawn on tension side", "hogging (−), drawn on tension side")),
    ]
    c_pos, c_neg = COLORS["pos"], COLORS["neg"]

    member_ids = sorted({eid for (_t, eid) in all_data}, key=lambda e: _member_sort_key(e, elements, nodes))
    blocks = [] if value_labels else _table_blocks(member_ids, elements)
    rmax = max((len(bl) for bl in blocks), default=0)
    fig_h = max(7.5, 0.20 * (rmax + 2) + 1.6)

    for fig_title, unit, idx_data, scale, vmax, (lab_pos, lab_neg) in diagrams:
        if value_labels:
            fig, ax = plt.subplots(figsize=(11, 7.5))
            tax = []
        else:
            ncol_t = (7 if idx_data == 4 else 5) if env_curves else (5 if idx_data == 4 else 3)
            bw = (0.56 if env_curves else 0.62) * ncol_t + 0.3
            fig = plt.figure(figsize=(10.5 + bw * len(blocks), fig_h))
            gs = fig.add_gridspec(1, 1 + len(blocks), width_ratios=[10.5] + [bw] * len(blocks), wspace=0.04)
            ax = fig.add_subplot(gs[0, 0])
            tax = [fig.add_subplot(gs[0, k + 1]) for k in range(len(blocks))]
        draw_walls(ax, nodes, geom, walls, label=False)
        draw_frame(ax, nodes, elements, color="#5C6770", lw_column=1.8, lw_beam=1.6, zorder=5)
        thr = 0.02 * max(vmax, 1e-12)

        for (set_tag, eid), data in all_data.items():
            xg, yg, N_arr, V_arr, M_arr, L, c, s = data
            values = data[idx_data] / 1000.0
            ts = idx_data == 4
            if eid in env_curves:
                # envelope: largest positive (from the max curve) and largest negative (from the
                # min curve) filled; max curve solid, min curve dashed
                hi, lo = (np.asarray(a, dtype=float) / 1000.0 for a in env_curves[eid][idx_data - 2])
                for part, colr in ((np.where(hi > 0, hi, 0.0), c_pos), (np.where(lo < 0, lo, 0.0), c_neg)):
                    if np.max(np.abs(part)) <= 0:
                        continue
                    fx, fy, _, _ = _diagram_offset(xg, yg, c, s, part, scale, tension_side=ts)
                    ax.fill(fx, fy, color=colr, alpha=0.28, lw=0, zorder=2)
                for curve, ls in ((hi, "-"), (lo, (0, (3.0, 2.0)))):
                    _, _, lx, ly = _diagram_offset(xg, yg, c, s, curve, scale, tension_side=ts)
                    ax.plot(lx, ly, color="#333333", lw=0.9, ls=ls, zorder=3)
                    for k in (0, -1):
                        ax.plot([xg[k], lx[k]], [yg[k], ly[k]], color="#333333", lw=0.7, ls=ls, zorder=3)
                if not value_labels:
                    ax.text(0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1]), str(eid), fontsize=6.5, ha="center",
                            va="center", color="#2D3748", zorder=8,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#A0AEC0", lw=0.5, alpha=0.95))
                    continue
                values = np.where(np.abs(hi) >= np.abs(lo), hi, lo)    # labels (older layout) only
            fill_xs, fill_ys, line_xs, line_ys = _diagram_offset(xg, yg, c, s, values, scale, tension_side=ts)
            if eid not in env_curves:
                # positive / negative parts filled separately
                vpos = np.where(values > 0, values, 0.0)
                vneg = np.where(values < 0, values, 0.0)
                for part, colr in ((vpos, c_pos), (vneg, c_neg)):
                    if np.max(np.abs(part)) <= 0:
                        continue
                    fx, fy, _, _ = _diagram_offset(xg, yg, c, s, part, scale, tension_side=ts)
                    ax.fill(fx, fy, color=colr, alpha=0.28, lw=0, zorder=2)
                ax.plot(line_xs, line_ys, color="#333333", lw=0.9, zorder=3)
                # closing ticks at member ends
                for k in (0, -1):
                    ax.plot([xg[k], line_xs[k]], [yg[k], line_ys[k]], color="#333333", lw=0.7, zorder=3)

            if not value_labels:     # member ID on the frame; the values go in the table
                ax.text(0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1]), str(eid), fontsize=6.5, ha="center",
                        va="center", color="#2D3748", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#A0AEC0", lw=0.5, alpha=0.95))
                continue

            # labels: peak value, plus end values for M; for a curved M diagram (load along
            # the member) also its maximum and minimum inside the member, marked with a dot
            idx_lab = {int(np.argmax(np.abs(values)))}
            inner = set()
            if idx_data == 4:
                idx_lab |= {0, len(values) - 1}
                imax, imin = int(np.argmax(values)), int(np.argmin(values))
                for ii, ok in ((imax, values[imax] > 0), (imin, values[imin] < 0)):   # peaks of |M| only
                    if ok and 0 < ii < len(values) - 1:
                        idx_lab.add(ii)
                        inner.add(ii)
            placed = []
            for ip in sorted(idx_lab, key=lambda i: (i not in inner, i)):
                v = values[ip]
                if abs(v) < thr:
                    continue
                if ip not in inner and any(abs(v - pv) < 0.02 * max(vmax, 1e-12)
                                           and abs(ip - pi) < len(values) * 0.15 for pi, pv in placed):
                    continue
                placed.append((ip, v))
                colr = c_pos if v > 0 else c_neg
                if ip in inner:
                    ax.plot(line_xs[ip], line_ys[ip], "o", ms=3.2, color=colr, zorder=6)
                    txt = ("max " if v == np.max(values) else "min ") + f"{v:.1f}"
                    dx, dy = _label_offset(c, s, v, ts, 0.018 * Lref)
                    ax.text(line_xs[ip] + dx, line_ys[ip] + dy, txt, fontsize=label_size, ha="center",
                            va="center", color=colr, fontweight="bold", zorder=7,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colr, lw=0.6, alpha=0.9))
                    continue
                ax.text(line_xs[ip], line_ys[ip], f"{v:.1f}", fontsize=label_size, ha="center", va="center",
                        color=colr, fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colr, lw=0.6, alpha=0.9))

        draw_nodes(ax, nodes, size=3.2)
        draw_supports(ax, nodes)
        finish_frame_axes(ax, nodes, pad=0.16)
        handles = [Patch(fc=c_pos, alpha=0.4, label=lab_pos), Patch(fc=c_neg, alpha=0.4, label=lab_neg)]
        if env_curves:
            handles += [Line2D([], [], color="#333333", lw=1.0, label="max"),
                        Line2D([], [], color="#333333", lw=1.0, ls=(0, (3.0, 2.0)), label="min")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles))
        ax.set_title(f"{fig_title} [{unit}]  —  {combo_name}")
        sub = f"max |{fig_title.split()[-1]}| = {vmax:.1f} {unit}"
        if env:
            sub += "   ·   Range Add / Envelope: max (solid) and min (dashed) at each section"
        if idx_data == 4 and value_labels:
            sub += "   ·   ● max / min inside a member"
        subtitle(ax, sub)
        if tax:
            vals = {}
            for (_t, eid), data in all_data.items():
                if eid in env_curves:      # envelope: (max, min) curves
                    v = tuple(np.asarray(a, dtype=float) / 1000.0 for a in env_curves[eid][idx_data - 2])
                else:
                    v = (np.asarray(data[idx_data], dtype=float) / 1000.0,)
                if _member_reversed(eid, elements, nodes):
                    v = tuple(a[::-1] for a in v)
                vals[eid] = v
            peak = {e: max(float(np.max(np.abs(a))) for a in v) for e, v in vals.items()}
            pmax = max(peak.values())
            top = {e for e, pk in peak.items() if pk >= pmax * (1 - 5e-4)}
            for k, (axt, bl) in enumerate(zip(tax, blocks)):
                _draw_member_table(axt, bl, vals, unit, idx_data == 4, top, rmax, c_pos, c_neg,
                                   note=(k == len(blocks) - 1), envelope=bool(env_curves))
        plt.show()


_TABLE_ROWS = 40
_TYPE_TITLES = {"column": "Columns", "beam": "Beams"}


def _member_sort_key(eid, elements, nodes):
    """Columns first (by storey, then from left), then beams (by floor, then from left)."""
    ni, nj, et = elements[eid]
    (x1, y1), (x2, y2) = nodes[ni], nodes[nj]
    et = str(et).lower()
    order = {"column": 0, "beam": 1}.get(et, 2)
    return (order, et, round(min(y1, y2), 3), round(min(x1, x2), 3), eid)


def _member_reversed(eid, elements, nodes):
    """True when the member runs top-to-bottom or right-to-left (table: start = bottom / left)."""
    ni, nj, _ = elements[eid]
    (x1, y1), (x2, y2) = nodes[ni], nodes[nj]
    return (y2 - y1) < -1e-9 or (abs(y2 - y1) <= 1e-9 and (x2 - x1) < 0)


def _table_blocks(member_ids, elements):
    """Rows of the value tables, split in blocks that fit beside the frame.

    A row is ("hdr", title) or ("row", eid). Small frames: one block; otherwise
    one block per member type, split every _TABLE_ROWS rows."""
    groups = []
    for eid in member_ids:
        et = str(elements[eid][2]).lower()
        if not groups or groups[-1][0] != et:
            groups.append((et, []))
        groups[-1][1].append(eid)
    total = sum(len(g) + 1 for _, g in groups)
    if total <= _TABLE_ROWS:
        rows = []
        for et, ids in groups:
            rows.append(("hdr", _TYPE_TITLES.get(et, et.capitalize())))
            rows += [("row", e) for e in ids]
        return [rows]
    blocks = []
    for et, ids in groups:
        title = _TYPE_TITLES.get(et, et.capitalize())
        n = _TABLE_ROWS - 1
        chunks = [ids[i:i + n] for i in range(0, len(ids), n)]
        for k, ch in enumerate(chunks):
            t = title if len(chunks) == 1 else f"{title} ({k + 1}/{len(chunks)})"
            blocks.append([("hdr", t)] + [("row", e) for e in ch])
    return blocks


def _draw_member_table(ax, rows, vals, unit, with_extremes, top_ids, rmax, c_pos, c_neg, note=False,
                       envelope=False):
    """Draw one value table in an empty axes.

    vals: {eid: (curve,)} for an ordinary combination -> ID, start, end [, max, min];
          {eid: (max curve, min curve)} for a Range Add / Envelope combination (envelope=True)
          -> ID, start max, start min, end max, end min [, max, min along the member].
    top_ids: members with the largest value (highlighted)."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if envelope:
        heads = ["ID", "max", "min", "max", "min"] + (["max", "min"] if with_extremes else [])
        groups = [("start", 1, 2), ("end", 3, 4)] + ([("member", 5, 6)] if with_extremes else [])
    else:
        heads = ["ID", "start", "end"] + (["max", "min"] if with_extremes else [])
        groups = []
    ncol = len(heads)
    xs = [0.10] + list(np.linspace(0.36 if ncol <= 5 else 0.27, 0.97, ncol - 1)) if ncol > 3 else [0.14, 0.58, 0.95]
    h = min(0.93 / (rmax + 2 + (1 if groups else 0)), 0.034)    # same row height in every block
    fh = ax.get_figure().get_size_inches()[1] * ax.get_position().height * 72 * h
    fs = float(np.clip(0.55 * fh, 6.0, 9.0))
    y = 0.985
    ax.text(0.0, y, f"values [{unit}]", fontsize=fs, ha="left", va="top", color="#4A5568", fontstyle="italic")
    y -= h
    if groups:
        step = xs[2] - xs[1]
        for label, k1, k2 in groups:
            xa, xb = xs[k1] - 0.8 * step, xs[k2]
            ax.text(0.5 * (xa + xb), y, label, fontsize=fs, ha="center", va="top", fontweight="bold", color="#1A202C")
            ax.plot([xa + 0.02, xb], [y - 0.85 * h, y - 0.85 * h], color="#A0AEC0", lw=0.6)
        y -= h
    for k, hd in enumerate(heads):
        ax.text(xs[k], y, hd, fontsize=fs, ha="center" if k == 0 else "right", va="top",
                fontweight="normal" if groups and k > 0 else "bold", color="#1A202C")
    y -= h
    ax.plot([0.0, 1.0], [y + 0.12 * h, y + 0.12 * h], color="#4A5568", lw=0.8)
    stripe = 0
    for kind, item in rows:
        if kind == "hdr":
            ax.text(0.0, y - 0.1 * h, item, fontsize=fs, ha="left", va="top", fontweight="bold", color="#2D3748")
            y -= h
            stripe = 0
            continue
        v = vals[item]
        if envelope and len(v) == 2:
            hi, lo = v
            nums = [hi[0], lo[0], hi[-1], lo[-1]] + ([float(np.max(hi)), float(np.min(lo))] if with_extremes else [])
        else:
            c = v[0]
            nums = [c[0], c[-1]] + ([float(np.max(c)), float(np.min(c))] if with_extremes else [])
        if item in top_ids:
            ax.add_patch(plt.Rectangle((0.0, y - h), 1.0, h, color="#FEEBC8", lw=0, zorder=0))
        elif stripe % 2:
            ax.add_patch(plt.Rectangle((0.0, y - h), 1.0, h, color="#F2F4F7", lw=0, zorder=0))
        stripe += 1
        bold = "bold" if item in top_ids else "normal"
        ax.text(xs[0], y - 0.15 * h, str(item), fontsize=fs, ha="center", va="top", color="#1A202C",
                fontweight=bold)
        for k, x in enumerate(nums, start=1):
            x = 0.0 if abs(x) < 0.05 else float(x)          # no "-0.0"
            col = c_pos if x > 0 else (c_neg if x < 0 else "#718096")
            ax.text(xs[k], y - 0.15 * h, f"{x:.1f}", fontsize=fs, ha="right", va="top", color=col,
                    fontweight=bold)
        y -= h
    if note:
        txt = "start = bottom of columns, left end of beams\nhighlighted: largest value"
        if envelope:
            txt += "\nmax / min: envelope of the combination at each section"
        ax.text(0.0, y - 0.4 * h, txt, fontsize=max(fs - 1.0, 6.0), ha="left", va="top", color="#718096")


def plot_all_combos_NVM(
    nodes,
    elements,
    combos,
    names,
    Ec_Pa,
    Es_Pa,
    column_sec,
    beam_sec,
    geom,
    ops_build_model,
    ops_apply_combo_loads,
    ops_run_static,
    walls=None,
    wall_tw_mm=200.0,
    wall_E_Pa=30e9,
    npts=60
):
    for cname in names:
        loads_combo = combos[cname]["loads"]
        env = nvm_envelope_for_combo(nodes, elements, loads_combo, Ec_Pa, Es_Pa, column_sec, beam_sec, geom,
                                     ops_build_model, ops_apply_combo_loads, ops_run_static, walls=walls,
                                     wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa, npts=npts)
        if env is not None:
            print(f"✔ Plotting N/V/M envelope for: {cname}")
            plot_structure_NVM(nodes, elements, loads_combo, combo_name=cname, npts=npts, geom=geom, walls=walls,
                               env=env)
            continue

        ops_build_model(
            nodes,
            elements,
            Ec_Pa,
            Es_Pa,
            column_sec,
            beam_sec,
            geom=geom,
            use_rigid_diaphragm=False,
            walls=walls,
            wall_tw_mm=wall_tw_mm,
            wall_E_Pa=wall_E_Pa
        )

        ops_apply_combo_loads(nodes, elements, loads_combo)
        ok = ops_run_static()

        if ok != 0:
            print(f"❌ Static analysis failed for {cname} — skipping.")
            continue

        print(f"✔ Plotting N/V/M for: {cname}")
        plot_structure_NVM(
            nodes,
            elements,
            loads_combo,
            combo_name=cname,
            npts=npts,
            geom=geom,
            walls=walls
        )


def nvm_envelope_for_combo(nodes, elements, loads_combo, Ec_Pa, Es_Pa, column_sec, beam_sec, geom,
                           ops_build_model, ops_apply_combo_loads, ops_run_static, walls=None, wall_tw_mm=200.0,
                           wall_E_Pa=30e9, element_sections=None, npts=60, element_ids=None):
    """Envelope N/V/M of a result-level combination, or None for an ordinary one."""
    from framekit.superposition import is_group_combo, nvm_by_case, nvm_envelope
    if not is_group_combo(loads_combo):
        return None

    def build():
        ops_build_model(nodes, elements, Ec_Pa, Es_Pa, column_sec, beam_sec, geom=geom, use_rigid_diaphragm=False,
                        walls=walls, wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa, element_sections=element_sections)

    per = nvm_by_case(loads_combo, elements, nodes, build, lambda L: ops_apply_combo_loads(nodes, elements, L),
                      ops_run_static, npts=npts, element_ids=element_ids)
    if per is None:
        return {}
    ids = list(element_ids) if element_ids is not None else list(elements)
    return nvm_envelope(loads_combo, {e: elements[e] for e in ids}, nodes, per)


def envelope_to_dataframe(env_entry, gov=None):
    """Report station data of an envelope: the governing curves (see superposition.governing_curve)."""
    from framekit.superposition import governing
    xg, yg, Nt, Vt, Mt, L, c, s = env_entry
    n = len(Nt[0])
    N, V, M = gov if gov is not None else (governing(*Nt), governing(*Vt), governing(*Mt))
    return pd.DataFrame({"x_m": np.linspace(0.0, float(L), n), "N_kN": np.asarray(N) / 1000.0,
                         "V_kN": np.asarray(V) / 1000.0, "M_kNm": np.asarray(M) / 1000.0})
