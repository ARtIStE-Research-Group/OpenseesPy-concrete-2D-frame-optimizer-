import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from framekit.opensees_model import ops_ele_end_forces_local, _elem_cos_sin
from framekit.units import _kNpm_to_Npm


def _nvm_arrays_for_element(eid, elements, loads_combo, nodes, npts=60):
    ni, nj, _ = elements[eid]
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
):
    from framekit.plot_style import (
        COLORS, draw_frame, draw_supports, draw_nodes, draw_walls,
        finish_frame_axes, subtitle,
    )
    from matplotlib.patches import Patch

    all_data = {}
    for eid in elements:
        result = _nvm_arrays_for_element(eid, elements, loads_combo, nodes, npts=npts)
        if result is not None:
            all_data[eid] = result

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

    for fig_title, unit, idx_data, scale, vmax, (lab_pos, lab_neg) in diagrams:
        fig, ax = plt.subplots(figsize=(11, 7.5))
        draw_walls(ax, nodes, geom, walls, label=False)
        draw_frame(ax, nodes, elements, color="#5C6770", lw_column=1.8, lw_beam=1.6, zorder=5)
        thr = 0.02 * max(vmax, 1e-12)

        for eid, data in all_data.items():
            xg, yg, N_arr, V_arr, M_arr, L, c, s = data
            values = data[idx_data] / 1000.0
            ts = idx_data == 4
            fill_xs, fill_ys, line_xs, line_ys = _diagram_offset(xg, yg, c, s, values, scale, tension_side=ts)

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

            # labels: peak value, plus end values for M
            idx_lab = {int(np.argmax(np.abs(values)))}
            if idx_data == 4:
                idx_lab |= {0, len(values) - 1}
            placed = []
            for ip in sorted(idx_lab):
                v = values[ip]
                if abs(v) < thr:
                    continue
                if any(abs(v - pv) < 0.02 * max(vmax, 1e-12) and abs(ip - pi) < len(values) * 0.15 for pi, pv in placed):
                    continue
                placed.append((ip, v))
                colr = c_pos if v > 0 else c_neg
                ax.text(line_xs[ip], line_ys[ip], f"{v:.1f}", fontsize=label_size, ha="center", va="center",
                        color=colr, fontweight="bold", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=colr, lw=0.6, alpha=0.9))

        draw_nodes(ax, nodes, size=3.2)
        draw_supports(ax, nodes)
        finish_frame_axes(ax, nodes, pad=0.16)
        ax.legend(handles=[Patch(fc=c_pos, alpha=0.4, label=lab_pos), Patch(fc=c_neg, alpha=0.4, label=lab_neg)],
                  loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
        ax.set_title(f"{fig_title} [{unit}]  —  {combo_name}")
        subtitle(ax, f"max |{fig_title.split()[-1]}| = {vmax:.1f} {unit}")
        plt.show()


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