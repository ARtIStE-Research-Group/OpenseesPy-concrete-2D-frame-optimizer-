from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openseespy.opensees as ops


# ============================================================
# INPUT
# ============================================================
def ask_time_history_params() -> Dict:
    print("\n--- Linear Time-History Analysis Input ---")
    print("1) Single record")
    print("2) Multiple records")

    from framekit.input_helpers import ask_float_def, ask_int_def, ask_choice_def, ask_existing_file
    mode = ask_choice_def("Select option (1/2) [1]: ", ["1", "2"], "1")

    records = []

    if mode == "2":
        nrec = ask_int_def("Number of earthquake records: ", None, 1, 200)
        for i in range(nrec):
            print(f"\nRecord {i+1}/{nrec}")
            file_path = ask_existing_file("Acceleration file path (.txt / .csv): ")
            dt = ask_float_def("Time step dt of record [s]: ", None, 1e-5, 1.0)
            scale_factor = ask_float_def("Scale factor for acceleration [default 1.0]: ", 1.0)
            accel_unit = ask_choice_def("Acceleration unit in file [g / m/s2] (default g): ", ["g", "m/s2"], "g")

            records.append({
                "name": _record_name_from_path(file_path),
                "file_path": file_path,
                "dt": dt,
                "scale_factor": scale_factor,
                "accel_unit": accel_unit,
            })
    else:
        file_path = ask_existing_file("Acceleration file path (.txt / .csv): ")
        dt = ask_float_def("Time step dt of record [s]: ", None, 1e-5, 1.0)
        scale_factor = ask_float_def("Scale factor for acceleration [default 1.0]: ", 1.0)
        accel_unit = ask_choice_def("Acceleration unit in file [g / m/s2] (default g): ", ["g", "m/s2"], "g")

        records.append({
            "name": _record_name_from_path(file_path),
            "file_path": file_path,
            "dt": dt,
            "scale_factor": scale_factor,
            "accel_unit": accel_unit,
        })

    # 2D frame: horizontal excitation along the frame plane (X = DOF 1)
    direction = 1
    print("Excitation direction: X (in-plane horizontal).")

    xi = ask_float_def("Damping ratio ξ [default 0.05]: ", 0.05, 0.0, 0.5)
    mode_i = ask_int_def("Rayleigh damping mode i [default 1]: ", 1, 1, 500)
    mode_j = ask_int_def("Rayleigh damping mode j [default 3]: ", 3, 1, 500)

    dt_analysis = ask_float_def("Analysis time step [s] (Enter = use each record dt): ", 0.0, 0.0, 1.0)
    dt_analysis = dt_analysis if dt_analysis > 0 else None

    return {
        "records": records,
        "direction": direction,
        "damping_ratio": xi,
        "mode_i": mode_i,
        "mode_j": mode_j,
        "dt_analysis": dt_analysis,
    }


# ============================================================
# RECORD IO
# ============================================================
def _record_name_from_path(file_path: str) -> str:
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    return name if name else "record"


def load_acceleration_record(file_path: str) -> np.ndarray:
    file_path = str(file_path).strip().strip('"').strip("'")

    try:
        arr = np.loadtxt(file_path, dtype=float, delimiter=",")
    except Exception:
        arr = np.loadtxt(file_path, dtype=float)

    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 0:
        arr = np.array([float(arr)], dtype=float)
    elif arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            arr = arr.reshape(-1)

    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        raise ValueError(f"No valid acceleration values found in file: {file_path}")

    return arr


def convert_accel_to_mps2(accel: np.ndarray, unit: str) -> np.ndarray:
    unit = unit.strip().lower()
    if unit in ("g", "grav", "gravity"):
        return np.asarray(accel, dtype=float) * 9.81
    if unit in ("m/s2", "mps2", "ms2"):
        return np.asarray(accel, dtype=float)
    raise ValueError(f"Unsupported acceleration unit: {unit}")


def build_time_vector(n: int, dt: float) -> np.ndarray:
    return np.arange(n, dtype=float) * float(dt)


# ============================================================
# DAMPING
# ============================================================
def compute_rayleigh_coefficients_from_periods(
    periods: np.ndarray,
    damping_ratio: float = 0.05,
    mode_i: int = 1,
    mode_j: int = 3,
) -> Tuple[float, float]:
    periods = np.asarray(periods, dtype=float)
    valid = periods[np.isfinite(periods) & (periods > 0.0)]

    if valid.size == 0:
        raise RuntimeError("Cannot compute Rayleigh damping: no valid modal periods available.")

    mode_i = max(1, int(mode_i))
    mode_j = max(1, int(mode_j))

    if mode_i > valid.size:
        mode_i = valid.size
    if mode_j > valid.size:
        mode_j = valid.size

    if mode_i == mode_j:
        if mode_j < valid.size:
            mode_j += 1
        elif mode_i > 1:
            mode_i -= 1

    wi = 2.0 * math.pi / valid[mode_i - 1]
    wj = 2.0 * math.pi / valid[mode_j - 1]


    a0 = 2.0 * damping_ratio * wi * wj / (wi + wj)
    a1 = 2.0 * damping_ratio / (wi + wj)

    return float(a0), float(a1)


def apply_rayleigh_damping(periods: np.ndarray, damping_ratio: float, mode_i: int, mode_j: int):
    a0, a1 = compute_rayleigh_coefficients_from_periods(
        periods=periods,
        damping_ratio=damping_ratio,
        mode_i=mode_i,
        mode_j=mode_j,
    )
    ops.rayleigh(a0, 0.0, 0.0, a1)
    return a0, a1


# ============================================================
# NODE / ELEMENT HELPERS
# ============================================================
def _get_story_nodes_from_nodes(nodes, tol: float = 1e-9) -> List[Tuple[str, int, int]]:
    node_data = []
    for tag, val in nodes.items():
        x, y = float(val[0]), float(val[1])
        node_data.append((int(tag), x, y))

    if not node_data:
        return []

    ys = sorted(y for _, _, y in node_data)

    level_values = []
    for y in ys:
        if not level_values or abs(y - level_values[-1]) > tol:
            level_values.append(y)

    if len(level_values) < 2:
        return []

    levels = []
    for y_ref in level_values:
        level_nodes = [(tag, x, y) for tag, x, y in node_data if abs(y - y_ref) <= tol]
        level_nodes.sort(key=lambda r: (r[1], r[0]))
        levels.append((y_ref, level_nodes))

    pairs = []
    for i in range(1, len(levels)):
        lower_node = levels[i - 1][1][0][0]
        upper_node = levels[i][1][0][0]
        pairs.append((f"Story {i}", lower_node, upper_node))

    return pairs


def _get_node_tags(nodes) -> List[int]:
    return sorted(int(k) for k in nodes.keys())


def _get_ele_tags(elements) -> List[int]:
    return sorted(int(k) for k in elements.keys())


def _compute_base_shear(direction: int) -> float:
    ops.reactions()
    V = 0.0
    for nid in ops.getNodeTags():
        if abs(ops.nodeCoord(nid, 2)) > 1e-9:
            continue
        try:
            V += float(ops.nodeReaction(nid, direction))
        except Exception:
            pass
    return V   # signed: history plots show both directions; peaks use |V|


# ============================================================
# PEAK EXTRACTION
# ============================================================
def _update_peak_node_disp(peak_node_disp: Dict[int, np.ndarray], node_tags: List[int], ndf: int):
    for node in node_tags:
        vals = np.array([ops.nodeDisp(node, dof + 1) for dof in range(ndf)], dtype=float)
        if node not in peak_node_disp:
            peak_node_disp[node] = np.abs(vals)
        else:
            peak_node_disp[node] = np.maximum(peak_node_disp[node], np.abs(vals))


def _update_peak_element_forces(peak_element_forces: Dict[int, np.ndarray], ele_tags: List[int]):
    for ele in ele_tags:
        try:
            vals = np.array(ops.eleResponse(ele, "localForce"), dtype=float)
        except Exception:
            vals = np.array([], dtype=float)

        if ele not in peak_element_forces:
            peak_element_forces[ele] = np.abs(vals)
        else:
            if len(vals) == 0:
                continue
            if len(peak_element_forces[ele]) == 0:
                peak_element_forces[ele] = np.abs(vals)
            else:
                peak_element_forces[ele] = np.maximum(peak_element_forces[ele], np.abs(vals))


def _update_peak_story_drifts(
    peak_story_drifts: Dict[str, float],
    story_nodes: List[Tuple[str, int, int]],
    direction_index_zero_based: int = 0,
):
    for story_name, n_bot, n_top in story_nodes:
        u_bot = float(ops.nodeDisp(n_bot, direction_index_zero_based + 1))
        u_top = float(ops.nodeDisp(n_top, direction_index_zero_based + 1))
        drift = abs(u_top - u_bot)
        peak_story_drifts[story_name] = max(peak_story_drifts.get(story_name, 0.0), drift)


# ============================================================
# MAIN THA
# ============================================================
def run_linear_time_history_analysis(
    geom,
    nodes,
    elements,
    periods,
    accel_series_mps2: np.ndarray,
    dt_record: float,
    direction: int = 1,
    damping_ratio: float = 0.05,
    mode_i: int = 1,
    mode_j: int = 3,
    dt_analysis: Optional[float] = None,
    g_const: float = 9.81,
) -> Dict:
    del geom
    del g_const

    accel_series_mps2 = np.asarray(accel_series_mps2, dtype=float)
    if accel_series_mps2.size == 0:
        raise ValueError("Empty acceleration series.")

    if dt_analysis is None:
        dt_analysis = float(dt_record)

    node_tags = _get_node_tags(nodes)
    ele_tags = _get_ele_tags(elements)
    story_nodes = _get_story_nodes_from_nodes(nodes)

    try:
        ndf = len(ops.nodeDisp(node_tags[0]))
    except Exception:
        ndf = 3

    ops.wipeAnalysis()
    ops.reset()
    ops.setTime(0.0)

    ts_tag = 1001
    pat_tag = 1001

    try:
        ops.remove("loadPattern", pat_tag)
    except Exception:
        pass

    try:
        ops.remove("timeSeries", ts_tag)
    except Exception:
        pass

    ops.timeSeries("Path", ts_tag, "-dt", float(dt_record), "-values", *accel_series_mps2.tolist())
    ops.pattern("UniformExcitation", pat_tag, direction, "-accel", ts_tag)

    a0, a1 = apply_rayleigh_damping(periods, damping_ratio, mode_i, mode_j)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 50)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    n_steps = int(math.ceil(len(accel_series_mps2) * float(dt_record) / float(dt_analysis) - 1e-9))
    time = []
    base_shear_hist = []
    roof_disp_hist = []

    peak_node_disp = {}
    peak_element_forces = {}
    peak_story_drifts = {}

    roof_node = max(nodes.keys(), key=lambda nid: (nodes[nid][1], nodes[nid][0]))

    ok_final = 0
    failed_step = None

    for i in range(n_steps):
        ok = ops.analyze(1, float(dt_analysis))
        if ok != 0:
            ok_final = ok
            failed_step = i
            break

        t = ops.getTime()
        time.append(float(t))
        roof_disp_hist.append(float(ops.nodeDisp(roof_node, direction)))
        base_shear_hist.append(_compute_base_shear(direction))

        _update_peak_node_disp(peak_node_disp, node_tags, ndf)
        _update_peak_element_forces(peak_element_forces, ele_tags)
        _update_peak_story_drifts(peak_story_drifts, story_nodes, direction_index_zero_based=direction - 1)

    return {
        "ok": ok_final == 0,
        "error_code": ok_final,
        "failed_step": failed_step,
        "time": np.array(time, dtype=float),
        "roof_disp_history": np.array(roof_disp_hist, dtype=float),
        "base_shear_history": np.array(base_shear_hist, dtype=float),
        "peak_story_drifts": peak_story_drifts,
        "peak_node_displacements": peak_node_disp,
        "peak_element_forces": peak_element_forces,
        "rayleigh_a0": a0,
        "rayleigh_a1": a1,
        "direction": direction,
        "dt_record": float(dt_record),
        "dt_analysis": float(dt_analysis),
        "n_steps_record": n_steps,
        "n_steps_completed": len(time),
    }


def run_multiple_time_histories(
    geom,
    nodes,
    elements,
    periods,
    records: List[Dict],
    direction: int = 1,
    damping_ratio: float = 0.05,
    mode_i: int = 1,
    mode_j: int = 3,
    dt_analysis: Optional[float] = None,
) -> Dict:
    results_by_record = {}
    summary_rows = []
    all_peak_drift_rows = []

    env_story_drifts = {}

    for rec in records:
        accel_raw = load_acceleration_record(rec["file_path"])
        accel_mps2 = convert_accel_to_mps2(accel_raw, rec["accel_unit"])
        accel_mps2 = float(rec["scale_factor"]) * accel_mps2

        res = run_linear_time_history_analysis(
            geom=geom,
            nodes=nodes,
            elements=elements,
            periods=periods,
            accel_series_mps2=accel_mps2,
            dt_record=rec["dt"],
            direction=direction,
            damping_ratio=damping_ratio,
            mode_i=mode_i,
            mode_j=mode_j,
            dt_analysis=dt_analysis,
        )

        rec_name = rec["name"]
        results_by_record[rec_name] = res

        peak_roof = float(np.max(np.abs(res["roof_disp_history"]))) if len(res["roof_disp_history"]) else np.nan
        peak_base = float(np.max(np.abs(res["base_shear_history"]))) if len(res["base_shear_history"]) else np.nan

        summary_rows.append({
            "Record": rec_name,
            "OK": res["ok"],
            "Error code": res["error_code"],
            "Failed step": res["failed_step"],
            "Completed steps": res["n_steps_completed"],
            "Record steps": res["n_steps_record"],
            "Peak roof disp [m]": peak_roof,
            "Peak base shear [N]": peak_base,
        })

        df_rec = th_peak_drifts_to_dataframe(res["peak_story_drifts"], case_name=rec_name)
        df_rec.insert(0, "Record", rec_name)
        all_peak_drift_rows.append(df_rec)

        for story, drift in res["peak_story_drifts"].items():
            env_story_drifts[story] = max(env_story_drifts.get(story, 0.0), float(drift))

    df_summary = pd.DataFrame(summary_rows)

    if all_peak_drift_rows:
        df_peak_drifts_all = pd.concat(all_peak_drift_rows, ignore_index=True)
    else:
        df_peak_drifts_all = pd.DataFrame(columns=["Record", "Case", "Story", "Peak Drift [m]", "Peak Drift [mm]"])

    df_envelope = th_peak_drifts_to_dataframe(env_story_drifts, case_name="TH_ENVELOPE")
    df_envelope.insert(0, "Record", "ENVELOPE")

    return {
        "results_by_record": results_by_record,
        "summary": df_summary,
        "peak_drifts_all": df_peak_drifts_all,
        "envelope_story_drifts": env_story_drifts,
        "envelope_drifts_df": df_envelope,
    }


# ============================================================
# DATAFRAMES
# ============================================================
def th_peak_drifts_to_dataframe(peak_story_drifts: Dict[str, float], case_name: str = "TH") -> pd.DataFrame:
    rows = []
    for story, drift_m in peak_story_drifts.items():
        rows.append({
            "Case": case_name,
            "Story": story,
            "Peak Drift [m]": drift_m,
            "Peak Drift [mm]": drift_m * 1000.0,
        })
    return pd.DataFrame(rows)


def th_history_to_dataframe(time: np.ndarray, values: np.ndarray, value_name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Time [s]": np.asarray(time, dtype=float),
        value_name: np.asarray(values, dtype=float),
    })


def _node_disp_to_dataframe(node_disp: Dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    for node, vec in node_disp.items():
        row = {"Node": node}
        for i, val in enumerate(vec, start=1):
            row[f"U{i}_peak"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _element_forces_to_dataframe(element_forces: Dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    for ele, vec in element_forces.items():
        row = {"Element": ele}
        for i, val in enumerate(vec, start=1):
            row[f"F{i}_peak"] = val
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================
def plot_time_history_response(time: np.ndarray, values: np.ndarray, ylabel: str, title: str):
    from framekit.plot_style import COLORS
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    n = min(len(t), len(v))
    t, v = t[:n], v[:n]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.fill_between(t, 0, v, color=COLORS["beam"], alpha=0.12, lw=0)
    ax.plot(t, v, color=COLORS["column"], lw=1.1)
    if n:
        k = int(np.nanargmax(np.abs(v)))
        ax.plot(t[k], v[k], "o", ms=7, mfc="white", mec=COLORS["deformed"], mew=1.8, zorder=5)
        ax.annotate(f"peak {v[k]:.4g} at t = {t[k]:.2f} s", (t[k], v[k]), xytext=(10, 8 if v[k] >= 0 else -14),
                    textcoords="offset points", fontsize=8.5, color=COLORS["deformed"], fontweight="bold")
        ax.set_xlim(t[0], t[-1])
        vm = np.nanmax(np.abs(v)) * 1.18 or 1.0
        ax.set_ylim(-vm, vm)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()


# ============================================================
# EXPORT
# ============================================================
def export_time_history_results(results: Dict, filename: str = "time_history_results.xlsx"):
    df_drifts = th_peak_drifts_to_dataframe(results["peak_story_drifts"], case_name="TH")
    df_roof = th_history_to_dataframe(results["time"], results["roof_disp_history"], "Roof displacement [m]")
    df_base = th_history_to_dataframe(results["time"], results["base_shear_history"], "Base shear [N]")
    df_nodes = _node_disp_to_dataframe(results["peak_node_displacements"])
    df_elem = _element_forces_to_dataframe(results["peak_element_forces"])

    df_info = pd.DataFrame([{
        "OK": results["ok"],
        "Error code": results["error_code"],
        "Failed step": results["failed_step"],
        "Direction": results["direction"],
        "dt record [s]": results["dt_record"],
        "dt analysis [s]": results["dt_analysis"],
        "Record steps": results["n_steps_record"],
        "Completed steps": results["n_steps_completed"],
        "Rayleigh a0": results["rayleigh_a0"],
        "Rayleigh a1": results["rayleigh_a1"],
    }])

    try:
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            df_info.to_excel(writer, sheet_name="Info", index=False)
            df_drifts.to_excel(writer, sheet_name="Peak_Drifts", index=False)
            df_roof.to_excel(writer, sheet_name="Roof_History", index=False)
            df_base.to_excel(writer, sheet_name="BaseShear_History", index=False)
            df_nodes.to_excel(writer, sheet_name="Peak_Node_Disp", index=False)
            df_elem.to_excel(writer, sheet_name="Peak_Elem_Forces", index=False)
        print(f"✔ Time-history results exported: {filename}")
    except ModuleNotFoundError:
        csv_base = filename.rsplit(".", 1)[0]
        df_info.to_csv(csv_base + "_info.csv", index=False)
        df_drifts.to_csv(csv_base + "_peak_drifts.csv", index=False)
        df_roof.to_csv(csv_base + "_roof_history.csv", index=False)
        df_base.to_csv(csv_base + "_base_history.csv", index=False)
        df_nodes.to_csv(csv_base + "_peak_node_disp.csv", index=False)
        df_elem.to_csv(csv_base + "_peak_elem_forces.csv", index=False)
        print(f"✔ Time-history results exported as CSV files with prefix: {csv_base}")


def export_multiple_time_history_results(batch_results: Dict, filename: str = "time_history_batch_results.xlsx"):
    df_summary = batch_results["summary"]
    df_peak = batch_results["peak_drifts_all"]
    df_env = batch_results["envelope_drifts_df"]

    try:
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
            df_peak.to_excel(writer, sheet_name="Peak_Drifts_All", index=False)
            df_env.to_excel(writer, sheet_name="Envelope_Drifts", index=False)

            for rec_name, res in batch_results["results_by_record"].items():
                sheet = rec_name[:24] if rec_name else "record"
                th_history_to_dataframe(res["time"], res["roof_disp_history"], "Roof displacement [m]").to_excel(
                    writer, sheet_name=f"{sheet}_roof"[:31], index=False
                )
                th_history_to_dataframe(res["time"], res["base_shear_history"], "Base shear [N]").to_excel(
                    writer, sheet_name=f"{sheet}_base"[:31], index=False
                )

        print(f"✔ Batch time-history results exported: {filename}")

    except ModuleNotFoundError:
        base = filename.rsplit(".", 1)[0]
        df_summary.to_csv(base + "_summary.csv", index=False)
        df_peak.to_csv(base + "_peak_drifts_all.csv", index=False)
        df_env.to_csv(base + "_envelope_drifts.csv", index=False)
        print(f"✔ Batch time-history results exported as CSV files with prefix: {base}")