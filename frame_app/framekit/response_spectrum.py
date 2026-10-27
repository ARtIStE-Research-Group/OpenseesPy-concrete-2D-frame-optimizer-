from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openseespy.opensees as ops


G = 9.81  # m/s2


@dataclass
class EC8SpectrumParams:
    ag: float
    soil_class: str
    spectrum_type: int = 1
    damping_ratio: float = 0.05
    direction: int = 1
    importance_factor: float = 1.0
    combo_method: str = "CQC"


@dataclass
class RSAConfig:
    modal_combination: str = "CQC"
    modes_to_include: Optional[int] = None
    verbose: bool = True


EC8_SOIL_TABLE = {
    1: {
        "A": {"S": 1.00, "TB": 0.15, "TC": 0.40, "TD": 2.00},
        "B": {"S": 1.20, "TB": 0.15, "TC": 0.50, "TD": 2.00},
        "C": {"S": 1.15, "TB": 0.20, "TC": 0.60, "TD": 2.00},
        "D": {"S": 1.35, "TB": 0.20, "TC": 0.80, "TD": 2.00},
        "E": {"S": 1.40, "TB": 0.15, "TC": 0.50, "TD": 2.00},
    },
    2: {
        "A": {"S": 1.00, "TB": 0.05, "TC": 0.25, "TD": 1.20},
        "B": {"S": 1.35, "TB": 0.05, "TC": 0.25, "TD": 1.20},
        "C": {"S": 1.50, "TB": 0.10, "TC": 0.25, "TD": 1.20},
        "D": {"S": 1.80, "TB": 0.10, "TC": 0.30, "TD": 1.20},
        "E": {"S": 1.60, "TB": 0.05, "TC": 0.25, "TD": 1.20},
    },
}


def damping_correction_eta(xi: float) -> float:
    xi_pct = 100.0 * xi
    eta = math.sqrt(10.0 / (5.0 + xi_pct))
    return max(eta, 0.55)


def ec8_elastic_spectrum_ordinate(T: float, params: EC8SpectrumParams) -> float:
    soil = EC8_SOIL_TABLE[params.spectrum_type][params.soil_class.upper()]
    S = soil["S"]
    TB = soil["TB"]
    TC = soil["TC"]
    TD = soil["TD"]

    ag = params.ag * params.importance_factor
    eta = damping_correction_eta(params.damping_ratio)

    if T < 0.0:
        raise ValueError("Period cannot be negative.")

    if T <= TB:
        return ag * S * (1.0 + (T / TB) * (eta * 2.5 - 1.0))
    elif T <= TC:
        return ag * S * eta * 2.5
    elif T <= TD:
        return ag * S * eta * 2.5 * (TC / T)
    else:
        return ag * S * eta * 2.5 * (TC * TD / (T * T))


def build_ec8_spectrum(periods: np.ndarray, params: EC8SpectrumParams) -> np.ndarray:
    return np.array(
        [ec8_elastic_spectrum_ordinate(float(T), params) for T in periods],
        dtype=float
    )


def ask_ec8_spectrum_params() -> Dict:
    print("\n--- EC8 Response Spectrum Input ---")

    from framekit.input_helpers import ask_float_def, ask_int_def, ask_choice_def
    ag_g = ask_float_def("Design ground acceleration ag/g [e.g. 0.35]: ", None, 0.0, 3.0)
    soil_class = ask_choice_def("Soil class [A/B/C/D/E]: ", ["A", "B", "C", "D", "E"]).upper()
    spectrum_type = ask_int_def("Spectrum type [1 or 2] (default 1): ", 1, 1, 2)
    damping_ratio = ask_float_def("Damping ratio ξ as decimal [default 0.05]: ", 0.05, 0.0, 0.5)
    # 2D frame: the horizontal spectrum acts along the frame plane (X = DOF 1)
    direction = 1
    print("Direction: X (in-plane horizontal) — the only horizontal direction of a 2D frame.")
    importance_factor = ask_float_def("Importance factor γI [default 1.0]: ", 1.0, 0.1, 3.0)
    combo_method = ask_choice_def("Modal combination [CQC/SRSS] (default CQC): ", ["CQC", "SRSS"], "CQC").upper()

    return {
        "ag": ag_g * G,
        "soil_class": soil_class,
        "spectrum_type": spectrum_type,
        "damping_ratio": damping_ratio,
        "direction": direction,
        "importance_factor": importance_factor,
        "combo_method": combo_method,
    }


def _params_from_dict(rsa_params: Dict) -> EC8SpectrumParams:
    return EC8SpectrumParams(
        ag=rsa_params["ag"],
        soil_class=rsa_params["soil_class"],
        spectrum_type=rsa_params["spectrum_type"],
        damping_ratio=rsa_params["damping_ratio"],
        direction=rsa_params["direction"],
        importance_factor=rsa_params["importance_factor"],
        combo_method=rsa_params["combo_method"],
    )


def build_ec8_spectrum_for_periods(periods: np.ndarray, rsa_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    params = _params_from_dict(rsa_params)
    periods = np.asarray(periods, dtype=float)
    valid_mask = np.isfinite(periods) & (periods > 0.0)
    periods = periods[valid_mask]
    Sa = build_ec8_spectrum(periods, params)
    return periods, Sa


def build_ec8_plot_spectrum(
    rsa_params: Dict,
    t_max: Optional[float] = None,
    npts: int = 600
) -> Tuple[np.ndarray, np.ndarray]:
    params = _params_from_dict(rsa_params)
    soil = EC8_SOIL_TABLE[params.spectrum_type][params.soil_class.upper()]
    TD = soil["TD"]

    if t_max is None:
        t_max = max(4.0, 1.5 * TD)

    T = np.linspace(0.0, t_max, npts)
    Sa = build_ec8_spectrum(T, params)
    return T, Sa


def plot_ec8_spectrum(T: np.ndarray, Sa: np.ndarray, title: str = "EC8 Response Spectrum",
                      periods=None, rsa_params: Dict = None, Sd=None, q=None):
    from framekit.plot_style import COLORS
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.fill_between(T, 0, Sa, color=COLORS["beam"], alpha=0.10, lw=0)
    ax.plot(T, Sa, color=COLORS["column"], lw=2.2, label="Elastic spectrum Se(T)")
    S_marks = Sa
    if Sd is not None:
        Sd = np.asarray(Sd, dtype=float)
        ax.fill_between(T, 0, Sd, color=COLORS["deformed"], alpha=0.10, lw=0)
        ax.plot(T, Sd, color=COLORS["deformed"], lw=2.2, ls="-",
                label=f"Design spectrum Sd(T)" + (f", q = {q:.2f}" if q else ""))
        S_marks = Sd
    sub = []
    if rsa_params is not None:
        try:
            prm = _params_from_dict(rsa_params)
            soil = EC8_SOIL_TABLE[prm.spectrum_type][prm.soil_class.upper()]
            for key in ("TB", "TC", "TD"):
                ax.axvline(soil[key], color="#AAAAAA", lw=0.9, ls="--", zorder=0)
                ax.text(soil[key], max(Sa) * 1.04, key, ha="center",
                        va="bottom", fontsize=8.5, color=COLORS["muted"])
            sub.append(f"Type {prm.spectrum_type} · ground {prm.soil_class.upper()} · "
                       f"ag = {prm.ag / G:.3f} g · ξ = {100 * prm.damping_ratio:.0f}% · γI = {prm.importance_factor:.2f}")
        except Exception:
            pass
    if periods is not None and len(periods):
        Tm = np.asarray(periods, dtype=float)
        Tm = Tm[np.isfinite(Tm) & (Tm > 0)]
        Sm = np.interp(Tm, T, S_marks)
        ax.plot(Tm, Sm, "o", ms=7, mfc="white", mec="#222222", mew=1.6, zorder=5, label="Structure modes")
        for k, (t, sa) in enumerate(zip(Tm, Sm), start=1):
            if k <= 3:
                ax.annotate(f"T{k} = {t:.3f} s", (t, sa), xytext=(8, 10 + 14 * (k - 1)), textcoords="offset points",
                            fontsize=8.5, color="#222222",
                            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6))
    ax.set_xlim(0, T.max())
    ax.set_ylim(0, max(Sa) * 1.12)
    ax.set_xlabel("Period T [s]")
    ax.set_ylabel("Spectral acceleration Sa [m/s²]")
    sec = ax.secondary_yaxis("right", functions=(lambda x: x / G, lambda x: x * G))
    sec.set_ylabel("Sa [g]")
    ax.legend(loc="upper right", bbox_to_anchor=(0.97, 0.93))
    ax.set_title(title)
    if sub:
        from framekit.plot_style import subtitle
        subtitle(ax, sub[0])
    plt.show()


def cqc_correlation(beta: float, xi: float) -> float:
    num = 8.0 * xi * xi * (1.0 + beta) * beta ** 1.5
    den = (1.0 - beta * beta) ** 2 + 4.0 * xi * xi * beta * (1.0 + beta) ** 2
    if abs(den) < 1e-14:
        return 1.0
    return num / den


def combine_modal_response(
    modal_values: np.ndarray,
    omegas: np.ndarray,
    method: str = "CQC",
    damping_ratio: float = 0.05,
) -> float:
    method = method.upper()

    if method == "SRSS":
        return float(np.sqrt(np.sum(modal_values ** 2)))

    if method != "CQC":
        raise ValueError(f"Unsupported modal combination method: {method}")

    total = 0.0
    n = len(modal_values)

    for i in range(n):
        wi = omegas[i]
        for j in range(n):
            wj = omegas[j]
            beta = wj / wi
            rho = cqc_correlation(beta, damping_ratio)
            total += rho * modal_values[i] * modal_values[j]

    return float(np.sqrt(max(total, 0.0)))


def _guess_ndf_from_model() -> int:
    try:
        tags = ops.getNodeTags()
        if not tags:
            return 3
        return len(ops.nodeDisp(tags[0]))
    except Exception:
        return 3


def _iter_node_items(nodes):
    if isinstance(nodes, dict):
        for tag, val in nodes.items():
            yield int(tag), val
    else:
        for item in nodes:
            if isinstance(item, dict):
                yield int(item["id"]), item
            else:
                yield int(getattr(item, "id")), item


def _node_xy_from_value(val):
    if isinstance(val, dict):
        if "x" in val and "y" in val:
            return float(val["x"]), float(val["y"])
        if "X" in val and "Y" in val:
            return float(val["X"]), float(val["Y"])
        if "coord" in val and len(val["coord"]) >= 2:
            return float(val["coord"][0]), float(val["coord"][1])

    if hasattr(val, "x") and hasattr(val, "y"):
        return float(val.x), float(val.y)

    if hasattr(val, "X") and hasattr(val, "Y"):
        return float(val.X), float(val.Y)

    if isinstance(val, (tuple, list)) and len(val) >= 2:
        return float(val[0]), float(val[1])

    raise ValueError(f"Cannot extract node coordinates from node value: {val}")


def _get_story_nodes_from_nodes(nodes, tol: float = 1e-9) -> List[Tuple[str, int, int]]:
    node_data = []
    for tag, val in _iter_node_items(nodes):
        x, y = _node_xy_from_value(val)
        node_data.append((tag, x, y))

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
        lower_level_nodes = levels[i - 1][1]
        upper_level_nodes = levels[i][1]

        lower_node = lower_level_nodes[0][0]
        upper_node = upper_level_nodes[0][0]

        pairs.append((f"Story {i}", lower_node, upper_node))

    return pairs


def _get_node_tags(nodes) -> List[int]:
    if isinstance(nodes, dict):
        return sorted(int(k) for k in nodes.keys())
    return sorted(int(n["id"]) if isinstance(n, dict) else int(getattr(n, "id")) for n in nodes)


def _get_ele_tags(elements) -> List[int]:
    if isinstance(elements, dict):
        return sorted(int(k) for k in elements.keys())
    return sorted(int(e["id"]) if isinstance(e, dict) else int(getattr(e, "id")) for e in elements)


def _node_disp_vector(node_tags: List[int], ndf: int) -> Dict[int, np.ndarray]:
    out = {}
    for node in node_tags:
        vals = [ops.nodeDisp(node, dof + 1) for dof in range(ndf)]
        out[node] = np.array(vals, dtype=float)
    return out


def _element_force_vector(ele_tags: List[int]) -> Dict[int, np.ndarray]:
    out = {}
    for ele in ele_tags:
        try:
            vals = ops.eleResponse(ele, "localForce")
            out[ele] = np.array(vals, dtype=float)
        except Exception:
            out[ele] = np.array([], dtype=float)
    return out


def _compute_base_shear(node_tags, direction):
    ops.reactions()

    V = 0.0
    for nid in node_tags:
        if abs(ops.nodeCoord(nid, 2)) > 1e-9:
            continue
        try:
            rxn = ops.nodeReaction(nid, direction)
            V += rxn
        except Exception:
            continue

    return V


def _combine_dict_of_vectors(
    modal_dict: Dict[int, Dict[int, np.ndarray]],
    omegas: np.ndarray,
    method: str,
    damping_ratio: float,
) -> Dict[int, np.ndarray]:
    first_mode = next(iter(modal_dict))
    tags = list(modal_dict[first_mode].keys())

    combined = {}
    for tag in tags:
        vec_len = len(modal_dict[first_mode][tag])

        if vec_len == 0:
            combined[tag] = np.array([], dtype=float)
            continue

        arr = np.zeros(vec_len, dtype=float)
        for k in range(vec_len):
            modal_values = np.array(
                [modal_dict[m][tag][k] for m in sorted(modal_dict)],
                dtype=float
            )
            arr[k] = combine_modal_response(
                modal_values,
                omegas,
                method=method,
                damping_ratio=damping_ratio
            )
        combined[tag] = arr

    return combined


def compute_story_drifts_from_displacements(
    combined_node_disp: Dict[int, np.ndarray],
    story_nodes: List[Tuple[str, int, int]],
    direction_index_zero_based: int = 0,
) -> Dict[str, float]:
    drifts = {}
    for story_name, n_bot, n_top in story_nodes:
        u_bot = combined_node_disp[n_bot][direction_index_zero_based]
        u_top = combined_node_disp[n_top][direction_index_zero_based]
        drifts[story_name] = abs(u_top - u_bot)
    return drifts


def _prepare_modal_data(periods, spectrum_Sa, num_modes=None):
    periods = np.asarray(periods, dtype=float)
    spectrum_Sa = np.asarray(spectrum_Sa, dtype=float)

    valid_mask = np.isfinite(periods) & (periods > 0.0) & np.isfinite(spectrum_Sa)
    periods = periods[valid_mask]
    spectrum_Sa = spectrum_Sa[valid_mask]

    if periods.size == 0:
        raise RuntimeError("No valid positive finite modal periods available for response spectrum analysis.")

    if num_modes is not None:
        periods = periods[:num_modes]
        spectrum_Sa = spectrum_Sa[:num_modes]

    return periods, spectrum_Sa


def run_response_spectrum_analysis(
    geom,
    nodes,
    elements,
    periods,
    spectrum_T,
    spectrum_Sa,
    direction=1,
    damping_ratio=0.05,
    combo_method="CQC",
    num_modes=None,
):
    del geom
    del spectrum_T

    node_tags = _get_node_tags(nodes)
    ele_tags = _get_ele_tags(elements)
    ndf = _guess_ndf_from_model()

    periods, spectrum_Sa = _prepare_modal_data(periods, spectrum_Sa, num_modes=num_modes)
    omegas = 2.0 * np.pi / periods
    story_nodes = _get_story_nodes_from_nodes(nodes)

    modal_node_disp = {}
    modal_element_forces = {}
    modal_base_shear = {}

    for i_mode in range(1, len(periods) + 1):
        ops.responseSpectrumAnalysis(
            direction,
            "-Tn", *periods[np.argsort(periods)].tolist(),
            "-Sa", *spectrum_Sa[np.argsort(periods)].tolist(),
            "-mode", i_mode
        )

        modal_node_disp[i_mode] = _node_disp_vector(node_tags, ndf)
        modal_element_forces[i_mode] = _element_force_vector(ele_tags)
        modal_base_shear[i_mode] = _compute_base_shear(node_tags, direction)

        print(f"[RSA] mode {i_mode}/{len(periods)} processed")

    combined_node_disp = _combine_dict_of_vectors(
        modal_node_disp,
        omegas,
        combo_method,
        damping_ratio
    )

    combined_element_forces = _combine_dict_of_vectors(
        modal_element_forces,
        omegas,
        combo_method,
        damping_ratio
    )

    combined_story_drifts = {}
    for story_name, n_bot, n_top in story_nodes:
        d_modal = np.array([
            modal_node_disp[m][n_top][direction - 1] - modal_node_disp[m][n_bot][direction - 1]
            for m in sorted(modal_node_disp)
        ], dtype=float)
        combined_story_drifts[story_name] = combine_modal_response(
            d_modal, omegas, method=combo_method, damping_ratio=damping_ratio
        )

    modal_V = np.array([modal_base_shear[m] for m in sorted(modal_base_shear)], dtype=float)
    V_combined = combine_modal_response(
        modal_V,
        omegas,
        method=combo_method,
        damping_ratio=damping_ratio
    )

    return {
        "periods": periods,
        "omegas": omegas,
        "spectral_accel": spectrum_Sa,
        "node_displacements": combined_node_disp,
        "story_drifts": combined_story_drifts,
        "element_forces": combined_element_forces,
        "modal_node_disp": modal_node_disp,
        "modal_element_forces": modal_element_forces,
        "modal_base_shear": modal_base_shear,
        "base_shear": V_combined,
    }


def rsa_drifts_to_dataframe(story_drifts: Dict[str, float], case_name: str = "RSA") -> pd.DataFrame:
    rows = []
    for story, drift_m in story_drifts.items():
        rows.append({
            "Case": case_name,
            "Story": story,
            "Drift [m]": drift_m,
            "Drift [mm]": drift_m * 1000.0,
        })
    return pd.DataFrame(rows)


def _node_disp_to_dataframe(node_disp: Dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    for node, vec in node_disp.items():
        row = {"Node": node}
        for i, val in enumerate(vec, start=1):
            row[f"U{i}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _element_forces_to_dataframe(element_forces: Dict[int, np.ndarray]) -> pd.DataFrame:
    rows = []
    for ele, vec in element_forces.items():
        row = {"Element": ele}
        for i, val in enumerate(vec, start=1):
            row[f"F{i}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _modal_node_disp_to_dataframe(modal_node_disp: Dict[int, Dict[int, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for mode, node_map in modal_node_disp.items():
        for node, vec in node_map.items():
            row = {"Mode": mode, "Node": node}
            for i, val in enumerate(vec, start=1):
                row[f"U{i}"] = val
            rows.append(row)
    return pd.DataFrame(rows)


def _modal_element_forces_to_dataframe(modal_ele_forces: Dict[int, Dict[int, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for mode, ele_map in modal_ele_forces.items():
        for ele, vec in ele_map.items():
            row = {"Mode": mode, "Element": ele}
            for i, val in enumerate(vec, start=1):
                row[f"F{i}"] = val
            rows.append(row)
    return pd.DataFrame(rows)


def export_response_spectrum_results(rsa_results: Dict, filename: str = "response_spectrum_results.xlsx"):
    df_drifts = rsa_drifts_to_dataframe(rsa_results["story_drifts"], case_name="RSA")
    df_nodes = _node_disp_to_dataframe(rsa_results["node_displacements"])
    df_elem = _element_forces_to_dataframe(rsa_results["element_forces"])
    df_modal_nodes = _modal_node_disp_to_dataframe(rsa_results["modal_node_disp"])
    df_modal_elem = _modal_element_forces_to_dataframe(rsa_results["modal_element_forces"])

    df_spec = pd.DataFrame({
        "Period [s]": rsa_results["periods"],
        "Omega [rad/s]": rsa_results["omegas"],
        "Sa [m/s2]": rsa_results["spectral_accel"],
    })

    df_base = pd.DataFrame({
        "Mode": list(rsa_results["modal_base_shear"].keys()),
        "Modal Base Shear [N]": list(rsa_results["modal_base_shear"].values())
    })

    df_base_comb = pd.DataFrame({
        "Combined Base Shear [N]": [rsa_results["base_shear"]]
    })

    try:
        engine = "xlsxwriter"
        with pd.ExcelWriter(filename, engine=engine) as writer:
            df_spec.to_excel(writer, sheet_name="Spectrum", index=False)
            df_drifts.to_excel(writer, sheet_name="Story_Drifts", index=False)
            df_nodes.to_excel(writer, sheet_name="Node_Displacements", index=False)
            df_elem.to_excel(writer, sheet_name="Element_Forces", index=False)
            df_modal_nodes.to_excel(writer, sheet_name="Modal_Node_Disp", index=False)
            df_modal_elem.to_excel(writer, sheet_name="Modal_Elem_Forces", index=False)
            df_base.to_excel(writer, sheet_name="Modal_Base_Shear", index=False)
            df_base_comb.to_excel(writer, sheet_name="Base_Shear_Combined", index=False)
        print(f"✔ RSA detailed results exported: {filename}")

    except ModuleNotFoundError:
        engine = "openpyxl"
        with pd.ExcelWriter(filename, engine=engine) as writer:
            df_spec.to_excel(writer, sheet_name="Spectrum", index=False)
            df_drifts.to_excel(writer, sheet_name="Story_Drifts", index=False)
            df_nodes.to_excel(writer, sheet_name="Node_Displacements", index=False)
            df_elem.to_excel(writer, sheet_name="Element_Forces", index=False)
            df_modal_nodes.to_excel(writer, sheet_name="Modal_Node_Disp", index=False)
            df_modal_elem.to_excel(writer, sheet_name="Modal_Elem_Forces", index=False)
            df_base.to_excel(writer, sheet_name="Modal_Base_Shear", index=False)
            df_base_comb.to_excel(writer, sheet_name="Base_Shear_Combined", index=False)
        print(f"✔ RSA detailed results exported: {filename}")


def _combine_scalar_dict_abs(a: Dict, b: Dict, ax: float, by: float) -> Dict:
    keys = sorted(set(a.keys()) | set(b.keys()))
    out = {}
    for k in keys:
        va = abs(float(a.get(k, 0.0)))
        vb = abs(float(b.get(k, 0.0)))
        out[k] = ax * va + by * vb
    return out


def _combine_vector_dict_abs(
    a: Dict[int, np.ndarray],
    b: Dict[int, np.ndarray],
    ax: float,
    by: float
) -> Dict[int, np.ndarray]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    out = {}

    for k in keys:
        va = np.asarray(a.get(k, np.array([], dtype=float)), dtype=float)
        vb = np.asarray(b.get(k, np.array([], dtype=float)), dtype=float)

        n = max(len(va), len(vb))
        if n == 0:
            out[k] = np.array([], dtype=float)
            continue

        aa = np.zeros(n, dtype=float)
        bb = np.zeros(n, dtype=float)

        aa[:len(va)] = np.abs(va)
        bb[:len(vb)] = np.abs(vb)

        out[k] = ax * aa + by * bb

    return out


def _envelope_scalar_dict(a: Dict, b: Dict) -> Dict:
    keys = sorted(set(a.keys()) | set(b.keys()))
    out = {}
    for k in keys:
        out[k] = max(float(a.get(k, 0.0)), float(b.get(k, 0.0)))
    return out


def _envelope_vector_dict(
    a: Dict[int, np.ndarray],
    b: Dict[int, np.ndarray]
) -> Dict[int, np.ndarray]:
    keys = sorted(set(a.keys()) | set(b.keys()))
    out = {}

    for k in keys:
        va = np.asarray(a.get(k, np.array([], dtype=float)), dtype=float)
        vb = np.asarray(b.get(k, np.array([], dtype=float)), dtype=float)

        n = max(len(va), len(vb))
        if n == 0:
            out[k] = np.array([], dtype=float)
            continue

        aa = np.zeros(n, dtype=float)
        bb = np.zeros(n, dtype=float)

        aa[:len(va)] = va
        bb[:len(vb)] = vb

        out[k] = np.maximum(aa, bb)

    return out


def _build_directional_case(
    rsa_x: Dict,
    rsa_y: Dict,
    fx: float,
    fy: float,
    case_name: str
) -> Dict:
    return {
        "case_name": case_name,
        "story_drifts": _combine_scalar_dict_abs(
            rsa_x["story_drifts"],
            rsa_y["story_drifts"],
            fx,
            fy
        ),
        "node_displacements": _combine_vector_dict_abs(
            rsa_x["node_displacements"],
            rsa_y["node_displacements"],
            fx,
            fy
        ),
        "element_forces": _combine_vector_dict_abs(
            rsa_x["element_forces"],
            rsa_y["element_forces"],
            fx,
            fy
        ),
        "base_shear": fx * abs(float(rsa_x["base_shear"])) + fy * abs(float(rsa_y["base_shear"])),
    }


def combine_two_directional_rsa_results(
    rsa_x: Dict,
    rsa_y: Dict,
    orthogonal_factor: float = 0.30
) -> Dict:
    case_x_30y = _build_directional_case(
        rsa_x, rsa_y,
        fx=1.0,
        fy=orthogonal_factor,
        case_name=f"RSA_X_plus_{orthogonal_factor:.2f}Y"
    )

    case_30x_y = _build_directional_case(
        rsa_x, rsa_y,
        fx=orthogonal_factor,
        fy=1.0,
        case_name=f"RSA_{orthogonal_factor:.2f}X_plus_Y"
    )

    envelope = {
        "case_name": "RSA_DIR_ENVELOPE",
        "story_drifts": _envelope_scalar_dict(
            case_x_30y["story_drifts"],
            case_30x_y["story_drifts"]
        ),
        "node_displacements": _envelope_vector_dict(
            case_x_30y["node_displacements"],
            case_30x_y["node_displacements"]
        ),
        "element_forces": _envelope_vector_dict(
            case_x_30y["element_forces"],
            case_30x_y["element_forces"]
        ),
        "base_shear": max(
            float(case_x_30y["base_shear"]),
            float(case_30x_y["base_shear"])
        ),
    }

    return {
        "orthogonal_factor": orthogonal_factor,
        "cases": {
            case_x_30y["case_name"]: case_x_30y,
            case_30x_y["case_name"]: case_30x_y,
        },
        "envelope": envelope,
    }


def export_directional_response_spectrum_results(
    directional_results: Dict,
    filename: str = "response_spectrum_directional_results.xlsx"
):
    case_items = directional_results["cases"]
    env = directional_results["envelope"]

    rows_base = []
    for cname, cres in case_items.items():
        rows_base.append({
            "Case": cname,
            "Combined Base Shear [N]": cres["base_shear"]
        })

    rows_base.append({
        "Case": env["case_name"],
        "Combined Base Shear [N]": env["base_shear"]
    })

    df_base = pd.DataFrame(rows_base)

    try:
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            for cname, cres in case_items.items():
                rsa_drifts_to_dataframe(cres["story_drifts"], case_name=cname).to_excel(
                    writer, sheet_name=f"{cname[:20]}_Drifts", index=False
                )
                _node_disp_to_dataframe(cres["node_displacements"]).to_excel(
                    writer, sheet_name=f"{cname[:20]}_Nodes", index=False
                )
                _element_forces_to_dataframe(cres["element_forces"]).to_excel(
                    writer, sheet_name=f"{cname[:20]}_Elems", index=False
                )

            rsa_drifts_to_dataframe(env["story_drifts"], case_name=env["case_name"]).to_excel(
                writer, sheet_name="Envelope_Drifts", index=False
            )
            _node_disp_to_dataframe(env["node_displacements"]).to_excel(
                writer, sheet_name="Envelope_Nodes", index=False
            )
            _element_forces_to_dataframe(env["element_forces"]).to_excel(
                writer, sheet_name="Envelope_Elems", index=False
            )
            df_base.to_excel(writer, sheet_name="Base_Shear", index=False)

        print(f"✔ Directional RSA results exported: {filename}")

    except ModuleNotFoundError:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for cname, cres in case_items.items():
                rsa_drifts_to_dataframe(cres["story_drifts"], case_name=cname).to_excel(
                    writer, sheet_name=f"{cname[:20]}_Drifts", index=False
                )
                _node_disp_to_dataframe(cres["node_displacements"]).to_excel(
                    writer, sheet_name=f"{cname[:20]}_Nodes", index=False
                )
                _element_forces_to_dataframe(cres["element_forces"]).to_excel(
                    writer, sheet_name=f"{cname[:20]}_Elems", index=False
                )

            rsa_drifts_to_dataframe(env["story_drifts"], case_name=env["case_name"]).to_excel(
                writer, sheet_name="Envelope_Drifts", index=False
            )
            _node_disp_to_dataframe(env["node_displacements"]).to_excel(
                writer, sheet_name="Envelope_Nodes", index=False
            )
            _element_forces_to_dataframe(env["element_forces"]).to_excel(
                writer, sheet_name="Envelope_Elems", index=False
            )
            df_base.to_excel(writer, sheet_name="Base_Shear", index=False)

        print(f"✔ Directional RSA results exported: {filename}")