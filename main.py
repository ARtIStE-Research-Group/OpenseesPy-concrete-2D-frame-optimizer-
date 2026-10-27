import sys
import os
import inspect
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)   # all CSV/XLSX/report outputs go next to main.py
sys.path.insert(0, current_dir)

frame_app_dir = os.path.join(current_dir, "frame_app")
if os.path.isdir(frame_app_dir):
    sys.path.insert(0, frame_app_dir)

# Spyder keeps modules in memory between runs: if 'framekit' was imported from
# another ARCOPY folder, forget it so this folder's version is used.
_fk = sys.modules.get("framekit")
if _fk is not None and not os.path.abspath(str(getattr(_fk, "__file__", "") or "")).startswith(
        os.path.abspath(frame_app_dir)):
    for _m in [m for m in list(sys.modules) if m == "framekit" or m.startswith("framekit.")]:
        del sys.modules[_m]
    print("(framekit from another folder was in memory - reloaded from this folder)")

from framekit.plot_style import apply_style, enable_figure_autosave
apply_style()
from framekit.geometry import ask_geometry, generate_nodes, generate_elements
from framekit.plotting import plot_structure
from framekit.section_library import ask_sections_rc, plot_selected_rc_sections, plot_jacketed_section_comparisons
from framekit.loads import ask_structure_type, ask_material, compute_self_weight, compute_wall_self_weight, ask_loads_by_cases
from framekit.combinations import build_eurocode_combinations, print_combinations_detailed
from framekit.walls import ask_shear_walls, ask_shear_wall_section
from framekit.opensees_model import ops_build_model, ops_apply_combo_loads, ops_run_static
from framekit.units import _MPa_to_Pa
from framekit.static_analysis import (
    get_node_disps,
    compute_story_drifts,
    drifts_to_dataframe,
    plot_deformed_shape,
)
from framekit.diagrams import (
    plot_structure_NVM,
    plot_all_combos_NVM,
    collect_member_force_diagrams_for_report,
)


from framekit.ec2_checks import (
    ask_ec2_materials_default_or_custom,
    run_combo_and_check_all,
    worst_table_to_dataframe,
    export_ec2_worst_table_colored,
    plot_verification_table,
    plot_utilization_on_structure,
    add_stirrup_columns_to_worst_table,
)

from framekit.dynamics import (
    build_mass_loadset_from_cases,
    compute_floor_masses_kg_from_line_loads,
    print_floor_mass_report,
    ops_assign_floor_masses,
    ops_run_eigen_and_print_and_plot,
    modal_participation_table,
)

from framekit.response_spectrum import (
    ask_ec8_spectrum_params,
    build_ec8_spectrum_for_periods,
    build_ec8_plot_spectrum,
    plot_ec8_spectrum,
    run_response_spectrum_analysis,
    rsa_drifts_to_dataframe,
    export_response_spectrum_results,
    combine_two_directional_rsa_results,
    export_directional_response_spectrum_results,
)

from framekit.planar_directions import (
    clone_geom_for_direction,
    clone_nodes_for_direction,
    clone_elements_for_direction,
)

from framekit.ec8_drift_checks import (
    ask_ec8_drift_check_params,
    compute_story_heights,
    ec8_damage_limitation_drifts_from_rsa,
    check_ec8_drift_limits,
    summarize_ec8_drift_check,
    export_ec8_drift_check,
)

from framekit.time_history import (
    ask_time_history_params,
    run_multiple_time_histories,
    plot_time_history_response,
    export_multiple_time_history_results,
)

from framekit.performance_summary import (
    build_story_performance_table,
    build_story_failures_table,
    build_member_performance_table,
    build_member_failures_table,
    build_global_performance_summary,
    export_performance_summary_results,
)

from framekit.constants import PSI_Q2
from framekit.checkpoint import WorkflowCheckpoint, STEP_DESCRIPTIONS
from framekit.ec8_seismic import (
    ask_code_basis,
    apply_confidence_factor,
    ask_behaviour_factor,
    ask_dls_params,
    ask_seismic_settings,
    design_spectrum,
    modal_x_properties,
    rsa_equivalent_floor_forces,
    floor_forces_to_loads,
    add_seismic_combos,
    elastic_storey_drifts_from_loads,
    storey_checks,
    strong_column_weak_beam_check,
    modal_mass_check,
)

WORKFLOW_STEPS = list(STEP_DESCRIPTIONS.keys())

from framekit.retrofit_recommendation import (
    recommend_story_retrofits,
    recommend_member_retrofits,
    build_retrofit_priority_table,
    export_retrofit_recommendations,)

from framekit.retrofit_optimizer import (
    DEFAULT_UNIT_COSTS,
    ask_retrofit_cost_parameters,
    ask_max_global_candidates,
    optimize_shear_wall_positions,
    optimize_global_retrofit_strategy,
    build_global_optimization_dataset,
    export_global_optimization_dataset,
    optimize_concrete_jacketing,
    make_jacketed_element_sections,
    run_ec2_member_assessment,
    estimate_jacketing_cost,
    plot_shear_wall_retrofit_layout,
    plot_jacketed_members,
    plot_final_retrofit_utilization,
)

from framekit.retrofit_optimization_report import (
    export_retrofit_optimization_report,
)
from framekit.report_builder import export_retrofit_report_v2




# ============================================================
# Robust console input (re-asks instead of crashing)
# ============================================================
def _yn(prompt, default=None):
    """Return 'y' or 'n'. Accepts y/yes/n/no; Enter gives the default if any."""
    while True:
        a = input(prompt).strip().lower()
        if not a and default is not None:
            return default
        if a in ("y", "yes", "s", "si"):
            return "y"
        if a in ("n", "no"):
            return "n"
        print("❌ Please answer y or n.")


def _ask_run_name(home):
    """Ask the run (folder) name; existing runs with saved steps are listed."""
    import datetime
    runs = sorted(d for d in os.listdir(home)
                  if os.path.isdir(os.path.join(home, d, "_checkpoints")))
    try:
        from framekit import __version__ as _arcopy_version
    except ImportError:
        _arcopy_version = "?"
    print(f"\n=== ARCOPY v{_arcopy_version} run ===")
    if runs:
        print("Existing runs (type the name to continue one of them):")
        for r in runs:
            print(f"  - {r}")
    default = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    while True:
        name = input(f"Run name / folder [{default}]: ").strip() or default
        clean = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in name).strip("._")
        if clean:
            if clean != name:
                print(f"Folder name adjusted to: {clean}")
            return clean
        print("❌ Please enter a valid name.")


def _num(prompt, cast=float, default=None, min_val=None, max_val=None):
    while True:
        a = input(prompt).strip().replace(",", ".")
        if not a and default is not None:
            return cast(default)
        try:
            v = cast(float(a)) if cast is int else cast(a)
        except ValueError:
            print("❌ Please enter a valid number.")
            continue
        if cast is int and float(a) != int(float(a)):
            print("❌ Please enter an integer.")
            continue
        if min_val is not None and v < min_val:
            print(f"❌ Must be >= {min_val}.")
            continue
        if max_val is not None and v > max_val:
            print(f"❌ Must be <= {max_val}.")
            continue
        return v


# ============================================================
# Robust exact N/V/M collection helpers for report export
# ============================================================
def _first_existing_column_for_report(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _critical_member_ids_for_report_nvm(df, max_members=6):
    """Return the most critical element IDs from an EC2/member table.

    This is used to guarantee that the report has real OpenSees N/V/M
    diagrams even when no member is still failing after retrofit.  The list is
    sorted by largest available utilization ratio.
    """
    if df is None or getattr(df, "empty", True):
        return []
    ele_col = _first_existing_column_for_report(df, ["Element", "ElementID", "Member", "Member ID"])
    if ele_col is None:
        return []
    eta_col = _first_existing_column_for_report(
        df,
        [
            "eta", "Static_EC2_Utilization", "Utilization", "Worst eta", "Max_Eta",
            "Max_EC2_Utilization", "Final_Max_EC2_Utilization",
        ],
    )
    d = df.copy()
    d[ele_col] = pd.to_numeric(d[ele_col], errors="coerce")
    d = d.dropna(subset=[ele_col])
    if d.empty:
        return []
    if eta_col is not None:
        d[eta_col] = pd.to_numeric(d[eta_col], errors="coerce")
        d = d.sort_values(eta_col, ascending=False, na_position="last")
    out = []
    for val in d[ele_col].tolist():
        try:
            eid = int(val)
        except Exception:
            continue
        if eid not in out:
            out.append(eid)
        if len(out) >= int(max_members):
            break
    return out


def _nvm_result_dataframe_for_report(result, governing_combo=""):
    if result is None:
        return None
    try:
        _xg, _yg, n_arr, v_arr, m_arr, length_m, _c, _s = result
        npts = len(n_arr)
        df = pd.DataFrame({
            "x_m": np.linspace(0.0, float(length_m), npts),
            "N_kN": np.asarray(n_arr, dtype=float) / 1000.0,
            "V_kN": np.asarray(v_arr, dtype=float) / 1000.0,
            "M_kNm": np.asarray(m_arr, dtype=float) / 1000.0,
        })
        if governing_combo:
            df["Governing_Combo"] = str(governing_combo)
        return df
    except Exception:
        return None


def _run_model_for_report_nvm_stage(
    *,
    nodes,
    elements,
    geom,
    Ec_Pa,
    Es_Pa,
    column_sec,
    beam_sec,
    ops_build_model_fn,
    walls,
    wall_tw_mm,
    wall_E_Pa,
    use_rigid_diaphragm=False,
    element_sections=None,
):
    """Build the model for an N/V/M report stage, compatible with old signatures."""
    try:
        sig = inspect.signature(ops_build_model_fn)
        supports_element_sections = "element_sections" in sig.parameters
    except Exception:
        supports_element_sections = True

    kwargs = dict(
        geom=geom,
        use_rigid_diaphragm=bool(use_rigid_diaphragm),
        walls=walls,
        wall_tw_mm=float(wall_tw_mm),
        wall_E_Pa=float(wall_E_Pa),
    )
    if supports_element_sections:
        kwargs["element_sections"] = element_sections

    try:
        return ops_build_model_fn(
            nodes,
            elements,
            Ec_Pa,
            Es_Pa,
            column_sec,
            beam_sec,
            **kwargs,
        )
    except TypeError:
        # Fallback for older opensees_model.py versions that do not accept
        # element_sections.  The report will still get exact post-wall diagrams.
        kwargs.pop("element_sections", None)
        return ops_build_model_fn(
            nodes,
            elements,
            Ec_Pa,
            Es_Pa,
            column_sec,
            beam_sec,
            **kwargs,
        )


def _collect_member_force_diagrams_by_uls_scan(
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
    ops_build_model_fn,
    ops_apply_combo_loads_fn,
    ops_run_static_fn,
    target_member_ids,
    stage_specs,
    npts=80,
):
    """Fallback exact N/V/M collector for the Word report.

    The primary collector uses the governing combination stored in the EC2
    worst table.  Some EC2 tables do not keep that column, so the primary
    collector may return no data.  This fallback scans all ULS combinations,
    runs the real OpenSees model for each retrofit stage, and stores the exact
    station-by-station diagram from the most demanding ULS combination for
    every requested member.
    """
    try:
        from framekit.diagrams import _nvm_arrays_for_element
    except Exception as exc:
        print(f"WARNING: exact N/V/M fallback is unavailable: {exc}")
        return {}

    target_ids = []
    for eid in target_member_ids or []:
        try:
            eid_int = int(eid)
        except Exception:
            continue
        if eid_int in elements and eid_int not in target_ids:
            target_ids.append(eid_int)
    if not target_ids:
        return {}

    all_names = [str(c) for c in (combo_names or []) if str(c) in combos]
    uls_names = [c for c in all_names if str(c).upper().startswith("ULS")]
    scan_names = uls_names if uls_names else all_names
    if not scan_names:
        return {}

    out = {}
    for stage_name, spec in (stage_specs or {}).items():
        if spec is None:
            continue
        best_by_member = {int(eid): (None, -np.inf, "") for eid in target_ids}
        for cname in scan_names:
            try:
                _run_model_for_report_nvm_stage(
                    nodes=nodes,
                    elements=elements,
                    geom=geom,
                    Ec_Pa=Ec_Pa,
                    Es_Pa=Es_Pa,
                    column_sec=column_sec,
                    beam_sec=beam_sec,
                    ops_build_model_fn=ops_build_model_fn,
                    walls=spec.get("walls", None),
                    wall_tw_mm=spec.get("wall_tw_mm", 200.0),
                    wall_E_Pa=spec.get("wall_E_Pa", 30e9),
                    use_rigid_diaphragm=spec.get("use_rigid_diaphragm", False),
                    element_sections=spec.get("element_sections", None),
                )
                ops_apply_combo_loads_fn(nodes, elements, combos[cname]["loads"])
                ok = ops_run_static_fn()
                if ok != 0:
                    continue
                for eid in target_ids:
                    result = _nvm_arrays_for_element(eid, elements, combos[cname]["loads"], nodes, npts=npts)
                    df = _nvm_result_dataframe_for_report(result, governing_combo=cname)
                    if df is None or df.empty:
                        continue
                    # Select a real governing combination.  The score is a
                    # transparent demand proxy using the absolute station maxima.
                    max_m = pd.to_numeric(df["M_kNm"], errors="coerce").abs().max()
                    max_v = pd.to_numeric(df["V_kN"], errors="coerce").abs().max()
                    max_n = pd.to_numeric(df["N_kN"], errors="coerce").abs().max()
                    score = float(max_m if pd.notna(max_m) else 0.0)
                    score += 0.01 * float(max_v if pd.notna(max_v) else 0.0)
                    score += 0.001 * float(max_n if pd.notna(max_n) else 0.0)
                    if score > best_by_member[int(eid)][1]:
                        best_by_member[int(eid)] = (df, score, cname)
            except Exception as exc:
                print(f"WARNING: report N/V/M fallback skipped {stage_name} / {cname}: {exc}")
                continue

        for eid, (df, score, cname) in best_by_member.items():
            if df is not None and np.isfinite(score):
                out.setdefault(int(eid), {})[str(stage_name)] = df

    return out


if __name__ == "__main__":
    # storage for global summary
    df_worst = None
    rsa_results = None
    dir_results = None
    df_ec8_single_all = None
    df_ec8_dir_all = None
    batch_th = None
    df_dir_env = None
    df_worst_after_sw = None
    df_worst_final = None
    member_fail_after_sw = None
    member_fail_final = None
    jacket_cost = None
    failed_member_ids = []
    sw_opt_df = None
    optimized_walls = None
    jacketing_iteration_worst_dfs = []
    member_force_diagrams = None
    optimization_dataset_df = None

    # ------------------------------------------------------------
    # Automatic checkpoints: state saved after every step
    # ------------------------------------------------------------
    # Run folder: all results of this run go into <main folder>/<run name>/
    os.environ["ARCOPY_HOME"] = current_dir
    run_name = _ask_run_name(current_dir)
    run_dir = os.path.join(current_dir, run_name)
    report_dir = os.path.join(run_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    os.chdir(run_dir)
    enable_figure_autosave(os.path.join(run_dir, "figures"))
    print(f"\nAll results of this run are written to:\n  {run_dir}\n  (report in: {report_dir})")

    ckpt = WorkflowCheckpoint(
        folder=os.path.join(run_dir, "_checkpoints"),
        steps=WORKFLOW_STEPS,
        namespace=globals(),
    )
    ckpt.start()


    if ckpt.need("model"):
        # 1) Structure type
        stype = ask_structure_type()

        # 2) Geometry
        geom = ask_geometry()
        nodes = generate_nodes(geom)
        elements = generate_elements(geom)

        plot_structure(nodes, elements, geom=geom, title="Structure (IDs)")

        # 3) Shear walls
        walls = ask_shear_walls(geom)

        wall_tw_mm = 200.0
        wall_E_MPa = 30000.0
        wall_density = 2500.0

        if walls:
            wall_tw_mm, wall_E_MPa, wall_density = ask_shear_wall_section()
            for _w in walls:
                _w.tw_mm = float(wall_tw_mm)
            plot_structure(
                nodes,
                elements,
                geom=geom,
                walls=walls,
                title="Structure + Shear Walls"
            )

        # 4) Sections
        if stype == "rc":
            column_sec, beam_sec = ask_sections_rc()
            ec2mat = ask_ec2_materials_default_or_custom()
        else:
            print("\nSteel section input is not connected yet in this modular version.")
            print("For now, using RC section input block temporarily.")
            column_sec, beam_sec = ask_sections_rc()
            ec2mat = None

        # Seismic code basis: new design (EN 1998-1) or existing building (EN 1998-3)
        code_info = ask_code_basis()
        if code_info["code"] == "EN1998-3" and ec2mat is not None:
            ec2mat_input = ec2mat
            ec2mat = apply_confidence_factor(ec2mat, code_info["CF"])
            print(f"EC2 strengths used in all checks: fck = {ec2mat.fck:.2f} MPa, fyk = {ec2mat.fyk:.1f} MPa "
                  f"(mean values / CF {code_info['CF']:.2f})")

        print("\nColumn section:")
        print(column_sec)
        print("\nBeam section:")
        print(beam_sec)

        plot_sections_now = _yn("\nPlot selected RC beam/column sections? (y/n) [y]: ", "y")
        if plot_sections_now == "y":
            plot_selected_rc_sections(column_sec, beam_sec)

        # 5) Material
        material = ask_material("Concrete" if stype == "rc" else "Steel")
        ckpt.done("model")


    if ckpt.need("loads"):
        # 6) Load cases
        gravity_loads = compute_self_weight(elements, material, beam_sec, column_sec)
        cases = ask_loads_by_cases(nodes, elements, geom)
        if walls:
            gravity_loads += compute_wall_self_weight(nodes, geom, walls, wall_tw_mm, wall_density)
        cases["SW"] = gravity_loads

        print("\n=== LOAD CASE SUMMARY ===")
        for k, v in cases.items():
            print(f"{k}: {len(v)} loads")

        print("\nShowing each load case separately...")
        for case_name in ["SW", "G", "Q", "W+", "W-"]:
            if cases.get(case_name):
                plot_structure(
                    nodes,
                    elements,
                    geom=geom,
                    walls=walls,
                    loads=cases[case_name],
                    title=f"Structure + Loads : {case_name}"
                )

        # 7) Combinations
        combos = build_eurocode_combinations(cases)
        names = print_combinations_detailed(combos)
        ckpt.done("loads")


    if ckpt.need("static"):
        # 8) Run all combinations
        Ec_Pa = _MPa_to_Pa(material.E)
        Es_Pa = 200e9
        wall_E_Pa = _MPa_to_Pa(wall_E_MPa)

        print("\n--- Running static analysis for ALL combinations ---")

        all_drift_tables = []
        results_ok = []
        results_failed = []

        for cname in names:
            print(f"\nRunning: {cname}")
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
                print(f"❌ Failed: {cname}")
                results_failed.append(cname)
                continue

            print(f"✔ Success: {cname}")
            results_ok.append(cname)

            U = get_node_disps(nodes)
            drifts = compute_story_drifts(geom, nodes, U)

            df = drifts_to_dataframe(drifts, cname)
            all_drift_tables.append(df)

            if cname.startswith("ULS"):
                plot_deformed_shape(
                    geom,
                    nodes,
                    elements,
                    scale=None,
                    title=f"Deformed Shape - {cname}",
                    walls=walls,
                )

        # 9) Summary
        print("\n=== STATIC ANALYSIS SUMMARY ===")
        print(f"Successful combinations: {len(results_ok)}")
        print(f"Failed combinations: {len(results_failed)}")

        if results_ok:
            print("\nSuccessful:")
            for cname in results_ok:
                print(f"  - {cname}")

        if results_failed:
            print("\nFailed:")
            for cname in results_failed:
                print(f"  - {cname}")

        # 10) Drift tables
        if all_drift_tables:
            df_all = pd.concat(all_drift_tables, ignore_index=True)

            print("\n=== MAX STORY DRIFTS ===")
            print(df_all.to_string(index=False))

            df_all.to_csv("story_drifts.csv", index=False)
            print("\n✔ Drift table exported: story_drifts.csv")
        else:
            df_all = None
        ckpt.done("static")


    if ckpt.need("ec2"):
        # 11) EC2 checks
        if stype == "rc":
            run_checks = _yn("\nRun EC2 checks for ULS combinations? (y/n): ")
            if run_checks == "y":
                checks_by_combo = {}

                for cname in names:
                    if not cname.startswith("ULS"):
                        continue

                    print(f"\nChecking: {cname}")
                    loads_combo = combos[cname]["loads"]

                    checks, ok = run_combo_and_check_all(
                        nodes,
                        elements,
                        loads_combo,
                        Ec_Pa,
                        Es_Pa,
                        column_sec,
                        beam_sec,
                        ec2mat,
                        ops_build_model,
                        ops_apply_combo_loads,
                        ops_run_static,
                        geom=geom,
                        use_rigid_diaphragm=False,
                        walls=walls,
                        wall_tw_mm=wall_tw_mm,
                        wall_E_Pa=wall_E_Pa
                    )

                    if ok != 0 or checks is None:
                        print(f"❌ Check failed for {cname}")
                        continue

                    checks_by_combo[cname] = checks
                    print(f"✔ Check completed for {cname}")

                if checks_by_combo:
                    df_worst = worst_table_to_dataframe(checks_by_combo)
                
                    df_worst = add_stirrup_columns_to_worst_table(
        df_worst,
        beam_sec=beam_sec,
        column_sec=column_sec,
        elements=elements
    )

                    print("\n=== EC2 WORST UTILIZATION TABLE ===")
                    print(df_worst.to_string(index=False))

                    show_table = _yn("Show colored verification table? (y/n): ")
                    if show_table == "y":
                        plot_verification_table(
                            checks_by_combo,
                            title="EC2 Verification (ULS worst eta per member)",
                            max_rows=40
                        )

                    show_structure_map = _yn("Show utilization colored on structure? (y/n): ")
                    if show_structure_map == "y":
                        plot_utilization_on_structure(
                            nodes,
                            elements,
                            checks_by_combo,
                            title="Utilization map (ULS worst eta)",
                            geom=geom,
                            walls=walls,
                        )

                    export_ec2_worst_table_colored(
                        df_worst,
                        filename="ec2_worst_utilization.xlsx"
                    )
        ckpt.done("ec2")


    if ckpt.need("nvm"):
        # 13) N/V/M menu
        print("\n--- N/V/M diagrams menu ---")
        print("1) Plot N/V/M for ONE combination")
        print("2) Plot N/V/M for ALL combinations")
        print("3) Skip diagrams")
        opt = str(_num("Select option (1/2/3): ", int, min_val=1, max_val=3))

        if opt == "1":
            for i, cname in enumerate(names, start=1):
                print(f"{i:2d}) {cname}")

            idx = _num(f"Choose combination (1..{len(names)}): ", int, min_val=1, max_val=len(names))
            cname = names[idx - 1]
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

            if ok == 0:
                plot_structure_NVM(
                    nodes,
                    elements,
                    loads_combo,
                    combo_name=cname,
                    npts=60,
                    geom=geom,
                    walls=walls,
                )
            else:
                print("❌ Analysis failed for selected combination.")

        elif opt == "2":
            plot_all_combos_NVM(
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
                walls=walls,
                wall_tw_mm=wall_tw_mm,
                wall_E_Pa=wall_E_Pa,
                npts=60
            )
        ckpt.done("nvm")


    if ckpt.need("modal"):
        # 14) Linear dynamic analysis: masses + eigen + response spectrum + time history
        run_dynamics = _yn("\nGo to dynamic analysis now? (y/n): ")
        if run_dynamics == "y":
            psi_mass_Q = _num(f"Mass live-load factor psi [recommended {PSI_Q2}]: ", float, default=PSI_Q2, min_val=0.0, max_val=1.0)
            extra_mass_floor = _num("Additional lumped mass per floor [kg] (0 if none): ", float, default=0.0, min_val=0.0)

            use_diaphragm = _yn("Use rigid diaphragm per floor? (y/n): ") == "y"
            crack_factor = _num("Cracked stiffness factor EI_eff/EI for seismic analyses (EC8 4.3.1(7)) [default 0.5]: ",
                                float, default=0.5, min_val=0.1, max_val=1.0)

            loads_for_mass = build_mass_loadset_from_cases(cases, psi_mass_Q=psi_mass_Q)
            floor_masses = compute_floor_masses_kg_from_line_loads(
                geom,
                nodes,
                elements,
                loads_for_mass,
                add_lumped_mass_per_floor_kg=extra_mass_floor
            )

            print_floor_mass_report(floor_masses)

            master_nodes = ops_build_model(
                nodes,
                elements,
                Ec_Pa,
                Es_Pa,
                column_sec,
                beam_sec,
                geom=geom,
                use_rigid_diaphragm=use_diaphragm,
                walls=walls,
                wall_tw_mm=wall_tw_mm,
                wall_E_Pa=wall_E_Pa,
                stiffness_factor=crack_factor
            )

            ops_assign_floor_masses(
                nodes,
                geom,
                floor_masses,
                master_nodes=(master_nodes if use_diaphragm else None)
            )

            nmodes = _num(f"Number of eigen modes to compute [max {geom.num_floors * (geom.num_bays + 1) * 3}]: ", int, min_val=1, max_val=geom.num_floors * (geom.num_bays + 1) * 3)
            plot_floor_shapes = _yn("Plot floor mode-shape curves? (y/n): ") == "y"
            plot_struct_modes = _yn("Plot full deformed structure for each mode? (y/n): ") == "y"

            periods, shapes = ops_run_eigen_and_print_and_plot(
                geom,
                nodes,
                elements,
                num_modes=nmodes,
                ref_grid_x=0,
                normalize=True,
                plot_floor_shapes=plot_floor_shapes,
                plot_structure_each_mode=plot_struct_modes
            )

            df_modal_part = modal_participation_table(geom, floor_masses, len(periods))
            print("\n=== MODAL PARTICIPATION TABLE ===")
            print(df_modal_part.to_string(index=False))

            df_modal_part.to_csv("modal_participation.csv", index=False)
            print("\n✔ Modal participation exported: modal_participation.csv")

            df_periods = pd.DataFrame({
                "Mode": list(range(1, len(periods) + 1)),
                "Period [s]": periods
            })
            df_periods.to_csv("modal_periods.csv", index=False)
            print("✔ Modal periods exported: modal_periods.csv")
        ckpt.done("modal")


    if ckpt.need("rsa"):
        if run_dynamics == "y":
            run_rsa = _yn("\nRun EC8 response spectrum analysis (design spectrum, q-factor)? (y/n): ")
            if run_rsa == "y":
                rsa_params = ask_ec8_spectrum_params()
                q_info = ask_behaviour_factor(geom, walls, code_info)
                q_beh = float(q_info["q"])

                master_nodes = ops_build_model(
                    nodes, elements, Ec_Pa, Es_Pa, column_sec, beam_sec,
                    geom=geom, use_rigid_diaphragm=use_diaphragm, walls=walls,
                    wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa, stiffness_factor=crack_factor,
                )
                ops_assign_floor_masses(nodes, geom, floor_masses,
                                        master_nodes=(master_nodes if use_diaphragm else None))
                periods, shapes = ops_run_eigen_and_print_and_plot(
                    geom, nodes, elements, num_modes=nmodes, ref_grid_x=0, normalize=True,
                    plot_floor_shapes=False, plot_structure_each_mode=False,
                )

                T_rsa = np.asarray(periods, dtype=float)
                Sa_rsa = design_spectrum(T_rsa, rsa_params, q_beh)      # Sd(T), EC8 3.2.2.5
                T_plot, Sa_plot = build_ec8_plot_spectrum(rsa_params)

                if _yn("Plot EC8 spectrum? (y/n): ") == "y":
                    plot_ec8_spectrum(T_plot, Sa_plot, title="EC8 response spectrum", periods=periods,
                                      rsa_params=rsa_params, Sd=design_spectrum(T_plot, rsa_params, q_beh), q=q_beh)

                rsa_results = run_response_spectrum_analysis(
                    geom=geom, nodes=nodes, elements=elements, periods=periods,
                    spectrum_T=T_rsa, spectrum_Sa=Sa_rsa, direction=1,
                    damping_ratio=rsa_params["damping_ratio"], combo_method=rsa_params["combo_method"],
                    num_modes=len(periods),
                )

                if rsa_results is not None:
                    # Design-spectrum response -> displacements ds = q·de (EC8 4.3.4)
                    ds_story = {k: q_beh * float(v) for k, v in rsa_results["story_drifts"].items()}
                    rsa_results["story_drifts_design_spectrum"] = dict(rsa_results["story_drifts"])
                    rsa_results["story_drifts"] = ds_story
                    rsa_results["q"] = q_beh

                    df_rsa_drifts = rsa_drifts_to_dataframe(ds_story, case_name="RSA ds=q·de")
                    print(f"\n=== RESPONSE SPECTRUM STOREY DRIFTS ds = q·de  (q = {q_beh:.2f}) ===")
                    print(df_rsa_drifts.to_string(index=False))
                    df_rsa_drifts.to_csv("response_spectrum_drifts.csv", index=False)

                    print("\n=== RSA BASE SHEAR (design spectrum) ===")
                    print(f"Base shear: {rsa_results['base_shear'] / 1000.0:.2f} kN")
                    export_response_spectrum_results(rsa_results, filename="response_spectrum_results.xlsx")

                    if "df_modal_part" in globals() and df_modal_part is not None:
                        tot_ratio, mass_ok = modal_mass_check(df_modal_part)
                        print(f"\nEffective modal mass of the {len(periods)} modes: {100 * tot_ratio:.1f}% "
                              f"({'OK ≥ 90%' if mass_ok else 'LESS than 90% - increase the number of modes (EC8 4.3.3.3.1)'})")

                    # ---- damage limitation (EC8 4.4.3.2) ----
                    dls = ask_dls_params()
                    story_heights = compute_story_heights(geom)
                    dls_drifts = ec8_damage_limitation_drifts_from_rsa(ds_story, nu=dls["nu"])
                    df_ec8_drift = check_ec8_drift_limits(
                        drift_m=dls_drifts, story_heights_m=story_heights, limit_ratio=dls["dls_limit"],
                        case_name="EC8_DLS_RSA", nse_class=dls["dls_class"],
                    )
                    print(f"\n=== EC8 DAMAGE LIMITATION : {dls['dls_class']}  (ν = {dls['nu']:.2f}) ===")
                    print(df_ec8_drift.to_string(index=False))
                    summary = summarize_ec8_drift_check(df_ec8_drift)
                    print(f"Status: {'OK' if summary['ok'] else 'FAIL'}   worst utilization: {summary['worst_utilization']:.3f}")
                    df_ec8_single_all = df_ec8_drift
                    export_ec8_drift_check(df_ec8_single_all, filename="ec8_drift_check_single_direction.xlsx")

                    seis_settings = {**code_info, "rsa_params": dict(rsa_params), **q_info, **dls, "beta": 0.2,
                                     "stiffness_factor": crack_factor, "psi2": psi_mass_Q,
                                     "extra_mass_floor_kg": extra_mass_floor}

                    # ---- seismic action as static floor forces (RSA-equivalent storey shears) ----
                    master_nodes = ops_build_model(
                        nodes, elements, Ec_Pa, Es_Pa, column_sec, beam_sec,
                        geom=geom, use_rigid_diaphragm=use_diaphragm, walls=walls,
                        wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa, stiffness_factor=crack_factor,
                    )
                    ops_assign_floor_masses(nodes, geom, floor_masses,
                                            master_nodes=(master_nodes if use_diaphragm else None))
                    seis_modes = modal_x_properties(geom, nmodes, dense=use_diaphragm)
                    eq_forces = rsa_equivalent_floor_forces(
                        geom, floor_masses, seis_modes, rsa_params, q_beh,
                        combo_method=rsa_params["combo_method"], damping_ratio=rsa_params["damping_ratio"],
                    )
                    E_loads = floor_forces_to_loads(geom, eq_forces["forces_kN"])
                    de_story = elastic_storey_drifts_from_loads(
                        nodes=nodes, elements=elements, geom=geom, E_loads=E_loads, Ec_Pa=Ec_Pa, Es_Pa=Es_Pa,
                        column_sec=column_sec, beam_sec=beam_sec, walls=walls, wall_tw_mm=wall_tw_mm,
                        wall_E_Pa=wall_E_Pa, element_sections=None, stiffness_factor=crack_factor,
                        ops_build_model=ops_build_model,
                    )
                    df_storey_seis = storey_checks(geom, floor_masses, eq_forces["forces_kN"], de_story, seis_settings)
                    print("\n=== EC8 STOREY CHECKS (seismic floor forces, ds = q·de, θ sensitivity) ===")
                    print(df_storey_seis.round(4).to_string(index=False))
                    theta_max = float(df_storey_seis["theta"].max())
                    if theta_max > 0.10:
                        amp = 1.0 / (1.0 - min(theta_max, 0.5))
                        print(f"⚠ θmax = {theta_max:.3f} > 0.10: seismic action effects amplified by 1/(1-θ) = {amp:.3f}")
                        if theta_max > 0.20:
                            print("⚠ θ > 0.20: EC8 requires a second-order analysis (θ > 0.30 is not allowed).")
                        E_loads = floor_forces_to_loads(geom, eq_forces["forces_kN"], factor=amp)

                    seis_combos, seis_names = add_seismic_combos(combos, [], E_loads)
                    try:
                        df_storey_seis.to_excel("ec8_seismic_storey_checks.xlsx", index=False)
                    except Exception as exc:
                        print(f"⚠ Could not export ec8_seismic_storey_checks.xlsx: {exc}")

                    # ---- seismic member verification: G + ψ2·Q ± E ----
                    if ec2mat is not None:
                        checks_seis = {}
                        for cname in seis_names:
                            checks, ok = run_combo_and_check_all(
                                nodes, elements, seis_combos[cname]["loads"], Ec_Pa, Es_Pa, column_sec, beam_sec,
                                ec2mat, ops_build_model, ops_apply_combo_loads, ops_run_static,
                                geom=geom, use_rigid_diaphragm=False, walls=walls,
                                wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa,
                            )
                            if ok == 0 and checks is not None:
                                checks_seis[cname] = checks
                        if checks_seis:
                            df_worst_seis = add_stirrup_columns_to_worst_table(
                                worst_table_to_dataframe(checks_seis), beam_sec=beam_sec,
                                column_sec=column_sec, elements=elements)
                            print("\n=== EC8 SEISMIC COMBINATION: MEMBER CHECKS (G + ψ2·Q ± E) ===")
                            print(df_worst_seis.to_string(index=False))
                            export_ec2_worst_table_colored(df_worst_seis, filename="ec8_seismic_member_checks.xlsx")
                            plot_utilization_on_structure(
                                nodes, elements, checks_seis,
                                title=f"Utilization map - EC8 seismic combination (q = {q_beh:.2f})",
                                geom=geom, walls=walls,
                            )
                            # the global member tables now include the seismic combinations
                            _base_checks = checks_by_combo if ("checks_by_combo" in globals() and checks_by_combo) else {}
                            df_worst = add_stirrup_columns_to_worst_table(
                                worst_table_to_dataframe({**_base_checks, **checks_seis}),
                                beam_sec=beam_sec, column_sec=column_sec, elements=elements)

                        if code_info["code"] == "EN1998-1" and geom.num_floors >= 2:
                            df_scwb = strong_column_weak_beam_check(
                                nodes=nodes, elements=elements, geom=geom, combos=seis_combos,
                                Ec_Pa=Ec_Pa, Es_Pa=Es_Pa, column_sec=column_sec, beam_sec=beam_sec,
                                ec2mat=ec2mat, walls=walls, wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa,
                                ops_build_model=ops_build_model,
                            )
                            print("\n=== EC8 CAPACITY DESIGN: ΣMRc ≥ 1.3 ΣMRb (EN 1998-1 4.4.2.3) ===")
                            print(df_scwb.round(2).to_string(index=False))
                            try:
                                df_scwb.to_excel("ec8_strong_column_weak_beam.xlsx", index=False)
                            except Exception as exc:
                                print(f"⚠ Could not export ec8_strong_column_weak_beam.xlsx: {exc}")
        ckpt.done("rsa")


    if ckpt.need("th"):
        if run_dynamics == "y":
            run_th = _yn("\nRun linear time-history analysis? (y/n): ")
            if run_th == "y":
                th_params = ask_time_history_params()

                master_nodes = ops_build_model(
                    nodes,
                    elements,
                    Ec_Pa,
                    Es_Pa,
                    column_sec,
                    beam_sec,
                    geom=geom,
                    use_rigid_diaphragm=use_diaphragm,
                    walls=walls,
                    wall_tw_mm=wall_tw_mm,
                    wall_E_Pa=wall_E_Pa,
                    stiffness_factor=crack_factor
                )

                ops_assign_floor_masses(
                    nodes,
                    geom,
                    floor_masses,
                    master_nodes=(master_nodes if use_diaphragm else None)
                )

                periods_th, _ = ops_run_eigen_and_print_and_plot(
                    geom,
                    nodes,
                    elements,
                    num_modes=nmodes,
                    ref_grid_x=0,
                    normalize=True,
                    plot_floor_shapes=False,
                    plot_structure_each_mode=False
                )

                batch_th = run_multiple_time_histories(
                    geom=geom,
                    nodes=nodes,
                    elements=elements,
                    periods=periods_th,
                    records=th_params["records"],
                    direction=th_params["direction"],
                    damping_ratio=th_params["damping_ratio"],
                    mode_i=th_params["mode_i"],
                    mode_j=th_params["mode_j"],
                    dt_analysis=th_params["dt_analysis"],
                )

                print("\n=== MULTI-RECORD TIME-HISTORY SUMMARY ===")
                print(batch_th["summary"].to_string(index=False))

                print("\n=== MULTI-RECORD PEAK STORY DRIFTS ===")
                print(batch_th["peak_drifts_all"].to_string(index=False))

                print("\n=== ENVELOPE STORY DRIFTS ===")
                print(batch_th["envelope_drifts_df"].to_string(index=False))

                batch_th["summary"].to_csv("time_history_batch_summary.csv", index=False)
                batch_th["peak_drifts_all"].to_csv("time_history_batch_peak_drifts.csv", index=False)
                batch_th["envelope_drifts_df"].to_csv("time_history_batch_envelope_drifts.csv", index=False)

                print("\n✔ TH batch summary exported: time_history_batch_summary.csv")
                print("✔ TH batch peak drifts exported: time_history_batch_peak_drifts.csv")
                print("✔ TH batch envelope drifts exported: time_history_batch_envelope_drifts.csv")

                show_th_plots = _yn("Plot roof displacement and base shear histories for each record? (y/n): ")
                if show_th_plots == "y":
                    for rec_name, res in batch_th["results_by_record"].items():
                        if len(res["roof_disp_history"]):
                            plot_time_history_response(
                                res["time"],
                                res["roof_disp_history"],
                                ylabel="Roof displacement [m]",
                                title=f"Linear TH - Roof displacement - {rec_name}"
                            )
                        if len(res["base_shear_history"]):
                            plot_time_history_response(
                                res["time"],
                                np.asarray(res["base_shear_history"], dtype=float) / 1000.0,
                                ylabel="Base shear [kN]",
                                title=f"Linear TH - Base shear - {rec_name}"
                            )

                export_multiple_time_history_results(
                    batch_th,
                    filename="time_history_batch_results.xlsx"
                )
        ckpt.done("th")


    if ckpt.need("perf"):
        # 15) Global performance summary
        run_perf_summary = _yn("\nBuild global performance summary (static + RSA + TH)? (y/n): ")
        if run_perf_summary == "y":
            static_uls_df = None
            if df_all is not None and not df_all.empty and "Case" in df_all.columns:
                static_uls_df = df_all[df_all["Case"].astype(str).str.startswith("ULS", na=False)].copy()
                if static_uls_df.empty:
                    static_uls_df = df_all.copy()

            rsa_single_df = None
            if rsa_results is not None:
                rsa_single_df = rsa_drifts_to_dataframe(rsa_results["story_drifts"], case_name="RSA")

            rsa_dir_df = df_dir_env.copy() if df_dir_env is not None else None

            th_env_df = None
            if batch_th is not None:
                th_env_df = batch_th["envelope_drifts_df"].copy()

            story_perf_df = build_story_performance_table(
                static_drifts_df=static_uls_df,
                rsa_single_drifts_df=rsa_single_df,
                rsa_dir_drifts_df=rsa_dir_df,
                th_envelope_drifts_df=th_env_df,
                ec8_single_df=df_ec8_single_all,
                ec8_dir_df=df_ec8_dir_all,
            )

            story_fail_df = build_story_failures_table(story_perf_df)

            member_perf_df = build_member_performance_table(
                ec2_worst_df=df_worst,
                rsa_single_results=rsa_results,
                rsa_dir_results=dir_results,
                batch_th=batch_th,
            )

            member_fail_df = build_member_failures_table(member_perf_df)

            global_summary_df = build_global_performance_summary(
                story_perf_df=story_perf_df,
                member_perf_df=member_perf_df,
            )

            if global_summary_df is not None and not global_summary_df.empty:
                print("\n=== GLOBAL PERFORMANCE SUMMARY ===")
                print(global_summary_df.to_string(index=False))

            if story_perf_df is not None and not story_perf_df.empty:
                print("\n=== STORY PERFORMANCE TABLE ===")
                print(story_perf_df.to_string(index=False))

            if story_fail_df is not None and not story_fail_df.empty:
                print("\n=== STORY FAILURES TABLE ===")
                print(story_fail_df.to_string(index=False))

            if member_perf_df is not None and not member_perf_df.empty:
                print("\n=== MEMBER PERFORMANCE TABLE ===")
                print(member_perf_df.to_string(index=False))

            if member_fail_df is not None and not member_fail_df.empty:
                print("\n=== MEMBER FAILURES TABLE ===")
                print(member_fail_df.to_string(index=False))

            export_performance_summary_results(
                story_perf_df=story_perf_df,
                story_fail_df=story_fail_df,
                member_perf_df=member_perf_df,
                member_fail_df=member_fail_df,
                global_summary_df=global_summary_df,
                filename="performance_summary.xlsx",
            )
        ckpt.done("perf")


    if ckpt.need("retrofit_search"):
        # 16) Retrofit optimization: shear walls first, then concrete jacketing
        run_retrofit_opt = _yn("\nRun retrofit optimization? (y/n): ")

        if run_retrofit_opt == "y" and ec2mat is None:
            print("⚠ Retrofit optimization needs the RC/EC2 material model; skipped for this structure type.")
            run_retrofit_opt = "n"

        if run_retrofit_opt == "y":
            if "story_fail_df" not in locals():
                story_fail_df = None
            if "member_fail_df" not in locals():
                member_fail_df = None

            # --------------------------------------------
            # A) Get failed drift stories
            # --------------------------------------------
            failed_stories = []
            if story_fail_df is not None and not story_fail_df.empty and "Story" in story_fail_df.columns:
                failed_stories = (
                    pd.to_numeric(story_fail_df["Story"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
                    .dropna()
                    .astype(int)
                    .unique()
                    .tolist()
                )

            if failed_stories:
                print("\nFailed drift stories:")
                print(failed_stories)
            else:
                print("\nNo failed drift stories detected from performance summary.")

            # --------------------------------------------
            # B-D) Global retrofit optimization
            # --------------------------------------------
            # The optimizer now compares complete strategies:
            #   wall layout + required jacketing + total retrofit cost.
            # Cost is the primary objective; drift, EC2 member safety,
            # constructability and jacketing limits are constraints/penalties.
            drift_limit_ratio = 0.005
            max_walls_per_layout = min(2, int(getattr(geom, "num_bays", 2)))
            wall_layout_mode = "hybrid"
            wall_thicknesses = [150.0, 200.0, 250.0, 300.0]

            # Runtime control: the user chooses how many screened wall layouts
            # enter the expensive global EC2 + jacketing stage. This prevents the
            # default run from becoming an exhaustive many-hour dataset run.
            print("\n--- Wall search method ---")
            print("  1) Screening: rank all layouts with a quick drift check, then full check of the best N")
            print("  2) Smart: Bayesian optimization (TPE) - each new layout learns from all previous trials")
            search_method = _num("Choose search method [default 2]: ", int, default=2, min_val=1, max_val=2)
            if search_method == 2:
                try:
                    import optuna  # noqa: F401
                except ImportError:
                    print("\n⚠ The smart search needs the 'optuna' package, which is not installed in this Python.")
                    print("  Install it by typing in the Spyder console:   %pip install optuna")
                    print("  then restart the kernel and resume this run from step 10.")
                    if _yn("Continue now with the screening method instead? (y/n) [n]: ", "n") == "y":
                        search_method = 1
                    else:
                        raise SystemExit("Stopped: install optuna, restart the kernel and resume from step 10.")
            smart_history_df = None
            if search_method == 2:
                smart_budget = _num("Number of smart trials (complete strategies) [default 60]: ", int,
                                    default=60, min_val=5, max_val=2000)
                max_global_candidates = None
            else:
                max_global_candidates = ask_max_global_candidates(default=20)

            retrofit_cost_parameters = ask_retrofit_cost_parameters(DEFAULT_UNIT_COSTS)

            # Jacket size limits
            print("\n--- Concrete jacketing limits ---")
            max_jacket_dim_ratio = _num("Max ratio jacketed / original section dimension [default 1.5]: ",
                                        float, default=1.5, min_val=1.05, max_val=3.0)
            max_jacket_t = _num("Max jacket thickness [mm] [default 200]: ", float, default=200.0,
                                min_val=75.0, max_val=400.0)

            # EC8 seismic action in the retrofit (lateral force method per candidate)
            seismic_settings = None
            if _yn("\nInclude the EC8 seismic action in the retrofit (lateral force method per candidate)? (y/n) [y]: ", "y") == "y":
                _prev = seis_settings if ("seis_settings" in globals() and seis_settings) else None
                if _prev is not None and _yn(
                        f"Use the seismic parameters of the RSA step (q = {_prev['q']:.2f}, "
                        f"ag = {_prev['rsa_params']['ag'] / 9.81:.3f} g)? (y/n) [y]: ", "y") == "y":
                    seismic_settings = dict(_prev)
                else:
                    seismic_settings = ask_seismic_settings(
                        geom, walls, code_info,
                        stiffness_factor=(crack_factor if "crack_factor" in globals() else None),
                        extra_mass_floor_kg=(extra_mass_floor if "extra_mass_floor" in globals() else 0.0),
                    )
                print(f"Seismic action: {seismic_settings['code']}, q = {seismic_settings['q']:.2f}, "
                      f"EI factor = {seismic_settings['stiffness_factor']:.2f}, DLS limit ν·dr/h ≤ {seismic_settings['dls_limit']}")

            print("\nRunning GLOBAL cost-minimizing retrofit optimization...")
            print(f"  Primary objective: minimum total retrofit cost")
            print(f"  Drift limit ratio: {drift_limit_ratio:.5f}")
            print(f"  Max wall segments per trial layout: {max_walls_per_layout}")
            print(f"  Vertical layout search mode: {wall_layout_mode}")
            print(f"  Wall thickness candidates [mm]: {wall_thicknesses}")
            if search_method == 2:
                print(f"  Search: Bayesian optimization, {smart_budget} trials")
            elif max_global_candidates is None:
                print("  Global wall candidates evaluated: ALL screened candidates plus no-new-wall option")
            else:
                print(f"  Global wall candidates evaluated: best {max_global_candidates} screened candidates plus no-new-wall option")

            retrofit_kwargs = dict(
                nodes=nodes,
                elements=elements,
                geom=geom,
                combos=combos,
                combo_names=names,
                Ec_Pa=Ec_Pa,
                Es_Pa=Es_Pa,
                column_sec=column_sec,
                beam_sec=beam_sec,
                ec2mat=ec2mat,
                ops_build_model=ops_build_model,
                ops_apply_combo_loads=ops_apply_combo_loads,
                ops_run_static=ops_run_static,
                base_walls=walls,
                wall_tw_mm=wall_tw_mm,
                wall_E_Pa=wall_E_Pa,
                use_rigid_diaphragm=False,
                drift_limit_ratio=drift_limit_ratio,
                failed_stories=failed_stories if failed_stories else None,
                thicknesses_mm=wall_thicknesses,
                max_walls_per_layout=max_walls_per_layout,
                wall_layout_mode=wall_layout_mode,
                include_no_new_wall_candidate=True,
                max_global_candidates=max_global_candidates,
                unit_costs=retrofit_cost_parameters,
                concrete_density_kgm3=material.density,
                wall_density_kgm3=wall_density,
                seismic_settings=seismic_settings,
                max_jacketed_section_dimension_ratio=max_jacket_dim_ratio,
                max_jacket_thickness_mm=max_jacket_t,
            )
            if search_method == 2:
                from framekit.smart_search import smart_wall_search
                global_opt_df, smart_history_df = smart_wall_search(
                    global_kwargs=retrofit_kwargs, geom=geom, base_walls=walls, wall_E_Pa=wall_E_Pa,
                    thicknesses_mm=wall_thicknesses, n_trials=smart_budget,
                    max_segments=max_walls_per_layout,
                    failed_stories=failed_stories if failed_stories else None,
                )
                try:
                    smart_history_df.drop(columns=["_segments"], errors="ignore").to_csv(
                        os.path.join(run_dir, "smart_search_history.csv"), index=False)
                except Exception:
                    pass
                try:
                    from framekit.smart_search import show_smart_search_plots
                    show_smart_search_plots(smart_history_df, geom, global_opt_df)
                except Exception as exc:
                    print(f"⚠ smart-search plots failed: {exc}")
            else:
                global_opt_df = optimize_global_retrofit_strategy(**retrofit_kwargs)
        # keep the heavy per-candidate objects only for the 5 best strategies
        # (otherwise the saved step can reach several GB with many candidates)
        if "global_opt_df" in globals() and global_opt_df is not None and not global_opt_df.empty:
            for _c in [c for c in global_opt_df.columns if str(c).startswith("_")]:
                global_opt_df[_c] = global_opt_df[_c].astype(object)
                global_opt_df.loc[global_opt_df.index[5:], _c] = None
        ckpt.done("retrofit_search")


    if ckpt.need("retrofit_select"):
        if run_retrofit_opt == "y":
            if global_opt_df is None or global_opt_df.empty:
                print("\nNo global retrofit candidates were generated.")
                optimized_walls = walls
                optimized_wall_tw_mm = wall_tw_mm
                optimized_wall_E_Pa = wall_E_Pa
                df_worst_after_sw = None
                df_worst_final = None
                member_fail_after_sw = None
                member_fail_final = None
                jacketed_element_sections = None
                all_jacketed_member_ids = []
                jacketing_rows = []
                member_jacketing_report = None
                jacket_cost = None
                jacketing_iteration_worst_dfs = []
                sw_opt_df = None
            else:
                internal_cols = [
                    "_candidate", "_drifts", "_baseline_drifts", "_walls",
                    "_df_worst_after_walls", "_member_perf_after_walls", "_member_fail_after_walls",
                    "_cj_opt_df", "_df_worst_final", "_member_fail_final", "_element_sections",
                    "_member_report", "_jacket_cost", "_jacketing_iteration_worst_dfs",
                ]
                show_cols = [c for c in [
                    "Global_Rank", "Candidate_ID", "Description", "Complete_Feasible",
                    "Search_Max_Global_Candidates", "Search_Candidates_Sent_To_Global_Stage",
                    "Drift_Utilization", "Member_Failures_After_Walls", "Jacketing_Required",
                    "Final_Max_EC2_Utilization", "Final_Failed_Members",
                    "Wall_Cost", "Jacketing_Cost", "Total_Retrofit_Cost",
                    "Constructability_Class", "Penalty_Jacketing_Limit", "Global_Objective_Score",
                ] if c in global_opt_df.columns]
                print("\n=== GLOBAL RETROFIT OPTIMIZATION RESULTS ===")
                print(global_opt_df[show_cols].to_string(index=False))

                top5_cols = [c for c in [
                    "Global_Rank", "Selected", "Candidate_ID", "Complete_Feasible",
                    "Wall_Cost", "Jacketing_Cost", "Total_Retrofit_Cost",
                    "Drift_Utilization", "Final_Max_EC2_Utilization", "Final_Failed_Members",
                    "Constructability_Class",
                ] if c in global_opt_df.columns]
                print("\n=== TOP 5 COMPLETE RETROFIT ALTERNATIVES FOR REPORT ===")
                print(global_opt_df.head(5)[top5_cols].to_string(index=False))

                global_export_df = global_opt_df.drop(columns=internal_cols, errors="ignore")
                global_export_df.to_csv("retrofit_global_optimization.csv", index=False)
                global_export_df.head(5).to_csv("retrofit_top5_alternatives.csv", index=False)

                # Step 1 toward a research/ML workflow: save every evaluated
                # complete retrofit strategy in a clean dataset format.
                optimization_dataset_df = build_global_optimization_dataset(
                    global_opt_df,
                    structure_id="current_structure",
                    include_cost_assumptions=True,
                )
                if optimization_dataset_df is not None and not optimization_dataset_df.empty:
                    optimization_dataset_df.to_csv("global_retrofit_optimization_dataset_later_ML.csv", index=False)
                    try:
                        optimization_dataset_df.to_excel("global_retrofit_optimization_dataset_later_ML.xlsx", index=False)
                        print("✔ Exported: global_retrofit_optimization_dataset_later_ML.xlsx")
                    except Exception as exc:
                        print(f"⚠ Could not export XLSX optimization dataset: {exc}")

                print("\n✔ Exported: retrofit_global_optimization.csv")
                print("✔ Exported: retrofit_top5_alternatives.csv")
                if optimization_dataset_df is not None and not optimization_dataset_df.empty:
                    print("✔ Exported: global_retrofit_optimization_dataset_later_ML.csv")

                best_global = global_opt_df.iloc[0]
                best_sw = best_global.get("_candidate", None)
                optimized_walls = best_global.get("_walls", list(walls or []))
                optimized_wall_tw_mm = float(getattr(best_sw, "wall_tw_mm", wall_tw_mm)) if best_sw is not None else wall_tw_mm
                optimized_wall_E_Pa = float(getattr(best_sw, "wall_E_Pa", wall_E_Pa)) if best_sw is not None else wall_E_Pa

                df_worst_after_sw = best_global.get("_df_worst_after_walls", None)
                member_fail_after_sw = best_global.get("_member_fail_after_walls", None)
                df_worst_final = best_global.get("_df_worst_final", df_worst_after_sw)
                member_fail_final = best_global.get("_member_fail_final", member_fail_after_sw)
                jacketed_element_sections = best_global.get("_element_sections", None)
                member_jacketing_report = best_global.get("_member_report", None)
                jacket_cost = best_global.get("_jacket_cost", None)
                jacketing_iteration_worst_dfs = best_global.get("_jacketing_iteration_worst_dfs", []) or []
                cj_opt_df = best_global.get("_cj_opt_df", pd.DataFrame())

                if cj_opt_df is not None and not getattr(cj_opt_df, "empty", True):
                    jacketing_rows = cj_opt_df.drop(
                        columns=["_element_sections", "_df_worst", "_member_perf", "_member_fail", "_member_report", "_original_df_worst", "_original_member_fail"],
                        errors="ignore"
                    ).to_dict("records")
                else:
                    jacketing_rows = []

                all_jacketed_member_ids = [
                    int(x) for x in str(best_global.get("Jacketed_Members", "")).split(",")
                    if str(x).strip()
                ]
                failed_member_ids = list(all_jacketed_member_ids)

                # Use the selected global table as the selected wall/strategy table for report compatibility.
                sw_opt_df = global_opt_df.copy()

                print("\nSelected global retrofit strategy:")
                print(f"  Candidate: {best_global.get('Candidate_ID', '')}")
                print(f"  Wall layout: {best_global.get('Description', '')}")
                print(f"  Complete feasible: {best_global.get('Complete_Feasible', '')}")
                print(f"  Wall cost: {float(best_global.get('Wall_Cost', 0.0)):.2f}")
                print(f"  Jacketing cost: {float(best_global.get('Jacketing_Cost', 0.0)):.2f}")
                print(f"  Total retrofit cost: {float(best_global.get('Total_Retrofit_Cost', 0.0)):.2f}")
                print(f"  Final failed members: {best_global.get('Final_Failed_Members', '')}")
                if seismic_settings:
                    print(f"  EC8 seismic (LFM): T1 = {float(best_global.get('Seismic_T1_s', float('nan'))):.3f} s, "
                          f"Fb = {float(best_global.get('Seismic_Fb_kN', float('nan'))):.1f} kN, "
                          f"DLS utilization = {float(best_global.get('Seismic_DLS_Utilization', float('nan'))):.2f}, "
                          f"θmax = {float(best_global.get('Seismic_Theta_Max', float('nan'))):.3f}")

                if best_sw is not None:
                    plot_shear_wall_retrofit_layout(
                        nodes,
                        elements,
                        geom,
                        base_walls=walls,
                        selected_candidate=best_sw,
                        title="Global Optimized Retrofit Layout - Selected Shear Walls",
                    )

                if member_jacketing_report is not None and not getattr(member_jacketing_report, "empty", True):
                    print("\n=== PER-MEMBER JACKETING DECISION REPORT ===")
                    print(member_jacketing_report.to_string(index=False))
                    member_jacketing_report.to_csv("retrofit_jacketing_member_report.csv", index=False)
                    print("✔ Exported: retrofit_jacketing_member_report.csv")

                if member_fail_after_sw is not None and not getattr(member_fail_after_sw, "empty", True):
                    print("\n=== MEMBERS FAILING AFTER SELECTED WALL LAYOUT ===")
                    print(member_fail_after_sw.to_string(index=False))
                else:
                    print("\nNo member failures after selected wall layout.")

                if member_fail_final is not None and not getattr(member_fail_final, "empty", True):
                    print("\n=== MEMBERS STILL FAILING AFTER GLOBAL RETROFIT STRATEGY ===")
                    fail_cols = [c for c in [
                        "Element", "Type", "eta", "etaV", "eta_NM",
                        "Static_EC2_Utilization", "NEd_kN", "VEd_kN", "MEd_kNm",
                    ] if c in member_fail_final.columns]
                    print(member_fail_final[fail_cols].to_string(index=False))
                else:
                    print("\n✔ All members verified after selected global retrofit strategy.")

                if df_worst_final is not None and not getattr(df_worst_final, "empty", True):
                    plot_final_retrofit_utilization(
                        nodes, elements, df_worst_final,
                        title="Final Member Utilization After Global Retrofit Strategy",
                        geom=geom, walls=optimized_walls,
                        highlight_ids=all_jacketed_member_ids,
                    )
                    export_ec2_worst_table_colored(
                        df_worst_final,
                        filename="ec2_worst_utilization_after_retrofit.xlsx"
                    )
                    print("✔ Exported: ec2_worst_utilization_after_retrofit.xlsx")

                if all_jacketed_member_ids:
                    plot_jacketed_members(
                        nodes, elements, all_jacketed_member_ids,
                        title="Concrete Jacketing Layout - Global Cost-Selected Members",
                        geom=geom, walls=optimized_walls,
                    )
                    plot_jacketed_sections_now = _yn(
                        "\nPlot before/after RC sections for jacketed members? (y/n) [y]: ", "y"
                    )
                    if plot_jacketed_sections_now == "y" and jacketed_element_sections is not None:
                        plot_jacketed_section_comparisons(
                            elements=elements,
                            column_sec=column_sec,
                            beam_sec=beam_sec,
                            jacketed_element_sections=jacketed_element_sections,
                            jacketed_member_ids=all_jacketed_member_ids,
                        )
        ckpt.done("retrofit_select")


    if ckpt.need("report"):
        if run_retrofit_opt == "y":
            # --------------------------------------------
            # E) Professional retrofit report
            # --------------------------------------------
            make_report = _yn("\nGenerate the retrofit report (Word + Excel)? (y/n) [y]: ", "y")
            if make_report == "y":
                print("\nGenerating the retrofit report...")
                try:
                    report = export_retrofit_report_v2(
                        output_dir=report_dir,
                        run_name=run_name,
                        nodes=nodes, elements=elements, geom=geom,
                        combos=combos, names=names, Ec_Pa=Ec_Pa, Es_Pa=Es_Pa,
                        column_sec=column_sec, beam_sec=beam_sec, ec2mat=ec2mat,
                        ops_build_model=ops_build_model, ops_apply_combo_loads=ops_apply_combo_loads,
                        ops_run_static=ops_run_static,
                        base_walls=walls, wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa,
                        concrete_density=material.density, wall_density=wall_density,
                        drift_limit=drift_limit_ratio,
                        smart_history=(smart_history_df if "smart_history_df" in globals() else None),
                    global_opt_df=(global_opt_df if "global_opt_df" in globals() else None),
                        df_worst_before=df_worst,
                        seismic_settings=(seismic_settings if "seismic_settings" in globals() else None),
                        code_info=code_info,
                        logo_polito=os.path.join(current_dir, "Logo_PoliTo_dal_2021_blu.png"),
                        logo_artiste=os.path.join(current_dir, "artiste_logo_ridotto-768x512.jpg"),
                        logo_arcopy=os.path.join(current_dir, "arcopy_logo.png"),
                        title="Seismic Assessment and Retrofit Optimization Report",
                        subtitle="AI and Structural Engineering",
                        student="Arman Hajiha",
                        role="Research assistant at Politecnico di Torino",
                        supervisor="Prof. Giuseppe Carlo Marano",
                    )
                except PermissionError as exc:
                    print(f"\n❌ Report could not be written (is the Word/Excel file open?): {exc}")
                    print("   Close the file and run main.py again: press Enter to resume at the report step.")
                    raise
                except Exception as exc:
                    print(f"\n❌ Report generation failed: {exc}")
                    print("   All previous steps are saved: run main.py again and press Enter to resume at the report step.")
                    raise

            print("\nRetrofit optimization stage completed.")
        ckpt.done("report")

    ckpt.finish()
