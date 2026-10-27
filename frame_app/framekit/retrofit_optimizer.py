from __future__ import annotations

from dataclasses import dataclass, field
from copy import copy, deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import inspect
import math
from itertools import combinations

import numpy as np
import pandas as pd

from framekit.data_models import ShearWall
from framekit.geometry import node_id_at
from framekit.static_analysis import get_node_disps, compute_story_drifts, drifts_to_dataframe
from framekit.ec2_checks import run_combo_and_check_all, worst_table_to_dataframe, add_stirrup_columns_to_worst_table
from framekit.performance_summary import build_member_performance_table, build_member_failures_table


# ============================================================
# DATA OBJECTS
# ============================================================
@dataclass
class ShearWallCandidate:
    candidate_id: str
    walls: List[ShearWall]
    wall_tw_mm: float
    wall_E_Pa: float
    description: str = ""


@dataclass
class JacketingOption:
    option_id: str
    element_ids: List[int]
    t_jacket_mm: float
    fc_jacket_mpa: Optional[float] = None
    longitudinal_bar_increase_ratio: float = 0.0
    stirrup_spacing_mm: Optional[float] = None


@dataclass
class RetrofitCostBreakdown:
    concrete_m3: float = 0.0
    rebar_kg: float = 0.0
    formwork_m2: float = 0.0
    material_cost: float = 0.0
    labor_cost: float = 0.0
    indirect_cost: float = 0.0
    total_cost: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


DEFAULT_UNIT_COSTS = {
    # Material unit costs
    "wall_concrete_per_m3": 180.0,
    "wall_rebar_per_kg": 2.20,
    "wall_formwork_per_m2": 35.0,
    "jacket_concrete_per_m3": 220.0,
    "jacket_rebar_per_kg": 2.40,
    "jacket_formwork_per_m2": 45.0,

    # Percentage add-ons, expressed as decimal ratios
    "labor_factor": 0.30,
    "indirect_factor": 0.10,

    # Fixed/constructability allowances
    "foundation_cost_per_wall_line": 2500.0,
    "constructability_cost_per_wall_line": 1000.0,

    # Time/productivity assumptions used for site-overhead cost
    "site_overhead_per_day": 300.0,
    "wall_productivity_m3_per_day": 3.0,
    "jacketing_productivity_m3_per_day": 1.0,
    "jacketing_productivity_members_per_day": 2.0,
}


_COST_PARAMETER_LABELS = [
    ("wall_concrete_per_m3", "Wall concrete cost", "EUR/m3", "material cost for added shear-wall concrete"),
    ("wall_rebar_per_kg", "Wall reinforcement cost", "EUR/kg", "material cost for shear-wall reinforcement"),
    ("wall_formwork_per_m2", "Wall formwork cost", "EUR/m2", "two-sided wall formwork cost"),
    ("jacket_concrete_per_m3", "Jacketing concrete cost", "EUR/m3", "material cost for concrete jacket"),
    ("jacket_rebar_per_kg", "Jacketing reinforcement cost", "EUR/kg", "jacket reinforcement cost"),
    ("jacket_formwork_per_m2", "Jacketing formwork cost", "EUR/m2", "formwork cost for jacketing"),
    ("labor_factor", "Labor factor", "%", "percentage of direct material/formwork cost"),
    ("indirect_factor", "Indirect factor", "%", "percentage of direct material/formwork cost"),
    ("foundation_cost_per_wall_line", "Foundation/connection allowance", "EUR/wall line", "extra cost for each new wall line"),
    ("constructability_cost_per_wall_line", "Constructability allowance", "EUR/wall line", "coordination/detailing allowance per wall line"),
    ("site_overhead_per_day", "Site overhead", "EUR/day", "daily site overhead due to retrofit duration"),
    ("wall_productivity_m3_per_day", "Wall productivity", "m3/day", "added wall concrete volume placed per day"),
    ("jacketing_productivity_m3_per_day", "Jacketing volume productivity", "m3/day", "jacket concrete volume placed per day"),
    ("jacketing_productivity_members_per_day", "Jacketing member productivity", "members/day", "number of members jacketed per day"),
]


def _format_cost_parameter_value(key: str, value: float) -> str:
    if key in {"labor_factor", "indirect_factor"}:
        return f"{100.0 * float(value):.1f}"
    if abs(float(value)) >= 100.0:
        return f"{float(value):.1f}"
    return f"{float(value):.3g}"


def print_retrofit_cost_parameters(unit_costs: Optional[Dict[str, float]] = None) -> None:
    """Print the retrofit cost assumptions before optimization starts."""
    params = dict(DEFAULT_UNIT_COSTS)
    if unit_costs:
        params.update({k: float(v) for k, v in unit_costs.items() if k in params})

    print("\n=== RETROFIT COST PARAMETERS USED FOR OPTIMIZATION ===")
    print("These values are used for comparing retrofit alternatives.")
    print("You can keep them as default values or edit them before optimization.\n")
    print(f"{'Parameter':38s} {'Value':>14s}  Unit")
    print("-" * 64)
    for key, label, unit, _desc in _COST_PARAMETER_LABELS:
        val = _format_cost_parameter_value(key, params.get(key, 0.0))
        print(f"{label:38s} {val:>14s}  {unit}")
    print("-" * 64)


def ask_max_global_candidates(default: int = 20) -> Optional[int]:
    """Ask how many screened wall candidates should enter the expensive global stage.

    The global optimizer first ranks generated wall layouts by the wall/drift
    screening step. Then only the best N screened candidates are sent to the
    expensive EC2 + concrete-jacketing stage. The no-new-wall jacketing-only
    option is added separately when enabled.

    Returning None means exhaustive mode: all screened wall candidates are
    sent to the global stage.
    """
    default = max(1, int(default or 20))
    print("\n=== RETROFIT OPTIMIZATION SEARCH SIZE ===")
    print("This controls the expensive global stage: EC2 checks + automatic jacketing.")
    print("The no-new-wall jacketing-only option is added separately.")
    print("Suggested values:")
    print("  10  = fast engineering run")
    print("  20  = balanced research run [recommended default]")
    print("  50  = larger research dataset")
    print("  ALL = exhaustive later-ML dataset mode; can take many hours")
    while True:
        raw = input(f"Maximum screened wall candidates for global optimization [{default}, or ALL]: " ).strip()
        if not raw:
            value = default
            print(f"Using maximum {value} screened wall candidates for the expensive global stage.\n")
            return value
        raw_l = raw.lower()
        if raw_l in {"all", "a", "none", "no limit", "nolimit"}:
            print("Using exhaustive mode: all screened wall candidates will enter the expensive global stage.\n")
            return None
        try:
            value = int(float(raw_l.replace(",", ".")))
            if value < 1:
                print("Please enter a positive integer, or ALL for exhaustive mode.")
                continue
            print(f"Using maximum {value} screened wall candidates for the expensive global stage.\n")
            return value
        except Exception:
            print("Invalid input. Enter a positive integer such as 20, or ALL.")


def _format_global_candidate_limit(value: Optional[int]) -> str:
    return "ALL" if value is None else str(int(value))


def ask_retrofit_cost_parameters(defaults: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Ask whether the user wants to keep or edit retrofit cost assumptions.

    The defaults are printed first, so the user can see the basic values before
    deciding. If the user presses Enter or answers 'y', the default values are
    used. If the user answers 'n', every parameter can be edited interactively.
    """
    params = dict(DEFAULT_UNIT_COSTS)
    if defaults:
        params.update({k: float(v) for k, v in defaults.items() if k in params})

    print_retrofit_cost_parameters(params)
    use_default = input("Use default retrofit cost parameters? (y/n) [y]: ").strip().lower() or "y"
    if use_default != "n":
        print("Using default retrofit cost parameters.\n")
        return params

    print("\nEnter new values. Press Enter to keep the shown default.")
    for key, label, unit, _desc in _COST_PARAMETER_LABELS:
        current = params[key]
        shown = _format_cost_parameter_value(key, current)
        raw = input(f"{label} [{unit}] default {shown}: ").strip()
        if not raw:
            continue
        try:
            value = float(raw.replace(",", "."))
            if key in {"labor_factor", "indirect_factor"}:
                # User enters percentages, e.g. 30 means 30%.
                value = value / 100.0
            params[key] = value
        except Exception:
            print(f"  Invalid value. Keeping default {shown} {unit}.")

    print_retrofit_cost_parameters(params)
    return params


DEFAULT_SW_OBJECTIVE_WEIGHTS = {
    # Cost is normalized by the most expensive candidate in the current search.
    "cost": 0.35,
    # Drift is based on max_drift / drift_limit, with strong penalty above 1.0.
    "drift": 0.35,
    # Constructability penalizes elevated/discontinuous wall segments.
    "constructability": 0.20,
    # Wall quantity discourages unnecessarily large/heavy layouts.
    "wall_quantity": 0.10,
}


# ============================================================
# GENERIC HELPERS
# ============================================================
def _call_build_model(
    build_model_fn: Callable,
    *,
    nodes,
    elements,
    Ec_Pa,
    Es_Pa,
    column_sec,
    beam_sec,
    geom,
    use_rigid_diaphragm=False,
    walls=None,
    wall_tw_mm=200.0,
    wall_E_Pa=30e9,
    element_sections=None,
):
    """Call ops_build_model while staying compatible with old and patched signatures."""
    kwargs = dict(
        nodes=nodes,
        elements=elements,
        Ec_Pa=Ec_Pa,
        Es_Pa=Es_Pa,
        column_sec=column_sec,
        beam_sec=beam_sec,
        geom=geom,
        use_rigid_diaphragm=use_rigid_diaphragm,
        walls=walls,
        wall_tw_mm=wall_tw_mm,
        wall_E_Pa=wall_E_Pa,
    )
    sig = inspect.signature(build_model_fn)
    if "element_sections" in sig.parameters:
        kwargs["element_sections"] = element_sections
    return build_model_fn(**kwargs)


def _seismic_eval(seismic_settings, **kw):
    """EC8 lateral-force-method evaluation; returns None if disabled/failed."""
    if not seismic_settings:
        return None
    from framekit.ec8_seismic import evaluate_lateral_force_method
    try:
        return evaluate_lateral_force_method(settings=seismic_settings, **kw)
    except Exception as exc:
        print(f"WARNING: EC8 seismic evaluation failed: {exc}")
        return None


def _combos_with_extra_self_weight(
    combos,
    *,
    elements,
    column_sec,
    beam_sec,
    element_sections=None,
    concrete_density_kgm3=None,
    nodes=None,
    geom=None,
    new_walls=None,
    wall_tw_mm=200.0,
    wall_density_kgm3=None,
):
    """Return combos with the self-weight of jackets / new wall panels added.

    The extra loads are scaled by each combination's own "SW" factor, so ULS
    (1.35) and SLS (1.0) remain consistent with the original SW case.
    """
    from framekit.data_models import Load
    from framekit.constants import G_STD
    from framekit.combinations import scale_loads
    from framekit.loads import compute_wall_self_weight

    extra = []
    if concrete_density_kgm3 and element_sections:
        for eid, sec in element_sections.items():
            _ni, _nj, etype = elements[int(eid)]
            base = column_sec if etype == "column" else beam_sec
            dA_m2 = (_section_b_mm(sec) * _section_h_mm(sec) - _section_b_mm(base) * _section_h_mm(base)) * 1e-6
            if dA_m2 > 0.0:
                w_kNpm = float(concrete_density_kgm3) * G_STD * dA_m2 / 1000.0
                extra.append(Load("uniform", -w_kNpm, element=int(eid), direction="vertical"))
    if wall_density_kgm3 and new_walls:
        extra += compute_wall_self_weight(nodes, geom, new_walls, wall_tw_mm, wall_density_kgm3)
    if not extra:
        return combos
    out = {}
    for name, c in combos.items():
        f_sw = float((c.get("factors") or {}).get("SW", 1.0))
        out[name] = {**c, "loads": list(c["loads"]) + scale_loads(extra, f_sw)}
    return out


def _get_attr(obj, names: Iterable[str], default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _set_existing_or_new_attr(obj, preferred_name: str, fallback_names: Iterable[str], value):
    for name in fallback_names:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return
    setattr(obj, preferred_name, value)


def _section_b_mm(sec) -> float:
    return float(_get_attr(sec, ["b_mm", "b"], 0.0))


def _section_h_mm(sec) -> float:
    return float(_get_attr(sec, ["h_mm", "h"], 0.0))


def _section_perimeter_m(sec) -> float:
    return 2.0 * (_section_b_mm(sec) + _section_h_mm(sec)) / 1000.0


def _element_length_m(nodes, elements, eid: int) -> float:
    ni, nj, _ = elements[eid]
    x1, y1 = nodes[ni]
    x2, y2 = nodes[nj]
    return float(math.hypot(x2 - x1, y2 - y1))


def _max_numeric_column(df: Optional[pd.DataFrame], candidates: List[str]) -> float:
    if df is None or df.empty:
        return np.nan
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                return float(vals.max())
    return np.nan


def _failed_member_ids(member_fail_df: Optional[pd.DataFrame]) -> List[int]:
    if member_fail_df is None or member_fail_df.empty or "Element" not in member_fail_df.columns:
        return []
    vals = pd.to_numeric(member_fail_df["Element"], errors="coerce").dropna().astype(int)
    return sorted(vals.unique().tolist())


# ============================================================
# STATIC DRIFT ASSESSMENT
# ============================================================
def run_static_drift_assessment(
    *,
    nodes,
    elements,
    geom,
    combos: Dict[str, Dict[str, Any]],
    combo_names: List[str],
    Ec_Pa: float,
    Es_Pa: float,
    column_sec,
    beam_sec,
    ops_build_model,
    ops_apply_combo_loads,
    ops_run_static,
    walls=None,
    wall_tw_mm: float = 200.0,
    wall_E_Pa: float = 30e9,
    use_rigid_diaphragm: bool = False,
    element_sections: Optional[Dict[int, Any]] = None,
    combo_prefix: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Run static analyses and return the combined drift table."""
    all_drift_tables = []
    results_ok = []
    results_failed = []

    for cname in combo_names:
        if combo_prefix and not str(cname).startswith(combo_prefix):
            continue

        loads_combo = combos[cname]["loads"]

        _call_build_model(
            ops_build_model,
            nodes=nodes,
            elements=elements,
            Ec_Pa=Ec_Pa,
            Es_Pa=Es_Pa,
            column_sec=column_sec,
            beam_sec=beam_sec,
            geom=geom,
            use_rigid_diaphragm=use_rigid_diaphragm,
            walls=walls,
            wall_tw_mm=wall_tw_mm,
            wall_E_Pa=wall_E_Pa,
            element_sections=element_sections,
        )
        ops_apply_combo_loads(nodes, elements, loads_combo)
        ok = ops_run_static()

        if ok != 0:
            results_failed.append(cname)
            continue

        results_ok.append(cname)
        U = get_node_disps(nodes)
        drifts = compute_story_drifts(geom, nodes, U)
        all_drift_tables.append(drifts_to_dataframe(drifts, cname))

    if all_drift_tables:
        return pd.concat(all_drift_tables, ignore_index=True), results_ok, results_failed
    return pd.DataFrame(), results_ok, results_failed


# ============================================================
# EC2 MEMBER ASSESSMENT
# ============================================================
def run_ec2_member_assessment(
    *,
    nodes,
    elements,
    geom,
    combos: Dict[str, Dict[str, Any]],
    combo_names: List[str],
    Ec_Pa: float,
    Es_Pa: float,
    column_sec,
    beam_sec,
    ec2mat,
    ops_build_model,
    ops_apply_combo_loads,
    ops_run_static,
    walls=None,
    wall_tw_mm: float = 200.0,
    wall_E_Pa: float = 30e9,
    use_rigid_diaphragm: bool = False,
    element_sections: Optional[Dict[int, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run EC2 checks for ULS combinations and return worst/member/failure tables."""
    checks_by_combo = {}

    run_check_sig = inspect.signature(run_combo_and_check_all)
    supports_element_sections = "element_sections" in run_check_sig.parameters

    if element_sections is not None and not supports_element_sections:
        print("WARNING: element_sections were provided, but current run_combo_and_check_all does not support them yet.")

    for cname in combo_names:
        if not str(cname).startswith(("ULS", "SEIS")):
            continue

        check_kwargs = dict(
            geom=geom,
            use_rigid_diaphragm=use_rigid_diaphragm,
            walls=walls,
            wall_tw_mm=wall_tw_mm,
            wall_E_Pa=wall_E_Pa,
        )
        if supports_element_sections:
            check_kwargs["element_sections"] = element_sections

        checks, ok = run_combo_and_check_all(
            nodes,
            elements,
            combos[cname]["loads"],
            Ec_Pa,
            Es_Pa,
            column_sec,
            beam_sec,
            ec2mat,
            ops_build_model,
            ops_apply_combo_loads,
            ops_run_static,
            **check_kwargs,
        )
        if ok == 0 and checks is not None:
            checks_by_combo[cname] = checks

    if not checks_by_combo:
        empty = pd.DataFrame()
        return empty, empty, empty

    df_worst = worst_table_to_dataframe(checks_by_combo)
    df_worst = add_stirrup_columns_to_worst_table(
        df_worst,
        beam_sec=beam_sec,
        column_sec=column_sec,
        elements=elements,
    )
    member_perf_df = build_member_performance_table(ec2_worst_df=df_worst)
    member_fail_df = build_member_failures_table(member_perf_df)
    return df_worst, member_perf_df, member_fail_df


# ============================================================
# SHEAR WALL CANDIDATES AND COST
# ============================================================
def _target_top_story_from_failed_stories(geom, failed_stories: Optional[List[int]]) -> int:
    """Return the highest story to be considered by the wall candidate generator."""
    n_floors = int(getattr(geom, "num_floors"))
    if failed_stories:
        valid = [int(s) for s in failed_stories if 1 <= int(s) <= n_floors]
        if valid:
            return max(1, min(max(valid), n_floors))
    return n_floors


def _contiguous_failed_story_blocks(failed_stories: Optional[List[int]], target_top_story: int) -> List[Tuple[int, int]]:
    """Return contiguous failed-story blocks as (start_story, end_story)."""
    if not failed_stories:
        return []
    vals = sorted({int(s) for s in failed_stories if 1 <= int(s) <= target_top_story})
    if not vals:
        return []
    blocks: List[Tuple[int, int]] = []
    start = prev = vals[0]
    for st in vals[1:]:
        if st == prev + 1:
            prev = st
        else:
            blocks.append((start, prev))
            start = prev = st
    blocks.append((start, prev))
    return blocks


def _wall_segment_catalog(
    *,
    n_bays: int,
    target_top_story: int,
    failed_stories: Optional[List[int]],
    wall_layout_mode: str,
) -> List[Tuple[int, int, int]]:
    """Build candidate wall segments as (bay, start_story, end_story).

    Story numbers refer to wall panels: start_story=1 means the wall starts
    at the foundation and covers the panel from floor 0 to floor 1.  A segment
    (bay, 2, 4) covers story panels 2, 3 and 4 only; this is useful for
    research sensitivity studies, but it is less constructable unless a transfer
    mechanism exists below the first panel.
    """
    mode = str(wall_layout_mode or "bottom_anchored").strip().lower()
    if mode in {"continuous", "full", "fullheight", "full_height"}:
        mode = "full_height"
    elif mode in {"anchored", "from_bottom", "bottom", "bottomanchored"}:
        mode = "bottom_anchored"
    elif mode in {"segments", "flexible", "vertical", "vertical_segment"}:
        mode = "vertical_segments"
    elif mode in {"single", "single_story", "story"}:
        mode = "single_story"
    elif mode in {"failed", "failed_blocks", "failed_story_blocks"}:
        mode = "failed_story_blocks"
    elif mode in {"hybrid", "all"}:
        mode = "hybrid"

    segments: set[Tuple[int, int, int]] = set()

    def add_for_all_bays(start_story: int, end_story: int) -> None:
        if 1 <= start_story <= end_story <= target_top_story:
            for bay in range(1, n_bays + 1):
                segments.add((bay, int(start_story), int(end_story)))

    if mode in {"full_height", "hybrid"}:
        add_for_all_bays(1, target_top_story)

    if mode in {"bottom_anchored", "hybrid"}:
        for end_story in range(1, target_top_story + 1):
            add_for_all_bays(1, end_story)

    if mode in {"single_story", "hybrid"}:
        for story in range(1, target_top_story + 1):
            add_for_all_bays(story, story)

    if mode in {"failed_story_blocks", "hybrid"}:
        for start_story, end_story in _contiguous_failed_story_blocks(failed_stories, target_top_story):
            add_for_all_bays(start_story, end_story)

    if mode == "vertical_segments":
        for start_story in range(1, target_top_story + 1):
            for end_story in range(start_story, target_top_story + 1):
                add_for_all_bays(start_story, end_story)

    if not segments:
        # Safe fallback: reproduce the old behaviour.
        add_for_all_bays(1, target_top_story)

    return sorted(segments, key=lambda x: (x[0], x[1], x[2]))


def _wall_segments_are_non_overlapping(segments: Iterable[Tuple[int, int, int]]) -> bool:
    """Avoid duplicate/overlapping wall panels in the same bay."""
    occupied: set[Tuple[int, int]] = set()
    for bay, start_story, end_story in segments:
        for story in range(int(start_story), int(end_story) + 1):
            key = (int(bay), int(story))
            if key in occupied:
                return False
            occupied.add(key)
    return True


def _walls_from_segments(segments: Iterable[Tuple[int, int, int]]) -> List[ShearWall]:
    walls: List[ShearWall] = []
    for bay, start_story, end_story in segments:
        for story in range(int(start_story), int(end_story) + 1):
            walls.append(
                ShearWall(
                    bay_left=int(bay),
                    bay_right=int(bay) + 1,
                    floor_bot=int(story) - 1,
                    floor_top=int(story),
                )
            )
    return walls


def _wall_panel_stories_by_bay(walls: Iterable[ShearWall]) -> Dict[int, List[int]]:
    by_bay: Dict[int, List[int]] = {}
    for w in walls or []:
        bay = int(w.bay_left)
        # A wall panel from floor k-1 to k is story panel k.
        story = int(w.floor_top)
        by_bay.setdefault(bay, []).append(story)
    return {bay: sorted(set(stories)) for bay, stories in by_bay.items()}


def _is_contiguous(values: List[int]) -> bool:
    if not values:
        return True
    vals = sorted(set(int(v) for v in values))
    return vals == list(range(vals[0], vals[-1] + 1))


def _candidate_constructability_score(candidate: ShearWallCandidate) -> Tuple[float, str]:
    """Return a 0..1 constructability penalty and a readable class label.

    The score is deliberately simple and transparent for a research optimizer:
    bottom-anchored continuous wall segments are preferred; elevated or
    discontinuous segments are penalized because they usually require transfer
    mechanisms, collectors, or additional detailing.
    """
    by_bay = _wall_panel_stories_by_bay(candidate.walls)
    if not by_bay:
        return 1.0, "no_wall_panels"

    elevated = False
    discontinuous = False
    single_story_elevated = False
    bottom_anchored_all = True
    full_height_like = True
    lengths = []

    for _bay, stories in by_bay.items():
        if not stories:
            continue
        start = min(stories)
        end = max(stories)
        lengths.append(len(stories))
        if start > 1:
            elevated = True
            bottom_anchored_all = False
        if not _is_contiguous(stories):
            discontinuous = True
        if len(stories) == 1 and start > 1:
            single_story_elevated = True
        if start != 1:
            full_height_like = False

    n_lines = len(by_bay)
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0

    if bottom_anchored_all and not discontinuous:
        if max_len == min_len and max_len > 1:
            label = "bottom_anchored_continuous"
            score = 0.00
        elif max_len == 1:
            label = "bottom_story_panel"
            score = 0.08
        else:
            label = "bottom_anchored_variable_height"
            score = 0.12
    elif single_story_elevated:
        label = "elevated_single_story"
        score = 0.75
    elif elevated and not discontinuous:
        label = "elevated_continuous_segment"
        score = 0.65
    elif discontinuous:
        label = "discontinuous_segment"
        score = 0.90
    elif full_height_like:
        label = "full_height_like"
        score = 0.05
    else:
        label = "mixed_layout"
        score = 0.50

    # More separate wall lines can be constructable, but they increase
    # foundation/coordination complexity, so apply a small capped penalty.
    score += min(0.10 * max(n_lines - 1, 0), 0.25)
    return float(min(max(score, 0.0), 1.0)), label


def _penalized_utilization(value: Any, *, fail_penalty: float = 5.0, missing: float = 10.0) -> float:
    try:
        v = float(value)
        if not np.isfinite(v):
            return float(missing)
    except Exception:
        return float(missing)
    if v <= 1.0:
        return max(v, 0.0)
    return 1.0 + float(fail_penalty) * (v - 1.0)


def _apply_shear_wall_objective_scores(
    df: pd.DataFrame,
    *,
    objective_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Add transparent objective-function columns for wall-layout ranking."""
    if df is None or df.empty:
        return df
    out = df.copy()
    weights = dict(DEFAULT_SW_OBJECTIVE_WEIGHTS)
    if objective_weights:
        for key, value in objective_weights.items():
            if key in weights:
                weights[key] = float(value)
    total_w = sum(max(float(v), 0.0) for v in weights.values())
    if total_w <= 0.0:
        weights = dict(DEFAULT_SW_OBJECTIVE_WEIGHTS)
        total_w = sum(weights.values())
    weights = {k: max(float(v), 0.0) / total_w for k, v in weights.items()}

    cost = pd.to_numeric(out.get("Total_Cost", np.nan), errors="coerce")
    max_cost = float(cost.max()) if cost.notna().any() else np.nan
    if np.isfinite(max_cost) and max_cost > 0.0:
        out["Objective_Cost_Score"] = cost / max_cost
    else:
        out["Objective_Cost_Score"] = np.nan

    drift_util = pd.to_numeric(out.get("Drift_Utilization", np.nan), errors="coerce")
    out["Objective_Drift_Score"] = drift_util.apply(_penalized_utilization)

    panels = pd.to_numeric(out.get("Num_Wall_Panels", np.nan), errors="coerce")
    max_panels = float(panels.max()) if panels.notna().any() else np.nan
    if np.isfinite(max_panels) and max_panels > 0.0:
        out["Objective_Wall_Quantity_Score"] = panels / max_panels
    else:
        out["Objective_Wall_Quantity_Score"] = np.nan

    if "Objective_Constructability_Score" not in out.columns:
        out["Objective_Constructability_Score"] = np.nan

    # Candidates with failed analyses are intentionally pushed down, even if
    # their partial drift values look attractive.
    fail_count = pd.to_numeric(out.get("Analysis_Failed_Count", 0), errors="coerce").fillna(0)
    failed_analysis_mask = fail_count > 0
    out.loc[failed_analysis_mask, "Objective_Drift_Score"] = out.loc[failed_analysis_mask, "Objective_Drift_Score"] + 10.0

    out["Objective_Total_Score"] = (
        weights["cost"] * pd.to_numeric(out["Objective_Cost_Score"], errors="coerce").fillna(10.0)
        + weights["drift"] * pd.to_numeric(out["Objective_Drift_Score"], errors="coerce").fillna(10.0)
        + weights["constructability"] * pd.to_numeric(out["Objective_Constructability_Score"], errors="coerce").fillna(1.0)
        + weights["wall_quantity"] * pd.to_numeric(out["Objective_Wall_Quantity_Score"], errors="coerce").fillna(1.0)
    )

    out["Objective_Weights"] = (
        f"cost={weights['cost']:.2f}; drift={weights['drift']:.2f}; "
        f"constructability={weights['constructability']:.2f}; wall_quantity={weights['wall_quantity']:.2f}"
    )
    out["Objective_Rank"] = out["Objective_Total_Score"].rank(method="dense", ascending=True).astype(int)
    return out


def _segment_description(segments: Iterable[Tuple[int, int, int]]) -> str:
    parts = []
    for bay, start_story, end_story in segments:
        if int(start_story) == int(end_story):
            parts.append(f"B{bay}:S{start_story}")
        else:
            parts.append(f"B{bay}:S{start_story}-{end_story}")
    return ", ".join(parts)


def _segment_candidate_id(segments: Iterable[Tuple[int, int, int]], tw: float) -> str:
    seg_txt = "_".join(f"B{bay}S{start}to{end}" for bay, start, end in segments)
    return f"SW_{seg_txt}_T{int(tw)}"


def generate_shear_wall_candidates(
    geom,
    *,
    failed_stories: Optional[List[int]] = None,
    thicknesses_mm: Iterable[float] = (150.0, 200.0, 250.0, 300.0),
    wall_E_Pa: float = 30e9,
    max_walls_per_layout: Optional[int] = None,
    wall_layout_mode: str = "bottom_anchored",
) -> List[ShearWallCandidate]:
    """Generate shear-wall retrofit candidates.

    Previous versions generated only full vertical wall lines: one selected bay
    was closed from the foundation to the target top story. This function keeps
    that option, but it can also explore variable-height and story-level wall
    segments. Available modes are:

    - ``full_height``: old behaviour; each segment is foundation-to-target-top.
    - ``bottom_anchored``: walls start at the foundation but may stop at any story.
    - ``single_story``: one-story wall panels in different bays/stories.
    - ``failed_story_blocks``: contiguous blocks based on failed drift stories.
    - ``vertical_segments``: every contiguous vertical segment, including elevated starts.
    - ``hybrid``: full-height + bottom-anchored + single-story + failed blocks.

    ``max_walls_per_layout`` is interpreted as the maximum number of wall
    segments in one candidate layout. For the old full-height mode, this is
    equivalent to the previous maximum number of wall lines.
    """
    n_bays = int(getattr(geom, "num_bays"))
    target_top_story = _target_top_story_from_failed_stories(geom, failed_stories)

    max_segments = n_bays if max_walls_per_layout is None else max(1, int(max_walls_per_layout))

    segment_catalog = _wall_segment_catalog(
        n_bays=n_bays,
        target_top_story=target_top_story,
        failed_stories=failed_stories,
        wall_layout_mode=wall_layout_mode,
    )

    candidates: List[ShearWallCandidate] = []
    for tw in thicknesses_mm:
        tw = float(tw)
        for n_segments in range(1, min(max_segments, len(segment_catalog)) + 1):
            for segment_combo in combinations(segment_catalog, n_segments):
                if not _wall_segments_are_non_overlapping(segment_combo):
                    continue
                walls = _walls_from_segments(segment_combo)
                for _w in walls:
                    _w.tw_mm = tw
                seg_desc = _segment_description(segment_combo)
                candidates.append(
                    ShearWallCandidate(
                        candidate_id=_segment_candidate_id(segment_combo, tw),
                        walls=walls,
                        wall_tw_mm=tw,
                        wall_E_Pa=float(wall_E_Pa),
                        description=(
                            f"Wall segment(s): {seg_desc}; "
                            f"mode={wall_layout_mode}, tw={tw:.0f} mm"
                        ),
                    )
                )

    return candidates


def _story_height_m_from_geom(geom) -> float:
    for name in ("floor_height", "floor_height_m", "story_height", "story_height_m"):
        if hasattr(geom, name):
            val = float(getattr(geom, name))
            if val > 0:
                return val
    raise AttributeError("Cannot determine story height from geom.")


def _bay_width_from_wall_and_nodes(wall, geom, nodes) -> float:
    if nodes is None:
        # fallback for older regular-geometry definitions
        for name in ("bay_width", "bay_width_m"):
            if hasattr(geom, name):
                val = float(getattr(geom, name))
                if val > 0:
                    return val
        for name in ("bay_widths", "spans"):
            if hasattr(geom, name):
                vals = list(getattr(geom, name))
                idx = int(wall.bay_left) - 1
                if 0 <= idx < len(vals):
                    return float(vals[idx])
        for name in ("total_width", "width"):
            if hasattr(geom, name) and hasattr(geom, "num_bays"):
                return float(getattr(geom, name)) / max(float(getattr(geom, "num_bays")), 1.0)
        raise AttributeError("Cannot determine bay width. Pass nodes to estimate_shear_wall_cost.")

    floor = int(wall.floor_bot)
    bay_left = int(wall.bay_left)
    bay_right = int(wall.bay_right)
    n_left = node_id_at(geom, floor, bay_left - 1)
    n_right = node_id_at(geom, floor, bay_right - 1)
    x_left = float(nodes[n_left][0])
    x_right = float(nodes[n_right][0])
    return abs(x_right - x_left)

def estimate_shear_wall_cost(
    candidate: ShearWallCandidate,
    geom,
    *,
    nodes=None,
    unit_costs: Optional[Dict[str, float]] = None,
    rebar_kg_per_m3: float = 140.0,
    foundation_cost_per_wall_line: float = 2500.0,
    constructability_cost_per_wall_line: float = 1000.0,
) -> RetrofitCostBreakdown:
    """
    Research-oriented comparative cost model.

    Cost components:
    - concrete volume
    - reinforcement weight, estimated per concrete volume
    - two-sided formwork
    - labor and indirect factors
    - foundation/constructability penalties per distinct wall line

    The absolute values are still unit-cost assumptions, but the model is
    consistent for ranking alternatives and penalizes overly scattered layouts.
    """
    unit_costs = {**DEFAULT_UNIT_COSTS, **(unit_costs or {})}
    foundation_cost_per_wall_line = float(unit_costs.get("foundation_cost_per_wall_line", foundation_cost_per_wall_line))
    constructability_cost_per_wall_line = float(unit_costs.get("constructability_cost_per_wall_line", constructability_cost_per_wall_line))
    tw_m = candidate.wall_tw_mm / 1000.0
    concrete_m3 = 0.0
    formwork_m2 = 0.0
    wall_lines = set()

    for w in candidate.walls:
        length_m = _bay_width_from_wall_and_nodes(w, geom, nodes)
        height_m = _story_height_m_from_geom(geom)
        vol = length_m * height_m * tw_m
        concrete_m3 += vol
        formwork_m2 += 2.0 * length_m * height_m
        wall_lines.add(int(w.bay_left))

    rebar_kg = concrete_m3 * rebar_kg_per_m3
    material = (
        concrete_m3 * unit_costs["wall_concrete_per_m3"]
        + rebar_kg * unit_costs["wall_rebar_per_kg"]
        + formwork_m2 * unit_costs["wall_formwork_per_m2"]
    )
    labor = unit_costs["labor_factor"] * material
    indirect = unit_costs["indirect_factor"] * material
    foundation = foundation_cost_per_wall_line * len(wall_lines)
    constructability = constructability_cost_per_wall_line * len(wall_lines)
    wall_productivity = float(unit_costs.get("wall_productivity_m3_per_day", 0.0) or 0.0)
    duration_days = concrete_m3 / wall_productivity if wall_productivity > 0.0 else 0.0
    site_overhead = duration_days * float(unit_costs.get("site_overhead_per_day", 0.0) or 0.0)
    total = material + labor + indirect + foundation + constructability + site_overhead

    return RetrofitCostBreakdown(
        concrete_m3=concrete_m3,
        rebar_kg=rebar_kg,
        formwork_m2=formwork_m2,
        material_cost=material,
        labor_cost=labor,
        indirect_cost=indirect,
        total_cost=total,
        details={
            "num_wall_panels": len(candidate.walls),
            "num_wall_lines": len(wall_lines),
            "foundation_cost": foundation,
            "constructability_cost": constructability,
            "duration_days": duration_days,
            "site_overhead_cost": site_overhead,
            "unit_costs": dict(unit_costs),
        },
    )

def optimize_shear_wall_positions(
    *,
    nodes,
    elements,
    geom,
    combos,
    combo_names,
    Ec_Pa,
    Es_Pa,
    column_sec,
    beam_sec,
    ops_build_model,
    ops_apply_combo_loads,
    ops_run_static,
    base_walls=None,
    wall_tw_mm: float = 200.0,
    wall_E_Pa=30e9,
    use_rigid_diaphragm=False,
    drift_limit_ratio: float = 0.005,
    failed_stories: Optional[List[int]] = None,
    thicknesses_mm: Iterable[float] = (150.0, 200.0, 250.0, 300.0),
    max_walls_per_layout: Optional[int] = None,
    wall_layout_mode: str = "bottom_anchored",
    early_stop_by_wall_count: bool = False,
    objective_weights: Optional[Dict[str, float]] = None,
    unit_costs: Optional[Dict[str, float]] = None,
    seismic_settings: Optional[Dict[str, Any]] = None,
    wall_density_kgm3: Optional[float] = None,
    candidates_override: Optional[List[ShearWallCandidate]] = None,
) -> pd.DataFrame:
    """
    Research-level shear-wall layout search.

    The optimizer explores shear-wall layouts generated by
    ``generate_shear_wall_candidates``. In addition to old full-height wall
    lines, it can evaluate variable-height and elevated wall segments through
    ``wall_layout_mode``. The resulting table still includes the columns
    expected by main.py: Wall_Lines, Drift_Reduction_pct,
    Minimum_Wall_Count_Feasible and Pareto_Efficient. It also evaluates a
    scalar objective function combining normalized cost, drift utilization,
    wall quantity and constructability, then ranks/selects by the lowest score.
    """
    base_walls = list(base_walls or [])
    for _w in base_walls:
        if getattr(_w, "tw_mm", None) is None:
            _w.tw_mm = float(wall_tw_mm)
    base_panels = {(int(w.bay_left), int(w.floor_top)) for w in base_walls}

    # Baseline drift with the currently existing wall layout. This is used for
    # drift-reduction reporting and for before/after plotting.
    # Use the caller-supplied wall_tw_mm for the baseline. This keeps
    # drift-reduction reporting consistent with any existing wall thickness.
    baseline_drifts, baseline_ok_names, baseline_failed_names = run_static_drift_assessment(
        nodes=nodes,
        elements=elements,
        geom=geom,
        combos=combos,
        combo_names=combo_names,
        Ec_Pa=Ec_Pa,
        Es_Pa=Es_Pa,
        column_sec=column_sec,
        beam_sec=beam_sec,
        ops_build_model=ops_build_model,
        ops_apply_combo_loads=ops_apply_combo_loads,
        ops_run_static=ops_run_static,
        walls=base_walls,
        wall_tw_mm=wall_tw_mm,
        wall_E_Pa=wall_E_Pa,
        use_rigid_diaphragm=use_rigid_diaphragm,
    )
    baseline_max_drift_ratio = _max_numeric_column(
        baseline_drifts,
        ["MaxDriftRatio", "DriftRatio", "drift_ratio"],
    )

    candidates = list(candidates_override) if candidates_override is not None else generate_shear_wall_candidates(
        geom,
        failed_stories=failed_stories,
        thicknesses_mm=thicknesses_mm,
        wall_E_Pa=wall_E_Pa,
        max_walls_per_layout=max_walls_per_layout,
        wall_layout_mode=wall_layout_mode,
    )

    rows = []
    # FIX 3: Early-exit tracking.
    # Once we have found at least one feasible solution with `n_lines` wall
    # lines we can skip all candidates that use strictly more wall lines AND
    # a higher (or equal) cost — they can never be the best answer. The
    # cheapest feasible n_lines solution is kept; then we keep evaluating
    # only cheaper/better candidates at the same line count.
    _min_feasible_lines: Optional[int] = None

    for cand in candidates:
        # Skip if this candidate uses more lines than the best feasible we've
        # already found (wall lines is the primary optimisation objective).
        cand_lines = len({int(w.bay_left) for w in cand.walls})
        if any((int(w.bay_left), int(w.floor_top)) in base_panels for w in cand.walls):
            continue
        if early_stop_by_wall_count and _min_feasible_lines is not None and cand_lines > _min_feasible_lines:
            continue
        trial_walls = base_walls + cand.walls
        df_drifts, ok_names, failed_names = run_static_drift_assessment(
            nodes=nodes,
            elements=elements,
            geom=geom,
            combos=combos,
            combo_names=combo_names,
            Ec_Pa=Ec_Pa,
            Es_Pa=Es_Pa,
            column_sec=column_sec,
            beam_sec=beam_sec,
            ops_build_model=ops_build_model,
            ops_apply_combo_loads=ops_apply_combo_loads,
            ops_run_static=ops_run_static,
            walls=trial_walls,
            wall_tw_mm=cand.wall_tw_mm,
            wall_E_Pa=cand.wall_E_Pa,
            use_rigid_diaphragm=use_rigid_diaphragm,
        )

        max_drift_ratio = _max_numeric_column(df_drifts, ["MaxDriftRatio", "DriftRatio", "drift_ratio"])
        max_drift_mm = _max_numeric_column(df_drifts, ["MaxDrift_mm", "Drift [mm]", "Max Drift [mm]"])
        cost = estimate_shear_wall_cost(cand, geom, nodes=nodes, unit_costs=unit_costs)
        wall_lines = len({int(w.bay_left) for w in cand.walls})
        constructability_score, constructability_class = _candidate_constructability_score(cand)
        feasible = bool(np.isfinite(max_drift_ratio) and max_drift_ratio <= drift_limit_ratio and not failed_names)
        static_util = max_drift_ratio / drift_limit_ratio if drift_limit_ratio > 0 and np.isfinite(max_drift_ratio) else np.nan
        ev = None
        if seismic_settings:
            ev = _seismic_eval(
                seismic_settings,
                nodes=nodes, elements=elements, geom=geom,
                combos=_combos_with_extra_self_weight(
                    combos, elements=elements, column_sec=column_sec, beam_sec=beam_sec, nodes=nodes, geom=geom,
                    new_walls=list(cand.walls), wall_tw_mm=float(cand.wall_tw_mm), wall_density_kgm3=wall_density_kgm3),
                combo_names=combo_names, Ec_Pa=Ec_Pa, Es_Pa=Es_Pa, column_sec=column_sec, beam_sec=beam_sec,
                walls=trial_walls, wall_tw_mm=cand.wall_tw_mm, wall_E_Pa=cand.wall_E_Pa,
                ops_build_model=ops_build_model,
            )
            seis_ok = bool(ev is not None and ev["dls_utilization"] <= 1.0 and ev["theta_ok"])
            feasible = bool(feasible and seis_ok)
        drift_util_total = float(np.nanmax([static_util, ev["dls_utilization"] if ev else np.nan])) if np.isfinite(static_util) or ev else np.nan

        if np.isfinite(baseline_max_drift_ratio) and abs(baseline_max_drift_ratio) > 1e-12 and np.isfinite(max_drift_ratio):
            drift_reduction_pct = 100.0 * (baseline_max_drift_ratio - max_drift_ratio) / abs(baseline_max_drift_ratio)
        else:
            drift_reduction_pct = np.nan

        # FIX 3 (continued): record the minimum feasible wall-line count so
        # subsequent iterations can skip costlier multi-line candidates early.
        if feasible and (_min_feasible_lines is None or cand_lines < _min_feasible_lines):
            _min_feasible_lines = cand_lines

        rows.append({
            "Candidate_ID": cand.candidate_id,
            "Description": cand.description,
            "Num_Wall_Panels": len(cand.walls),
            "Num_Wall_Lines": wall_lines,
            "Wall_Lines": wall_lines,
            "Wall_Layout_Mode": wall_layout_mode,
            "Wall_Segments": cand.description,
            "Constructability_Class": constructability_class,
            "Objective_Constructability_Score": constructability_score,
            "Wall_Thickness_mm": cand.wall_tw_mm,
            "Max_Drift_Ratio": max_drift_ratio,
            "Max_Drift_mm": max_drift_mm,
            "Drift_Limit_Ratio": drift_limit_ratio,
            "Drift_Utilization": drift_util_total,
            "Static_Drift_Utilization": static_util,
            "Seismic_T1_s": ev["T1"] if ev else np.nan,
            "Seismic_Fb_kN": ev["Fb_kN"] if ev else np.nan,
            "Seismic_DLS_Utilization": ev["dls_utilization"] if ev else np.nan,
            "Seismic_Theta_Max": ev["theta_max"] if ev else np.nan,
            "Drift_Reduction_pct": drift_reduction_pct,
            "Feasible": "YES" if feasible else "NO",
            "Analysis_OK_Count": len(ok_names),
            "Analysis_Failed_Count": len(failed_names),
            "Concrete_m3": cost.concrete_m3,
            "Rebar_kg": cost.rebar_kg,
            "Formwork_m2": cost.formwork_m2,
            "Duration_Days": cost.details.get("duration_days", np.nan),
            "Site_Overhead_Cost": cost.details.get("site_overhead_cost", np.nan),
            "Total_Cost": cost.total_cost,
            "Cost_per_Drift_Reduction": cost.total_cost / max(drift_reduction_pct, 1e-9) if np.isfinite(drift_reduction_pct) and drift_reduction_pct > 0 else np.nan,
            "Cost_per_Drift_Utilization_Reduction": np.nan,
            "Minimum_Wall_Count_Feasible": "NO",
            "Pareto_Efficient": "NO",
            "_candidate": cand,
            "_drifts": df_drifts,
            "_baseline_drifts": baseline_drifts,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Minimum wall-count feasible flag.
    feasible_mask = df["Feasible"].astype(str).str.upper().eq("YES")
    if feasible_mask.any():
        min_lines = int(df.loc[feasible_mask, "Wall_Lines"].min())
        df.loc[feasible_mask & (df["Wall_Lines"] == min_lines), "Minimum_Wall_Count_Feasible"] = "YES"

    # Pareto efficiency among feasible candidates using three objectives:
    # fewer wall lines, lower cost, lower drift utilization.
    obj_cols = ["Wall_Lines", "Total_Cost", "Drift_Utilization"]
    feasible_idx = df.index[feasible_mask].tolist()
    for i in feasible_idx:
        xi = df.loc[i, obj_cols].astype(float).to_numpy()
        dominated = False
        for j in feasible_idx:
            if i == j:
                continue
            xj = df.loc[j, obj_cols].astype(float).to_numpy()
            if np.all(xj <= xi) and np.any(xj < xi):
                dominated = True
                break
        if not dominated:
            df.at[i, "Pareto_Efficient"] = "YES"

    # Objective-function ranking. Feasible candidates are still kept first,
    # but the preferred feasible alternative is now the best balanced solution
    # rather than simply the cheapest minimum-line layout.
    df = _apply_shear_wall_objective_scores(df, objective_weights=objective_weights)
    feasible_mask = df["Feasible"].astype(str).str.upper().eq("YES")
    if feasible_mask.any():
        df = df.sort_values(
            ["Feasible", "Objective_Total_Score", "Drift_Utilization", "Total_Cost", "Wall_Lines"],
            ascending=[False, True, True, True, True],
        )
    else:
        df = df.sort_values(
            ["Objective_Total_Score", "Drift_Utilization", "Total_Cost", "Wall_Lines"],
            ascending=[True, True, True, True],
        )

    df = df.reset_index(drop=True)
    df["Objective_Rank"] = np.arange(1, len(df) + 1)
    return df


# ============================================================
# CONCRETE JACKETING
# ============================================================
def make_jacketed_element_sections(
    *,
    elements,
    column_sec,
    beam_sec,
    failed_member_ids: List[int],
    t_jacket_mm: float = 75.0,
    fc_jacket_mpa: Optional[float] = None,
    stirrup_spacing_mm: Optional[float] = 100.0,
    jacket_thickness_by_element: Optional[Dict[int, float]] = None,
    stirrup_spacing_by_element: Optional[Dict[int, float]] = None,
    longitudinal_rebar_ratio_by_element: Optional[Dict[int, float]] = None,
) -> Dict[int, Any]:
    """
    Build per-element section map for jacketed members.

    Backward compatible with the old uniform-thickness call, but also supports
    per-member jacket thicknesses for iterative jacketing. This avoids enlarging
    members that already passed in an earlier jacketing iteration.
    """
    out: Dict[int, Any] = {}
    failed_set = set(int(e) for e in failed_member_ids)
    thickness_map = {int(k): float(v) for k, v in (jacket_thickness_by_element or {}).items()}
    spacing_map = {int(k): float(v) for k, v in (stirrup_spacing_by_element or {}).items()}
    rebar_ratio_map = {int(k): float(v) for k, v in (longitudinal_rebar_ratio_by_element or {}).items()}

    for eid, (_, _, etype) in elements.items():
        eid_int = int(eid)
        base = column_sec if etype == "column" else beam_sec
        sec = copy(base)
        if eid_int in failed_set:
            from framekit.jacketing import build_jacketed_section
            local_t = float(thickness_map.get(eid_int, t_jacket_mm))
            local_s = spacing_map.get(eid_int, stirrup_spacing_mm)
            local_rebar_ratio = float(rebar_ratio_map.get(eid_int, 0.0))
            sec = build_jacketed_section(base, str(etype), local_t, stirrup_spacing_mm=local_s,
                                         ratio=local_rebar_ratio)
            if fc_jacket_mpa is not None:
                setattr(sec, "fc_mpa", fc_jacket_mpa)

        out[eid_int] = sec

    return out

def _increase_longitudinal_reinforcement(sec, increase_ratio: float, member_type: Optional[str] = None) -> None:
    """
    Increase longitudinal reinforcement on common RC-section attribute names.

    Beams are strengthened mainly in the top and bottom rows. Columns are
    strengthened on all four faces: top row, bottom row, left side and right
    side. This is important because a column flexure/axial retrofit is not a
    beam-only detail.
    """
    ratio = max(float(increase_ratio), 0.0)
    if ratio <= 0.0:
        return

    # IMPORTANT: when the caller provides member_type from elements[eid][2],
    # it must override section attributes. Some beam section objects may carry
    # generic side-bar fields for compatibility/reporting. Those fields must
    # not make a beam behave like a column retrofit.
    if member_type is not None and str(member_type).strip():
        mtype = str(member_type).strip().lower()
        is_column = "column" in mtype
    else:
        mtype = str(_get_attr(sec, ["section_role", "role", "member_type"], "")).lower()
        is_column = "column" in mtype or any(
            hasattr(sec, name) for name in ["n_left", "n_right", "phi_side_mm", "left_bars", "right_bars"]
        )

    area_names = [
        "As_top_mm2", "As_bot_mm2", "As_bottom_mm2", "As_top", "As_bot", "As_bottom",
        "As1_mm2", "As2_mm2", "As1", "As2", "As_mm2", "As", "Ast_mm2", "Ast",
        "As_long_mm2", "As_long", "As_total_mm2", "As_total",
    ]
    if is_column:
        area_names += [
            "As_left_mm2", "As_right_mm2", "As_side_mm2", "As_left", "As_right", "As_side",
        ]

    face_count_names = [
        "n_top", "n_bot", "n_bottom", "n_bars_top", "n_bars_bot", "n_bars_bottom",
        "num_top_bars", "num_bottom_bars", "num_bars_top", "num_bars_bottom",
        "n_long", "n_bars", "num_bars",
    ]

    touched = False
    for name in area_names:
        if hasattr(sec, name):
            try:
                val = float(getattr(sec, name))
                if np.isfinite(val) and val > 0.0:
                    setattr(sec, name, val * (1.0 + ratio))
                    touched = True
            except Exception:
                pass

    original_counts: Dict[str, float] = {}
    for name in face_count_names:
        if hasattr(sec, name):
            try:
                val = float(getattr(sec, name))
                original_counts[name] = val
                if np.isfinite(val) and val > 0.0:
                    setattr(sec, name, int(math.ceil(val * (1.0 + ratio))))
                    touched = True
            except Exception:
                pass

    if is_column:
        # Ensure columns visibly and numerically receive bars on both side faces,
        # not only beam-style top/bottom bars.
        top_count = _get_attr(sec, ["n_top", "n_bars_top", "num_top_bars"], None)
        bot_count = _get_attr(sec, ["n_bot", "n_bottom", "n_bars_bot", "n_bars_bottom", "num_bottom_bars"], None)
        try:
            reference_face_bars = max(float(top_count or 0.0), float(bot_count or 0.0), 2.0)
        except Exception:
            reference_face_bars = 2.0

        side_phi = _get_attr(sec, ["phi_side_mm", "side_bar_d", "phi_side"], None)
        if side_phi is None or _safe_is_nonpositive(side_phi):
            side_phi = max(
                float(_get_attr(sec, ["phi_top_mm", "top_bar_d"], 0.0) or 0.0),
                float(_get_attr(sec, ["phi_bot_mm", "bottom_bar_d"], 0.0) or 0.0),
            )
        if side_phi and float(side_phi) > 0.0:
            setattr(sec, "phi_side_mm", float(side_phi))

        # Existing side bars are interpreted as intermediate bars excluding
        # corners. If the original column had no explicit side-bar attributes,
        # create side bars proportional to the requested longitudinal increase.
        for side_name, aliases in [
            ("n_left", ["n_left", "left_bars", "n_side_left"]),
            ("n_right", ["n_right", "right_bars", "n_side_right"]),
        ]:
            existing_name = next((nm for nm in aliases if hasattr(sec, nm)), side_name)
            try:
                old_side = float(getattr(sec, existing_name, 0.0) or 0.0)
            except Exception:
                old_side = 0.0
            if old_side > 0.0:
                new_side = int(math.ceil(old_side * (1.0 + ratio)))
            else:
                # Minimum one added intermediate bar per side for column
                # flexure/axial retrofit; larger ratios add more side bars.
                new_side = max(1, int(math.ceil(reference_face_bars * ratio / 2.0)))
            setattr(sec, existing_name, new_side)
            setattr(sec, side_name, new_side)
            touched = True

        setattr(sec, "section_role", "column")
        setattr(sec, "member_type", "column")

    setattr(sec, "retrofit_longitudinal_rebar_ratio", ratio)
    setattr(sec, "longitudinal_bar_increase_ratio", ratio)
    setattr(sec, "additional_longitudinal_rebar_ratio", ratio)
    setattr(sec, "longitudinal_reinforcement_upgraded", True)
    if is_column:
        setattr(sec, "longitudinal_rebar_distribution", "top_bottom_left_right")
    else:
        setattr(sec, "longitudinal_rebar_distribution", "top_bottom")
    if not touched:
        setattr(sec, "retrofit_note", "longitudinal reinforcement increase requested")


def _safe_is_nonpositive(value: Any) -> bool:
    try:
        return float(value) <= 0.0 or not np.isfinite(float(value))
    except Exception:
        return True


# ============================================================
# EC2-style longitudinal rebar spacing/detailing helpers
# ============================================================
def _bar_count(sec, names: Iterable[str], default: int = 0) -> int:
    val = _get_attr(sec, names, default)
    try:
        if val is None:
            return int(default)
        if not np.isfinite(float(val)):
            return int(default)
        return max(int(round(float(val))), 0)
    except Exception:
        return int(default)


def _bar_diameter_mm(sec, names: Iterable[str], default: float = 0.0) -> float:
    val = _get_attr(sec, names, default)
    try:
        val = float(val)
        return val if np.isfinite(val) and val > 0.0 else float(default)
    except Exception:
        return float(default)


def _ec2_min_clear_rebar_spacing_mm(
    bar_diameter_mm: float,
    *,
    aggregate_size_mm: float = 20.0,
    ec2_spacing_k1: float = 1.0,
    ec2_spacing_k2_mm: float = 5.0,
) -> float:
    """Return the EC2-style minimum clear spacing between parallel bars.

    EN 1992-1-1 clause 8.2(2) is commonly applied as:
        clear spacing >= max(k1 * bar diameter, aggregate size + k2, 20 mm)
    with recommended values k1 = 1.0 and k2 = 5 mm, unless the National Annex
    or project specifications require different values.
    """
    phi = max(float(bar_diameter_mm), 0.0)
    dg = max(float(aggregate_size_mm), 0.0)
    return float(max(float(ec2_spacing_k1) * phi, dg + float(ec2_spacing_k2_mm), 20.0))


def _face_clear_spacing_mm(face_width_mm: float, n_bars: int, phi_mm: float, cover_mm: float, stirrup_phi_mm: float) -> float:
    """Clear spacing between bars in a single row on one face.

    The check is intentionally single-layer. If this check fails, the current
    optimizer does not silently place bars in multiple layers because the EC2
    verification model and section drawings do not yet represent layered bars.
    """
    n = int(n_bars)
    phi = float(phi_mm)
    if n <= 1 or phi <= 0.0:
        return float("inf")
    available_clear = float(face_width_mm) - 2.0 * (float(cover_mm) + float(stirrup_phi_mm)) - n * phi
    return float(available_clear / max(n - 1, 1))


def _longitudinal_rebar_spacing_check(
    sec,
    *,
    member_type: Optional[str] = None,
    aggregate_size_mm: float = 20.0,
    ec2_spacing_k1: float = 1.0,
    ec2_spacing_k2_mm: float = 5.0,
) -> Dict[str, Any]:
    """Check whether the modeled longitudinal bars fit with minimum clear spacing.

    This is a detailing feasibility check, not a full reinforcement detailing
    design. It prevents the automatic jacketing optimizer from accepting a
    solution where the required additional longitudinal bars are geometrically
    impossible in a single row.
    """
    if float(getattr(sec, "jacket_t_mm", 0.0) or 0.0) > 0.0 and hasattr(sec, "jacket_check"):
        return dict(sec.jacket_check)
    b = _section_b_mm(sec)
    h = _section_h_mm(sec)
    cover = _bar_diameter_mm(sec, ["cover_mm", "cover"], 30.0)
    stirrup_phi = _bar_diameter_mm(sec, ["stirrup_phi_mm", "stirrup_phi"], 8.0)
    phi_top = _bar_diameter_mm(sec, ["phi_top_mm", "top_bar_d", "phi_top"], 0.0)
    phi_bot = _bar_diameter_mm(sec, ["phi_bot_mm", "bottom_bar_d", "phi_bot"], phi_top)
    phi_side = _bar_diameter_mm(sec, ["phi_side_mm", "side_bar_d", "phi_side"], max(phi_top, phi_bot, 0.0))
    n_top = _bar_count(sec, ["n_top", "top_bars", "n_bars_top", "num_top_bars"], 0)
    n_bot = _bar_count(sec, ["n_bot", "n_bottom", "bottom_bars", "n_bars_bot", "num_bottom_bars"], 0)
    n_left = _bar_count(sec, ["n_left", "left_bars", "n_side_left"], 0)
    n_right = _bar_count(sec, ["n_right", "right_bars", "n_side_right"], 0)

    if member_type is not None and str(member_type).strip():
        is_column = "column" in str(member_type).strip().lower()
    else:
        role = str(_get_attr(sec, ["section_role", "role", "member_type"], "")).lower()
        is_column = "column" in role or any(hasattr(sec, nm) for nm in ["n_left", "n_right", "phi_side_mm"])

    checks: List[Tuple[str, int, float, float, float]] = []
    if n_top > 1 and phi_top > 0.0:
        req = _ec2_min_clear_rebar_spacing_mm(phi_top, aggregate_size_mm=aggregate_size_mm, ec2_spacing_k1=ec2_spacing_k1, ec2_spacing_k2_mm=ec2_spacing_k2_mm)
        checks.append(("top_face", n_top, phi_top, _face_clear_spacing_mm(b, n_top, phi_top, cover, stirrup_phi), req))
    if n_bot > 1 and phi_bot > 0.0:
        req = _ec2_min_clear_rebar_spacing_mm(phi_bot, aggregate_size_mm=aggregate_size_mm, ec2_spacing_k1=ec2_spacing_k1, ec2_spacing_k2_mm=ec2_spacing_k2_mm)
        checks.append(("bottom_face", n_bot, phi_bot, _face_clear_spacing_mm(b, n_bot, phi_bot, cover, stirrup_phi), req))

    if is_column and phi_side > 0.0:
        # Side-face bars are modeled as intermediate side bars plus corner bars.
        # Use the largest diameter at the side line for a conservative clear-spacing check.
        side_phi = max(phi_side, phi_top, phi_bot)
        req_side = _ec2_min_clear_rebar_spacing_mm(side_phi, aggregate_size_mm=aggregate_size_mm, ec2_spacing_k1=ec2_spacing_k1, ec2_spacing_k2_mm=ec2_spacing_k2_mm)
        for face, n_side in [("left_face", n_left), ("right_face", n_right)]:
            n_total = int(n_side) + 2 if int(n_side) > 0 else 2
            if n_total > 1:
                checks.append((face, n_total, side_phi, _face_clear_spacing_mm(h, n_total, side_phi, cover, stirrup_phi), req_side))

    if not checks:
        return {
            "status": "OK",
            "min_clear_spacing_mm": np.nan,
            "required_clear_spacing_mm": np.nan,
            "critical_face": "",
            "note": "No modeled parallel longitudinal bar row requiring spacing check.",
        }

    critical = min(checks, key=lambda item: item[3] - item[4])
    face, n_bars, phi, provided, required = critical
    ok = bool(np.isfinite(provided) and provided + 1e-9 >= required)
    notes = []
    for f, n, ph, prov, req in checks:
        notes.append(f"{f}: n={n}, phi={ph:.0f}, clear={prov:.1f} mm, required={req:.1f} mm")
    return {
        "status": "OK" if ok else "FAIL",
        "min_clear_spacing_mm": provided,
        "required_clear_spacing_mm": required,
        "critical_face": face,
        "note": "; ".join(notes),
    }


def _spacing_check_for_jacketed_member(
    *,
    elements,
    column_sec,
    beam_sec,
    eid: int,
    jacket_thickness_mm: float,
    stirrup_spacing_mm: float,
    longitudinal_rebar_ratio: float,
    aggregate_size_mm: float = 20.0,
    ec2_spacing_k1: float = 1.0,
    ec2_spacing_k2_mm: float = 5.0,
) -> Dict[str, Any]:
    """Build a trial jacketed section and return its longitudinal-bar spacing check."""
    from framekit.jacketing import build_jacketed_section
    eid = int(eid)
    _ni, _nj, etype = elements[eid]
    base = column_sec if str(etype).strip().lower() == "column" else beam_sec
    sec = build_jacketed_section(base, str(etype), float(jacket_thickness_mm),
                                 stirrup_spacing_mm=float(stirrup_spacing_mm),
                                 ratio=float(longitudinal_rebar_ratio),
                                 aggregate_size_mm=aggregate_size_mm, k1=ec2_spacing_k1, k2_mm=ec2_spacing_k2_mm)
    return dict(sec.jacket_check)


def _spacing_map_for_jacketed_members(
    *,
    elements,
    column_sec,
    beam_sec,
    member_ids: List[int],
    jacket_thickness_by_element: Dict[int, float],
    stirrup_spacing_by_element: Dict[int, float],
    longitudinal_rebar_ratio_by_element: Dict[int, float],
    aggregate_size_mm: float = 20.0,
    ec2_spacing_k1: float = 1.0,
    ec2_spacing_k2_mm: float = 5.0,
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for eid in sorted({int(e) for e in member_ids if int(e) in elements}):
        out[eid] = _spacing_check_for_jacketed_member(
            elements=elements,
            column_sec=column_sec,
            beam_sec=beam_sec,
            eid=eid,
            jacket_thickness_mm=float(jacket_thickness_by_element.get(eid, 0.0)),
            stirrup_spacing_mm=float(stirrup_spacing_by_element.get(eid, np.nan)),
            longitudinal_rebar_ratio=float(longitudinal_rebar_ratio_by_element.get(eid, 0.0)),
            aggregate_size_mm=aggregate_size_mm,
            ec2_spacing_k1=ec2_spacing_k1,
            ec2_spacing_k2_mm=ec2_spacing_k2_mm,
        )
    return out


def _format_member_spacing_map(spacing_checks: Dict[int, Dict[str, Any]], member_ids: List[int], key: str) -> str:
    parts = []
    for mid in sorted(int(m) for m in member_ids if int(m) in spacing_checks):
        value = spacing_checks[mid].get(key, np.nan)
        if isinstance(value, str):
            parts.append(f"{mid}:{value}")
        else:
            try:
                if np.isfinite(float(value)):
                    parts.append(f"{mid}:{float(value):.1f}")
            except Exception:
                pass
    return "; ".join(parts)

def _section_longitudinal_area_mm2(sec) -> float:
    """Best-effort estimate/read of longitudinal reinforcement area."""
    names = [
        "As_longitudinal_mm2",
        "As_top_mm2", "As_bot_mm2", "As_bottom_mm2", "As_top", "As_bot", "As_bottom",
        "As1_mm2", "As2_mm2", "As1", "As2", "As_mm2", "As", "Ast_mm2", "Ast",
        "As_long_mm2", "As_long", "As_total_mm2", "As_total",
    ]
    vals = []
    for name in names:
        if hasattr(sec, name):
            try:
                val = float(getattr(sec, name))
                if np.isfinite(val) and val > 0.0:
                    vals.append(val)
            except Exception:
                pass
    if vals:
        # If top/bottom fields exist they should add together. If only total
        # fields exist, max(vals) is safer than summing duplicate aliases.
        top_bot_names = ["As_top_mm2", "As_bot_mm2", "As_bottom_mm2", "As_top", "As_bot", "As_bottom", "As1_mm2", "As2_mm2", "As1", "As2"]
        top_bot = []
        for name in top_bot_names:
            if hasattr(sec, name):
                try:
                    val = float(getattr(sec, name))
                    if np.isfinite(val) and val > 0.0:
                        top_bot.append(val)
                except Exception:
                    pass
        return float(sum(top_bot)) if len(top_bot) >= 2 else float(max(vals))

    # Fallback only for cost ranking: assume about 1 percent longitudinal steel.
    return 0.01 * _section_b_mm(sec) * _section_h_mm(sec)


def _format_member_percent_map(values: Dict[int, float], member_ids: List[int]) -> str:
    return "; ".join(f"{int(mid)}:{100.0 * float(values[int(mid)]):.0f}%" for mid in sorted(int(m) for m in member_ids if int(m) in values))

def estimate_jacketing_cost(
    *,
    nodes,
    elements,
    column_sec,
    beam_sec,
    failed_member_ids: List[int],
    t_jacket_mm: float = 75.0,
    unit_costs: Optional[Dict[str, float]] = None,
    jacket_thickness_by_element: Optional[Dict[int, float]] = None,
    longitudinal_rebar_ratio_by_element: Optional[Dict[int, float]] = None,
    rebar_kg_per_m3: float = 160.0,
) -> RetrofitCostBreakdown:
    unit_costs = {**DEFAULT_UNIT_COSTS, **(unit_costs or {})}
    thickness_map = {int(k): float(v) for k, v in (jacket_thickness_by_element or {}).items()}
    rebar_ratio_map = {int(k): float(v) for k, v in (longitudinal_rebar_ratio_by_element or {}).items()}
    concrete_m3 = 0.0
    formwork_m2 = 0.0
    added_longitudinal_rebar_kg = 0.0

    for eid in failed_member_ids:
        if eid not in elements:
            continue
        _, _, etype = elements[eid]
        sec = column_sec if etype == "column" else beam_sec
        L = _element_length_m(nodes, elements, eid)
        b = _section_b_mm(sec) / 1000.0
        h = _section_h_mm(sec) / 1000.0
        t = thickness_map.get(int(eid), t_jacket_mm) / 1000.0
        old_area = b * h
        new_area = (b + 2.0 * t) * (h + 2.0 * t)
        concrete_m3 += max(new_area - old_area, 0.0) * L
        formwork_m2 += 2.0 * ((b + 2.0 * t) + (h + 2.0 * t)) * L
        ratio = float(rebar_ratio_map.get(int(eid), 0.0))
        if ratio > 0.0:
            from framekit.jacketing import build_jacketed_section, jacket_steel_mm2, bar_area, MIN_CAGE_PHI_MM
            js = build_jacketed_section(sec, etype, t * 1000.0, ratio=ratio)
            extra_mm2 = max(jacket_steel_mm2(js) - 4.0 * bar_area(MIN_CAGE_PHI_MM), 0.0)
            added_longitudinal_rebar_kg += (extra_mm2 * 1e-6) * L * 7850.0

    rebar_kg = concrete_m3 * rebar_kg_per_m3 + added_longitudinal_rebar_kg
    material = (
        concrete_m3 * unit_costs["jacket_concrete_per_m3"]
        + rebar_kg * unit_costs["jacket_rebar_per_kg"]
        + formwork_m2 * unit_costs["jacket_formwork_per_m2"]
    )
    labor = unit_costs["labor_factor"] * material
    indirect = unit_costs["indirect_factor"] * material
    volume_productivity = float(unit_costs.get("jacketing_productivity_m3_per_day", 0.0) or 0.0)
    member_productivity = float(unit_costs.get("jacketing_productivity_members_per_day", 0.0) or 0.0)
    duration_by_volume = concrete_m3 / volume_productivity if volume_productivity > 0.0 else 0.0
    duration_by_members = len(failed_member_ids) / member_productivity if member_productivity > 0.0 else 0.0
    duration_days = max(duration_by_volume, duration_by_members)
    site_overhead = duration_days * float(unit_costs.get("site_overhead_per_day", 0.0) or 0.0)
    return RetrofitCostBreakdown(
        concrete_m3=concrete_m3,
        rebar_kg=rebar_kg,
        formwork_m2=formwork_m2,
        material_cost=material,
        labor_cost=labor,
        indirect_cost=indirect,
        total_cost=material + labor + indirect + site_overhead,
        details={
            "num_jacketed_members": len(failed_member_ids),
            "added_longitudinal_rebar_kg": added_longitudinal_rebar_kg,
            "duration_days": duration_days,
            "site_overhead_cost": site_overhead,
            "unit_costs": dict(unit_costs),
        },
    )



def _format_member_float_map(values: Dict[int, float], member_ids: List[int]) -> str:
    return "; ".join(f"{int(mid)}:{float(values[int(mid)]):.0f}" for mid in sorted(int(m) for m in member_ids if int(m) in values))


def _remaining_failed_member_ids(member_fail_df: Optional[pd.DataFrame], tracked_member_ids: List[int]) -> List[int]:
    failed = set(_failed_member_ids(member_fail_df))
    tracked = {int(e) for e in tracked_member_ids}
    return sorted(failed.intersection(tracked))


def _governing_failure_mode_for_member(member_fail_df: Optional[pd.DataFrame], df_worst: Optional[pd.DataFrame], eid: int) -> str:
    """
    Diagnose why a member controls.

    Returns:
        'shear'          when shear utilization controls,
        'flexure_axial'  when moment/flexure/axial utilization controls,
        'unknown'        when the table has no usable utilization components.
    """
    eid = int(eid)

    def _row_for_element(df: Optional[pd.DataFrame]):
        if df is None or df.empty:
            return None
        element_col = next((c for c in ["Element", "ElementID", "element", "element_id"] if c in df.columns), None)
        if element_col is None:
            return None
        tmp = df.copy()
        tmp[element_col] = pd.to_numeric(tmp[element_col], errors="coerce")
        hit = tmp[tmp[element_col] == eid]
        if hit.empty:
            return None
        # use the row with the largest total eta if there are several combos
        eta_col = next((c for c in ["eta", "Static_EC2_Utilization", "Utilization", "Max_Utilization"] if c in hit.columns), None)
        if eta_col is not None:
            hit = hit.copy()
            hit[eta_col] = pd.to_numeric(hit[eta_col], errors="coerce")
            hit = hit.sort_values(eta_col, ascending=False)
        return hit.iloc[0]

    row = _row_for_element(member_fail_df)
    if row is None:
        row = _row_for_element(df_worst)
    if row is None:
        return "unknown"

    def _first_numeric(row, names):
        for c in names:
            if c in row.index:
                val = pd.to_numeric(row.get(c), errors="coerce")
                try:
                    val = float(val)
                    if np.isfinite(val):
                        return val
                except Exception:
                    pass
        return np.nan

    eta_v = _first_numeric(row, [
        "etaV", "EtaV", "eta_V", "Shear_Utilization", "V_Utilization", "util_shear", "ShearUtilization",
    ])
    eta_nm = _first_numeric(row, [
        "eta_NM", "Eta_NM", "etaNM", "NM_Utilization", "Flexure_Axial_Utilization", "Moment_Utilization",
        "M_Utilization", "N_M_Utilization", "util_NM", "FlexuralUtilization",
    ])

    if np.isfinite(eta_v) and np.isfinite(eta_nm):
        return "shear" if eta_v >= eta_nm else "flexure_axial"
    if np.isfinite(eta_v):
        return "shear"
    if np.isfinite(eta_nm):
        return "flexure_axial"
    return "unknown"


def _governing_mode_label(mode: str) -> str:
    if mode == "shear":
        return "SHEAR"
    if mode == "flexure_axial":
        return "MOMENT/FLEXURE-AXIAL"
    return "UNKNOWN"


def _member_component_eta_map(df: Optional[pd.DataFrame], member_ids: List[int]) -> Dict[int, Dict[str, float]]:
    out = {int(e): {"eta_shear": np.nan, "eta_flexure_axial": np.nan} for e in member_ids}
    if df is None or df.empty:
        return out
    element_col = next((c for c in ["Element", "ElementID", "element", "element_id"] if c in df.columns), None)
    if element_col is None:
        return out
    tmp = df.copy()
    tmp[element_col] = pd.to_numeric(tmp[element_col], errors="coerce")
    shear_col = next((c for c in ["etaV", "EtaV", "eta_V", "Shear_Utilization", "V_Utilization", "util_shear", "ShearUtilization"] if c in tmp.columns), None)
    flex_col = next((c for c in ["eta_NM", "Eta_NM", "etaNM", "NM_Utilization", "Flexure_Axial_Utilization", "Moment_Utilization", "M_Utilization", "N_M_Utilization", "util_NM", "FlexuralUtilization"] if c in tmp.columns), None)
    if shear_col is not None:
        tmp[shear_col] = pd.to_numeric(tmp[shear_col], errors="coerce")
    if flex_col is not None:
        tmp[flex_col] = pd.to_numeric(tmp[flex_col], errors="coerce")
    for eid in member_ids:
        hit = tmp[tmp[element_col] == int(eid)]
        if hit.empty:
            continue
        if shear_col is not None:
            vals = hit[shear_col].dropna()
            if not vals.empty:
                out[int(eid)]["eta_shear"] = float(vals.max())
        if flex_col is not None:
            vals = hit[flex_col].dropna()
            if not vals.empty:
                out[int(eid)]["eta_flexure_axial"] = float(vals.max())
    return out

def _next_lower_spacing(current_spacing_mm: float, allowed_spacings_mm: List[float]) -> Optional[float]:
    allowed = sorted({float(s) for s in allowed_spacings_mm}, reverse=True)
    lower = [s for s in allowed if s < float(current_spacing_mm) - 1e-9]
    return lower[0] if lower else None


def _element_eta_map(df: Optional[pd.DataFrame], member_ids: List[int]) -> Dict[int, float]:
    """Return worst total utilization by member from any EC2/member table shape."""
    out = {int(e): np.nan for e in member_ids}
    if df is None or df.empty:
        return out
    element_col = next((c for c in ["Element", "ElementID", "element", "element_id"] if c in df.columns), None)
    eta_col = next((c for c in ["eta", "Static_EC2_Utilization", "Utilization", "Max_Utilization"] if c in df.columns), None)
    if element_col is None or eta_col is None:
        return out
    tmp = df.copy()
    tmp[element_col] = pd.to_numeric(tmp[element_col], errors="coerce")
    tmp[eta_col] = pd.to_numeric(tmp[eta_col], errors="coerce")
    for eid in member_ids:
        vals = tmp.loc[tmp[element_col] == int(eid), eta_col].dropna()
        if not vals.empty:
            out[int(eid)] = float(vals.max())
    return out


def _element_governing_mode_map(df: Optional[pd.DataFrame], member_ids: List[int]) -> Dict[int, str]:
    """Return original/final governing mode by comparing shear and flexure/axial utilizations."""
    out = {int(e): "unknown" for e in member_ids}
    if df is None or df.empty:
        return out
    element_col = next((c for c in ["Element", "ElementID", "element", "element_id"] if c in df.columns), None)
    if element_col is None:
        return out
    tmp = df.copy()
    tmp[element_col] = pd.to_numeric(tmp[element_col], errors="coerce")
    for eid in member_ids:
        hit = tmp[tmp[element_col] == int(eid)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        mode = _governing_failure_mode_for_member(hit, hit, int(eid))
        out[int(eid)] = mode if mode in ("shear", "flexure_axial") else "unknown"
    return out


def _build_member_jacketing_report(
    *,
    member_ids: List[int],
    original_df_worst: Optional[pd.DataFrame],
    final_df_worst: Optional[pd.DataFrame],
    jacket_thickness_by_element: Dict[int, float],
    stirrup_spacing_by_element: Dict[int, float],
    longitudinal_rebar_ratio_by_element: Dict[int, float],
    final_fail_ids: List[int],
    elements=None,
    column_sec=None,
    beam_sec=None,
    aggregate_size_mm: float = 20.0,
    ec2_spacing_k1: float = 1.0,
    ec2_spacing_k2_mm: float = 5.0,
) -> pd.DataFrame:
    """Build the per-member retrofit report requested by main.py/report exports."""
    member_ids = sorted({int(e) for e in member_ids})
    original_eta = _element_eta_map(original_df_worst, member_ids)
    final_eta = _element_eta_map(final_df_worst, member_ids)
    original_modes = _element_governing_mode_map(original_df_worst, member_ids)
    final_modes = _element_governing_mode_map(final_df_worst, member_ids)
    original_components = _member_component_eta_map(original_df_worst, member_ids)
    final_components = _member_component_eta_map(final_df_worst, member_ids)
    final_failed = {int(e) for e in final_fail_ids}
    if elements is not None and column_sec is not None and beam_sec is not None:
        spacing_checks = _spacing_map_for_jacketed_members(
            elements=elements,
            column_sec=column_sec,
            beam_sec=beam_sec,
            member_ids=member_ids,
            jacket_thickness_by_element=jacket_thickness_by_element,
            stirrup_spacing_by_element=stirrup_spacing_by_element,
            longitudinal_rebar_ratio_by_element=longitudinal_rebar_ratio_by_element,
            aggregate_size_mm=aggregate_size_mm,
            ec2_spacing_k1=ec2_spacing_k1,
            ec2_spacing_k2_mm=ec2_spacing_k2_mm,
        )
    else:
        spacing_checks = {}
    rows = []
    for eid in member_ids:
        rows.append({
            "Element": int(eid),
            "Original_Eta": original_eta.get(int(eid), np.nan),
            "Original_Eta_Shear": original_components.get(int(eid), {}).get("eta_shear", np.nan),
            "Original_Eta_Flexure_Axial": original_components.get(int(eid), {}).get("eta_flexure_axial", np.nan),
            "Problem_Before": _governing_mode_label(original_modes.get(int(eid), "unknown")),
            "Original_Governing_Mode": _governing_mode_label(original_modes.get(int(eid), "unknown")),
            "Chosen_Jacket_Thickness_mm": float(jacket_thickness_by_element.get(int(eid), np.nan)),
            "Chosen_Stirrup_Spacing_mm": float(stirrup_spacing_by_element.get(int(eid), np.nan)),
            "Chosen_Longitudinal_Rebar_Increase_pct": 100.0 * float(longitudinal_rebar_ratio_by_element.get(int(eid), 0.0)),
            "Final_Eta": final_eta.get(int(eid), np.nan),
            "Final_Eta_Shear": final_components.get(int(eid), {}).get("eta_shear", np.nan),
            "Final_Eta_Flexure_Axial": final_components.get(int(eid), {}).get("eta_flexure_axial", np.nan),
            "Problem_After": _governing_mode_label(final_modes.get(int(eid), "unknown")),
            "Final_Governing_Mode": _governing_mode_label(final_modes.get(int(eid), "unknown")),
            "Final_Rebar_Min_Clear_Spacing_mm": spacing_checks.get(int(eid), {}).get("min_clear_spacing_mm", np.nan),
            "Final_Rebar_Required_Clear_Spacing_mm": spacing_checks.get(int(eid), {}).get("required_clear_spacing_mm", np.nan),
            "Final_Rebar_Spacing_Status": spacing_checks.get(int(eid), {}).get("status", ""),
            "Final_Rebar_Spacing_Critical_Face": spacing_checks.get(int(eid), {}).get("critical_face", ""),
            "Final_Rebar_Spacing_Note": spacing_checks.get(int(eid), {}).get("note", ""),
            "Final_Status": "FAIL" if int(eid) in final_failed or spacing_checks.get(int(eid), {}).get("status", "OK") == "FAIL" else "PASS",
        })
    return pd.DataFrame(rows)



def _next_higher_rebar_ratio(current_ratio: float, allowed_ratios: List[float]) -> Optional[float]:
    higher = [r for r in sorted(float(x) for x in allowed_ratios) if r > float(current_ratio) + 1e-9]
    return higher[0] if higher else None


def _sec_spacing_or(sec, default):
    try:
        return float(_get_attr(sec, ["stirrup_spacing_mm", "stirrup_spacing"], default))
    except Exception:
        return float(default)


def _member_base_section(elements, column_sec, beam_sec, eid: int):
    """Return the original section object for a member."""
    _ni, _nj, etype = elements[int(eid)]
    return column_sec if str(etype).strip().lower() == "column" else beam_sec


def _member_max_jacket_thickness_mm(
    *,
    elements,
    column_sec,
    beam_sec,
    eid: int,
    max_jacket_thickness_mm: float,
    max_jacketed_section_dimension_ratio: Optional[float],
) -> float:
    """Per-member concrete-jacket thickness cap.

    The cap is the smaller of the global thickness limit and the geometric
    limit implied by a final/original dimension ratio. The ratio is checked
    independently in both local section directions:

        b_final <= ratio * b_original
        h_final <= ratio * h_original

    Because the jacket is added on both sides, the member cap is:

        t <= min((ratio - 1) * b / 2, (ratio - 1) * h / 2)

    Example: for a 400 x 400 member and ratio = 1.50, the cap is 100 mm,
    so the largest jacketed section is 600 x 600. For a 300 x 500 member,
    the cap is 75 mm, so the width does not exceed 1.50 times its original
    value.
    """
    cap = float(max_jacket_thickness_mm)
    if max_jacketed_section_dimension_ratio is None:
        return max(cap, 0.0)
    try:
        ratio = float(max_jacketed_section_dimension_ratio)
        if not np.isfinite(ratio) or ratio <= 1.0:
            return 0.0
        sec = _member_base_section(elements, column_sec, beam_sec, int(eid))
        b = float(_section_b_mm(sec))
        h = float(_section_h_mm(sec))
        ratio_cap = 0.5 * (ratio - 1.0) * min(b, h)
        cap = min(cap, ratio_cap)
    except Exception:
        pass
    return max(float(cap), 0.0)


def _format_member_limit_map(values: Dict[int, float], member_ids: List[int]) -> str:
    return "; ".join(
        f"{int(mid)}:{float(values[int(mid)]):.0f}"
        for mid in sorted(int(m) for m in member_ids if int(m) in values)
    )


def _next_higher_thickness_with_cap(current_t: float, step_t: float, cap_t: float) -> Optional[float]:
    """Return next jacket thickness only if it stays within the member cap."""
    next_t = float(current_t) + float(step_t)
    if next_t <= float(cap_t) + 1e-9:
        return float(next_t)
    if float(current_t) < float(cap_t) - 1e-9:
        return float(cap_t)
    return None


def _jacketing_alternative_recommendation(mode: str) -> str:
    mode_txt = str(mode or "").lower()
    if "spacing" in mode_txt or "detailing" in mode_txt:
        return "Longitudinal rebar spacing/detailing limit reached; consider larger jacket dimensions, smaller/more distributed bars, multiple layers with explicit detailing, steel/FRP strengthening, or member replacement."
    if "shear" in mode_txt:
        return "Jacket size/detailing limit reached; consider steel/FRP wrapping, added wall/bracing, or member replacement for shear."
    if "flexure" in mode_txt or "axial" in mode_txt:
        return "Jacket size/detailing limit reached; consider steel plates/FRP flexural strengthening, larger longitudinal bars with anchorage check, added wall/bracing, or member replacement."
    return "Jacket size/detailing limit reached; consider an alternative local/global retrofit strategy."


def optimize_concrete_jacketing(
    *,
    nodes,
    elements,
    geom,
    combos: Dict[str, Dict[str, Any]],
    combo_names: List[str],
    Ec_Pa: float,
    Es_Pa: float,
    column_sec,
    beam_sec,
    ec2mat,
    ops_build_model,
    ops_apply_combo_loads,
    ops_run_static,
    failed_member_ids: List[int],
    walls=None,
    wall_tw_mm: float = 200.0,
    wall_E_Pa: float = 30e9,
    use_rigid_diaphragm: bool = False,
    initial_jacket_thickness_mm: float = 75.0,
    max_jacket_thickness_mm: float = 200.0,
    max_jacketed_section_dimension_ratio: Optional[float] = 1.50,
    jacket_thickness_step_mm: float = 25.0,
    initial_stirrup_spacing_mm: float = 150.0,
    min_stirrup_spacing_mm: float = 50.0,
    stirrup_spacing_steps_mm: Iterable[float] = (150.0, 100.0, 75.0, 50.0),
    longitudinal_rebar_ratio_steps: Iterable[float] = (0.0, 0.25, 0.50, 0.75, 1.00, 1.50),
    max_longitudinal_rebar_ratio: float = 1.50,
    prefer_rebar_over_thickness_for_flexure: bool = True,
    max_iterations: int = 30,
    fc_jacket_mpa: Optional[float] = None,
    unit_costs: Optional[Dict[str, float]] = None,
    aggregate_size_mm: float = 20.0,
    ec2_spacing_k1: float = 1.0,
    ec2_spacing_k2_mm: float = 5.0,
    enforce_longitudinal_rebar_spacing: bool = True,
    concrete_density_kgm3: Optional[float] = None,
    seismic_settings: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Smart iterative automatic concrete-jacketing optimizer.

    Strategy:
    1. Run a baseline EC2 assessment of the post-wall, pre-jacketing state.
    2. Start each initially failed member at a 75 mm jacket and 150 mm stirrups
       (or the existing spacing, if smaller).
    3. At every iteration, freeze members that pass.
    4. Upgrade only still-failing members according to their governing mode:
       shear -> reduce stirrup spacing; flexure/axial -> increase longitudinal rebars first, then jacket thickness.
    5. Stop when all tracked members pass, engineering limits are reached, or
       max_iterations is reached. Jacket thickness is also capped by a
       final/original section-dimension ratio, so a 400 x 400 member with
       a 1.50 ratio limit cannot receive more than 100 mm of jacket.
       Longitudinal-bar upgrades are accepted only if the modeled bars satisfy
       the EC2-style minimum clear spacing check; otherwise the optimizer first
       tries more jacket thickness and then recommends an alternative detail.
    6. Mark the cheapest feasible iteration as Selected. If no feasible row
       exists, select the row with the fewest remaining failures, lowest max eta,
       then lowest cost.

    Returned rows are the iteration table. Each row keeps main.py-compatible
    internal objects: _element_sections, _df_worst, _member_perf, _member_fail.
    """
    tracked_member_ids = sorted({int(e) for e in failed_member_ids if int(e) in elements})
    if not tracked_member_ids:
        return pd.DataFrame()

    allowed_spacings = sorted(
        {float(s) for s in stirrup_spacing_steps_mm if float(s) >= float(min_stirrup_spacing_mm)},
        reverse=True,
    )
    if float(initial_stirrup_spacing_mm) not in allowed_spacings:
        allowed_spacings.append(float(initial_stirrup_spacing_mm))
        allowed_spacings = sorted(set(allowed_spacings), reverse=True)

    allowed_rebar_ratios = sorted({
        float(r) for r in longitudinal_rebar_ratio_steps
        if 0.0 <= float(r) <= float(max_longitudinal_rebar_ratio) + 1e-9
    })
    if 0.0 not in allowed_rebar_ratios:
        allowed_rebar_ratios.insert(0, 0.0)

    # Per-member jacket thickness cap. This prevents unrealistic enlargement
    # by limiting the final width and height to a ratio of the original
    # member dimensions. With the default ratio of 1.50, a 400 x 400 member
    # cannot grow beyond 600 x 600.
    max_jacket_thickness_by_element = {
        int(eid): _member_max_jacket_thickness_mm(
            elements=elements,
            column_sec=column_sec,
            beam_sec=beam_sec,
            eid=int(eid),
            max_jacket_thickness_mm=float(max_jacket_thickness_mm),
            max_jacketed_section_dimension_ratio=max_jacketed_section_dimension_ratio,
        )
        for eid in tracked_member_ids
    }

    # If the requested initial jacket exceeds the member-specific cap, start at
    # the cap instead of creating an oversized section.
    initial_jacket_by_element = {
        int(eid): min(float(initial_jacket_thickness_mm), float(max_jacket_thickness_by_element[int(eid)]))
        for eid in tracked_member_ids
    }

    # Baseline = after selected shear-wall layout, before any concrete jacketing.
    # This is used only for reporting the original eta and original governing mode.
    original_df_worst, _, original_member_fail = run_ec2_member_assessment(
        nodes=nodes,
        elements=elements,
        geom=geom,
        combos=combos,
        combo_names=combo_names,
        Ec_Pa=Ec_Pa,
        Es_Pa=Es_Pa,
        column_sec=column_sec,
        beam_sec=beam_sec,
        ec2mat=ec2mat,
        ops_build_model=ops_build_model,
        ops_apply_combo_loads=ops_apply_combo_loads,
        ops_run_static=ops_run_static,
        walls=walls,
        wall_tw_mm=wall_tw_mm,
        wall_E_Pa=wall_E_Pa,
        use_rigid_diaphragm=use_rigid_diaphragm,
        element_sections=None,
    )

    jacket_thickness_by_element = dict(initial_jacket_by_element)
    stirrup_spacing_by_element = {
        eid: min(float(initial_stirrup_spacing_mm), _sec_spacing_or(_member_base_section(elements, column_sec, beam_sec, eid), initial_stirrup_spacing_mm))
        for eid in tracked_member_ids
    }
    longitudinal_rebar_ratio_by_element = {eid: 0.0 for eid in tracked_member_ids}
    all_jacketed_member_ids = list(tracked_member_ids)
    upgraded_this_iteration = list(tracked_member_ids)
    rows = []
    final_reason = "maximum_iterations_reached"

    for iteration in range(1, int(max_iterations) + 1):
        element_sections = make_jacketed_element_sections(
            elements=elements,
            column_sec=column_sec,
            beam_sec=beam_sec,
            failed_member_ids=all_jacketed_member_ids,
            t_jacket_mm=float(initial_jacket_thickness_mm),
            fc_jacket_mpa=fc_jacket_mpa,
            stirrup_spacing_mm=float(initial_stirrup_spacing_mm),
            jacket_thickness_by_element=jacket_thickness_by_element,
            stirrup_spacing_by_element=stirrup_spacing_by_element,
            longitudinal_rebar_ratio_by_element=longitudinal_rebar_ratio_by_element,
        )

        cost = estimate_jacketing_cost(
            nodes=nodes,
            elements=elements,
            column_sec=column_sec,
            beam_sec=beam_sec,
            failed_member_ids=all_jacketed_member_ids,
            t_jacket_mm=float(initial_jacket_thickness_mm),
            jacket_thickness_by_element=jacket_thickness_by_element,
            longitudinal_rebar_ratio_by_element=longitudinal_rebar_ratio_by_element,
            unit_costs=unit_costs,
        )

        combos_it = _combos_with_extra_self_weight(
            combos,
            elements=elements,
            column_sec=column_sec,
            beam_sec=beam_sec,
            element_sections=element_sections,
            concrete_density_kgm3=concrete_density_kgm3,
        )
        names_it = combo_names
        if seismic_settings:
            # seismic action recomputed with the jacketed (stiffer, heavier) sections
            ev_it = _seismic_eval(
                seismic_settings, nodes=nodes, elements=elements, geom=geom, combos=combos_it,
                combo_names=combo_names, Ec_Pa=Ec_Pa, Es_Pa=Es_Pa, column_sec=column_sec, beam_sec=beam_sec,
                walls=walls, wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa,
                element_sections=element_sections, ops_build_model=ops_build_model,
            )
            if ev_it is not None:
                combos_it, names_it = ev_it["combos"], ev_it["combo_names"]
        df_worst, member_perf, member_fail = run_ec2_member_assessment(
            nodes=nodes,
            elements=elements,
            geom=geom,
            combos=combos_it,
            combo_names=names_it,
            Ec_Pa=Ec_Pa,
            Es_Pa=Es_Pa,
            column_sec=column_sec,
            beam_sec=beam_sec,
            ec2mat=ec2mat,
            ops_build_model=ops_build_model,
            ops_apply_combo_loads=ops_apply_combo_loads,
            ops_run_static=ops_run_static,
            walls=walls,
            wall_tw_mm=wall_tw_mm,
            wall_E_Pa=wall_E_Pa,
            use_rigid_diaphragm=use_rigid_diaphragm,
            element_sections=element_sections,
        )

        remaining_failed_ids = _remaining_failed_member_ids(member_fail, tracked_member_ids)
        max_eta = _max_numeric_column(df_worst, ["eta", "Static_EC2_Utilization", "Utilization"])
        governing_modes = {
            int(eid): _governing_failure_mode_for_member(None, df_worst, int(eid))
            for eid in remaining_failed_ids
        }
        spacing_checks = _spacing_map_for_jacketed_members(
            elements=elements,
            column_sec=column_sec,
            beam_sec=beam_sec,
            member_ids=all_jacketed_member_ids,
            jacket_thickness_by_element=jacket_thickness_by_element,
            stirrup_spacing_by_element=stirrup_spacing_by_element,
            longitudinal_rebar_ratio_by_element=longitudinal_rebar_ratio_by_element,
            aggregate_size_mm=aggregate_size_mm,
            ec2_spacing_k1=ec2_spacing_k1,
            ec2_spacing_k2_mm=ec2_spacing_k2_mm,
        ) if enforce_longitudinal_rebar_spacing else {}
        rebar_spacing_limit_members = [
            int(eid) for eid, chk in spacing_checks.items()
            if str(chk.get("status", "OK")).upper() == "FAIL"
        ]
        member_report = _build_member_jacketing_report(
            member_ids=tracked_member_ids,
            original_df_worst=original_df_worst,
            final_df_worst=df_worst,
            jacket_thickness_by_element=jacket_thickness_by_element,
            stirrup_spacing_by_element=stirrup_spacing_by_element,
            longitudinal_rebar_ratio_by_element=longitudinal_rebar_ratio_by_element,
            final_fail_ids=remaining_failed_ids,
            elements=elements,
            column_sec=column_sec,
            beam_sec=beam_sec,
            aggregate_size_mm=aggregate_size_mm,
            ec2_spacing_k1=ec2_spacing_k1,
            ec2_spacing_k2_mm=ec2_spacing_k2_mm,
        )
        thickness_map_txt = _format_member_float_map(jacket_thickness_by_element, all_jacketed_member_ids)
        spacing_map_txt = _format_member_float_map(stirrup_spacing_by_element, all_jacketed_member_ids)
        rebar_map_txt = _format_member_percent_map(longitudinal_rebar_ratio_by_element, all_jacketed_member_ids)
        max_allowed_t_txt = _format_member_limit_map(max_jacket_thickness_by_element, all_jacketed_member_ids)
        members_at_thickness_limit = [
            int(eid) for eid in all_jacketed_member_ids
            if float(jacket_thickness_by_element.get(int(eid), 0.0)) >= float(max_jacket_thickness_by_element.get(int(eid), 0.0)) - 1e-9
        ]
        alternative_required_now = [
            int(eid) for eid in remaining_failed_ids
            if int(eid) in members_at_thickness_limit
            and _next_higher_rebar_ratio(longitudinal_rebar_ratio_by_element.get(int(eid), 0.0), allowed_rebar_ratios) is None
            and _next_lower_spacing(stirrup_spacing_by_element.get(int(eid), initial_stirrup_spacing_mm), allowed_spacings) is None
        ]
        alternative_required_now = sorted(set(alternative_required_now).union(set(rebar_spacing_limit_members)))
        alternative_recommendations = "; ".join(
            f"{eid}:{_jacketing_alternative_recommendation('rebar_spacing' if int(eid) in rebar_spacing_limit_members else governing_modes.get(int(eid), 'unknown'))}"
            for eid in alternative_required_now
        )
        feasible = bool(
            len(remaining_failed_ids) == 0
            and np.isfinite(max_eta)
            and max_eta <= 1.0
            and len(rebar_spacing_limit_members) == 0
        )

        rows.append({
            "Option_ID": f"CJ_ITER_{iteration:03d}",
            "Iteration": iteration,
            "Jacketed_Members": ",".join(map(str, all_jacketed_member_ids)),
            "Upgraded_Members": ",".join(map(str, upgraded_this_iteration)),
            "Members_Targeted_This_Iteration": ",".join(map(str, upgraded_this_iteration)),
            "Num_Jacketed_Members": len(all_jacketed_member_ids),
            "Jacket_Thickness_mm": max(jacket_thickness_by_element.values()) if jacket_thickness_by_element else np.nan,
            "Stirrup_Spacing_mm": min(stirrup_spacing_by_element.values()) if stirrup_spacing_by_element else np.nan,
            "Per_Member_Jacket_Thickness_mm": thickness_map_txt,
            "Max_Allowed_Jacket_Thickness_By_Member_mm": max_allowed_t_txt,
            "Max_Jacketed_Section_Dimension_Ratio": max_jacketed_section_dimension_ratio,
            "Members_At_Jacket_Size_Limit": ",".join(map(str, sorted(members_at_thickness_limit))),
            "Alternative_Required_Members": ",".join(map(str, sorted(alternative_required_now))),
            "Alternative_Recommendation": alternative_recommendations,
            "Per_Member_Stirrup_Spacing_mm": spacing_map_txt,
            "Per_Member_Longitudinal_Rebar_Increase": rebar_map_txt,
            "EC2_Rebar_Spacing_Rule": f"clear spacing >= max({ec2_spacing_k1:.2g}*phi, dg+{ec2_spacing_k2_mm:.0f} mm, 20 mm), dg={aggregate_size_mm:.0f} mm",
            "Rebar_Spacing_Limit_Members": ",".join(map(str, sorted(rebar_spacing_limit_members))),
            "Per_Member_Min_Clear_Rebar_Spacing_mm": _format_member_spacing_map(spacing_checks, all_jacketed_member_ids, "min_clear_spacing_mm"),
            "Per_Member_Required_Clear_Rebar_Spacing_mm": _format_member_spacing_map(spacing_checks, all_jacketed_member_ids, "required_clear_spacing_mm"),
            "Per_Member_Rebar_Spacing_Status": _format_member_spacing_map(spacing_checks, all_jacketed_member_ids, "status"),
            "Governing_Mode_By_Member": "; ".join(f"{eid}:{_governing_mode_label(mode)}" for eid, mode in sorted(governing_modes.items())),
            "Optimization_Action_Rule": "SHEAR -> stirrup spacing; MOMENT/FLEXURE-AXIAL -> longitudinal rebars, then thickness",
            "Max_EC2_Utilization": max_eta,
            "Remaining_Failed_Members": len(remaining_failed_ids),
            "Remaining_Failed_Member_IDs": ",".join(map(str, remaining_failed_ids)),
            "Feasible": "YES" if feasible else "NO",
            "Selected": "NO",
            "Convergence_Reason": "all_members_pass" if feasible else "running",
            "Concrete_m3": cost.concrete_m3,
            "Rebar_kg": cost.rebar_kg,
            "Formwork_m2": cost.formwork_m2,
            "Total_Cost": cost.total_cost,
            "Duration_Days": cost.details.get("duration_days", 0.0),
            "Site_Overhead_Cost": cost.details.get("site_overhead_cost", 0.0),
            "_element_sections": element_sections,
            "_df_worst": df_worst,
            "_member_perf": member_perf,
            "_member_fail": member_fail,
            "_member_report": member_report,
            "_original_df_worst": original_df_worst,
            "_original_member_fail": original_member_fail,
        })

        if feasible:
            final_reason = "all_members_pass"
            break

        next_upgrades = []
        # Members that fail only after redistribution are added to the tracked set.
        for e_new in sorted(set(_failed_member_ids(member_fail)) - set(tracked_member_ids)):
            if e_new not in elements:
                continue
            tracked_member_ids.append(e_new)
            all_jacketed_member_ids.append(e_new)
            max_jacket_thickness_by_element[e_new] = _member_max_jacket_thickness_mm(
                elements=elements, column_sec=column_sec, beam_sec=beam_sec, eid=e_new,
                max_jacket_thickness_mm=float(max_jacket_thickness_mm),
                max_jacketed_section_dimension_ratio=max_jacketed_section_dimension_ratio,
            )
            jacket_thickness_by_element[e_new] = min(float(initial_jacket_thickness_mm), max_jacket_thickness_by_element[e_new])
            stirrup_spacing_by_element[e_new] = min(float(initial_stirrup_spacing_mm), _sec_spacing_or(_member_base_section(elements, column_sec, beam_sec, e_new), initial_stirrup_spacing_mm))
            longitudinal_rebar_ratio_by_element[e_new] = 0.0
            next_upgrades.append(e_new)
        blocked_by_limits = []
        for eid in remaining_failed_ids:
            eid = int(eid)
            mode = governing_modes.get(eid, "flexure_axial")
            upgraded = False

            if mode == "shear":
                current_s = stirrup_spacing_by_element[eid]
                next_s = _next_lower_spacing(current_s, allowed_spacings)
                if next_s is not None and next_s >= float(min_stirrup_spacing_mm):
                    stirrup_spacing_by_element[eid] = float(next_s)
                    upgraded = True
                else:
                    # If shear detailing is already at its limit, do not loop forever.
                    # A thicker jacket may still help via section geometry, so try it
                    # before declaring the member blocked.
                    next_t = _next_higher_thickness_with_cap(
                        jacket_thickness_by_element[eid],
                        jacket_thickness_step_mm,
                        max_jacket_thickness_by_element[eid],
                    )
                    if next_t is not None:
                        jacket_thickness_by_element[eid] = float(next_t)
                        upgraded = True
            else:
                # Moment/flexure/axial problem: increase longitudinal bars first.
                # This avoids unnecessary jacket-thickness growth for beams whose
                # demand is controlled by top/bottom flexural reinforcement.
                next_r = None
                if prefer_rebar_over_thickness_for_flexure:
                    next_r = _next_higher_rebar_ratio(longitudinal_rebar_ratio_by_element[eid], allowed_rebar_ratios)
                if next_r is not None and next_r <= float(max_longitudinal_rebar_ratio) + 1e-9:
                    spacing_ok = True
                    if enforce_longitudinal_rebar_spacing:
                        spacing_trial = _spacing_check_for_jacketed_member(
                            elements=elements,
                            column_sec=column_sec,
                            beam_sec=beam_sec,
                            eid=eid,
                            jacket_thickness_mm=jacket_thickness_by_element[eid],
                            stirrup_spacing_mm=stirrup_spacing_by_element[eid],
                            longitudinal_rebar_ratio=float(next_r),
                            aggregate_size_mm=aggregate_size_mm,
                            ec2_spacing_k1=ec2_spacing_k1,
                            ec2_spacing_k2_mm=ec2_spacing_k2_mm,
                        )
                        spacing_ok = str(spacing_trial.get("status", "OK")).upper() == "OK"
                    if spacing_ok:
                        longitudinal_rebar_ratio_by_element[eid] = float(next_r)
                        upgraded = True
                    else:
                        # More jacket thickness increases the available bar row length.
                        # Try geometry increase first instead of accepting a congested bar layout.
                        current_t = jacket_thickness_by_element[eid]
                        next_t = _next_higher_thickness_with_cap(
                            current_t,
                            jacket_thickness_step_mm,
                            max_jacket_thickness_by_element[eid],
                        )
                        if next_t is not None:
                            jacket_thickness_by_element[eid] = float(next_t)
                            upgraded = True
                else:
                    current_t = jacket_thickness_by_element[eid]
                    next_t = _next_higher_thickness_with_cap(
                        current_t,
                        jacket_thickness_step_mm,
                        max_jacket_thickness_by_element[eid],
                    )
                    if next_t is not None:
                        jacket_thickness_by_element[eid] = float(next_t)
                        upgraded = True
                    else:
                        # As a last resort, tighten shear links if that is still available,
                        # but keep the diagnosis visible as flexure/axial in the report.
                        next_s = _next_lower_spacing(stirrup_spacing_by_element[eid], allowed_spacings)
                        if next_s is not None and next_s >= float(min_stirrup_spacing_mm):
                            stirrup_spacing_by_element[eid] = float(next_s)
                            upgraded = True

            if upgraded:
                next_upgrades.append(eid)
            else:
                blocked_by_limits.append(eid)

        if not next_upgrades:
            final_reason = "engineering_limits_reached_or_section_size_limit_reached"
            rows[-1]["Convergence_Reason"] = final_reason
            rows[-1]["Limit_Blocked_Members"] = ",".join(map(str, blocked_by_limits))
            if blocked_by_limits:
                rows[-1]["Alternative_Required_Members"] = ",".join(map(str, sorted(set(blocked_by_limits))))
                rows[-1]["Alternative_Recommendation"] = "; ".join(
                    f"{eid}:{_jacketing_alternative_recommendation(governing_modes.get(int(eid), 'unknown'))}"
                    for eid in sorted(set(blocked_by_limits))
                )
            break

        rows[-1]["Convergence_Reason"] = "upgraded_remaining_failed_members"
        rows[-1]["Limit_Blocked_Members"] = ",".join(map(str, blocked_by_limits))
        upgraded_this_iteration = sorted(set(next_upgrades))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Cost-aware selection. Keep the table chronological, but explicitly mark
    # the chosen row for main.py. Monotonic costs normally mean the first fully
    # feasible row is cheapest; this also handles future non-monotonic cost
    # models or optional longitudinal-bar upgrades.
    feasible_mask = df["Feasible"].astype(str).str.upper().eq("YES")
    if feasible_mask.any():
        selected_idx = df.loc[feasible_mask].sort_values(
            ["Total_Cost", "Jacket_Thickness_mm", "Max_EC2_Utilization", "Iteration"],
            ascending=[True, True, True, True],
        ).index[0]
        df.at[df.index[-1], "Convergence_Reason"] = final_reason
    else:
        selected_idx = df.sort_values(
            ["Remaining_Failed_Members", "Max_EC2_Utilization", "Total_Cost", "Iteration"],
            ascending=[True, True, True, False],
        ).index[0]
        df.at[df.index[-1], "Convergence_Reason"] = final_reason
    df.loc[:, "Selected"] = "NO"
    df.at[selected_idx, "Selected"] = "YES"
    return df.reset_index(drop=True)



# ============================================================
# GLOBAL RETROFIT STRATEGY OPTIMIZATION
# ============================================================
def _selected_row(df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if df is None or getattr(df, "empty", True):
        return None
    if "Selected" in df.columns:
        sel = df[df["Selected"].astype(str).str.upper().eq("YES")]
        if not sel.empty:
            return sel.iloc[0]
    return df.iloc[0]


def _global_failed_count(member_fail_df: Optional[pd.DataFrame]) -> int:
    return len(_failed_member_ids(member_fail_df))



def _global_wall_strategy_key(candidate: ShearWallCandidate) -> str:
    """Stable key for the actual wall panels used by a global alternative.

    Different candidate generators can sometimes produce the same final wall
    panels with different candidate IDs/descriptions. This key prevents the
    report from showing duplicated alternatives.
    """
    walls = list(getattr(candidate, "walls", []) or [])
    panels = sorted(
        (
            int(getattr(w, "bay_left", 0)),
            int(getattr(w, "bay_right", 0)),
            int(getattr(w, "floor_bot", 0)),
            int(getattr(w, "floor_top", 0)),
        )
        for w in walls
    )
    if not panels:
        panel_txt = "NO_NEW_WALL"
    else:
        panel_txt = ";".join(f"B{a}-{b}:F{fb}-{ft}" for a, b, fb, ft in panels)
    try:
        tw = float(getattr(candidate, "wall_tw_mm", 0.0))
    except Exception:
        tw = 0.0
    return f"tw={tw:.3f}|{panel_txt}"


def _global_jacketing_strategy_key(cj_selected: Optional[pd.Series], jacketed_member_ids: List[int]) -> str:
    """Stable key for the concrete-jacketing decision of a global alternative."""
    if cj_selected is None or not jacketed_member_ids:
        return "NO_JACKETING"
    fields = [
        "Jacketed_Members",
        "Per_Member_Jacket_Thickness_mm",
        "Per_Member_Stirrup_Spacing_mm",
        "Per_Member_Longitudinal_Rebar_Increase",
    ]
    parts = []
    for field in fields:
        try:
            val = cj_selected.get(field, "")
        except Exception:
            val = ""
        parts.append(f"{field}={str(val).strip()}")
    if not str(parts[0]).split("=", 1)[-1].strip():
        parts[0] = "Jacketed_Members=" + ",".join(map(str, sorted(set(int(x) for x in jacketed_member_ids))))
    return "|".join(parts)


def _deduplicate_global_alternatives(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row for each unique complete retrofit strategy.

    A complete strategy is defined by the actual wall panels/thickness and the
    selected jacketing detailing. Among duplicates, keep the best row according
    to the same hierarchy used by the global optimization.
    """
    if df is None or df.empty or "Strategy_Key" not in df.columns:
        return df
    d = df.copy()
    d["_feasible_sort"] = d.get("Complete_Feasible", "NO").astype(str).str.upper().eq("YES").astype(int)
    d["_obj_sort"] = pd.to_numeric(d.get("Global_Objective_Score", np.nan), errors="coerce")
    d["_cost_sort"] = pd.to_numeric(d.get("Total_Retrofit_Cost", np.nan), errors="coerce")
    d["_drift_sort"] = pd.to_numeric(d.get("Drift_Utilization", np.nan), errors="coerce")
    d = d.sort_values(
        ["_feasible_sort", "_cost_sort", "_obj_sort", "_drift_sort"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    d = d.drop_duplicates(subset=["Strategy_Key"], keep="first")
    return d.drop(columns=["_feasible_sort", "_obj_sort", "_cost_sort", "_drift_sort"], errors="ignore")


def optimize_global_retrofit_strategy(
    *,
    nodes,
    elements,
    geom,
    combos,
    combo_names,
    Ec_Pa,
    Es_Pa,
    column_sec,
    beam_sec,
    ec2mat,
    ops_build_model,
    ops_apply_combo_loads,
    ops_run_static,
    base_walls=None,
    wall_tw_mm: float = 200.0,
    wall_E_Pa: float = 30e9,
    use_rigid_diaphragm: bool = False,
    drift_limit_ratio: float = 0.005,
    failed_stories: Optional[List[int]] = None,
    thicknesses_mm: Iterable[float] = (150.0, 200.0, 250.0, 300.0),
    max_walls_per_layout: Optional[int] = None,
    wall_layout_mode: str = "hybrid",
    include_no_new_wall_candidate: bool = True,
    max_global_candidates: Optional[int] = None,
    unit_costs: Optional[Dict[str, float]] = None,
    global_penalty_scale: float = 1.0e9,
    concrete_density_kgm3: Optional[float] = None,
    wall_density_kgm3: Optional[float] = None,
    seismic_settings: Optional[Dict[str, Any]] = None,
    max_jacketed_section_dimension_ratio: Optional[float] = 1.50,
    max_jacket_thickness_mm: float = 200.0,
    candidates_override: Optional[List[ShearWallCandidate]] = None,
) -> pd.DataFrame:
    """Optimize the complete retrofit strategy, not only the wall layout.

    This is the global layer above the existing wall and concrete-jacketing
    optimizers.  Each row represents a complete strategy:

        selected wall candidate + required concrete jacketing, if any

    The primary ranking objective is total retrofit cost.  Drift failure,
    remaining member failure, poor wall constructability and jacketing/detailing
    limits are treated as feasibility constraints and penalty terms.  The best
    row is explicitly marked with ``Selected = YES``.

    The returned dataframe keeps internal objects used by ``main.py`` and the
    report exporter: ``_candidate``, ``_walls``, ``_df_worst_after_walls``,
    ``_df_worst_final``, ``_element_sections``, ``_member_report``,
    ``_jacket_cost`` and ``_jacketing_iteration_worst_dfs``.
    """
    base_walls = list(base_walls or [])

    # First generate and rank wall candidates with the existing transparent
    # wall objective. The global optimizer then evaluates the complete
    # post-wall + jacketing outcome. By default it evaluates all generated
    # wall candidates; pass max_global_candidates only when a deliberately
    # limited research run is needed.
    if candidates_override is not None and len(candidates_override) == 0:
        sw_df = None
    else:
      sw_df = optimize_shear_wall_positions(
        candidates_override=candidates_override,
        nodes=nodes,
        elements=elements,
        geom=geom,
        combos=combos,
        combo_names=combo_names,
        Ec_Pa=Ec_Pa,
        Es_Pa=Es_Pa,
        column_sec=column_sec,
        beam_sec=beam_sec,
        ops_build_model=ops_build_model,
        ops_apply_combo_loads=ops_apply_combo_loads,
        ops_run_static=ops_run_static,
        base_walls=base_walls,
        wall_tw_mm=wall_tw_mm,
        wall_E_Pa=wall_E_Pa,
        use_rigid_diaphragm=use_rigid_diaphragm,
        drift_limit_ratio=drift_limit_ratio,
        failed_stories=failed_stories,
        thicknesses_mm=thicknesses_mm,
        max_walls_per_layout=max_walls_per_layout,
        wall_layout_mode=wall_layout_mode,
        early_stop_by_wall_count=False,
        unit_costs=unit_costs,
        seismic_settings=seismic_settings,
        wall_density_kgm3=wall_density_kgm3,
    )

    candidate_rows: List[Dict[str, Any]] = []

    generated_wall_candidate_count = int(len(sw_df)) if sw_df is not None else 0
    global_candidate_limit_label = _format_global_candidate_limit(max_global_candidates)

    if include_no_new_wall_candidate:
        no_wall = ShearWallCandidate(
            candidate_id="NO_NEW_WALL",
            walls=[],
            wall_tw_mm=float(wall_tw_mm),
            wall_E_Pa=float(wall_E_Pa),
            description="No additional shear walls; verify whether jacketing alone is sufficient.",
        )
        baseline_drifts, ok_names, failed_names = run_static_drift_assessment(
            nodes=nodes,
            elements=elements,
            geom=geom,
            combos=combos,
            combo_names=combo_names,
            Ec_Pa=Ec_Pa,
            Es_Pa=Es_Pa,
            column_sec=column_sec,
            beam_sec=beam_sec,
            ops_build_model=ops_build_model,
            ops_apply_combo_loads=ops_apply_combo_loads,
            ops_run_static=ops_run_static,
            walls=base_walls,
            wall_tw_mm=wall_tw_mm,
            wall_E_Pa=wall_E_Pa,
            use_rigid_diaphragm=use_rigid_diaphragm,
        )
        max_drift = _max_numeric_column(baseline_drifts, ["MaxDriftRatio", "DriftRatio", "drift_ratio"])
        max_drift_mm = _max_numeric_column(baseline_drifts, ["MaxDrift_mm", "Drift [mm]", "Max Drift [mm]"])
        feasible = bool(np.isfinite(max_drift) and max_drift <= drift_limit_ratio and not failed_names)
        candidate_rows.append({
            "Candidate_ID": no_wall.candidate_id,
            "Description": no_wall.description,
            "Num_Wall_Panels": 0,
            "Num_Wall_Lines": 0,
            "Wall_Lines": 0,
            "Wall_Layout_Mode": "none",
            "Wall_Segments": "none",
            "Constructability_Class": "no_new_wall",
            "Objective_Constructability_Score": 0.0,
            "Wall_Thickness_mm": no_wall.wall_tw_mm,
            "Max_Drift_Ratio": max_drift,
            "Max_Drift_mm": max_drift_mm,
            "Drift_Limit_Ratio": drift_limit_ratio,
            "Drift_Utilization": max_drift / drift_limit_ratio if drift_limit_ratio > 0 and np.isfinite(max_drift) else np.nan,
            "Feasible": "YES" if feasible else "NO",
            "Analysis_OK_Count": len(ok_names),
            "Analysis_Failed_Count": len(failed_names),
            "Concrete_m3": 0.0,
            "Rebar_kg": 0.0,
            "Formwork_m2": 0.0,
            "Total_Cost": 0.0,
            "Duration_Days": 0.0,
            "Site_Overhead_Cost": 0.0,
            "Objective_Total_Score": 0.0,
            "Objective_Rank": 0,
            "_candidate": no_wall,
            "_drifts": baseline_drifts,
            "_baseline_drifts": baseline_drifts,
        })

    if sw_df is not None and not sw_df.empty:
        eval_df = sw_df.copy()
        if max_global_candidates is not None:
            eval_df = eval_df.head(max(1, int(max_global_candidates))).copy()
        candidate_rows.extend(eval_df.to_dict("records"))

    candidates_sent_to_global_stage = int(len(candidate_rows))

    rows: List[Dict[str, Any]] = []
    for rank_i, sw_row in enumerate(candidate_rows, start=1):
        cand = sw_row.get("_candidate")
        if cand is None:
            continue
        trial_walls = base_walls + list(getattr(cand, "walls", []) or [])
        wall_cost = float(pd.to_numeric(sw_row.get("Total_Cost", 0.0), errors="coerce") or 0.0)
        wall_duration_days = float(pd.to_numeric(sw_row.get("Duration_Days", np.nan), errors="coerce")) if pd.notna(sw_row.get("Duration_Days", np.nan)) else 0.0
        drift_util = float(pd.to_numeric(sw_row.get("Drift_Utilization", np.nan), errors="coerce")) if pd.notna(sw_row.get("Drift_Utilization", np.nan)) else np.nan
        max_drift_ratio = float(pd.to_numeric(sw_row.get("Max_Drift_Ratio", np.nan), errors="coerce")) if pd.notna(sw_row.get("Max_Drift_Ratio", np.nan)) else np.nan
        drift_ok = bool(np.isfinite(max_drift_ratio) and drift_limit_ratio > 0 and max_drift_ratio <= drift_limit_ratio and int(sw_row.get("Analysis_Failed_Count", 0) or 0) == 0)

        combos_cand = _combos_with_extra_self_weight(
            combos,
            elements=elements,
            column_sec=column_sec,
            beam_sec=beam_sec,
            nodes=nodes,
            geom=geom,
            new_walls=list(getattr(cand, "walls", []) or []),
            wall_tw_mm=float(getattr(cand, "wall_tw_mm", wall_tw_mm)),
            wall_density_kgm3=wall_density_kgm3,
        )

        # EC8 seismic action (lateral force method) for this wall layout
        static_util = max_drift_ratio / drift_limit_ratio if drift_limit_ratio > 0 and np.isfinite(max_drift_ratio) else np.nan
        ev_walls = _seismic_eval(
            seismic_settings, nodes=nodes, elements=elements, geom=geom, combos=combos_cand,
            combo_names=combo_names, Ec_Pa=Ec_Pa, Es_Pa=Es_Pa, column_sec=column_sec, beam_sec=beam_sec,
            walls=trial_walls, wall_tw_mm=float(getattr(cand, "wall_tw_mm", wall_tw_mm)),
            wall_E_Pa=float(getattr(cand, "wall_E_Pa", wall_E_Pa)), ops_build_model=ops_build_model,
        )
        ev_final = ev_walls
        if seismic_settings:
            seis_util = ev_walls["dls_utilization"] if ev_walls else np.nan
            drift_util = float(np.nanmax([static_util, seis_util]))
            drift_ok = bool(drift_ok and ev_walls is not None and seis_util <= 1.0 and ev_walls["theta_ok"])
        names_after = ev_walls["combo_names"] if ev_walls else combo_names
        combos_after = ev_walls["combos"] if ev_walls else combos_cand

        df_worst_after, member_perf_after, member_fail_after = run_ec2_member_assessment(
            nodes=nodes,
            elements=elements,
            geom=geom,
            combos=combos_after,
            combo_names=names_after,
            Ec_Pa=Ec_Pa,
            Es_Pa=Es_Pa,
            column_sec=column_sec,
            beam_sec=beam_sec,
            ec2mat=ec2mat,
            ops_build_model=ops_build_model,
            ops_apply_combo_loads=ops_apply_combo_loads,
            ops_run_static=ops_run_static,
            walls=trial_walls,
            wall_tw_mm=float(getattr(cand, "wall_tw_mm", wall_tw_mm)),
            wall_E_Pa=float(getattr(cand, "wall_E_Pa", wall_E_Pa)),
            use_rigid_diaphragm=use_rigid_diaphragm,
        )
        fail_after_ids = _failed_member_ids(member_fail_after)

        cj_df = pd.DataFrame()
        cj_selected = None
        df_worst_final = df_worst_after
        member_fail_final = member_fail_after
        element_sections = None
        member_report = None
        jacket_cost_obj = RetrofitCostBreakdown()
        jacket_cost_total = 0.0
        jacketed_member_ids: List[int] = []
        jacket_duration_days = 0.0
        iteration_worst_dfs: List[Any] = []
        jacketing_required = bool(fail_after_ids)
        jacketing_limit_reached = False
        alternative_required = ""

        if fail_after_ids:
            cj_df = optimize_concrete_jacketing(
                nodes=nodes,
                elements=elements,
                geom=geom,
                combos=combos_cand,
                concrete_density_kgm3=concrete_density_kgm3,
                combo_names=combo_names,
                seismic_settings=seismic_settings,
                max_jacketed_section_dimension_ratio=max_jacketed_section_dimension_ratio,
                max_jacket_thickness_mm=max_jacket_thickness_mm,
                Ec_Pa=Ec_Pa,
                Es_Pa=Es_Pa,
                column_sec=column_sec,
                beam_sec=beam_sec,
                ec2mat=ec2mat,
                ops_build_model=ops_build_model,
                ops_apply_combo_loads=ops_apply_combo_loads,
                ops_run_static=ops_run_static,
                failed_member_ids=fail_after_ids,
                walls=trial_walls,
                wall_tw_mm=float(getattr(cand, "wall_tw_mm", wall_tw_mm)),
                wall_E_Pa=float(getattr(cand, "wall_E_Pa", wall_E_Pa)),
                use_rigid_diaphragm=use_rigid_diaphragm,
                unit_costs=unit_costs,
            )
            cj_selected = _selected_row(cj_df)
            if cj_selected is not None:
                df_worst_final = cj_selected.get("_df_worst", df_worst_after)
                member_fail_final = cj_selected.get("_member_fail", member_fail_after)
                element_sections = cj_selected.get("_element_sections", None)
                member_report = cj_selected.get("_member_report", None)
                jacket_cost_total = float(pd.to_numeric(cj_selected.get("Total_Cost", 0.0), errors="coerce") or 0.0)
                jacket_duration_days = float(pd.to_numeric(cj_selected.get("Duration_Days", np.nan), errors="coerce")) if pd.notna(cj_selected.get("Duration_Days", np.nan)) else 0.0
                jacket_cost_obj = RetrofitCostBreakdown(total_cost=jacket_cost_total)
                jacketed_member_ids = [int(x) for x in str(cj_selected.get("Jacketed_Members", "")).split(",") if str(x).strip()]
                jacketing_limit_reached = "limit" in str(cj_selected.get("Convergence_Reason", "")).lower()
                alternative_required = str(cj_selected.get("Alternative_Required_Members", "") or "")
                for _, iter_row in cj_df.iterrows():
                    iteration_worst_dfs.append({
                        "iteration": int(iter_row.get("Iteration", len(iteration_worst_dfs) + 1)),
                        "df_worst": iter_row.get("_df_worst", pd.DataFrame()).copy() if iter_row.get("_df_worst", None) is not None else pd.DataFrame(),
                        "label": f"After global jacketing iteration {int(iter_row.get('Iteration', len(iteration_worst_dfs) + 1))}",
                    })

        final_fail_count = _global_failed_count(member_fail_final)
        final_max_eta = _max_numeric_column(df_worst_final, ["eta", "Static_EC2_Utilization", "Utilization", "Max_Eta"])
        # Drift re-check with the jacketed (stiffer, heavier) sections
        if element_sections:
            try:
                combos_final = _combos_with_extra_self_weight(
                    combos_cand, elements=elements, column_sec=column_sec, beam_sec=beam_sec,
                    element_sections=element_sections, concrete_density_kgm3=concrete_density_kgm3,
                )
                df_dr_j, _ok_j, _failed_j = run_static_drift_assessment(
                    nodes=nodes, elements=elements, geom=geom, combos=combos_final,
                    combo_names=combo_names, Ec_Pa=Ec_Pa, Es_Pa=Es_Pa,
                    column_sec=column_sec, beam_sec=beam_sec,
                    ops_build_model=ops_build_model, ops_apply_combo_loads=ops_apply_combo_loads,
                    ops_run_static=ops_run_static, walls=trial_walls,
                    wall_tw_mm=float(getattr(cand, "wall_tw_mm", wall_tw_mm)),
                    wall_E_Pa=float(getattr(cand, "wall_E_Pa", wall_E_Pa)),
                    use_rigid_diaphragm=use_rigid_diaphragm, element_sections=element_sections,
                )
                md_j = _max_numeric_column(df_dr_j, ["MaxDriftRatio", "DriftRatio", "drift_ratio"])
                if np.isfinite(md_j):
                    max_drift_ratio = float(md_j)
                    drift_util = max_drift_ratio / drift_limit_ratio if drift_limit_ratio > 0 else np.nan
                    drift_ok = bool(max_drift_ratio <= drift_limit_ratio and not _failed_j)
                if seismic_settings:
                    ev_final = _seismic_eval(
                        seismic_settings, nodes=nodes, elements=elements, geom=geom, combos=combos_final,
                        combo_names=combo_names, Ec_Pa=Ec_Pa, Es_Pa=Es_Pa, column_sec=column_sec,
                        beam_sec=beam_sec, walls=trial_walls,
                        wall_tw_mm=float(getattr(cand, "wall_tw_mm", wall_tw_mm)),
                        wall_E_Pa=float(getattr(cand, "wall_E_Pa", wall_E_Pa)),
                        element_sections=element_sections, ops_build_model=ops_build_model,
                    )
                    seis_util = ev_final["dls_utilization"] if ev_final else np.nan
                    drift_util = float(np.nanmax([drift_util, seis_util]))
                    drift_ok = bool(drift_ok and ev_final is not None and seis_util <= 1.0 and ev_final["theta_ok"])
            except Exception as exc:
                print(f"WARNING: drift re-check after jacketing failed for {getattr(cand, 'candidate_id', '')}: {exc}")

        final_member_ok = bool(final_fail_count == 0 and np.isfinite(final_max_eta) and final_max_eta <= 1.0)
        complete_feasible = bool(drift_ok and final_member_ok and not jacketing_limit_reached)
        total_retrofit_cost = wall_cost + jacket_cost_total

        # Detailed cost components for transparent reporting of each alternative.
        # These are stored explicitly so the Word/Excel report can show the
        # formula behind the cost-comparison bar charts.
        _uc = {**DEFAULT_UNIT_COSTS, **(unit_costs or {})}
        wall_concrete_m3 = float(pd.to_numeric(sw_row.get("Concrete_m3", 0.0), errors="coerce") or 0.0)
        wall_rebar_kg = float(pd.to_numeric(sw_row.get("Rebar_kg", 0.0), errors="coerce") or 0.0)
        wall_formwork_m2 = float(pd.to_numeric(sw_row.get("Formwork_m2", 0.0), errors="coerce") or 0.0)
        wall_num_lines = int(pd.to_numeric(sw_row.get("Num_Wall_Lines", sw_row.get("Wall_Lines", 0)), errors="coerce") or 0)
        wall_material_cost = (
            wall_concrete_m3 * float(_uc.get("wall_concrete_per_m3", 0.0))
            + wall_rebar_kg * float(_uc.get("wall_rebar_per_kg", 0.0))
            + wall_formwork_m2 * float(_uc.get("wall_formwork_per_m2", 0.0))
        )
        wall_labor_cost = float(_uc.get("labor_factor", 0.0)) * wall_material_cost
        wall_indirect_cost = float(_uc.get("indirect_factor", 0.0)) * wall_material_cost
        wall_foundation_connection_cost = float(_uc.get("foundation_cost_per_wall_line", 0.0)) * wall_num_lines
        wall_constructability_cost = float(_uc.get("constructability_cost_per_wall_line", 0.0)) * wall_num_lines
        wall_site_overhead_cost = wall_duration_days * float(_uc.get("site_overhead_per_day", 0.0))

        if cj_selected is not None:
            jacket_concrete_m3 = float(pd.to_numeric(cj_selected.get("Concrete_m3", 0.0), errors="coerce") or 0.0)
            jacket_rebar_kg = float(pd.to_numeric(cj_selected.get("Rebar_kg", 0.0), errors="coerce") or 0.0)
            jacket_formwork_m2 = float(pd.to_numeric(cj_selected.get("Formwork_m2", 0.0), errors="coerce") or 0.0)
            jacket_num_members = int(pd.to_numeric(cj_selected.get("Num_Jacketed_Members", len(jacketed_member_ids)), errors="coerce") or len(jacketed_member_ids))
        else:
            jacket_concrete_m3 = 0.0
            jacket_rebar_kg = 0.0
            jacket_formwork_m2 = 0.0
            jacket_num_members = 0
        jacket_material_cost = (
            jacket_concrete_m3 * float(_uc.get("jacket_concrete_per_m3", 0.0))
            + jacket_rebar_kg * float(_uc.get("jacket_rebar_per_kg", 0.0))
            + jacket_formwork_m2 * float(_uc.get("jacket_formwork_per_m2", 0.0))
        )
        jacket_labor_cost = float(_uc.get("labor_factor", 0.0)) * jacket_material_cost
        jacket_indirect_cost = float(_uc.get("indirect_factor", 0.0)) * jacket_material_cost
        jacket_site_overhead_cost = jacket_duration_days * float(_uc.get("site_overhead_per_day", 0.0))

        drift_penalty = max(0.0, (drift_util if np.isfinite(drift_util) else 10.0) - 1.0) ** 2
        member_penalty = float(final_fail_count)
        constructability_penalty = float(pd.to_numeric(sw_row.get("Objective_Constructability_Score", 0.0), errors="coerce") or 0.0)
        jacketing_limit_penalty = 1.0 if jacketing_limit_reached or alternative_required else 0.0
        global_objective = (
            total_retrofit_cost
            + float(global_penalty_scale) * drift_penalty
            + float(global_penalty_scale) * member_penalty
            + 0.02 * float(global_penalty_scale) * constructability_penalty
            + 0.50 * float(global_penalty_scale) * jacketing_limit_penalty
        )

        row = dict(sw_row)
        wall_strategy_key = _global_wall_strategy_key(cand)
        jacketing_strategy_key = _global_jacketing_strategy_key(cj_selected, jacketed_member_ids)
        complete_strategy_key = f"WALL::{wall_strategy_key}||JACKET::{jacketing_strategy_key}"

        row.update({
            "Global_Evaluation_Rank_Input": rank_i,
            "Wall_Strategy_Key": wall_strategy_key,
            "Jacketing_Strategy_Key": jacketing_strategy_key,
            "Strategy_Key": complete_strategy_key,
            "Search_Max_Global_Candidates": global_candidate_limit_label,
            "Search_Generated_Wall_Candidates": generated_wall_candidate_count,
            "Search_Candidates_Sent_To_Global_Stage": candidates_sent_to_global_stage,
            "Search_No_New_Wall_Included": "YES" if include_no_new_wall_candidate else "NO",
            "Wall_Cost": wall_cost,
            "Wall_Concrete_m3": wall_concrete_m3,
            "Wall_Rebar_kg": wall_rebar_kg,
            "Wall_Formwork_m2": wall_formwork_m2,
            "Wall_Material_Cost": wall_material_cost,
            "Wall_Labor_Cost": wall_labor_cost,
            "Wall_Indirect_Cost": wall_indirect_cost,
            "Wall_Foundation_Connection_Cost": wall_foundation_connection_cost,
            "Wall_Constructability_Cost": wall_constructability_cost,
            "Wall_Site_Overhead_Cost": wall_site_overhead_cost,
            "Jacketing_Required": "YES" if jacketing_required else "NO",
            "Jacketing_Cost": jacket_cost_total,
            "Jacketing_Concrete_m3": jacket_concrete_m3,
            "Jacketing_Rebar_kg": jacket_rebar_kg,
            "Jacketing_Formwork_m2": jacket_formwork_m2,
            "Jacketing_Material_Cost": jacket_material_cost,
            "Jacketing_Labor_Cost": jacket_labor_cost,
            "Jacketing_Indirect_Cost": jacket_indirect_cost,
            "Jacketing_Site_Overhead_Cost": jacket_site_overhead_cost,
            "Jacketing_Num_Members": jacket_num_members,
            "Wall_Duration_Days": wall_duration_days,
            "Jacketing_Duration_Days": jacket_duration_days,
            "Total_Duration_Days": wall_duration_days + jacket_duration_days,
            "Total_Retrofit_Cost": total_retrofit_cost,
            "Final_Max_EC2_Utilization": final_max_eta,
            "Member_Failures_After_Walls": len(fail_after_ids),
            "Final_Failed_Members": final_fail_count,
            "Drift_OK": "YES" if drift_ok else "NO",
            "Drift_Utilization": drift_util,
            "Max_Drift_Ratio": max_drift_ratio,
            "Seismic_Action": "EC8 LFM" if seismic_settings else "none",
            "Seismic_T1_s": ev_final["T1"] if ev_final else np.nan,
            "Seismic_Fb_kN": ev_final["Fb_kN"] if ev_final else np.nan,
            "Seismic_DLS_Utilization": ev_final["dls_utilization"] if ev_final else np.nan,
            "Seismic_Theta_Max": ev_final["theta_max"] if ev_final else np.nan,
            "Final_Member_OK": "YES" if final_member_ok else "NO",
            "Complete_Feasible": "YES" if complete_feasible else "NO",
            "Penalty_Drift": drift_penalty,
            "Penalty_Member": member_penalty,
            "Penalty_Constructability": constructability_penalty,
            "Penalty_Jacketing_Limit": jacketing_limit_penalty,
            "Global_Objective_Score": global_objective,
            "Selected": "NO",
            "Jacketed_Members": ",".join(map(str, jacketed_member_ids)),
            "Jacketing_Convergence": str(cj_selected.get("Convergence_Reason", "not_required") if cj_selected is not None else "not_required"),
            "Alternative_Required_Members": alternative_required,
            "Cost_Assumptions": "; ".join(f"{k}={v}" for k, v in sorted(({**DEFAULT_UNIT_COSTS, **(unit_costs or {})}).items())),
            "_walls": trial_walls,
            "_df_worst_after_walls": df_worst_after,
            "_member_perf_after_walls": member_perf_after,
            "_member_fail_after_walls": member_fail_after,
            "_cj_opt_df": cj_df,
            "_df_worst_final": df_worst_final,
            "_member_fail_final": member_fail_final,
            "_element_sections": element_sections,
            "_member_report": member_report,
            "_jacket_cost": jacket_cost_obj,
            "_jacketing_iteration_worst_dfs": iteration_worst_dfs,
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    n_before_dedup = len(out)
    out = _deduplicate_global_alternatives(out)
    n_after_dedup = len(out)
    if n_before_dedup != n_after_dedup:
        out["Duplicate_Strategies_Removed"] = int(n_before_dedup - n_after_dedup)
    else:
        out["Duplicate_Strategies_Removed"] = 0
    out["Search_Unique_Complete_Strategies"] = int(len(out))

    feasible_mask = out["Complete_Feasible"].astype(str).str.upper().eq("YES")
    if feasible_mask.any():
        out = out.sort_values(
            ["Complete_Feasible", "Total_Retrofit_Cost", "Drift_Utilization", "Penalty_Constructability", "Final_Max_EC2_Utilization"],
            ascending=[False, True, True, True, True],
            na_position="last",
        )
    else:
        out = out.sort_values(
            ["Global_Objective_Score", "Final_Failed_Members", "Drift_Utilization", "Total_Retrofit_Cost"],
            ascending=[True, True, True, True],
            na_position="last",
        )
    out = out.reset_index(drop=True)
    out.loc[:, "Selected"] = "NO"
    out.at[0, "Selected"] = "YES"
    out["Global_Rank"] = np.arange(1, len(out) + 1)
    out["Alternative_Number"] = [f"Alternative {i}" for i in out["Global_Rank"]]

    def _explain_global_row(rr):
        feasible = str(rr.get("Complete_Feasible", "")).upper() == "YES"
        selected = str(rr.get("Selected", "")).upper() == "YES"
        if selected and feasible:
            return "Selected because it is the lowest-cost complete feasible retrofit strategy."
        if feasible:
            return "Feasible but not selected because a lower-cost feasible alternative exists."
        reasons = []
        if str(rr.get("Drift_OK", "")).upper() != "YES":
            reasons.append("drift limit not satisfied")
        if str(rr.get("Final_Member_OK", "")).upper() != "YES":
            reasons.append("final EC2 member safety not fully satisfied")
        if float(pd.to_numeric(rr.get("Penalty_Jacketing_Limit", 0.0), errors="coerce") or 0.0) > 0.0:
            reasons.append("jacketing or detailing limit reached")
        if not reasons:
            reasons.append("higher global penalty")
        return "Not selected: " + "; ".join(reasons) + "."

    out["Selection_Explanation"] = out.apply(_explain_global_row, axis=1)
    return out


# ============================================================
# OPTIMIZATION DATASET EXPORT
# ============================================================
def _dataset_safe_value(value: Any, default: Any = "") -> Any:
    """Return a CSV-friendly scalar value for optimization dataset export."""
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, (list, tuple, set, dict)):
        return str(value)
    return value


def build_global_optimization_dataset(
    global_opt_df: Optional[pd.DataFrame],
    *,
    structure_id: str = "current_structure",
    include_cost_assumptions: bool = True,
) -> pd.DataFrame:
    """Build a clean research dataset from all evaluated global alternatives.

    Each row represents one complete retrofit strategy after duplicate
    strategies have been removed:

        shear-wall layout + required concrete jacketing + final cost/performance

    The returned dataframe intentionally removes internal Python objects such
    as stored OpenSees result tables and section objects.  It is therefore safe
    to save as CSV/XLSX and to reuse later for sensitivity studies or future
    machine-learning datasets.
    """
    if global_opt_df is None or getattr(global_opt_df, "empty", True):
        return pd.DataFrame()

    df = global_opt_df.copy()
    # Remove internal columns that contain Python objects/dataframes.
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("_")], errors="ignore")

    # Keep a stable order that is useful for research post-processing.
    preferred = [
        "Global_Rank", "Alternative_Number", "Selected", "Complete_Feasible",
        "Candidate_ID", "Description", "Strategy_Key", "Wall_Strategy_Key", "Jacketing_Strategy_Key",
        "Wall_Layout_Mode", "Wall_Segments", "Wall_Thickness_mm", "Num_Wall_Panels", "Num_Wall_Lines", "Constructability_Class",
        "Max_Drift_Ratio", "Drift_Limit_Ratio", "Drift_Utilization", "Drift_OK",
        "Member_Failures_After_Walls", "Jacketing_Required", "Jacketed_Members",
        "Final_Max_EC2_Utilization", "Final_Failed_Members", "Final_Member_OK",
        "Concrete_m3", "Rebar_kg", "Formwork_m2",
        "Wall_Concrete_m3", "Wall_Rebar_kg", "Wall_Formwork_m2",
        "Wall_Material_Cost", "Wall_Labor_Cost", "Wall_Indirect_Cost",
        "Wall_Foundation_Connection_Cost", "Wall_Constructability_Cost", "Wall_Site_Overhead_Cost",
        "Jacketing_Concrete_m3", "Jacketing_Rebar_kg", "Jacketing_Formwork_m2",
        "Jacketing_Material_Cost", "Jacketing_Labor_Cost", "Jacketing_Indirect_Cost",
        "Jacketing_Site_Overhead_Cost", "Jacketing_Num_Members",
        "Wall_Cost", "Jacketing_Cost", "Total_Retrofit_Cost",
        "Wall_Duration_Days", "Jacketing_Duration_Days", "Total_Duration_Days",
        "Duration_Days", "Site_Overhead_Cost",
        "Penalty_Drift", "Penalty_Member", "Penalty_Constructability", "Penalty_Jacketing_Limit", "Global_Objective_Score",
        "Jacketing_Convergence", "Alternative_Required_Members", "Selection_Explanation",
    ]
    if include_cost_assumptions:
        preferred.append("Cost_Assumptions")

    ordered = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining].copy()

    df.insert(0, "Dataset_Record_ID", [f"ALT_{i:04d}" for i in range(1, len(df) + 1)])
    df.insert(1, "Structure_ID", str(structure_id or "current_structure"))

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: _dataset_safe_value(x, ""))
    return df


def export_global_optimization_dataset(
    global_opt_df: Optional[pd.DataFrame],
    filename: str = "global_retrofit_optimization_dataset_later_ML.csv",
    *,
    structure_id: str = "current_structure",
    include_cost_assumptions: bool = True,
) -> pd.DataFrame:
    """Export all evaluated global retrofit alternatives to a CSV dataset.

    Returns the dataframe that was exported.  If ``global_opt_df`` is empty, an
    empty dataframe is returned and no file is written.
    """
    dataset = build_global_optimization_dataset(
        global_opt_df,
        structure_id=structure_id,
        include_cost_assumptions=include_cost_assumptions,
    )
    if dataset is None or dataset.empty:
        return pd.DataFrame()
    dataset.to_csv(filename, index=False)
    return dataset

def plot_drift_before_after(
    df_compare: pd.DataFrame,
    drift_limit_ratio: Optional[float] = None,
    title: str = "Story Drift Before/After Shear Wall Retrofit",
):
    """
    Plot story drift comparison from a before/after dataframe.

    Accepts the column names generated by the retrofit workflow, including:
    - Story
    - Before_MaxDriftRatio / After_MaxDriftRatio
    - Before_Drift_Ratio / After_Drift_Ratio
    - Before_Drift_Percent / After_Drift_Percent
    """
    import matplotlib.pyplot as plt

    if df_compare is None or df_compare.empty:
        print("No drift comparison data to plot.")
        return None

    df = df_compare.copy()

    if "Story" not in df.columns:
        print("Cannot plot drift comparison: missing 'Story' column.")
        return None

    def _first_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    before_pct_col = _first_col(["Before_Drift_Percent", "Before_MaxDriftPercent"])
    after_pct_col = _first_col(["After_Drift_Percent", "After_MaxDriftPercent"])

    before_ratio_col = _first_col(["Before_MaxDriftRatio", "Before_Drift_Ratio", "BeforeDriftRatio"])
    after_ratio_col = _first_col(["After_MaxDriftRatio", "After_Drift_Ratio", "AfterDriftRatio"])

    if before_pct_col is not None and after_pct_col is not None:
        before_x = pd.to_numeric(df[before_pct_col], errors="coerce")
        after_x = pd.to_numeric(df[after_pct_col], errors="coerce")
        xlabel = "Story drift [%]"
        limit_x = 100.0 * drift_limit_ratio if drift_limit_ratio is not None else None
    elif before_ratio_col is not None and after_ratio_col is not None:
        before_x = 100.0 * pd.to_numeric(df[before_ratio_col], errors="coerce")
        after_x = 100.0 * pd.to_numeric(df[after_ratio_col], errors="coerce")
        xlabel = "Story drift [%]"
        limit_x = 100.0 * drift_limit_ratio if drift_limit_ratio is not None else None
    else:
        print("Cannot plot drift comparison: missing before/after drift columns.")
        return None

    stories = pd.to_numeric(df["Story"], errors="coerce")

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.plot(before_x, stories, marker="o", label="Before retrofit")
    ax.plot(after_x, stories, marker="s", label="After shear walls")

    if limit_x is not None and np.isfinite(limit_x) and limit_x > 0:
        ax.axvline(limit_x, linestyle="--", label="Drift limit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Story")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return ax


# ============================================================
# DRIFT COMPARISON TABLE
# ============================================================
def build_drift_comparison_table(
    before_drifts_df: Optional[pd.DataFrame],
    after_drifts_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build story-level before/after max drift comparison table."""
    if before_drifts_df is None or before_drifts_df.empty or after_drifts_df is None or after_drifts_df.empty:
        return pd.DataFrame()

    def _story_max(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if "Story" not in df.columns:
            return pd.DataFrame()
        ratio_col = None
        for c in ["MaxDriftRatio", "DriftRatio", "drift_ratio"]:
            if c in df.columns:
                ratio_col = c
                break
        mm_col = None
        for c in ["MaxDrift_mm", "Drift [mm]", "Max Drift [mm]", "MaxDrift_m"]:
            if c in df.columns:
                mm_col = c
                break
        out = df.copy()
        if ratio_col is None:
            out["__ratio__"] = np.nan
            ratio_col = "__ratio__"
        if mm_col is None:
            out["__mm__"] = np.nan
            mm_col = "__mm__"
        out[ratio_col] = pd.to_numeric(out[ratio_col], errors="coerce")
        out[mm_col] = pd.to_numeric(out[mm_col], errors="coerce")
        g = out.groupby("Story", as_index=False).agg({ratio_col: "max", mm_col: "max"})
        return g.rename(columns={ratio_col: f"{prefix}_MaxDriftRatio", mm_col: f"{prefix}_MaxDrift_mm"})

    b = _story_max(before_drifts_df, "Before")
    a = _story_max(after_drifts_df, "After")
    if b.empty or a.empty:
        return pd.DataFrame()
    out = b.merge(a, on="Story", how="outer")
    out["Before_Drift_Percent"] = 100.0 * pd.to_numeric(out["Before_MaxDriftRatio"], errors="coerce")
    out["After_Drift_Percent"] = 100.0 * pd.to_numeric(out["After_MaxDriftRatio"], errors="coerce")
    out["Drift_Reduction_Percent"] = 100.0 * (
        pd.to_numeric(out["Before_MaxDriftRatio"], errors="coerce") - pd.to_numeric(out["After_MaxDriftRatio"], errors="coerce")
    ) / pd.to_numeric(out["Before_MaxDriftRatio"], errors="coerce").replace(0.0, np.nan)
    return out.sort_values("Story").reset_index(drop=True)


# ============================================================
# RETROFIT PLOTTING HELPERS
# ============================================================
def plot_shear_wall_retrofit_layout(
    nodes,
    elements,
    geom,
    *,
    base_walls=None,
    selected_candidate: Optional[ShearWallCandidate] = None,
    title: str = "Optimized Retrofit Layout - Selected Shear Walls",
):
    """Plot the frame and show existing/selected shear wall panels."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Patch
    from framekit.geometry import node_id_at
    from framekit.plot_style import (COLORS, draw_frame, draw_supports, draw_nodes, label_elements,
                                     finish_frame_axes, subtitle, draw_dimensions)

    walls_existing = list(base_walls or [])
    walls_new = list(selected_candidate.walls if selected_candidate is not None else [])

    fig, ax = plt.subplots(figsize=(11, 7.5))

    def _draw_wall(w, fc, ec, hatch, tag):
        try:
            ids = [node_id_at(geom, w.floor_bot, w.bay_left - 1), node_id_at(geom, w.floor_bot, w.bay_right - 1),
                   node_id_at(geom, w.floor_top, w.bay_right - 1), node_id_at(geom, w.floor_top, w.bay_left - 1)]
            pts = [nodes[n] for n in ids]
        except Exception:
            return
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=2.0, alpha=0.55, hatch=hatch, zorder=1))
        tw = getattr(w, "tw_mm", None)
        txt = f"{tag}\nbay {w.bay_left} · storey {w.floor_top}" + (f"\nt = {float(tw):.0f} mm" if tw else "")
        ax.text(sum(p[0] for p in pts) / 4.0, sum(p[1] for p in pts) / 4.0, txt, fontsize=8, ha="center",
                va="center", color=ec, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=ec, lw=0.8, alpha=0.9))

    for w in walls_existing:
        _draw_wall(w, "#D6D6D6", "#6B6B6B", "////", "EXISTING")
    for w in walls_new:
        _draw_wall(w, COLORS["wall_fill"], COLORS["highlight"], "xx", "NEW WALL")

    draw_frame(ax, nodes, elements)
    draw_supports(ax, nodes)
    draw_nodes(ax, nodes)
    label_elements(ax, nodes, elements, fontsize=7)
    draw_dimensions(ax, nodes, geom)
    finish_frame_axes(ax, nodes, pad=0.14, pad_bottom=0.16)

    handles = []
    if walls_existing:
        handles.append(Patch(fc="#D6D6D6", ec="#6B6B6B", hatch="////", label="Existing wall"))
    if walls_new:
        handles.append(Patch(fc=COLORS["wall_fill"], ec=COLORS["highlight"], hatch="xx", label="Selected new wall"))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title)
    if selected_candidate is not None:
        subtitle(ax, str(selected_candidate.description))
    plt.show()
    return ax


def plot_jacketed_members(
    nodes,
    elements,
    jacketed_member_ids: List[int],
    *,
    title: str = "Concrete Jacketing Layout - Strengthened Members",
    geom=None,
    walls=None,
):
    """Plot frame with jacketed members highlighted."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from framekit.plot_style import (COLORS, draw_supports, draw_nodes, draw_walls,
                                     finish_frame_axes, subtitle)

    jacketed = {int(e) for e in jacketed_member_ids}
    fig, ax = plt.subplots(figsize=(11, 7.5))
    draw_walls(ax, nodes, geom, walls, label=False)
    for eid, (ni, nj, etype) in elements.items():
        x1, y1 = nodes[ni]
        x2, y2 = nodes[nj]
        is_j = int(eid) in jacketed
        if is_j:
            ax.plot([x1, x2], [y1, y2], lw=11, color=COLORS["highlight"], alpha=0.25, solid_capstyle="butt", zorder=2)
            ax.plot([x1, x2], [y1, y2], lw=4.0, color=COLORS["highlight"], solid_capstyle="butt", zorder=3)
        else:
            ax.plot([x1, x2], [y1, y2], lw=2.0, color="#9AA5B1", zorder=3)
        ax.text((x1 + x2) / 2.0, (y1 + y2) / 2.0, str(eid), ha="center", va="center", fontsize=8,
                color=COLORS["highlight"] if is_j else "#555555", fontweight="bold" if is_j else "normal", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS["highlight"] if is_j else "#BBBBBB", lw=0.8))
    draw_nodes(ax, nodes)
    draw_supports(ax, nodes)
    finish_frame_axes(ax, nodes, pad=0.1)
    ax.legend(handles=[Line2D([], [], color=COLORS["highlight"], lw=5, label="Jacketed member"),
                       Line2D([], [], color="#9AA5B1", lw=2, label="Unchanged member")],
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title(title)
    n_col = sum(1 for e in jacketed if e in elements and elements[e][2] == "column")
    subtitle(ax, f"{len(jacketed)} jacketed members  ·  {n_col} columns, {len(jacketed) - n_col} beams")
    plt.show()
    return ax


def _df_worst_to_checks_by_combo(df_worst: Optional[pd.DataFrame], combo_name: str = "RETROFIT_FINAL") -> Dict[str, Dict[int, Dict[str, Any]]]:
    checks = {}
    if df_worst is None or df_worst.empty:
        return {combo_name: checks}
    for _, row in df_worst.iterrows():
        try:
            eid = int(row.get("Element", row.get("ElementID")))
        except Exception:
            continue
        checks[eid] = {
            "eta": float(row.get("eta", row.get("Static_EC2_Utilization", 0.0)) or 0.0),
            "etype": row.get("Type", row.get("etype", "")),
            "combo": row.get("Combination", combo_name),
            "eta_NM": float(row.get("eta_NM", 0.0) or 0.0),
            "etaV": float(row.get("etaV", 0.0) or 0.0),
            "NEd_kN": float(row.get("NEd_kN", 0.0) or 0.0),
            "VEd_kN": float(row.get("VEd_kN", 0.0) or 0.0),
            "MEd_kNm": float(row.get("MEd_kNm", 0.0) or 0.0),
        }
    return {combo_name: checks}


def plot_final_retrofit_utilization(
    nodes,
    elements,
    df_worst_final: Optional[pd.DataFrame],
    *,
    title: str = "Final Member Utilization After Retrofit",
    geom=None,
    walls=None,
    highlight_ids=None,
):
    """Final utilisation map from a worst-utilisation dataframe (jacketed members highlighted)."""
    from framekit.plot_style import plot_utilization_frame
    checks = _df_worst_to_checks_by_combo(df_worst_final)["RETROFIT_FINAL"]
    eta_map = {eid: d["eta"] for eid, d in checks.items()}
    return plot_utilization_frame(nodes, elements, eta_map, title=title, geom=geom, walls=walls,
                                  highlight_ids=highlight_ids, info="after retrofit")
