from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


# ============================================================
# HELPERS
# ============================================================
def _safe_float(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _cost_class_to_score(cost_class: str) -> int:
    cost_class = str(cost_class).upper()
    if cost_class == "LOW":
        return 1
    if cost_class == "MEDIUM":
        return 2
    if cost_class == "HIGH":
        return 3
    return 0


def _priority_label(score: float) -> str:
    if score >= 85:
        return "URGENT"
    if score >= 65:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


# ============================================================
# STORY RETROFIT LOGIC
# ============================================================
def recommend_story_retrofits(story_perf_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if story_perf_df is None or story_perf_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in story_perf_df.iterrows():
        story = row.get("Story", "")
        util_dir = _safe_float(row.get("EC8_Directional_Utilization", np.nan))
        util_single = _safe_float(row.get("EC8_Single_Utilization", np.nan))
        gov = str(row.get("Governing_Drift_Source", ""))

        util = np.nanmax([util_dir, util_single]) if not (np.isnan(util_dir) and np.isnan(util_single)) else np.nan
        fail = bool((not np.isnan(util)) and util > 1.0)
        near = bool((not np.isnan(util)) and 0.80 < util <= 1.0)

        if fail:
            if gov == "TH_Envelope_Drift_mm":
                action = "Increase lateral stiffness; add shear wall or bracing in governing direction"
                strategy = "GLOBAL_STIFFNESS"
                cost = "HIGH"
            elif gov == "RSA_Directional_Drift_mm":
                action = "Add wall / core / stiffer column line to reduce directional drift"
                strategy = "DIRECTIONAL_STIFFNESS"
                cost = "HIGH"
            else:
                action = "Increase frame stiffness; enlarge columns and/or beams at this story"
                strategy = "FRAME_STIFFNESS"
                cost = "MEDIUM"

            rationale = "Story drift exceeds EC8 limit"
            base_score = 90.0

        elif near:
            if gov == "TH_Envelope_Drift_mm":
                action = "Moderately increase stiffness; consider local wall or column enlargement"
                strategy = "PREVENTIVE_STIFFNESS"
                cost = "MEDIUM"
            else:
                action = "Monitor drift; consider moderate stiffness increase"
                strategy = "PREVENTIVE_CHECK"
                cost = "LOW"

            rationale = "Story drift is near EC8 limit"
            base_score = 55.0

        else:
            action = "No retrofit needed at story level"
            strategy = "NONE"
            cost = "LOW"
            rationale = "Story drift is acceptable"
            base_score = 10.0

        # FIX 5 — Priority score formula with documented rationale:
        #   base_score   : 90 (fail) / 55 (near-fail) / 10 (OK) — mirrors
        #                  FEMA P-58 consequence-function tiers.
        #   +20 per unit of excess utilisation (util - 1.0): amplifies
        #                  urgency proportionally to how far over-stressed
        #                  the story is; weight chosen so util=1.5 → +10,
        #                  util=2.0 → +20 (capped at 100).
        #   +10 * util   : adds a continuous gradient within the 0-1 range
        #                  to differentiate near-limit cases from safe ones.
        # These weights are heuristic; adjust for your demand model.
        if not np.isnan(util):
            priority_score = min(100.0, base_score + 20.0 * max(util - 1.0, 0.0) + 10.0 * min(util, 1.0))
        else:
            priority_score = base_score

        rows.append({
            "Story": story,
            "Drift_Utilization": util,
            "Governing_Drift_Source": gov,
            "Story_Retrofit_Strategy": strategy,
            "Story_Retrofit_Action": action,
            "Rationale": rationale,
            "Cost_Class": cost,
            "Cost_Score": _cost_class_to_score(cost),
            "Priority_Score": priority_score,
            "Priority_Label": _priority_label(priority_score),
            "Needs_Retrofit": "YES" if (fail or near) else "NO",
        })

    return pd.DataFrame(rows)


# ============================================================
# MEMBER RETROFIT LOGIC
# ============================================================
def recommend_member_retrofits(member_perf_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if member_perf_df is None or member_perf_df.empty:
        return pd.DataFrame()

    rows = []

    for _, row in member_perf_df.iterrows():
        eid = row.get("Element", row.get("ElementID", ""))
        util = _safe_float(row.get("Static_EC2_Utilization", np.nan))
        static_pass = str(row.get("Static_EC2_Pass", row.get("Overall_Static_Status", ""))).upper()
        gov_dyn = str(row.get("Dynamic_Governing_Source", ""))
        etype = str(row.get("Type", row.get("etype", ""))).lower()

        stirrup_spacing = _safe_float(row.get("stirrup_spacing_mm", np.nan))
        eta_v = _safe_float(row.get("etaV", np.nan))
        eta_nm = _safe_float(row.get("eta_NM", np.nan))

        fail = (static_pass == "FAIL") or ((not np.isnan(util)) and util > 1.0)
        near = ((not np.isnan(util)) and 0.80 < util <= 1.0)

        shear_controls = (not np.isnan(eta_v) and not np.isnan(eta_nm) and eta_v >= eta_nm)
        flex_nm_controls = (not np.isnan(eta_v) and not np.isnan(eta_nm) and eta_nm > eta_v)

        if fail:
            if shear_controls:
                action = "Reduce stirrup spacing and/or increase stirrup diameter"
                strategy = "SHEAR_RETROFIT"
                cost = "LOW" if etype == "beam" else "MEDIUM"
                rationale = "Shear utilization governs"
            elif flex_nm_controls:
                if etype == "column":
                    action = "Enlarge column section or add jacketing / FRP / RC encasement"
                    strategy = "COLUMN_CAPACITY"
                    cost = "HIGH"
                    rationale = "Axial-flexural interaction governs"
                else:
                    action = "Increase beam section and/or longitudinal reinforcement"
                    strategy = "BEAM_CAPACITY"
                    cost = "MEDIUM"
                    rationale = "Flexural capacity governs"
            else:
                action = "Increase section capacity and review detailing"
                strategy = "GENERAL_MEMBER_UPGRADE"
                cost = "MEDIUM"
                rationale = "Member fails static verification"

            base_score = 88.0

        elif near:
            if shear_controls:
                action = "Tighten stirrup spacing in critical regions"
                strategy = "PREVENTIVE_SHEAR"
                cost = "LOW"
                rationale = "Shear demand is near capacity"
            else:
                action = "Moderate section/reinforcement increase; monitor governing demand"
                strategy = "PREVENTIVE_MEMBER"
                cost = "LOW" if etype == "beam" else "MEDIUM"
                rationale = "Member is near capacity"

            base_score = 55.0

        else:
            action = "No member retrofit needed"
            strategy = "NONE"
            cost = "LOW"
            rationale = "Member verification is acceptable"
            base_score = 10.0

        if gov_dyn == "TH_MaxAbsForce":
            base_score += 8.0
            rationale += " | Time-history governs dynamic demand"
        elif gov_dyn == "RSA_Directional_MaxAbsForce":
            base_score += 4.0
            rationale += " | Directional RSA governs dynamic demand"

        if not np.isnan(stirrup_spacing) and stirrup_spacing > 150.0 and (shear_controls or fail):
            rationale += " | Large stirrup spacing may be non-conservative"
            base_score += 4.0

        priority_score = min(100.0, base_score + 20.0 * max((_safe_float(util, 0.0) - 1.0), 0.0))

        rows.append({
            "Element": eid,
            "Type": etype,
            "Static_Utilization": util,
            "Static_Pass": static_pass,
            "etaV": eta_v,
            "eta_NM": eta_nm,
            "Dynamic_Governing_Source": gov_dyn,
            "Member_Retrofit_Strategy": strategy,
            "Member_Retrofit_Action": action,
            "Rationale": rationale,
            "Cost_Class": cost,
            "Cost_Score": _cost_class_to_score(cost),
            "Priority_Score": priority_score,
            "Priority_Label": _priority_label(priority_score),
            "Needs_Retrofit": "YES" if (fail or near) else "NO",
        })

    return pd.DataFrame(rows)


# ============================================================
# GLOBAL PRIORITY TABLE
# ============================================================
def build_retrofit_priority_table(
    story_retro_df: Optional[pd.DataFrame],
    member_retro_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    rows = []

    if story_retro_df is not None and not story_retro_df.empty:
        for _, r in story_retro_df.iterrows():
            if str(r.get("Needs_Retrofit", "NO")) != "YES":
                continue

            rows.append({
                "Target_Type": "Story",
                "Target_ID": r.get("Story", ""),
                "Strategy": r.get("Story_Retrofit_Strategy", ""),
                "Action": r.get("Story_Retrofit_Action", ""),
                "Priority_Score": _safe_float(r.get("Priority_Score", np.nan)),
                "Priority_Label": r.get("Priority_Label", ""),
                "Cost_Class": r.get("Cost_Class", ""),
                "Rationale": r.get("Rationale", ""),
            })

    if member_retro_df is not None and not member_retro_df.empty:
        for _, r in member_retro_df.iterrows():
            if str(r.get("Needs_Retrofit", "NO")) != "YES":
                continue

            rows.append({
                "Target_Type": "Member",
                "Target_ID": r.get("Element", ""),
                "Strategy": r.get("Member_Retrofit_Strategy", ""),
                "Action": r.get("Member_Retrofit_Action", ""),
                "Priority_Score": _safe_float(r.get("Priority_Score", np.nan)),
                "Priority_Label": r.get("Priority_Label", ""),
                "Cost_Class": r.get("Cost_Class", ""),
                "Rationale": r.get("Rationale", ""),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # FIX 4: cost_score is now used as a secondary sort key so that among
    # items with the same Priority_Score, cheaper interventions are listed
    # first (LOW cost = score 1 comes before HIGH cost = score 3).
    return df.sort_values(
        ["Priority_Score", "Cost_Score", "Target_Type"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


# ============================================================
# XLSX EXPORT WITH COLORS
# ============================================================
def _xlsx_col_letter(col_idx_zero_based: int) -> str:
    n = col_idx_zero_based + 1
    s = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        s = chr(65 + rem) + s
    return s


def _apply_basic_sheet_format(writer, df: pd.DataFrame, sheet_name: str):
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    header_fmt = workbook.add_format({
        "bold": True,
        "text_wrap": True,
        "valign": "vcenter",
        "align": "center",
        "border": 1,
        "bg_color": "#D9EAF7"
    })

    cell_fmt = workbook.add_format({"border": 1})
    num_fmt_3 = workbook.add_format({"border": 1, "num_format": "0.000"})
    num_fmt_2 = workbook.add_format({"border": 1, "num_format": "0.00"})

    for j, col in enumerate(df.columns):
        worksheet.write(0, j, col, header_fmt)

        series = df[col]
        max_len = max(len(str(col)), *(len(str(v)) for v in series.head(200).tolist())) if len(series) else len(str(col))
        width = min(max(max_len + 2, 12), 40)

        if pd.api.types.is_numeric_dtype(series.dropna()):
            if "score" in col.lower():
                worksheet.set_column(j, j, width, num_fmt_2)
            else:
                worksheet.set_column(j, j, width, num_fmt_3)
        else:
            worksheet.set_column(j, j, width, cell_fmt)

    worksheet.freeze_panes(1, 0)


def _apply_yes_no_colors(writer, df: pd.DataFrame, sheet_name: str):
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    green_fmt = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
    red_fmt = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})

    if df.empty:
        return

    nrows = len(df)

    for j, col in enumerate(df.columns):
        c = col.lower()
        if "needs_retrofit" in c:
            xl = _xlsx_col_letter(j)
            rng = f"{xl}2:{xl}{nrows+1}"

            worksheet.conditional_format(rng, {
                "type": "text",
                "criteria": "containing",
                "value": "NO",
                "format": green_fmt
            })
            worksheet.conditional_format(rng, {
                "type": "text",
                "criteria": "containing",
                "value": "YES",
                "format": red_fmt
            })


def _apply_priority_heatmap(writer, df: pd.DataFrame, sheet_name: str):
    worksheet = writer.sheets[sheet_name]
    if df.empty:
        return

    nrows = len(df)

    for target_col in ["Priority_Score", "Static_Utilization", "Drift_Utilization", "etaV", "eta_NM"]:
        if target_col in df.columns:
            j = df.columns.get_loc(target_col)
            xl = _xlsx_col_letter(j)
            rng = f"{xl}2:{xl}{nrows+1}"

            maxv = 100.0 if "score" in target_col.lower() else 1.0
            worksheet.conditional_format(rng, {
                "type": "3_color_scale",
                "min_type": "num",
                "min_value": 0.0,
                "mid_type": "num",
                "mid_value": 0.75 * maxv,
                "max_type": "num",
                "max_value": maxv,
                "min_color": "#C6EFCE",
                "mid_color": "#FFEB84",
                "max_color": "#F8696B",
            })


def export_retrofit_recommendations(
    story_retro_df: Optional[pd.DataFrame],
    member_retro_df: Optional[pd.DataFrame],
    priority_df: Optional[pd.DataFrame],
    filename: str = "retrofit_recommendations.xlsx"
):
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        if story_retro_df is not None and not story_retro_df.empty:
            story_retro_df.to_excel(writer, sheet_name="Story_Retrofit", index=False)
            _apply_basic_sheet_format(writer, story_retro_df, "Story_Retrofit")
            _apply_yes_no_colors(writer, story_retro_df, "Story_Retrofit")
            _apply_priority_heatmap(writer, story_retro_df, "Story_Retrofit")

        if member_retro_df is not None and not member_retro_df.empty:
            member_retro_df.to_excel(writer, sheet_name="Member_Retrofit", index=False)
            _apply_basic_sheet_format(writer, member_retro_df, "Member_Retrofit")
            _apply_yes_no_colors(writer, member_retro_df, "Member_Retrofit")
            _apply_priority_heatmap(writer, member_retro_df, "Member_Retrofit")

        if priority_df is not None and not priority_df.empty:
            priority_df.to_excel(writer, sheet_name="Priority_List", index=False)
            _apply_basic_sheet_format(writer, priority_df, "Priority_List")
            _apply_priority_heatmap(writer, priority_df, "Priority_List")

    print(f"✔ Retrofit recommendations exported: {filename}")