from __future__ import annotations

from typing import Dict, Optional, List
import numpy as np
import pandas as pd


# ============================================================
# HELPERS
# ============================================================
def _first_existing_column(df: Optional[pd.DataFrame], candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _story_order_key(val):
    s = str(val).lower().replace("story", "").strip()
    try:
        return int(s)
    except Exception:
        return 10**9


def _element_force_maxabs_from_dict(force_dict: Optional[Dict[int, np.ndarray]]) -> pd.DataFrame:
    if not force_dict:
        return pd.DataFrame(columns=["Element", "MaxAbsForce"])

    rows = []
    for ele, vec in force_dict.items():
        arr = np.asarray(vec, dtype=float)
        if arr.size == 0:
            maxabs = np.nan
        else:
            maxabs = float(np.max(np.abs(arr)))
        rows.append({"Element": int(ele), "MaxAbsForce": maxabs})

    return pd.DataFrame(rows)


def _th_envelope_element_forces_from_batch(batch_th: Optional[Dict]) -> pd.DataFrame:
    if not batch_th or "results_by_record" not in batch_th:
        return pd.DataFrame(columns=["Element", "TH_MaxAbsForce"])

    env = {}

    for _, res in batch_th["results_by_record"].items():
        ele_forces = res.get("peak_element_forces", {})
        for ele, vec in ele_forces.items():
            arr = np.asarray(vec, dtype=float)
            if arr.size == 0:
                val = np.nan
            else:
                val = float(np.max(np.abs(arr)))

            if ele not in env:
                env[ele] = val
            else:
                if np.isnan(env[ele]):
                    env[ele] = val
                elif np.isnan(val):
                    pass
                else:
                    env[ele] = max(env[ele], val)

    rows = [{"Element": int(ele), "TH_MaxAbsForce": val} for ele, val in env.items()]
    return pd.DataFrame(rows)


def _extract_ec2_member_table(df_worst: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_worst is None or df_worst.empty:
        return pd.DataFrame(columns=["Element", "Static_EC2_Utilization", "Static_EC2_Pass"])

    ele_col = _first_existing_column(df_worst, ["Element", "Member", "Element ID", "Member ID", "Elem"])
    util_col = _first_existing_column(df_worst, ["eta", "Eta", "Utilization", "utilization", "Worst eta", "worst_eta"])

    if ele_col is None:
        out = df_worst.copy()
        out.insert(0, "Element", range(1, len(out) + 1))
        ele_col = "Element"
    else:
        out = df_worst.copy()

    if util_col is None:
        out["Static_EC2_Utilization"] = np.nan
    else:
        out["Static_EC2_Utilization"] = pd.to_numeric(out[util_col], errors="coerce")

    out["Element"] = pd.to_numeric(out[ele_col], errors="coerce")
    out = out.dropna(subset=["Element"]).copy()
    out["Element"] = out["Element"].astype(int)
    out["Static_EC2_Pass"] = np.where(out["Static_EC2_Utilization"] <= 1.0, "OK", "FAIL")

    return out[["Element", "Static_EC2_Utilization", "Static_EC2_Pass"]].drop_duplicates(subset=["Element"])


# ============================================================
# STORY TABLES
# ============================================================
def build_story_performance_table(
    static_drifts_df: Optional[pd.DataFrame] = None,
    rsa_single_drifts_df: Optional[pd.DataFrame] = None,
    rsa_dir_drifts_df: Optional[pd.DataFrame] = None,
    th_envelope_drifts_df: Optional[pd.DataFrame] = None,
    ec8_single_df: Optional[pd.DataFrame] = None,
    ec8_dir_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    stories = set()

    for df in [
        static_drifts_df,
        rsa_single_drifts_df,
        rsa_dir_drifts_df,
        th_envelope_drifts_df,
        ec8_single_df,
        ec8_dir_df,
    ]:
        if df is not None and not df.empty and "Story" in df.columns:
            stories.update(df["Story"].astype(str).tolist())

    if not stories:
        return pd.DataFrame()

    out = pd.DataFrame({"Story": sorted(stories, key=_story_order_key)})

    if static_drifts_df is not None and not static_drifts_df.empty:
        if "Story" in static_drifts_df.columns and "Drift [mm]" in static_drifts_df.columns:
            df_stat = static_drifts_df.groupby("Story", as_index=False)["Drift [mm]"].max()
            df_stat = df_stat.rename(columns={"Drift [mm]": "Static_Max_Drift_mm"})
            out = out.merge(df_stat, on="Story", how="left")

    if rsa_single_drifts_df is not None and not rsa_single_drifts_df.empty:
        if "Story" in rsa_single_drifts_df.columns and "Drift [mm]" in rsa_single_drifts_df.columns:
            df_rsa = rsa_single_drifts_df.groupby("Story", as_index=False)["Drift [mm]"].max()
            df_rsa = df_rsa.rename(columns={"Drift [mm]": "RSA_Single_Drift_mm"})
            out = out.merge(df_rsa, on="Story", how="left")

    if rsa_dir_drifts_df is not None and not rsa_dir_drifts_df.empty:
        if "Story" in rsa_dir_drifts_df.columns and "Drift [mm]" in rsa_dir_drifts_df.columns:
            df_rsa_dir = rsa_dir_drifts_df.groupby("Story", as_index=False)["Drift [mm]"].max()
            df_rsa_dir = df_rsa_dir.rename(columns={"Drift [mm]": "RSA_Directional_Drift_mm"})
            out = out.merge(df_rsa_dir, on="Story", how="left")

    if th_envelope_drifts_df is not None and not th_envelope_drifts_df.empty:
        if "Story" in th_envelope_drifts_df.columns:
            drift_col = _first_existing_column(th_envelope_drifts_df, ["Peak Drift [mm]", "Drift [mm]"])
            if drift_col is not None:
                df_th = th_envelope_drifts_df.groupby("Story", as_index=False)[drift_col].max()
                df_th = df_th.rename(columns={drift_col: "TH_Envelope_Drift_mm"})
                out = out.merge(df_th, on="Story", how="left")

    if ec8_single_df is not None and not ec8_single_df.empty:
        cols = ["Story"]
        if "Drift ratio" in ec8_single_df.columns:
            cols.append("Drift ratio")
        if "Utilization" in ec8_single_df.columns:
            cols.append("Utilization")
        if "Pass" in ec8_single_df.columns:
            cols.append("Pass")

        df1 = ec8_single_df[cols].copy().sort_values("Story", key=lambda s: s.map(_story_order_key))
        if "Utilization" in df1.columns:
            df1["__util__"] = pd.to_numeric(df1["Utilization"], errors="coerce")
            df1 = df1.sort_values(["Story", "__util__"], ascending=[True, False]).drop_duplicates(subset=["Story"])
            df1 = df1.drop(columns=["__util__"])
        else:
            df1 = df1.drop_duplicates(subset=["Story"])

        rename_map = {}
        if "Drift ratio" in df1.columns:
            rename_map["Drift ratio"] = "EC8_Single_Drift_Ratio"
        if "Utilization" in df1.columns:
            rename_map["Utilization"] = "EC8_Single_Utilization"
        if "Pass" in df1.columns:
            rename_map["Pass"] = "EC8_Single_Pass"
        df1 = df1.rename(columns=rename_map)
        out = out.merge(df1, on="Story", how="left")

    if ec8_dir_df is not None and not ec8_dir_df.empty:
        cols = ["Story"]
        if "Drift ratio" in ec8_dir_df.columns:
            cols.append("Drift ratio")
        if "Utilization" in ec8_dir_df.columns:
            cols.append("Utilization")
        if "Pass" in ec8_dir_df.columns:
            cols.append("Pass")

        df2 = ec8_dir_df[cols].copy()
        if "Utilization" in df2.columns:
            df2["__util__"] = pd.to_numeric(df2["Utilization"], errors="coerce")
            df2 = df2.sort_values(["Story", "__util__"], ascending=[True, False]).drop_duplicates(subset=["Story"])
            df2 = df2.drop(columns=["__util__"])
        else:
            df2 = df2.drop_duplicates(subset=["Story"])

        rename_map = {}
        if "Drift ratio" in df2.columns:
            rename_map["Drift ratio"] = "EC8_Directional_Drift_Ratio"
        if "Utilization" in df2.columns:
            rename_map["Utilization"] = "EC8_Directional_Utilization"
        if "Pass" in df2.columns:
            rename_map["Pass"] = "EC8_Directional_Pass"
        df2 = df2.rename(columns=rename_map)
        out = out.merge(df2, on="Story", how="left")

    if "TH_Envelope_Drift_mm" in out.columns and "RSA_Directional_Drift_mm" in out.columns:
        out["TH_over_RSA_Directional"] = out["TH_Envelope_Drift_mm"] / out["RSA_Directional_Drift_mm"].replace(0, np.nan)

    if "TH_Envelope_Drift_mm" in out.columns and "RSA_Single_Drift_mm" in out.columns:
        out["TH_over_RSA_Single"] = out["TH_Envelope_Drift_mm"] / out["RSA_Single_Drift_mm"].replace(0, np.nan)

    def _gov_story(row):
        vals = {}
        for c in [
            "Static_Max_Drift_mm",
            "RSA_Single_Drift_mm",
            "RSA_Directional_Drift_mm",
            "TH_Envelope_Drift_mm",
        ]:
            if c in row.index and pd.notna(row[c]):
                vals[c] = float(row[c])
        if not vals:
            return ""
        return max(vals, key=vals.get)

    out["Governing_Drift_Source"] = out.apply(_gov_story, axis=1)
    return out.sort_values("Story", key=lambda s: s.map(_story_order_key)).reset_index(drop=True)


def build_story_failures_table(story_perf_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if story_perf_df is None or story_perf_df.empty:
        return pd.DataFrame()

    df = story_perf_df.copy()

    if "EC8_Directional_Pass" in df.columns:
        df["Directional_Drift_Failure"] = np.where(df["EC8_Directional_Pass"] == "FAIL", "YES", "NO")
    else:
        df["Directional_Drift_Failure"] = "NO"

    if "EC8_Single_Pass" in df.columns:
        df["Single_Drift_Failure"] = np.where(df["EC8_Single_Pass"] == "FAIL", "YES", "NO")
    else:
        df["Single_Drift_Failure"] = "NO"

    cols = ["Story"]
    for c in [
        "Directional_Drift_Failure",
        "Single_Drift_Failure",
        "Governing_Drift_Source",
        "TH_Envelope_Drift_mm",
        "RSA_Directional_Drift_mm",
        "RSA_Single_Drift_mm",
        "EC8_Directional_Utilization",
        "EC8_Single_Utilization",
    ]:
        if c in df.columns:
            cols.append(c)

    out = df[cols].copy()

    mask = pd.Series(False, index=out.index)
    if "Directional_Drift_Failure" in out.columns:
        mask = mask | (out["Directional_Drift_Failure"] == "YES")
    if "Single_Drift_Failure" in out.columns:
        mask = mask | (out["Single_Drift_Failure"] == "YES")

    out = out.loc[mask].copy()
    return out.reset_index(drop=True)


# ============================================================
# MEMBER TABLES
# ============================================================
def build_member_performance_table(
    ec2_worst_df: Optional[pd.DataFrame] = None,
    rsa_single_results: Optional[Dict] = None,
    rsa_dir_results: Optional[Dict] = None,
    batch_th: Optional[Dict] = None,
) -> pd.DataFrame:
    ec2_df = _extract_ec2_member_table(ec2_worst_df)

    rsa_single_df = pd.DataFrame(columns=["Element", "RSA_Single_MaxAbsForce"])
    if rsa_single_results is not None:
        tmp = _element_force_maxabs_from_dict(rsa_single_results.get("element_forces", {}))
        if not tmp.empty:
            rsa_single_df = tmp.rename(columns={"MaxAbsForce": "RSA_Single_MaxAbsForce"})

    rsa_dir_df = pd.DataFrame(columns=["Element", "RSA_Directional_MaxAbsForce"])
    if rsa_dir_results is not None:
        env = rsa_dir_results.get("envelope", {})
        tmp = _element_force_maxabs_from_dict(env.get("element_forces", {}))
        if not tmp.empty:
            rsa_dir_df = tmp.rename(columns={"MaxAbsForce": "RSA_Directional_MaxAbsForce"})

    th_df = _th_envelope_element_forces_from_batch(batch_th)

    all_elements = set()
    for df in [ec2_df, rsa_single_df, rsa_dir_df, th_df]:
        if not df.empty and "Element" in df.columns:
            all_elements.update(df["Element"].astype(int).tolist())

    if not all_elements:
        return pd.DataFrame()

    out = pd.DataFrame({"Element": sorted(all_elements)})
    for df in [ec2_df, rsa_single_df, rsa_dir_df, th_df]:
        if not df.empty:
            out = out.merge(df, on="Element", how="left")

    def _gov_dynamic(row):
        vals = {}
        for c in [
            "RSA_Single_MaxAbsForce",
            "RSA_Directional_MaxAbsForce",
            "TH_MaxAbsForce",
        ]:
            if c in row.index and pd.notna(row[c]):
                vals[c] = float(row[c])
        if not vals:
            return ""
        return max(vals, key=vals.get)

    out["Dynamic_Governing_Source"] = out.apply(_gov_dynamic, axis=1)

    def _overall_status(row):
        static_fail = False
        if "Static_EC2_Pass" in row.index and pd.notna(row["Static_EC2_Pass"]):
            static_fail = str(row["Static_EC2_Pass"]).upper() == "FAIL"
        return "FAIL" if static_fail else "OK"

    out["Overall_Static_Status"] = out.apply(_overall_status, axis=1)
    return out.sort_values("Element").reset_index(drop=True)


def build_member_failures_table(member_perf_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if member_perf_df is None or member_perf_df.empty:
        return pd.DataFrame()

    df = member_perf_df.copy()
    df["Static_Failure"] = np.where(df.get("Overall_Static_Status", "OK") == "FAIL", "YES", "NO")

    cols = ["Element"]
    for c in [
        "Static_Failure",
        "Static_EC2_Utilization",
        "Static_EC2_Pass",
        "RSA_Single_MaxAbsForce",
        "RSA_Directional_MaxAbsForce",
        "TH_MaxAbsForce",
        "Dynamic_Governing_Source",
    ]:
        if c in df.columns:
            cols.append(c)

    out = df[cols].copy()
    if "Static_Failure" in out.columns:
        out = out[out["Static_Failure"] == "YES"].copy()

    return out.reset_index(drop=True)


# ============================================================
# GLOBAL SUMMARY
# ============================================================
def build_global_performance_summary(
    story_perf_df: Optional[pd.DataFrame],
    member_perf_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    if story_perf_df is not None and not story_perf_df.empty:
        n_story_fail_dir = int((story_perf_df.get("EC8_Directional_Pass", pd.Series(dtype=str)) == "FAIL").sum()) \
            if "EC8_Directional_Pass" in story_perf_df.columns else 0
        n_story_fail_single = int((story_perf_df.get("EC8_Single_Pass", pd.Series(dtype=str)) == "FAIL").sum()) \
            if "EC8_Single_Pass" in story_perf_df.columns else 0

        rows.append({"Metric": "Stories failing EC8 directional drift", "Value": n_story_fail_dir})
        rows.append({"Metric": "Stories failing EC8 single-direction drift", "Value": n_story_fail_single})

    if member_perf_df is not None and not member_perf_df.empty:
        n_mem_fail = int((member_perf_df.get("Overall_Static_Status", pd.Series(dtype=str)) == "FAIL").sum()) \
            if "Overall_Static_Status" in member_perf_df.columns else 0
        rows.append({"Metric": "Members failing static EC2", "Value": n_mem_fail})

        if "Dynamic_Governing_Source" in member_perf_df.columns:
            for src in ["RSA_Single_MaxAbsForce", "RSA_Directional_MaxAbsForce", "TH_MaxAbsForce"]:
                rows.append({
                    "Metric": f"Members governed by {src}",
                    "Value": int((member_perf_df["Dynamic_Governing_Source"] == src).sum())
                })

    return pd.DataFrame(rows)


# ============================================================
# XLSX STYLING
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
    num_fmt_6 = workbook.add_format({"border": 1, "num_format": "0.000000"})
    int_fmt = workbook.add_format({"border": 1, "num_format": "0"})

    for j, col in enumerate(df.columns):
        worksheet.write(0, j, col, header_fmt)

        series = df[col]
        max_len = max(len(str(col)), *(len(str(v)) for v in series.head(200).tolist())) if len(series) else len(str(col))
        width = min(max(max_len + 2, 12), 28)

        if pd.api.types.is_integer_dtype(series.dropna()):
            worksheet.set_column(j, j, width, int_fmt)
        elif pd.api.types.is_numeric_dtype(series.dropna()):
            if "ratio" in col.lower() or "util" in col.lower():
                worksheet.set_column(j, j, width, num_fmt_6)
            else:
                worksheet.set_column(j, j, width, num_fmt_3)
        else:
            worksheet.set_column(j, j, width, cell_fmt)

    worksheet.freeze_panes(1, 0)


def _apply_pass_fail_colors(writer, df: pd.DataFrame, sheet_name: str):
    worksheet = writer.sheets[sheet_name]
    workbook = writer.book

    green_fmt = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
    red_fmt = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})

    if df.empty:
        return

    nrows = len(df)

    for j, col in enumerate(df.columns):
        c = col.lower()
        if ("pass" in c) or ("status" in c) or ("failure" in c):
            xl = _xlsx_col_letter(j)
            rng = f"{xl}2:{xl}{nrows+1}"

            worksheet.conditional_format(rng, {
                "type": "text",
                "criteria": "containing",
                "value": "OK",
                "format": green_fmt
            })
            worksheet.conditional_format(rng, {
                "type": "text",
                "criteria": "containing",
                "value": "PASS",
                "format": green_fmt
            })
            worksheet.conditional_format(rng, {
                "type": "text",
                "criteria": "containing",
                "value": "NO",
                "format": green_fmt
            })

            worksheet.conditional_format(rng, {
                "type": "text",
                "criteria": "containing",
                "value": "FAIL",
                "format": red_fmt
            })
            worksheet.conditional_format(rng, {
                "type": "text",
                "criteria": "containing",
                "value": "YES",
                "format": red_fmt
            })


def _apply_ratio_heatmaps(writer, df: pd.DataFrame, sheet_name: str):
    worksheet = writer.sheets[sheet_name]

    if df.empty:
        return

    nrows = len(df)

    ratio_like_cols = []
    for j, col in enumerate(df.columns):
        c = col.lower()
        if ("util" in c) or ("ratio" in c):
            ratio_like_cols.append(j)

    for j in ratio_like_cols:
        xl = _xlsx_col_letter(j)
        rng = f"{xl}2:{xl}{nrows+1}"
        worksheet.conditional_format(rng, {
            "type": "3_color_scale",
            "min_type": "num",
            "min_value": 0.0,
            "mid_type": "num",
            "mid_value": 0.75,
            "max_type": "num",
            "max_value": 1.0,
            "min_color": "#C6EFCE",
            "mid_color": "#FFEB84",
            "max_color": "#F8696B",
        })
        worksheet.conditional_format(rng, {
            "type": "cell",
            "criteria": ">",
            "value": 1.0,
            "format": writer.book.add_format({"bg_color": "#C00000", "font_color": "#FFFFFF"})
        })


def _apply_global_summary_colors(writer, df: pd.DataFrame, sheet_name: str):
    worksheet = writer.sheets[sheet_name]
    workbook = writer.book

    if df.empty or "Value" not in df.columns:
        return

    nrows = len(df)
    j = df.columns.get_loc("Value")
    xl = _xlsx_col_letter(j)
    rng = f"{xl}2:{xl}{nrows+1}"

    worksheet.conditional_format(rng, {
        "type": "3_color_scale",
        "min_type": "num",
        "min_value": 0,
        "mid_type": "percentile",
        "mid_value": 50,
        "max_type": "max",
        "min_color": "#C6EFCE",
        "mid_color": "#FFEB84",
        "max_color": "#F8696B",
    })

    red_fmt = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
    worksheet.conditional_format(rng, {
        "type": "cell",
        "criteria": ">",
        "value": 0,
        "format": red_fmt
    })


# ============================================================
# EXPORT
# ============================================================
def export_performance_summary_results(
    story_perf_df: Optional[pd.DataFrame],
    story_fail_df: Optional[pd.DataFrame],
    member_perf_df: Optional[pd.DataFrame],
    member_fail_df: Optional[pd.DataFrame],
    global_summary_df: Optional[pd.DataFrame],
    filename: str = "performance_summary.xlsx",
):
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        if global_summary_df is not None and not global_summary_df.empty:
            global_summary_df.to_excel(writer, sheet_name="Global_Summary", index=False)
            _apply_basic_sheet_format(writer, global_summary_df, "Global_Summary")
            _apply_global_summary_colors(writer, global_summary_df, "Global_Summary")

        if story_perf_df is not None and not story_perf_df.empty:
            story_perf_df.to_excel(writer, sheet_name="Story_Performance", index=False)
            _apply_basic_sheet_format(writer, story_perf_df, "Story_Performance")
            _apply_pass_fail_colors(writer, story_perf_df, "Story_Performance")
            _apply_ratio_heatmaps(writer, story_perf_df, "Story_Performance")

        if story_fail_df is not None and not story_fail_df.empty:
            story_fail_df.to_excel(writer, sheet_name="Story_Failures", index=False)
            _apply_basic_sheet_format(writer, story_fail_df, "Story_Failures")
            _apply_pass_fail_colors(writer, story_fail_df, "Story_Failures")
            _apply_ratio_heatmaps(writer, story_fail_df, "Story_Failures")

        if member_perf_df is not None and not member_perf_df.empty:
            member_perf_df.to_excel(writer, sheet_name="Member_Performance", index=False)
            _apply_basic_sheet_format(writer, member_perf_df, "Member_Performance")
            _apply_pass_fail_colors(writer, member_perf_df, "Member_Performance")
            _apply_ratio_heatmaps(writer, member_perf_df, "Member_Performance")

        if member_fail_df is not None and not member_fail_df.empty:
            member_fail_df.to_excel(writer, sheet_name="Member_Failures", index=False)
            _apply_basic_sheet_format(writer, member_fail_df, "Member_Failures")
            _apply_pass_fail_colors(writer, member_fail_df, "Member_Failures")
            _apply_ratio_heatmaps(writer, member_fail_df, "Member_Failures")

    print(f"✔ Performance summary exported: {filename}")