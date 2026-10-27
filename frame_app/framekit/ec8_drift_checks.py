from __future__ import annotations

from typing import Dict, Optional
import pandas as pd


EC8_DRIFT_LIMITS = {
    "brittle_attached": 0.0050,
    "ductile_attached": 0.0075,
    "detached_or_non_interfering": 0.0100,
}


def ask_ec8_drift_check_params() -> Dict:
    selected = []

    while True:
        print("\n--- EC8 Drift Check Input ---")
        print("Non-structural element class:")
        print("1) Brittle non-structural elements attached to structure")
        print("2) Ductile non-structural elements attached to structure")
        print("3) Non-structural elements detached / not interfering with structure")

        from framekit.input_helpers import ask_choice_def, ask_float_def
        opt = ask_choice_def("Select option (1/2/3) [1]: ", ["1", "2", "3"], "1")

        if opt == "1":
            nse_class = "brittle_attached"
        elif opt == "2":
            nse_class = "ductile_attached"
        elif opt == "3":
            nse_class = "detached_or_non_interfering"
        else:
            print("Invalid option. Using 1) brittle attached.")
            nse_class = "brittle_attached"

        if nse_class not in selected:
            selected.append(nse_class)

        again = ask_choice_def("Check another EC8 drift option too? (y/n) [n]: ", ["y", "n"], "n")
        if again != "y":
            break

    nu = ask_float_def("Reduction factor nu for damage limitation spectrum [default 0.50]: ", 0.50, 0.1, 1.0)

    return {
        "nse_classes": selected,
        "limit_ratios": {cls: EC8_DRIFT_LIMITS[cls] for cls in selected},
        "nu": nu,
    }


def _story_to_index(story_name: str) -> int:
    s = str(story_name).strip().lower().replace("story", "").strip()
    try:
        return int(s)
    except Exception:
        return 10**9


def compute_story_heights(geom) -> Dict[str, float]:
    return {
        f"Story {i}": float(geom.floor_height)
        for i in range(1, geom.num_floors + 1)
    }


def ec8_damage_limitation_drifts_from_rsa(
    rsa_story_drifts_m: Dict[str, float],
    nu: float = 0.50
) -> Dict[str, float]:
    return {
        story: float(nu) * float(drift_m)
        for story, drift_m in rsa_story_drifts_m.items()
    }


def check_ec8_drift_limits(
    drift_m: Dict[str, float],
    story_heights_m: Dict[str, float],
    limit_ratio: float,
    case_name: str = "EC8_DLS",
    nse_class: Optional[str] = None,
) -> pd.DataFrame:
    rows = []

    for story in sorted(drift_m.keys(), key=_story_to_index):
        d = float(drift_m[story])
        h = float(story_heights_m[story])

        ratio = d / h if h > 0.0 else 0.0
        ok = ratio <= limit_ratio + 1e-15

        rows.append({
            "Case": case_name,
            "Story": story,
            "Height [m]": h,
            "Drift [m]": d,
            "Drift [mm]": d * 1000.0,
            "Drift ratio": ratio,
            "Limit ratio": limit_ratio,
            "Utilization": (ratio / limit_ratio) if limit_ratio > 0 else 0.0,
            "Pass": "OK" if ok else "FAIL",
            "NSE class": nse_class if nse_class is not None else "",
        })

    return pd.DataFrame(rows)


def summarize_ec8_drift_check(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {
            "ok": True,
            "num_failed": 0,
            "worst_story": None,
            "worst_ratio": 0.0,
            "worst_utilization": 0.0,
        }

    worst_idx = df["Utilization"].astype(float).idxmax()
    worst_row = df.loc[worst_idx]

    num_failed = int((df["Pass"] == "FAIL").sum())

    return {
        "ok": num_failed == 0,
        "num_failed": num_failed,
        "worst_story": worst_row["Story"],
        "worst_ratio": float(worst_row["Drift ratio"]),
        "worst_utilization": float(worst_row["Utilization"]),
    }


def export_ec8_drift_check(df: pd.DataFrame, filename: str = "ec8_drift_check.xlsx"):
    try:
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="EC8_Drift_Check", index=False)
        print(f"✔ EC8 drift check exported: {filename}")
    except ModuleNotFoundError:
        csv_name = filename.rsplit(".", 1)[0] + ".csv"
        df.to_csv(csv_name, index=False)
        print(f"✔ EC8 drift check exported: {csv_name}")