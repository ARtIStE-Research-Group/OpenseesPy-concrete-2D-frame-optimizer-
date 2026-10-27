# PATCH MARKER: COLUMN_ALL_AROUND_LONGITUDINAL_BARS_PRESENT
from dataclasses import dataclass, asdict
import math


@dataclass
class RCSection:
    b_mm: float
    h_mm: float
    cover_mm: float

    # Longitudinal reinforcement on the top and bottom faces.
    # For beams these are the usual top/bottom layers.
    # For columns these are the bars on the upper/lower faces, including corner bars.
    n_top: int
    phi_top_mm: float
    n_bot: int
    phi_bot_mm: float

    # Transverse reinforcement (stirrups/ties)
    stirrup_phi_mm: float
    stirrup_spacing_mm: float
    stirrup_legs: int

    # Optional seismic end-zone detailing
    stirrup_spacing_critical_mm: float | None = None

    # Section role controls plotting/detailing logic.
    # "beam" keeps top/bottom layers only.
    # "column" distributes longitudinal bars around all four faces.
    section_role: str = "beam"

    # Column side-face longitudinal bars.
    # These are intermediate bars on left/right faces, excluding corner bars.
    # Corner bars are already counted by n_top and n_bot.
    n_left: int = 0
    n_right: int = 0
    phi_side_mm: float | None = None

    # RC jacket (0 = not jacketed). b_mm/h_mm are then the outer dimensions,
    # cover_mm locates the existing bars, jn_*/jphi_mm are the jacket bars.
    jacket_t_mm: float = 0.0
    jacket_cover_mm: float = 30.0
    jacket_stirrup_phi_mm: float = 8.0
    jn_top: int = 0
    jn_bot: int = 0
    jn_left: int = 0
    jn_right: int = 0
    jphi_mm: float = 0.0

    def to_dict(self):
        return asdict(self)

    @property
    def total_longitudinal_bars(self) -> int:
        return int(self.n_top) + int(self.n_bot) + int(self.n_left) + int(self.n_right)

    @property
    def phi_side_effective_mm(self) -> float:
        if self.phi_side_mm is not None:
            return float(self.phi_side_mm)
        return 0.5 * (float(self.phi_top_mm) + float(self.phi_bot_mm))

    @property
    def As_longitudinal_mm2(self) -> float:
        return (
            int(self.n_top) * math.pi * float(self.phi_top_mm) ** 2 / 4.0
            + int(self.n_bot) * math.pi * float(self.phi_bot_mm) ** 2 / 4.0
            + (int(self.n_left) + int(self.n_right)) * math.pi * float(self.phi_side_effective_mm) ** 2 / 4.0
            + (int(self.jn_top) + int(self.jn_bot) + int(self.jn_left) + int(self.jn_right))
            * math.pi * float(self.jphi_mm or 0.0) ** 2 / 4.0
        )

    @property
    def rho_longitudinal(self) -> float:
        Ac = max(float(self.b_mm) * float(self.h_mm), 1e-9)
        return self.As_longitudinal_mm2 / Ac



# ============================================================
# PRESET LIBRARIES
# Practical starter presets for normal RC buildings.
# They are not a substitute for final code-compliant detailing.
# ============================================================
BEAM_PRESETS = [
    {
        "label": "B1 light beam 250x400 - 3T/3B phi14",
        "b_mm": 250.0,
        "h_mm": 400.0,
        "cover_mm": 30.0,
        "n_top": 3,
        "phi_top_mm": 14.0,
        "n_bot": 3,
        "phi_bot_mm": 14.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 150.0,
        "stirrup_legs": 2,
        "stirrup_spacing_critical_mm": 100.0,
    },
    {
        "label": "B2 common beam 300x450 - 3T/3B phi16",
        "b_mm": 300.0,
        "h_mm": 450.0,
        "cover_mm": 30.0,
        "n_top": 3,
        "phi_top_mm": 16.0,
        "n_bot": 3,
        "phi_bot_mm": 16.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 150.0,
        "stirrup_legs": 2,
        "stirrup_spacing_critical_mm": 100.0,
    },
    {
        "label": "B3 common beam 300x500 - 3T/3B phi16",
        "b_mm": 300.0,
        "h_mm": 500.0,
        "cover_mm": 30.0,
        "n_top": 3,
        "phi_top_mm": 16.0,
        "n_bot": 3,
        "phi_bot_mm": 16.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 150.0,
        "stirrup_legs": 2,
        "stirrup_spacing_critical_mm": 100.0,
    },
    {
        "label": "B4 stronger beam 300x550 - 4T/4B phi16",
        "b_mm": 300.0,
        "h_mm": 550.0,
        "cover_mm": 30.0,
        "n_top": 4,
        "phi_top_mm": 16.0,
        "n_bot": 4,
        "phi_bot_mm": 16.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 125.0,
        "stirrup_legs": 2,
        "stirrup_spacing_critical_mm": 100.0,
    },
    {
        "label": "B5 deep beam 300x600 - 4T/4B phi18",
        "b_mm": 300.0,
        "h_mm": 600.0,
        "cover_mm": 30.0,
        "n_top": 4,
        "phi_top_mm": 18.0,
        "n_bot": 4,
        "phi_bot_mm": 18.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 125.0,
        "stirrup_legs": 2,
        "stirrup_spacing_critical_mm": 100.0,
    },
    {
        "label": "B6 wide beam 350x500 - 4T/4B phi16",
        "b_mm": 350.0,
        "h_mm": 500.0,
        "cover_mm": 30.0,
        "n_top": 4,
        "phi_top_mm": 16.0,
        "n_bot": 4,
        "phi_bot_mm": 16.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 150.0,
        "stirrup_legs": 2,
        "stirrup_spacing_critical_mm": 100.0,
    },
    {
        "label": "B7 strong beam 350x600 - 4T/4B phi20",
        "b_mm": 350.0,
        "h_mm": 600.0,
        "cover_mm": 30.0,
        "n_top": 4,
        "phi_top_mm": 20.0,
        "n_bot": 4,
        "phi_bot_mm": 20.0,
        "stirrup_phi_mm": 10.0,
        "stirrup_spacing_mm": 125.0,
        "stirrup_legs": 2,
        "stirrup_spacing_critical_mm": 100.0,
    },
    {
        "label": "B8 heavy beam 400x600 - 5T/5B phi18",
        "b_mm": 400.0,
        "h_mm": 600.0,
        "cover_mm": 35.0,
        "n_top": 5,
        "phi_top_mm": 18.0,
        "n_bot": 5,
        "phi_bot_mm": 18.0,
        "stirrup_phi_mm": 10.0,
        "stirrup_spacing_mm": 125.0,
        "stirrup_legs": 2,
        "stirrup_spacing_critical_mm": 100.0,
    },
    {
        "label": "B9 transfer-like beam 400x700 - 5T/5B phi20",
        "b_mm": 400.0,
        "h_mm": 700.0,
        "cover_mm": 35.0,
        "n_top": 5,
        "phi_top_mm": 20.0,
        "n_bot": 5,
        "phi_bot_mm": 20.0,
        "stirrup_phi_mm": 10.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 2,
        "stirrup_spacing_critical_mm": 75.0,
    },
    {
        "label": "B10 heavy transfer beam 450x750 - 6T/6B phi20",
        "b_mm": 450.0,
        "h_mm": 750.0,
        "cover_mm": 35.0,
        "n_top": 6,
        "phi_top_mm": 20.0,
        "n_bot": 6,
        "phi_bot_mm": 20.0,
        "stirrup_phi_mm": 10.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 75.0,
    },
]


COLUMN_PRESETS = [
    {
        "label": "C1 small column 300x300 - 3T/3B phi16",
        "b_mm": 300.0,
        "h_mm": 300.0,
        "cover_mm": 35.0,
        "n_top": 3,
        "phi_top_mm": 16.0,
        "n_bot": 3,
        "phi_bot_mm": 16.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 80.0,
    },
    {
        "label": "C2 common column 350x350 - 3T/3B phi16",
        "b_mm": 350.0,
        "h_mm": 350.0,
        "cover_mm": 35.0,
        "n_top": 3,
        "phi_top_mm": 16.0,
        "n_bot": 3,
        "phi_bot_mm": 16.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 80.0,
    },
    {
        "label": "C3 common column 400x400 - 4T/4B phi16",
        "b_mm": 400.0,
        "h_mm": 400.0,
        "cover_mm": 35.0,
        "n_top": 4,
        "phi_top_mm": 16.0,
        "n_bot": 4,
        "phi_bot_mm": 16.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 80.0,
    },
    {
        "label": "C4 rectangular column 300x500 - 4T/4B phi18",
        "b_mm": 300.0,
        "h_mm": 500.0,
        "cover_mm": 35.0,
        "n_top": 4,
        "phi_top_mm": 18.0,
        "n_bot": 4,
        "phi_bot_mm": 18.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 80.0,
    },
    {
        "label": "C5 medium column 450x450 - 4T/4B phi18",
        "b_mm": 450.0,
        "h_mm": 450.0,
        "cover_mm": 35.0,
        "n_top": 4,
        "phi_top_mm": 18.0,
        "n_bot": 4,
        "phi_bot_mm": 18.0,
        "stirrup_phi_mm": 8.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 80.0,
    },
    {
        "label": "C6 medium column 500x500 - 4T/4B phi20",
        "b_mm": 500.0,
        "h_mm": 500.0,
        "cover_mm": 40.0,
        "n_top": 4,
        "phi_top_mm": 20.0,
        "n_bot": 4,
        "phi_bot_mm": 20.0,
        "stirrup_phi_mm": 10.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 80.0,
    },
    {
        "label": "C7 rectangular column 400x600 - 5T/5B phi20",
        "b_mm": 400.0,
        "h_mm": 600.0,
        "cover_mm": 40.0,
        "n_top": 5,
        "phi_top_mm": 20.0,
        "n_bot": 5,
        "phi_bot_mm": 20.0,
        "stirrup_phi_mm": 10.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 80.0,
    },
    {
        "label": "C8 large column 600x600 - 5T/5B phi20",
        "b_mm": 600.0,
        "h_mm": 600.0,
        "cover_mm": 40.0,
        "n_top": 5,
        "phi_top_mm": 20.0,
        "n_bot": 5,
        "phi_bot_mm": 20.0,
        "stirrup_phi_mm": 10.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 75.0,
    },
    {
        "label": "C9 large rectangular column 500x700 - 6T/6B phi22",
        "b_mm": 500.0,
        "h_mm": 700.0,
        "cover_mm": 40.0,
        "n_top": 6,
        "phi_top_mm": 22.0,
        "n_bot": 6,
        "phi_bot_mm": 22.0,
        "stirrup_phi_mm": 10.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 75.0,
    },
    {
        "label": "C10 heavy column 700x700 - 6T/6B phi25",
        "b_mm": 700.0,
        "h_mm": 700.0,
        "cover_mm": 45.0,
        "n_top": 6,
        "phi_top_mm": 25.0,
        "n_bot": 6,
        "phi_bot_mm": 25.0,
        "stirrup_phi_mm": 10.0,
        "stirrup_spacing_mm": 100.0,
        "stirrup_legs": 4,
        "stirrup_spacing_critical_mm": 75.0,
    },
]


# Add backward-compatible reinforcement metadata to preset dictionaries.
# Existing code can still use n_top/n_bot exactly as before, while column plots
# and reports can now show all-around longitudinal reinforcement.
def _complete_section_presets() -> None:
    for p in BEAM_PRESETS:
        p.setdefault("section_role", "beam")
        p.setdefault("n_left", 0)
        p.setdefault("n_right", 0)
        p.setdefault("phi_side_mm", p.get("phi_top_mm", p.get("phi_bot_mm", 16.0)))

    for p in COLUMN_PRESETS:
        p.setdefault("section_role", "column")
        h = float(p.get("h_mm", 400.0))
        # Intermediate side bars excluding corner bars. The default grows with
        # section height so rectangular/tall columns are not drawn like beams.
        if "n_left" not in p:
            if h <= 350.0:
                n_side = 1
            elif h <= 500.0:
                n_side = 2
            elif h <= 650.0:
                n_side = 3
            else:
                n_side = 4
            p["n_left"] = n_side
            p["n_right"] = n_side
        p.setdefault("phi_side_mm", max(float(p.get("phi_top_mm", 16.0)), float(p.get("phi_bot_mm", 16.0))))


_complete_section_presets()


# ============================================================
# INPUT HELPERS
# ============================================================
def _ask_float(prompt: str, default: float) -> float:
    from framekit.input_helpers import ask_float_def
    return ask_float_def(f"{prompt} [{default}]: ", default)


def _ask_int(prompt: str, default: int) -> int:
    from framekit.input_helpers import ask_int_def
    return ask_int_def(f"{prompt} [{default}]: ", default)


def _ask_yes_no(prompt: str, default: str = "n") -> bool:
    ans = input(f"{prompt} [{default}]: ").strip().lower() or default.lower()
    return ans.startswith("y")


def _print_default_stirrup_note():
    print("\nTypical default stirrup values for normal RC buildings:")
    print("  Beams   : phi8-10 @ 100-150 mm, usually 2 legs")
    print("  Columns : phi8-10 @ 75-100 mm, usually 4 legs for rectangular sections")
    print("These are practical starter assumptions, not final code-verified detailing.")


def _print_preset_table(title: str, presets: list[dict]):
    print(f"\n--- {title} preset library ---")
    is_column_table = str(title).lower().startswith("column")
    for i, p in enumerate(presets, start=1):
        if is_column_table or str(p.get("section_role", "")).lower() == "column":
            total_bars = int(p.get("n_top", 0)) + int(p.get("n_bot", 0)) + int(p.get("n_left", 0)) + int(p.get("n_right", 0))
            rebar_txt = (
                f"bars all around: top {p.get('n_top', 0)}phi{p.get('phi_top_mm', 0):.0f}, "
                f"bottom {p.get('n_bot', 0)}phi{p.get('phi_bot_mm', 0):.0f}, "
                f"side L/R {p.get('n_left', 0)}/{p.get('n_right', 0)}phi{float(p.get('phi_side_mm', p.get('phi_top_mm', 0))):.0f}, "
                f"total={total_bars}"
            )
        else:
            rebar_txt = (
                f"top {p.get('n_top', 0)}phi{p.get('phi_top_mm', 0):.0f}, "
                f"bottom {p.get('n_bot', 0)}phi{p.get('phi_bot_mm', 0):.0f}"
            )
        print(
            f"{i:2d}) {p['label']} | {rebar_txt} | "
            f"stirrups phi{p['stirrup_phi_mm']:.0f}@{p['stirrup_spacing_mm']:.0f}, "
            f"legs={p['stirrup_legs']}"
        )


def _choose_preset(title: str, presets: list[dict], default_index: int) -> dict:
    _print_preset_table(title, presets)
    idx = _ask_int(f"Choose {title.lower()} preset number", default_index)
    if idx < 1 or idx > len(presets):
        print(f"Invalid choice. Using preset {default_index}.")
        idx = default_index
    p = dict(presets[idx - 1])
    p.pop("label", None)
    return p


def _ask_rc_section(name: str, defaults: dict) -> RCSection:
    print(f"\n--- {name} RC Section Input ---")

    is_column = name.strip().lower().startswith("column") or str(defaults.get("section_role", "")).lower() == "column"
    section_role = "column" if is_column else "beam"

    b_mm = _ask_float("Width b [mm]", defaults["b_mm"])
    h_mm = _ask_float("Height h [mm]", defaults["h_mm"])
    cover_mm = _ask_float("Concrete cover [mm]", defaults["cover_mm"])

    print("\nLongitudinal reinforcement:")
    if is_column:
        print("Column bars are distributed around the perimeter.")
        print("Top/bottom face bars include the corner bars; left/right side bars are intermediate bars only.")
        n_top = _ask_int("Top-face bars including corners", defaults["n_top"])
        phi_top_mm = _ask_float("Top-face bar diameter [mm]", defaults["phi_top_mm"])
        n_bot = _ask_int("Bottom-face bars including corners", defaults["n_bot"])
        phi_bot_mm = _ask_float("Bottom-face bar diameter [mm]", defaults["phi_bot_mm"])
        n_left = _ask_int("Left-side intermediate bars excluding corners", defaults.get("n_left", 0))
        n_right = _ask_int("Right-side intermediate bars excluding corners", defaults.get("n_right", n_left))
        phi_side_mm = _ask_float("Side-face bar diameter [mm]", defaults.get("phi_side_mm", max(phi_top_mm, phi_bot_mm)))
    else:
        n_top = _ask_int("Top bars number", defaults["n_top"])
        phi_top_mm = _ask_float("Top bar diameter [mm]", defaults["phi_top_mm"])
        n_bot = _ask_int("Bottom bars number", defaults["n_bot"])
        phi_bot_mm = _ask_float("Bottom bar diameter [mm]", defaults["phi_bot_mm"])
        n_left = int(defaults.get("n_left", 0))
        n_right = int(defaults.get("n_right", 0))
        phi_side_mm = defaults.get("phi_side_mm", max(phi_top_mm, phi_bot_mm))

    print("\nTransverse reinforcement (stirrups/ties):")
    stirrup_phi_mm = _ask_float("Stirrup/tie diameter [mm]", defaults["stirrup_phi_mm"])
    stirrup_spacing_mm = _ask_float("Stirrup/tie spacing s [mm]", defaults["stirrup_spacing_mm"])
    stirrup_legs = _ask_int("Number of stirrup/tie legs", defaults["stirrup_legs"])

    use_critical = input("Use smaller stirrup spacing in critical end zones? (y/n) [n]: ").strip().lower() or "n"
    if use_critical == "y":
        stirrup_spacing_critical_mm = _ask_float(
            "Critical-zone stirrup spacing [mm]",
            defaults.get("stirrup_spacing_critical_mm", stirrup_spacing_mm)
        )
    else:
        stirrup_spacing_critical_mm = stirrup_spacing_mm

    return RCSection(
        b_mm=b_mm,
        h_mm=h_mm,
        cover_mm=cover_mm,
        n_top=n_top,
        phi_top_mm=phi_top_mm,
        n_bot=n_bot,
        phi_bot_mm=phi_bot_mm,
        stirrup_phi_mm=stirrup_phi_mm,
        stirrup_spacing_mm=stirrup_spacing_mm,
        stirrup_legs=stirrup_legs,
        stirrup_spacing_critical_mm=stirrup_spacing_critical_mm,
        section_role=section_role,
        n_left=n_left,
        n_right=n_right,
        phi_side_mm=phi_side_mm,
    )


# ============================================================
# BASIC DETAILING WARNINGS
# ============================================================
# BASIC DETAILING WARNINGS
# ============================================================
def _bar_area_mm2(phi_mm: float) -> float:
    return math.pi * float(phi_mm) ** 2 / 4.0


def _is_column_section(sec: RCSection, title: str | None = None) -> bool:
    role = str(getattr(sec, "section_role", "")).strip().lower()
    if role == "column":
        return True
    if title and "column" in title.lower():
        return True
    return int(getattr(sec, "n_left", 0) or 0) > 0 or int(getattr(sec, "n_right", 0) or 0) > 0


def _phi_side(sec: RCSection) -> float:
    val = getattr(sec, "phi_side_mm", None)
    if val is not None:
        return float(val)
    return 0.5 * (float(sec.phi_top_mm) + float(sec.phi_bot_mm))


def _longitudinal_area_mm2(sec: RCSection) -> float:
    return (
        int(sec.n_top) * _bar_area_mm2(sec.phi_top_mm)
        + int(sec.n_bot) * _bar_area_mm2(sec.phi_bot_mm)
        + (int(getattr(sec, "n_left", 0) or 0) + int(getattr(sec, "n_right", 0) or 0)) * _bar_area_mm2(_phi_side(sec))
    )


def _bar_clear_spacing_along_face(face_length_mm: float, cover_mm: float, stirrup_phi_mm: float, n_bars: int, phi_mm: float) -> float | None:
    if n_bars <= 1:
        return None
    centerline_length = face_length_mm - 2.0 * (cover_mm + stirrup_phi_mm + 0.5 * phi_mm)
    return centerline_length / max(n_bars - 1, 1) - phi_mm


def _print_section_warnings(name: str, sec: RCSection):
    warnings = []
    is_column = name.lower().startswith("column") or _is_column_section(sec, name)

    if is_column:
        total_bars = int(sec.n_top) + int(sec.n_bot) + int(getattr(sec, "n_left", 0) or 0) + int(getattr(sec, "n_right", 0) or 0)
        rho = _longitudinal_area_mm2(sec) / max(sec.b_mm * sec.h_mm, 1e-9)
        if total_bars < 6:
            warnings.append("Column has fewer than 6 longitudinal bars around the perimeter.")
        if int(getattr(sec, "n_left", 0) or 0) == 0 and int(getattr(sec, "n_right", 0) or 0) == 0:
            warnings.append("Column has no intermediate side-face bars; the plot will look beam-like.")
        if sec.stirrup_legs < 4:
            warnings.append("Column tie legs lower than typical rectangular-column default of 4.")
        if rho < 0.002:
            warnings.append(f"Column longitudinal steel ratio is about {100*rho:.2f}%, below the usual EC2-style minimum range of 0.20%.")
        if rho > 0.04:
            warnings.append(f"Column longitudinal steel ratio is about {100*rho:.2f}%, above the usual EC2-style maximum range of 4.0%.")

        face_checks = [
            ("top face", sec.b_mm, sec.n_top, sec.phi_top_mm),
            ("bottom face", sec.b_mm, sec.n_bot, sec.phi_bot_mm),
            ("left side", sec.h_mm, int(getattr(sec, "n_left", 0) or 0) + 2, _phi_side(sec)),
            ("right side", sec.h_mm, int(getattr(sec, "n_right", 0) or 0) + 2, _phi_side(sec)),
        ]
        for face, L, n, phi in face_checks:
            sp = _bar_clear_spacing_along_face(L, sec.cover_mm, sec.stirrup_phi_mm, int(n), float(phi))
            if sp is not None and sp > 200.0:
                warnings.append(f"Column {face} clear bar spacing is about {sp:.1f} mm, greater than 200 mm.")
            if sp is not None and sp < max(20.0, float(phi)):
                warnings.append(f"Column {face} clear bar spacing is about {sp:.1f} mm, may be too small for concrete placing.")
    else:
        if sec.n_top < 3 or sec.n_bot < 3:
            warnings.append("Beam has fewer than 3 top or 3 bottom bars for this normal-building preset logic.")
        for row, n, phi in [("top", sec.n_top, sec.phi_top_mm), ("bottom", sec.n_bot, sec.phi_bot_mm)]:
            sp = _bar_clear_spacing_along_face(sec.b_mm, sec.cover_mm, sec.stirrup_phi_mm, int(n), float(phi))
            if sp is not None and sp > 200.0:
                warnings.append(f"{row.capitalize()} clear bar spacing is about {sp:.1f} mm, greater than 200 mm.")
            if sp is not None and sp < max(20.0, float(phi)):
                warnings.append(f"{row.capitalize()} clear bar spacing is about {sp:.1f} mm, may be too small for placing concrete.")

    if warnings:
        print(f"\nDetailing warnings for {name}:")
        for w in warnings:
            print(f"  - {w}")


# ============================================================
# PLOTTING
# ============================================================
# PLOTTING
# ============================================================
def _bar_positions_along_horizontal_face(sec: RCSection, n: int, phi: float, y: float):
    x_left = sec.cover_mm + sec.stirrup_phi_mm + 0.5 * phi
    x_right = sec.b_mm - sec.cover_mm - sec.stirrup_phi_mm - 0.5 * phi
    if n <= 1:
        xs = [0.5 * sec.b_mm]
    else:
        xs = [x_left + i * (x_right - x_left) / (n - 1) for i in range(n)]
    return [(x, y, phi) for x in xs]


def _bar_positions_along_vertical_face(sec: RCSection, n_intermediate: int, phi: float, x: float):
    # Side bars are intermediate bars only, excluding top/bottom corner bars.
    if n_intermediate <= 0:
        return []
    y_bot = sec.cover_mm + sec.stirrup_phi_mm + 0.5 * phi
    y_top = sec.h_mm - sec.cover_mm - sec.stirrup_phi_mm - 0.5 * phi
    step = (y_top - y_bot) / (n_intermediate + 1)
    return [(x, y_bot + (i + 1) * step, phi) for i in range(n_intermediate)]


def _all_longitudinal_bar_positions(sec: RCSection, *, title: str | None = None):
    is_column = _is_column_section(sec, title)
    y_top = sec.h_mm - sec.cover_mm - sec.stirrup_phi_mm - 0.5 * sec.phi_top_mm
    y_bot = sec.cover_mm + sec.stirrup_phi_mm + 0.5 * sec.phi_bot_mm
    bars = []
    bars.extend(_bar_positions_along_horizontal_face(sec, int(sec.n_top), float(sec.phi_top_mm), y_top))
    bars.extend(_bar_positions_along_horizontal_face(sec, int(sec.n_bot), float(sec.phi_bot_mm), y_bot))
    if is_column:
        phi_side = _phi_side(sec)
        x_left = sec.cover_mm + sec.stirrup_phi_mm + 0.5 * phi_side
        x_right = sec.b_mm - sec.cover_mm - sec.stirrup_phi_mm - 0.5 * phi_side
        bars.extend(_bar_positions_along_vertical_face(sec, int(getattr(sec, "n_left", 0) or 0), phi_side, x_left))
        bars.extend(_bar_positions_along_vertical_face(sec, int(getattr(sec, "n_right", 0) or 0), phi_side, x_right))
    return bars


def _section_rebar_summary(sec: RCSection, *, title: str | None = None) -> str:
    if _is_column_section(sec, title):
        total = int(sec.n_top) + int(sec.n_bot) + int(getattr(sec, "n_left", 0) or 0) + int(getattr(sec, "n_right", 0) or 0)
        return (
            f"Perimeter bars: top {sec.n_top}phi{sec.phi_top_mm:.0f}, bottom {sec.n_bot}phi{sec.phi_bot_mm:.0f}, "
            f"side L/R {int(getattr(sec, 'n_left', 0) or 0)}/{int(getattr(sec, 'n_right', 0) or 0)}phi{_phi_side(sec):.0f} | total {total}\n"
            f"Stirrups/ties: phi{sec.stirrup_phi_mm:.0f}@{sec.stirrup_spacing_mm:.0f}, legs={sec.stirrup_legs}"
        )
    return (
        f"Top: {sec.n_top}phi{sec.phi_top_mm:.0f} | Bottom: {sec.n_bot}phi{sec.phi_bot_mm:.0f} | "
        f"Stirrups: phi{sec.stirrup_phi_mm:.0f}@{sec.stirrup_spacing_mm:.0f}, legs={sec.stirrup_legs}"
    )


_SEC_COLORS = {
    "concrete": "#E4E1DB",
    "existing": "#F4F2EE",
    "jacket": "#BDB6AA",
    "edge": "#3D3D3D",
    "stirrup": "#1D3557",
    "bar": "#1B1B1B",
    "new_stirrup": "#C1121F",
    "dim": "#555555",
}


def _draw_rc_section(ax, sec, title, *, base_sec=None):
    """Engineering-style drawing of a rectangular RC section.

    If base_sec is given, sec is the jacketed section: the jacket ring is
    drawn darker and hatched, the existing section lighter inside it.
    """
    from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

    b = float(_section_value(sec, ["b_mm", "b"], 0.0))
    h = float(_section_value(sec, ["h_mm", "h"], 0.0))
    cover = float(_section_value(sec, ["cover_mm", "cover"], 30.0))
    st_phi = float(_section_value(sec, ["stirrup_phi_mm", "stirrup_phi"], 8.0))
    C = _SEC_COLORS
    m = max(b, h)

    if base_sec is None:
        ax.add_patch(Rectangle((0, 0), b, h, fc=C["concrete"], ec=C["edge"], lw=1.8, zorder=1))
        t_j = 0.0
    else:
        ob = float(_section_value(base_sec, ["b_mm", "b"], 0.0))
        oh = float(_section_value(base_sec, ["h_mm", "h"], 0.0))
        ox, oy = 0.5 * (b - ob), 0.5 * (h - oh)
        t_j = max(ox, oy, 0.0)
        ax.add_patch(Rectangle((0, 0), b, h, fc=C["jacket"], ec=C["edge"], lw=1.8, hatch="///", zorder=1))
        ax.add_patch(Rectangle((ox, oy), ob, oh, fc=C["existing"], ec=C["edge"], lw=1.0, ls="--", zorder=2))
        # jacket stirrup at the jacket cover from the outer face
        base_cover = float(_section_value(sec, ["jacket_cover_mm"], max(cover - t_j, 15.0)))
        so = base_cover + 0.5 * float(_section_value(sec, ["jacket_stirrup_phi_mm"], st_phi))
        ax.add_patch(FancyBboxPatch((so, so), b - 2 * so, h - 2 * so, boxstyle=f"round,pad=0,rounding_size={2.5 * st_phi}",
                                    fill=False, ec=C["new_stirrup"], lw=1.6, zorder=3))

    # stirrup / tie (centre line)
    so = cover + 0.5 * st_phi
    ax.add_patch(FancyBboxPatch((so, so), max(b - 2 * so, 1.0), max(h - 2 * so, 1.0),
                                boxstyle=f"round,pad=0,rounding_size={2.5 * st_phi}",
                                fill=False, ec=C["stirrup"], lw=1.6, zorder=3))

    for x, y, phi in _all_longitudinal_bar_positions(sec, title=title):
        ax.add_patch(Circle((x, y), max(0.5 * phi, 2.0), fc=C["bar"], ec="white", lw=0.6, zorder=5))
    if base_sec is not None:
        from framekit.jacketing import jacket_bar_positions
        for x, y, phi in jacket_bar_positions(sec):
            ax.add_patch(Circle((x, y), max(0.5 * phi, 2.0), fc=C["new_stirrup"], ec="white", lw=0.6, zorder=5))

    # dimensions
    d = 0.09 * m
    kw = dict(arrowstyle="<->", lw=0.8, color=C["dim"], shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(0, h + d), xytext=(b, h + d), arrowprops=kw)
    ax.text(0.5 * b, h + d + 0.015 * m, f"b = {b:.0f}", ha="center", va="bottom", fontsize=9, color=C["dim"])
    ax.annotate("", xy=(b + d, 0), xytext=(b + d, h), arrowprops=kw)
    ax.text(b + d + 0.02 * m, 0.5 * h, f"h = {h:.0f}", ha="left", va="center", rotation=90, fontsize=9, color=C["dim"])
    for (x0, y0, x1, y1) in [(0, h, 0, h + 1.3 * d), (b, h, b, h + 1.3 * d), (b, 0, b + 1.3 * d, 0), (b, h, b + 1.3 * d, h)]:
        ax.plot([x0, x1], [y0, y1], color=C["dim"], lw=0.5)

    # reinforcement summary
    n_top, n_bot = int(_section_value(sec, ["n_top"], 0)), int(_section_value(sec, ["n_bot"], 0))
    phi_top, phi_bot = float(_section_value(sec, ["phi_top_mm"], 0.0)), float(_section_value(sec, ["phi_bot_mm"], 0.0))
    n_side = int(_section_value(sec, ["n_left"], 0) or 0) + int(_section_value(sec, ["n_right"], 0) or 0)
    spacing = float(_section_value(sec, ["stirrup_spacing_mm", "stirrup_spacing"], 0.0))
    legs = int(_section_value(sec, ["stirrup_legs"], 0))
    As = 0.25 * math.pi * (n_top * phi_top ** 2 + n_bot * phi_bot ** 2)
    if n_side and _is_column_section(sec, title):
        ps = _phi_side(sec) if hasattr(sec, "n_left") else max(phi_top, phi_bot)
        As += 0.25 * math.pi * n_side * ps ** 2
    if hasattr(sec, "As_longitudinal_mm2"):
        As = float(sec.As_longitudinal_mm2)
    lines = [f"top {n_top}Ø{phi_top:.0f}   bottom {n_bot}Ø{phi_bot:.0f}" + (f"   sides {n_side}Ø{_phi_side(sec):.0f}" if n_side and hasattr(sec, 'n_left') else ""),
             f"stirrups Ø{st_phi:.0f}/{spacing:.0f} mm, {legs} legs",
             f"As = {As:.0f} mm²  (ρ = {100 * As / max(b * h, 1e-9):.2f}%)   cover = {cover - t_j:.0f} mm"]
    if base_sec is not None:
        nj = sum(int(_section_value(sec, [k], 0) or 0) for k in ("jn_top", "jn_bot", "jn_left", "jn_right"))
        jphi = float(_section_value(sec, ["jphi_mm"], 0.0) or 0.0)
        chk = getattr(sec, "jacket_check", {}) or {}
        lines.append(f"jacket t = {t_j:.0f} mm, cage {nj}Ø{jphi:.0f} (cover {float(_section_value(sec, ['jacket_cover_mm'], 30.0)):.0f} mm)")
        if chk.get("min_clear_spacing_mm") is not None and chk.get("status"):
            lines.append(f"jacket bar clear spacing {chk['min_clear_spacing_mm']:.0f} mm ≥ {chk['required_clear_spacing_mm']:.0f} mm: {chk['status']}")
    ax.text(0.5 * b, -0.07 * m, "\n".join(lines), ha="center", va="top", fontsize=8.3, color="#222222", linespacing=1.5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.12 * m, b + 0.24 * m)
    ax.set_ylim(-0.36 * m, h + 0.2 * m)
    ax.axis("off")
    ax.set_title(title, loc="center", fontsize=11.5)


def plot_rc_section(sec: RCSection, title: str = "RC Section"):
    """Plot one rectangular RC section (beams: top/bottom bars; columns: all-around bars)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.8, 6.4))
    _draw_rc_section(ax, sec, title)
    plt.show()


def plot_selected_rc_sections(column_sec: RCSection, beam_sec: RCSection):
    import matplotlib.pyplot as plt
    if not _is_column_section(column_sec, "Column"):
        column_sec.section_role = "column"
    beam_sec.section_role = "beam"
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2))
    _draw_rc_section(axes[0], beam_sec, "Beam section")
    _draw_rc_section(axes[1], column_sec, "Column section")
    fig.suptitle("Selected RC sections  (dimensions in mm)", x=0.01, ha="left")
    plt.show()


# ============================================================
# PUBLIC API
# ============================================================
# PUBLIC API
# ============================================================
def ask_sections_rc():
    print("\n=== RC SECTION INPUT ===")
    _print_default_stirrup_note()

    print("\nSection input options:")
    print("1) Choose beam and column from preset libraries")
    print("2) Use default normal-building presets")
    print("3) Fully manual input")
    opt = input("Select option (1/2/3) [1]: ").strip() or "1"

    # Good, ordinary defaults from the libraries.
    beam_defaults = dict(BEAM_PRESETS[2])
    beam_defaults.pop("label", None)
    column_defaults = dict(COLUMN_PRESETS[2])
    column_defaults.pop("label", None)

    if opt == "2":
        beam_sec = RCSection(**beam_defaults)
        column_sec = RCSection(**column_defaults)

        print("\nUsing default beam section:")
        print(beam_sec)
        print("\nUsing default column section:")
        print(column_sec)

        if _ask_yes_no("Modify default values manually? (y/n)", "n"):
            beam_sec = _ask_rc_section("Beam", beam_sec.to_dict())
            column_sec = _ask_rc_section("Column", column_sec.to_dict())

    elif opt == "3":
        beam_sec = _ask_rc_section("Beam", beam_defaults)
        column_sec = _ask_rc_section("Column", column_defaults)

    else:
        beam_defaults = _choose_preset("Beam", BEAM_PRESETS, default_index=3)
        column_defaults = _choose_preset("Column", COLUMN_PRESETS, default_index=3)

        beam_sec = RCSection(**beam_defaults)
        column_sec = RCSection(**column_defaults)

        if _ask_yes_no("Modify selected beam reinforcement/section? (y/n)", "n"):
            beam_sec = _ask_rc_section("Beam", beam_sec.to_dict())

        if _ask_yes_no("Modify selected column reinforcement/section? (y/n)", "n"):
            column_sec = _ask_rc_section("Column", column_sec.to_dict())

    beam_sec.section_role = "beam"
    column_sec.section_role = "column"

    _print_section_warnings("Beam", beam_sec)
    _print_section_warnings("Column", column_sec)

    print("\nSelected beam section:")
    print(beam_sec)
    print("\nSelected column section:")
    print(column_sec)

    # Plotting is handled by main.py after this function returns.
    return column_sec, beam_sec


# ============================================================
# RETROFIT SECTION COMPARISON PLOTTING
# ============================================================
def _section_value(sec, names, default=0.0):
    for name in names:
        if hasattr(sec, name):
            return getattr(sec, name)
    return default


def _report_bar_positions(sec: RCSection, n: int, phi: float, y: float):
    # Backward-compatible helper kept for older calls.
    return [(x, yy) for x, yy, _phi in _bar_positions_along_horizontal_face(sec, n, phi, y)]


def _draw_dimension(ax, x1, y1, x2, y2, text, *, text_offset=0.0, fontsize=7):
    """Draw a clean report-style dimension line with arrow heads."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="<->", lw=0.75, color="black", shrinkA=0, shrinkB=0),
    )
    xm = 0.5 * (x1 + x2)
    ym = 0.5 * (y1 + y2)
    if abs(y2 - y1) < 1e-9:
        ax.text(xm, ym + text_offset, text, ha="center", va="bottom", fontsize=fontsize)
    else:
        ax.text(xm + text_offset, ym, text, ha="left", va="center", rotation=90, fontsize=fontsize)


def _draw_report_style_rc_section(ax, sec: RCSection, subtitle: str, *, base_sec: RCSection | None = None):
    """Draw an RC section without coordinate axes, using dimension arrows.

    If base_sec is supplied, sec is treated as the jacketed section. The added
    concrete jacket is shown as solid gray and the original section remains
    white/transparent in the center.
    """
    try:
        from matplotlib.patches import Rectangle, Circle
    except Exception:
        return

    b = float(_section_value(sec, ["b_mm", "b"], 0.0))
    h = float(_section_value(sec, ["h_mm", "h"], 0.0))
    cover = float(_section_value(sec, ["cover_mm", "cover"], 30.0))
    stirrup_phi = float(_section_value(sec, ["stirrup_phi_mm", "stirrup_phi"], 8.0))
    n_top = int(_section_value(sec, ["n_top"], 0))
    n_bot = int(_section_value(sec, ["n_bot"], 0))
    n_left = int(_section_value(sec, ["n_left"], 0) or 0)
    n_right = int(_section_value(sec, ["n_right"], 0) or 0)
    phi_top = float(_section_value(sec, ["phi_top_mm"], 0.0))
    phi_bot = float(_section_value(sec, ["phi_bot_mm"], 0.0))
    phi_side = float(_section_value(sec, ["phi_side_mm"], max(phi_top, phi_bot)) or max(phi_top, phi_bot))
    spacing = float(_section_value(sec, ["stirrup_spacing_mm", "stirrup_spacing"], 0.0))
    legs = int(_section_value(sec, ["stirrup_legs"], 0))

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if base_sec is None:
        # Existing section: white/transparent body with black outline.
        ax.add_patch(Rectangle((0, 0), b, h, facecolor="white", edgecolor="black", linewidth=1.6, zorder=1))
        inner_bar_sec = sec
        inner_origin = (0.0, 0.0)
        summary_extra = f"cover = {cover:.0f} mm"
    else:
        # Jacketed section: full jacket is gray; original concrete remains white.
        ob = float(_section_value(base_sec, ["b_mm", "b"], 0.0))
        oh = float(_section_value(base_sec, ["h_mm", "h"], 0.0))
        ox = 0.5 * (b - ob)
        oy = 0.5 * (h - oh)
        t = max(0.5 * (b - ob), 0.5 * (h - oh), 0.0)
        ax.add_patch(Rectangle((0, 0), b, h, facecolor="0.78", edgecolor="black", linewidth=1.6, zorder=1))
        ax.add_patch(Rectangle((ox, oy), ob, oh, facecolor="white", edgecolor="black", linewidth=1.3, zorder=2))
        ax.text(ox + 0.5 * ob, oy + oh + 0.035 * h, "existing section", ha="center", va="bottom", fontsize=6.5, color="0.25")
        inner_bar_sec = sec
        inner_origin = (0.0, 0.0)
        summary_extra = f"jacket t = {t:.0f} mm"

    # Stirrups and bars for the section being checked.
    s_off = cover + 0.5 * stirrup_phi
    ax.add_patch(Rectangle((s_off, s_off), max(b - 2.0 * s_off, 1.0), max(h - 2.0 * s_off, 1.0),
                           fill=False, linestyle="--", edgecolor="0.35", linewidth=1.0, zorder=3))
    # Draw longitudinal bars. For columns this includes intermediate side-face bars.
    if not hasattr(inner_bar_sec, "section_role") and (n_left > 0 or n_right > 0):
        setattr(inner_bar_sec, "section_role", "column")
    if hasattr(inner_bar_sec, "n_left"):
        setattr(inner_bar_sec, "n_left", n_left)
        setattr(inner_bar_sec, "n_right", n_right)
        setattr(inner_bar_sec, "phi_side_mm", phi_side)
    for x, y, phi in _all_longitudinal_bar_positions(inner_bar_sec, title=subtitle):
        ax.add_patch(Circle((x + inner_origin[0], y + inner_origin[1]), max(0.5 * phi, 1.3), color="black", zorder=4))

    # Dimension arrows outside the section.
    m = max(b, h) * 0.28
    _draw_dimension(ax, 0, h + 0.12 * m, b, h + 0.12 * m, f"{b:.0f} mm", text_offset=0.02 * m, fontsize=7)
    _draw_dimension(ax, b + 0.13 * m, 0, b + 0.13 * m, h, f"{h:.0f} mm", text_offset=0.02 * m, fontsize=7)

    # Small extension lines for dimensions.
    ax.plot([0, 0], [h, h + 0.18 * m], color="black", lw=0.6)
    ax.plot([b, b], [h, h + 0.18 * m], color="black", lw=0.6)
    ax.plot([b, b + 0.18 * m], [0, 0], color="black", lw=0.6)
    ax.plot([b, b + 0.18 * m], [h, h], color="black", lw=0.6)

    ax.set_title(subtitle, fontsize=8.5, fontweight="bold", pad=4)
    ax.text(
        0.5 * b,
        -0.20 * m,
        f"b = {b:.0f} mm   h = {h:.0f} mm\nTop {n_top}Ø{phi_top:.0f} | Bot {n_bot}Ø{phi_bot:.0f}\nStirrups Ø{stirrup_phi:.0f}@{spacing:.0f}, legs={legs}\n{summary_extra}",
        ha="center",
        va="top",
        fontsize=6.8,
    )
    ax.set_xlim(-0.22 * m, b + 0.35 * m)
    ax.set_ylim(-0.44 * m, h + 0.38 * m)


def plot_rc_section_before_after(base_sec: RCSection, jacketed_sec: RCSection, title: str = "RC Section Before/After Jacketing"):
    """Before/after concrete-jacketing section comparison."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4))
    _draw_rc_section(axes[0], base_sec, "Before (existing)")
    _draw_rc_section(axes[1], jacketed_sec, "After (jacketed)", base_sec=base_sec)
    C = _SEC_COLORS
    fig.legend(handles=[Patch(fc=C["existing"], ec=C["edge"], ls="--", label="Existing concrete"),
                        Patch(fc=C["jacket"], ec=C["edge"], hatch="///", label="Added jacket"),
                        Line2D([], [], color=C["stirrup"], lw=1.6, label="Existing stirrup"),
                        Line2D([], [], color=C["new_stirrup"], lw=1.6, label="Jacket stirrup")],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle(title + "  (dimensions in mm)", x=0.01, ha="left")
    plt.show()


def plot_jacketed_section_comparisons(
    *,
    elements,
    column_sec: RCSection,
    beam_sec: RCSection,
    jacketed_element_sections,
    jacketed_member_ids,
    max_plots: int | None = None,
):
    """Plot report-style before/after section figures for jacketed members."""
    if jacketed_element_sections is None:
        print("No jacketed element section map available for plotting.")
        return

    ids = []
    for eid in jacketed_member_ids or []:
        try:
            eid_int = int(eid)
        except Exception:
            continue
        if eid_int in elements and eid_int in jacketed_element_sections:
            ids.append(eid_int)

    if not ids:
        print("No jacketed section figures to plot.")
        return

    ids_to_plot = ids if max_plots is None else ids[:max_plots]

    for eid in ids_to_plot:
        _, _, etype = elements[eid]
        base_sec = column_sec if str(etype).lower() == "column" else beam_sec
        new_sec = jacketed_element_sections[eid]
        plot_rc_section_before_after(
            base_sec,
            new_sec,
            title=f"Concrete Jacketing Section Comparison - Element {eid} ({etype})",
        )

    if max_plots is not None and len(ids) > max_plots:
        print(f"Displayed {max_plots} jacketed section comparisons out of {len(ids)} members.")
