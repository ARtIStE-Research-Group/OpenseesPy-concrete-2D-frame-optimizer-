"""
RC jacket detailing: geometry and longitudinal bars of a jacketed section.

A jacketed member is modelled as it is built:
  * the existing core keeps its bars in their original position
    (effective cover = original cover + jacket thickness);
  * the jacket has its own cage: new stirrups at the jacket cover and new
    longitudinal bars placed inside the jacket thickness.

The jacket bars are chosen (number and diameter) so that
  * the added steel area is at least `ratio` × existing longitudinal steel,
  * a minimum cage of corner bars is always present (4 Ø12),
  * EN 1992-1-1 8.2(2): clear spacing >= max(k1·φ, dg + k2, 20 mm),
  * columns: centre spacing of bars restrained by hoops <= 200 mm
    (EN 1998-1 5.4.3.2.2(11), DCM; 150 mm for DCH),
  * the bars fit inside the jacket: c_j + φ_st + φ + 10 mm <= t.
If no layout satisfies all rules the check returns FAIL and the optimizer
increases the jacket thickness instead.
"""
import math
from copy import copy

BAR_DIAMETERS_MM = (12.0, 14.0, 16.0, 20.0, 25.0)
MIN_CAGE_PHI_MM = 12.0
JACKET_COVER_MM = 30.0
COLUMN_MAX_CENTRE_SPACING_MM = 200.0
MIN_JACKET_THICKNESS_MM = 75.0
JACKET_BAR_CLEARANCE_MM = 10.0   # gap between jacket bars and the existing concrete surface


def bar_area(phi):
    return 0.25 * math.pi * float(phi) ** 2


def ec2_min_clear_spacing(phi, aggregate_size_mm=20.0, k1=1.0, k2_mm=5.0):
    """EN 1992-1-1 8.2(2)."""
    return max(k1 * float(phi), float(aggregate_size_mm) + k2_mm, 20.0)


def _face_layout(length, n, phi, edge):
    """Clear and centre spacing of n bars on a face of given length."""
    if n <= 1:
        return float("inf"), float("inf")
    avail = float(length) - 2.0 * edge
    clear = (avail - n * phi) / (n - 1)
    centre = (avail - phi) / (n - 1)
    return clear, centre


def _getf(sec, name, default=0.0):
    try:
        v = getattr(sec, name)
        return float(v) if v is not None else float(default)
    except Exception:
        return float(default)


def existing_steel_mm2(base, etype):
    a_top = int(_getf(base, "n_top")) * bar_area(_getf(base, "phi_top_mm"))
    a_bot = int(_getf(base, "n_bot")) * bar_area(_getf(base, "phi_bot_mm"))
    a_side = 0.0
    if "column" in str(etype).lower():
        phs = _getf(base, "phi_side_mm", 0.0) or 0.5 * (_getf(base, "phi_top_mm") + _getf(base, "phi_bot_mm"))
        a_side = (int(_getf(base, "n_left")) + int(_getf(base, "n_right"))) * bar_area(phs)
    return a_top, a_bot, a_side


def design_jacket_bars(base, etype, t_mm, ratio, *, jacket_cover_mm=JACKET_COVER_MM, stirrup_phi_mm=8.0,
                       aggregate_size_mm=20.0, k1=1.0, k2_mm=5.0,
                       column_max_spacing_mm=COLUMN_MAX_CENTRE_SPACING_MM):
    """Choose the jacket longitudinal bars. Returns a dict with the layout and the check."""
    is_col = "column" in str(etype).lower()
    b = _getf(base, "b_mm") + 2.0 * t_mm
    h = _getf(base, "h_mm") + 2.0 * t_mm
    edge = jacket_cover_mm + stirrup_phi_mm
    a_top, a_bot, a_side = existing_steel_mm2(base, etype)
    r = max(float(ratio), 0.0)
    best = None

    for phi in BAR_DIAMETERS_MM:
        if edge + phi + JACKET_BAR_CLEARANCE_MM > t_mm + 1e-9:   # bar inside the jacket, clear of old concrete
            continue
        if phi < MIN_CAGE_PHI_MM:
            continue
        smin = ec2_min_clear_spacing(phi, aggregate_size_mm, k1, k2_mm)
        if not is_col:
            sol = {}
            ok = True
            for face, req in (("top", r * a_top), ("bot", r * a_bot)):
                pick = None
                for n in range(2, 40):
                    clear, _ = _face_layout(b, n, phi, edge)
                    if clear + 1e-9 < smin:
                        break
                    if n * bar_area(phi) + 1e-9 >= max(req, 2 * bar_area(MIN_CAGE_PHI_MM) if r == 0 else req):
                        pick = n
                        break
                if pick is None:
                    ok = False
                    break
                sol[face] = pick
            if not ok:
                continue
            area = (sol["top"] + sol["bot"]) * bar_area(phi)
            cand = {"jn_top": sol["top"], "jn_bot": sol["bot"], "jn_left": 0, "jn_right": 0,
                    "jphi_mm": phi, "area": area, "n": sol["top"] + sol["bot"]}
        else:
            req = r * (a_top + a_bot + a_side)
            cand = None
            for n_face in range(2, 30):
                clear_b, cen_b = _face_layout(b, n_face, phi, edge)
                if clear_b + 1e-9 < smin:
                    break
                if cen_b > column_max_spacing_mm + 1e-9:
                    continue
                for k in range(0, 30):
                    clear_h, cen_h = _face_layout(h, k + 2, phi, edge)
                    if clear_h + 1e-9 < smin:
                        break
                    if cen_h > column_max_spacing_mm + 1e-9:
                        continue
                    n_tot = 2 * n_face + 2 * k
                    area = n_tot * bar_area(phi)
                    if area + 1e-9 >= req:
                        c = {"jn_top": n_face, "jn_bot": n_face, "jn_left": k, "jn_right": k,
                             "jphi_mm": phi, "area": area, "n": n_tot}
                        if cand is None or (c["area"], c["n"]) < (cand["area"], cand["n"]):
                            cand = c
                        break
            if cand is None:
                continue
        if best is None or (cand["area"], cand["n"], cand["jphi_mm"]) < (best["area"], best["n"], best["jphi_mm"]):
            best = cand

    if best is None:
        reason = ("jacket too thin for a longitudinal cage" if edge + MIN_CAGE_PHI_MM + JACKET_BAR_CLEARANCE_MM > t_mm
                  else "required bars do not fit with EC2 8.2(2) spacing")
        return {"status": "FAIL", "note": f"t = {t_mm:.0f} mm: {reason}", "jn_top": 0, "jn_bot": 0,
                "jn_left": 0, "jn_right": 0, "jphi_mm": 0.0, "min_clear_spacing_mm": float("nan"),
                "required_clear_spacing_mm": float("nan"), "critical_face": "jacket"}

    phi = best["jphi_mm"]
    clear_b, cen_b = _face_layout(b, best["jn_top"], phi, edge)
    clear_h = float("inf")
    cen_h = float("nan")
    if is_col:
        clear_h, cen_h = _face_layout(h, best["jn_left"] + 2, phi, edge)
    smin = ec2_min_clear_spacing(phi, aggregate_size_mm, k1, k2_mm)
    crit_face = "jacket top/bottom" if clear_b <= clear_h else "jacket side"
    note = (f"jacket cage {best['n']}Ø{phi:.0f} (top {best['jn_top']}, bottom {best['jn_bot']}"
            + (f", sides {best['jn_left']}+{best['jn_right']}" if is_col else "")
            + f"); clear spacing {min(clear_b, clear_h):.0f} mm ≥ {smin:.0f} mm (EC2 8.2(2))"
            + (f"; centre spacing ≤ {max(cen_b, cen_h if cen_h == cen_h else 0):.0f} mm ≤ {column_max_spacing_mm:.0f} mm (EC8)"
               if is_col else ""))
    return {"status": "OK", "note": note, "jn_top": best["jn_top"], "jn_bot": best["jn_bot"],
            "jn_left": best["jn_left"], "jn_right": best["jn_right"], "jphi_mm": phi,
            "min_clear_spacing_mm": min(clear_b, clear_h), "required_clear_spacing_mm": smin,
            "critical_face": crit_face}


def build_jacketed_section(base, etype, t_mm, *, stirrup_spacing_mm=None, ratio=0.0, **kw):
    """Jacketed section with existing bars in the core and the jacket cage."""
    sec = copy(base)
    t = float(t_mm)
    sec.b_mm = float(base.b_mm) + 2.0 * t
    sec.h_mm = float(base.h_mm) + 2.0 * t
    sec.cover_mm = float(base.cover_mm) + t
    if stirrup_spacing_mm is not None:
        sec.stirrup_spacing_mm = float(stirrup_spacing_mm)
    lay = design_jacket_bars(base, etype, t, ratio, stirrup_phi_mm=float(base.stirrup_phi_mm), **kw)
    sec.jacket_t_mm = t
    sec.jacket_cover_mm = kw.get("jacket_cover_mm", JACKET_COVER_MM)
    sec.jacket_stirrup_phi_mm = float(base.stirrup_phi_mm)
    sec.jn_top, sec.jn_bot = int(lay["jn_top"]), int(lay["jn_bot"])
    sec.jn_left, sec.jn_right = int(lay["jn_left"]), int(lay["jn_right"])
    sec.jphi_mm = float(lay["jphi_mm"])
    if "column" in str(etype).lower():
        sec.section_role = "column"
    sec.jacket_check = lay
    sec.retrofit_type = "concrete_jacketing"
    sec.jacket_thickness_mm = t
    sec.longitudinal_rebar_increase_ratio = float(ratio)
    return sec


def jacket_steel_mm2(sec):
    n = int(_getf(sec, "jn_top")) + int(_getf(sec, "jn_bot")) + int(_getf(sec, "jn_left")) + int(_getf(sec, "jn_right"))
    return n * bar_area(_getf(sec, "jphi_mm"))


def jacket_bar_positions(sec):
    """[(x, y, phi)] of the jacket bars, origin at the bottom-left corner."""
    n_t, n_b = int(_getf(sec, "jn_top")), int(_getf(sec, "jn_bot"))
    n_l, n_r = int(_getf(sec, "jn_left")), int(_getf(sec, "jn_right"))
    phi = _getf(sec, "jphi_mm")
    if phi <= 0 or (n_t + n_b) == 0:
        return []
    b, h = _getf(sec, "b_mm"), _getf(sec, "h_mm")
    e = _getf(sec, "jacket_cover_mm", JACKET_COVER_MM) + _getf(sec, "jacket_stirrup_phi_mm", 8.0) + 0.5 * phi
    pts = []

    def row(n, y):
        if n == 1:
            return [(0.5 * b, y, phi)]
        return [(e + i * (b - 2 * e) / (n - 1), y, phi) for i in range(n)]

    pts += row(n_t, h - e) + row(n_b, e)
    for k, x in ((n_l, e), (n_r, b - e)):
        for i in range(k):
            pts.append((x, e + (i + 1) * (h - 2 * e) / (k + 1), phi))
    return pts
