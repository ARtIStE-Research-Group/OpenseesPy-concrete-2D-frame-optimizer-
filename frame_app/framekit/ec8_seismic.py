"""
EC8 seismic design / assessment tools for the 2D frame.

Two code bases are available (chosen by the user):

  EN 1998-1  new design      q from ductility class (DCM/DCH), structural type
                             and regularity (5.2.2.2); capacity rule
                             ΣMRc >= 1.3 ΣMRb (4.4.2.3).
  EN 1998-3  existing bldg   knowledge level KL1/KL2/KL3 -> confidence factor
                             CF (Table 3.1); material strengths = mean / CF;
                             q-factor approach with q = 1.5 by default (4.2).

Common parts:
  - design spectrum Sd(T) (EN 1998-1 3.2.2.5) with lower bound β·ag
  - cracked stiffness factor for seismic analyses (4.3.1(7), default 0.5)
  - lateral force method (4.3.3.2) and RSA-equivalent floor forces
  - seismic combination G + ψ2·Q ± E (EN 1990 6.4.3.4) as two static
    combinations "SEIS_E+" / "SEIS_E-" checked with the EC2 routines
  - displacements ds = q·de (4.3.4), damage limitation ν·dr/h (4.4.3.2)
  - interstorey drift sensitivity θ (4.4.2.2) with 1/(1-θ) amplification
"""
import copy
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import openseespy.opensees as ops

from framekit.constants import G_STD, PSI_Q2
from framekit.data_models import Load
from framekit.geometry import node_id_at


SEIS_PREFIX = "SEIS"
CF_BY_KL = {"KL1": 1.35, "KL2": 1.20, "KL3": 1.00}
DLS_LIMITS = {
    "brittle_attached": 0.005,
    "ductile_attached": 0.0075,
    "detached_or_non_interfering": 0.010,
}


# ============================================================
# INPUT
# ============================================================
def _ask_choice(prompt, choices, default):
    from framekit.input_helpers import ask_choice_def
    return ask_choice_def(prompt, choices, default)


def _ask_float(prompt, default, lo=None, hi=None):
    from framekit.input_helpers import ask_float_def
    return ask_float_def(prompt, default, lo, hi)


def ask_code_basis() -> Dict:
    """Choose EN 1998-1 (new design) or EN 1998-3 (existing building)."""
    print("\n--- Seismic code basis ---")
    print("1) EN 1998-1  - new building design (ductility class DCM/DCH)")
    print("2) EN 1998-3  - assessment / retrofit of an existing building")
    opt = _ask_choice("Select (1/2) [2]: ", ["1", "2"], "2")
    if opt == "1":
        return {"code": "EN1998-1", "knowledge_level": None, "CF": 1.0}

    print("\nKnowledge level (EN 1998-3 Table 3.1):")
    print("  KL1 limited knowledge     -> CF = 1.35")
    print("  KL2 normal knowledge      -> CF = 1.20")
    print("  KL3 full knowledge        -> CF = 1.00")
    kl = _ask_choice("Knowledge level [KL1/KL2/KL3] (default KL2): ", ["KL1", "KL2", "KL3"], "KL2").upper()
    cf = CF_BY_KL[kl]
    print(f"Confidence factor CF = {cf:.2f}. The concrete/steel strengths entered for the EC2 checks are")
    print("treated as MEAN in-situ values and divided by CF (then by γc, γs as usual).")
    return {"code": "EN1998-3", "knowledge_level": kl, "CF": cf}


def apply_confidence_factor(ec2mat, cf: float):
    """Return a copy of the EC2 material with fck, fyk divided by CF."""
    if ec2mat is None or cf is None or abs(float(cf) - 1.0) < 1e-12:
        return ec2mat
    m = copy.copy(ec2mat)
    m.fck = float(ec2mat.fck) / float(cf)
    m.fyk = float(ec2mat.fyk) / float(cf)
    return m


def _wall_aspect_ratio(geom, walls) -> float:
    """α0 = Σ hw / Σ lw over the wall lines (EN 1998-1 5.2.2.2(11))."""
    if not walls:
        return np.nan
    by_bay = {}
    for w in walls:
        by_bay.setdefault(int(w.bay_left), set()).add(int(w.floor_top))
    hw = sum(len(st) * geom.floor_height for st in by_bay.values())
    lw = sum(float(geom.bay_lengths[b - 1]) for b in by_bay)
    return hw / max(lw, 1e-9)


def compute_behaviour_factor(geom, walls, code: str, ductility: str = "DCM",
                             regular_in_elevation: bool = True, q_existing: float = 1.5):
    """Behaviour factor q and a short explanation."""
    if code == "EN1998-3":
        return float(q_existing), f"EN 1998-3 q-factor approach: q = {q_existing:.2f}"

    nf, nb = int(geom.num_floors), int(geom.num_bays)
    if walls:
        # Infill RC walls with the frame: wall-equivalent dual system assumed
        # (conservative choice when the wall/frame shear split is unknown).
        au_a1 = 1.2
        alpha0 = _wall_aspect_ratio(geom, walls)
        kw = min(1.0, max(0.5, (1.0 + alpha0) / 3.0)) if np.isfinite(alpha0) else 1.0
        system = "wall-equivalent dual system"
    else:
        au_a1 = 1.1 if nf == 1 else (1.2 if nb == 1 else 1.3)
        kw = 1.0
        system = "frame system"
        alpha0 = np.nan
    q0 = (3.0 if str(ductility).upper() == "DCM" else 4.5) * au_a1
    q = q0 * kw
    red = 1.0 if regular_in_elevation else 0.8
    q = max(1.5, q * red)
    txt = (f"EN 1998-1 {ductility.upper()}, {system}: αu/α1 = {au_a1:.2f}, q0 = {q0:.2f}, "
           f"kw = {kw:.2f}" + ("" if regular_in_elevation else ", ×0.8 (not regular in elevation)")
           + f"  ->  q = {q:.2f}")
    return float(q), txt


def ask_behaviour_factor(geom, walls, code_info: Dict) -> Dict:
    code = code_info.get("code", "EN1998-1")
    print("\n--- Behaviour factor q ---")
    if code == "EN1998-3":
        q, txt = compute_behaviour_factor(geom, walls, code)
        print(txt)
        print("(EN 1998-3 4.2: q = 1.5 for RC unless a higher value is justified, max ≈ 3.0)")
        q = _ask_float(f"Behaviour factor q [default {q:.2f}]: ", q, 1.0, 3.0)
        return {"q": q, "ductility": None, "regular_in_elevation": None,
                "q_details": f"EN 1998-3 q-factor approach, q = {q:.2f}"}

    duct = _ask_choice("Ductility class [DCM/DCH] (default DCM): ", ["DCM", "DCH"], "DCM").upper()
    reg = _ask_choice("Regular in elevation (EN 1998-1 4.2.3.3)? (y/n) [y]: ", ["y", "n"], "y") == "y"
    q, txt = compute_behaviour_factor(geom, walls, code, duct, reg)
    print(txt)
    q = _ask_float(f"Behaviour factor q [Enter = {q:.2f}]: ", q, 1.0, 8.0)
    return {"q": q, "ductility": duct, "regular_in_elevation": reg, "q_details": txt}


def ask_dls_params() -> Dict:
    print("\n--- Damage limitation (EN 1998-1 4.4.3.2) ---")
    print("1) Brittle non-structural elements attached        ν·dr ≤ 0.005 h")
    print("2) Ductile non-structural elements attached        ν·dr ≤ 0.0075 h")
    print("3) Non-structural elements not interfering         ν·dr ≤ 0.010 h")
    opt = _ask_choice("Select (1/2/3) [1]: ", ["1", "2", "3"], "1")
    cls = {"1": "brittle_attached", "2": "ductile_attached", "3": "detached_or_non_interfering"}[opt]
    nu = _ask_float("Reduction factor ν [default 0.50 (importance I-II), 0.40 for III-IV]: ", 0.5, 0.1, 1.0)
    return {"dls_class": cls, "dls_limit": DLS_LIMITS[cls], "nu": nu}


def ask_seismic_settings(geom, walls, code_info: Dict, *, rsa_params=None, q_info=None,
                         dls=None, stiffness_factor=None, extra_mass_floor_kg=0.0) -> Dict:
    """Collect everything the lateral-force-method evaluation needs."""
    from framekit.response_spectrum import ask_ec8_spectrum_params
    if rsa_params is None:
        rsa_params = ask_ec8_spectrum_params()
    if q_info is None:
        q_info = ask_behaviour_factor(geom, walls, code_info)
    if dls is None:
        dls = ask_dls_params()
    if stiffness_factor is None:
        stiffness_factor = _ask_float("Cracked stiffness factor for seismic analysis [default 0.5]: ", 0.5, 0.1, 1.0)
    return {
        **code_info,
        "rsa_params": dict(rsa_params),
        **q_info,
        **dls,
        "beta": 0.2,
        "stiffness_factor": float(stiffness_factor),
        "psi2": PSI_Q2,
        "extra_mass_floor_kg": float(extra_mass_floor_kg or 0.0),
    }


# ============================================================
# SPECTRA
# ============================================================
def _soil(rsa_params):
    from framekit.response_spectrum import EC8_SOIL_TABLE
    return EC8_SOIL_TABLE[int(rsa_params["spectrum_type"])][str(rsa_params["soil_class"]).upper()]


def design_spectrum_ordinate(T: float, rsa_params: Dict, q: float, beta: float = 0.2) -> float:
    """EN 1998-1 3.2.2.5 design spectrum Sd(T) [m/s²]."""
    s = _soil(rsa_params)
    S, TB, TC, TD = s["S"], s["TB"], s["TC"], s["TD"]
    ag = float(rsa_params["ag"]) * float(rsa_params.get("importance_factor", 1.0))
    q = max(float(q), 1e-9)
    T = float(T)
    if T <= TB:
        return ag * S * (2.0 / 3.0 + T / TB * (2.5 / q - 2.0 / 3.0))
    if T <= TC:
        return ag * S * 2.5 / q
    if T <= TD:
        return max(ag * S * 2.5 / q * TC / T, beta * ag)
    return max(ag * S * 2.5 / q * TC * TD / (T * T), beta * ag)


def design_spectrum(periods, rsa_params, q, beta=0.2):
    return np.array([design_spectrum_ordinate(t, rsa_params, q, beta) for t in np.asarray(periods, float)])


# ============================================================
# MASSES / MODES
# ============================================================
def floor_masses_from_combos(geom, nodes, elements, combos, extra_mass_floor_kg=0.0,
                             base_combo="SLS_quasi_perm"):
    """Seismic masses = G + ψ2·Q, i.e. the quasi-permanent combination loads."""
    from framekit.dynamics import compute_floor_masses_kg_from_line_loads
    loads = combos[base_combo]["loads"] if base_combo in combos else []
    return compute_floor_masses_kg_from_line_loads(geom, nodes, elements, loads,
                                                   add_lumped_mass_per_floor_kg=extra_mass_floor_kg)


def _floor_nodes(geom, st):
    return [node_id_at(geom, st, ix) for ix in range(geom.num_bays + 1)]


def floor_shape_x(geom, mode):
    return np.array([np.mean([ops.nodeEigenvector(n, mode, 1) for n in _floor_nodes(geom, st)])
                     for st in range(1, geom.num_floors + 1)], dtype=float)


def _eigen(n_modes, dense=False):
    lam = None
    order = (("-fullGenLapack", n_modes), ("-genBandArpack", n_modes)) if dense else \
            (("-genBandArpack", n_modes), ("-fullGenLapack", n_modes))
    for args in order:
        try:
            lam = ops.eigen(*args)
            if lam is not None and len(np.atleast_1d(lam)) >= 1:
                break
        except Exception:
            lam = None
    if lam is None:
        raise RuntimeError("eigen analysis failed")
    return np.atleast_1d(np.asarray(lam, dtype=float))


def modal_x_properties(geom, n_modes, dense=False):
    """Periods, Γx and effective-mass ratios (X) of the current OpenSees model,
    computed from the assembled nodal masses: Γ = φᵀMr / φᵀMφ."""
    lam = _eigen(n_modes, dense)
    tags = ops.getNodeTags()
    mass = {n: [float(ops.nodeMass(n, d)) for d in (1, 2, 3)] for n in tags}
    m_tot = sum(v[0] for v in mass.values())
    out = []
    for k, l in enumerate(lam):
        if not (np.isfinite(l) and 0.0 < l < 1e100):
            continue
        mode = k + 1
        num = den = 0.0
        for n in tags:
            phi = [float(ops.nodeEigenvector(n, mode, d)) for d in (1, 2, 3)]
            mx, my, mr = mass[n]
            num += mx * phi[0]
            den += mx * phi[0] ** 2 + my * phi[1] ** 2 + mr * phi[2] ** 2
        gam = num / den if den > 0 else 0.0
        out.append({"mode": mode, "T": 2.0 * math.pi / math.sqrt(l), "omega": math.sqrt(l),
                    "gamma": gam, "ratio": (num * gam) / m_tot if m_tot > 0 else 0.0})
    return out


# ============================================================
# SEISMIC FORCES
# ============================================================
def lateral_force_method(geom, floor_masses, T1, rsa_params, q, beta=0.2):
    """EN 1998-1 4.3.3.2: Fb = Sd(T1)·m·λ, Fi = Fb·zi·mi/Σ zj·mj."""
    TC = _soil(rsa_params)["TC"]
    nf = int(geom.num_floors)
    m = np.array([floor_masses[s] for s in range(1, nf + 1)], dtype=float)
    z = np.array([s * geom.floor_height for s in range(1, nf + 1)], dtype=float)
    Sd = design_spectrum_ordinate(T1, rsa_params, q, beta)
    lam = 0.85 if (T1 <= 2.0 * TC and nf > 2) else 1.0
    Fb = Sd * m.sum() * lam
    F = Fb * z * m / max(np.sum(z * m), 1e-12)
    applicable = T1 <= min(4.0 * TC, 2.0)
    return {"Fb_kN": Fb / 1000.0, "Sd": Sd, "lambda": lam, "T1": T1,
            "forces_kN": {s: F[s - 1] / 1000.0 for s in range(1, nf + 1)},
            "applicable": bool(applicable)}


def rsa_equivalent_floor_forces(geom, floor_masses, modes, rsa_params, q, combo_method="CQC",
                                damping_ratio=0.05, beta=0.2):
    """Floor forces whose storey shears equal the CQC/SRSS-combined modal storey
    shears (signed loads for the static seismic combinations)."""
    from framekit.response_spectrum import combine_modal_response
    nf = int(geom.num_floors)
    m = np.array([floor_masses[s] for s in range(1, nf + 1)], dtype=float)
    shears = []
    omegas = []
    for md in modes:
        phi = floor_shape_x(geom, md["mode"])
        Sd = design_spectrum_ordinate(md["T"], rsa_params, q, beta)
        F = md["gamma"] * phi * m * Sd
        shears.append(np.cumsum(F[::-1])[::-1])   # storey shear, storey s carries floors >= s
        omegas.append(md["omega"])
    shears = np.array(shears)
    V = np.array([combine_modal_response(shears[:, s], np.array(omegas), method=combo_method,
                                         damping_ratio=damping_ratio) for s in range(nf)], dtype=float)
    F = V - np.r_[V[1:], 0.0]
    return {"Fb_kN": V[0] / 1000.0, "forces_kN": {s: F[s - 1] / 1000.0 for s in range(1, nf + 1)},
            "storey_shear_kN": {s: V[s - 1] / 1000.0 for s in range(1, nf + 1)}}


def floor_forces_to_loads(geom, forces_kN, factor=1.0):
    """Horizontal point loads, each floor force split equally over its nodes."""
    loads = []
    for st, F in forces_kN.items():
        ids = _floor_nodes(geom, st)
        for n in ids:
            loads.append(Load("point", float(factor) * float(F) / len(ids), node=n, direction="horizontal"))
    return loads


def strip_seismic_combos(combos, combo_names=None):
    c = {k: v for k, v in combos.items() if not str(k).startswith(SEIS_PREFIX)}
    n = [x for x in (combo_names or list(combos.keys())) if not str(x).startswith(SEIS_PREFIX)]
    return c, n


def add_seismic_combos(combos, combo_names, E_loads, base_combo="SLS_quasi_perm"):
    """Add SEIS_E+ / SEIS_E- = (G + ψ2·Q) ± E."""
    from framekit.combinations import scale_loads
    combos, names = strip_seismic_combos(combos, combo_names)
    base = combos.get(base_combo)
    if base is None:
        return combos, names
    out = dict(combos)
    for tag, sgn in (("SEIS_E+", 1.0), ("SEIS_E-", -1.0)):
        out[tag] = {
            "loads": list(base["loads"]) + scale_loads(E_loads, sgn),
            "note": f"EC8 seismic combination G + ψ2·Q {'+' if sgn > 0 else '-'} E",
            "factors": {**base.get("factors", {}), "E": sgn},
        }
    return out, names + ["SEIS_E+", "SEIS_E-"]


# ============================================================
# DISPLACEMENT CHECKS
# ============================================================
def storey_checks(geom, floor_masses, forces_kN, de_story_m, settings):
    """ds = q·de; damage limitation ν·ds/h; sensitivity θ = Ptot·ds/(Vtot·h)."""
    q = float(settings["q"])
    nu = float(settings.get("nu", 0.5))
    lim = float(settings.get("dls_limit", 0.005))
    nf = int(geom.num_floors)
    h = float(geom.floor_height)
    rows = []
    for s in range(1, nf + 1):
        de = float(de_story_m.get(s, 0.0))
        ds = q * de
        Ptot = G_STD * sum(floor_masses[k] for k in range(s, nf + 1)) / 1000.0
        Vtot = sum(forces_kN[k] for k in range(s, nf + 1))
        theta = Ptot * abs(ds) / max(abs(Vtot) * h, 1e-12)
        rows.append({
            "Story": s,
            "h [m]": h,
            "de [mm]": 1000.0 * de,
            "ds = q·de [mm]": 1000.0 * ds,
            "ν·ds/h [-]": nu * abs(ds) / h,
            "DLS limit [-]": lim,
            "DLS utilization": nu * abs(ds) / h / lim,
            "Ptot [kN]": Ptot,
            "Vtot [kN]": Vtot,
            "theta": theta,
            "theta_status": ("OK (≤0.10)" if theta <= 0.10 else
                             "amplify 1/(1-θ)" if theta <= 0.20 else
                             "2nd-order analysis needed" if theta <= 0.30 else "NOT ALLOWED (>0.30)"),
        })
    return pd.DataFrame(rows)


def _floor_avg_disp_x(geom):
    return {st: float(np.mean([ops.nodeDisp(n, 1) for n in _floor_nodes(geom, st)]))
            for st in range(0, geom.num_floors + 1)}


def elastic_storey_drifts_from_loads(*, nodes, elements, geom, E_loads, Ec_Pa, Es_Pa, column_sec,
                                     beam_sec, walls, wall_tw_mm, wall_E_Pa, element_sections,
                                     stiffness_factor, ops_build_model):
    """E-only static analysis on the cracked model -> storey drifts de [m]."""
    from framekit.opensees_model import ops_apply_combo_loads, ops_run_static
    ops_build_model(nodes, elements, Ec_Pa, Es_Pa, column_sec, beam_sec, geom=geom,
                    use_rigid_diaphragm=False, walls=walls, wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa,
                    element_sections=element_sections, stiffness_factor=stiffness_factor)
    ops_apply_combo_loads(nodes, elements, E_loads)
    if ops_run_static() != 0:
        raise RuntimeError("E-only static analysis failed")
    u = _floor_avg_disp_x(geom)
    return {s: u[s] - u[s - 1] for s in range(1, geom.num_floors + 1)}


# ============================================================
# CANDIDATE EVALUATION (retrofit optimizer)
# ============================================================
def evaluate_lateral_force_method(*, nodes, elements, geom, combos, combo_names, Ec_Pa, Es_Pa,
                                  column_sec, beam_sec, walls=None, wall_tw_mm=200.0, wall_E_Pa=30e9,
                                  element_sections=None, settings, ops_build_model, verbose=False):
    """Full EC8 lateral-force-method evaluation of one structural configuration.

    Returns combos/names with SEIS_E± added, T1, Fb, storey table and the
    governing damage-limitation and θ values.
    """
    from framekit.dynamics import ops_assign_floor_masses
    kf = float(settings.get("stiffness_factor", 0.5))
    q = float(settings["q"])
    rp = settings["rsa_params"]

    masses = floor_masses_from_combos(geom, nodes, elements, combos, settings.get("extra_mass_floor_kg", 0.0))
    ops_build_model(nodes, elements, Ec_Pa, Es_Pa, column_sec, beam_sec, geom=geom, use_rigid_diaphragm=False,
                    walls=walls, wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa,
                    element_sections=element_sections, stiffness_factor=kf)
    ops_assign_floor_masses(nodes, geom, masses, master_nodes=None)
    n_modes = max(1, min(3 * geom.num_floors, 6))
    modes = modal_x_properties(geom, n_modes)
    if not modes:
        raise RuntimeError("no valid modes")
    m1 = max(modes, key=lambda d: d["ratio"])
    lfm = lateral_force_method(geom, masses, m1["T"], rp, q, settings.get("beta", 0.2))

    E_loads = floor_forces_to_loads(geom, lfm["forces_kN"])
    de = elastic_storey_drifts_from_loads(
        nodes=nodes, elements=elements, geom=geom, E_loads=E_loads, Ec_Pa=Ec_Pa, Es_Pa=Es_Pa,
        column_sec=column_sec, beam_sec=beam_sec, walls=walls, wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa,
        element_sections=element_sections, stiffness_factor=kf, ops_build_model=ops_build_model)
    tab = storey_checks(geom, masses, lfm["forces_kN"], de, settings)

    theta_max = float(tab["theta"].max())
    amp = 1.0
    if 0.10 < theta_max <= 0.20:
        amp = 1.0 / (1.0 - theta_max)
    elif theta_max > 0.20:
        amp = 1.0 / (1.0 - min(theta_max, 0.5))
    if amp != 1.0:
        E_loads = floor_forces_to_loads(geom, lfm["forces_kN"], factor=amp)

    combos_s, names_s = add_seismic_combos(combos, combo_names, E_loads)
    dls_util = float(tab["DLS utilization"].max())
    if verbose:
        print(f"  EC8 LFM: T1 = {m1['T']:.3f} s, Sd = {lfm['Sd']:.3f} m/s², Fb = {lfm['Fb_kN']:.1f} kN, "
              f"max DLS util = {dls_util:.2f}, θmax = {theta_max:.3f}")
    return {
        "combos": combos_s, "combo_names": names_s, "E_loads": E_loads,
        "T1": m1["T"], "Sd": lfm["Sd"], "Fb_kN": lfm["Fb_kN"] * amp, "lambda": lfm["lambda"],
        "lfm_applicable": lfm["applicable"], "floor_masses": masses, "forces_kN": lfm["forces_kN"],
        "storey_table": tab, "dls_utilization": dls_util,
        "max_dls_drift_ratio": float(tab["ν·ds/h [-]"].max()), "theta_max": theta_max,
        "theta_amplification": amp, "theta_ok": theta_max <= 0.30,
    }


# ============================================================
# CAPACITY DESIGN CHECK (EN 1998-1 only)
# ============================================================
def _column_MRd_at_N(sec, mat, N_kN):
    from framekit.ec2_checks import ec2_NM_interaction_column
    lo, hi = 0.0, 1.0
    while ec2_NM_interaction_column(sec, mat, N_kN, hi)["eta_NM"] < 1.0 and hi < 1e5:
        hi *= 2.0
    if hi >= 1e5:
        return np.nan
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ec2_NM_interaction_column(sec, mat, N_kN, mid)["eta_NM"] < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def strong_column_weak_beam_check(*, nodes, elements, geom, combos, Ec_Pa, Es_Pa, column_sec, beam_sec,
                                  ec2mat, walls=None, wall_tw_mm=200.0, wall_E_Pa=30e9,
                                  element_sections=None, ops_build_model):
    """ΣMRc ≥ 1.3·ΣMRb at every beam-column joint below the roof (EN 1998-1 4.4.2.3(4)).
    Column MRd uses the minimum axial force of the two seismic combinations."""
    from framekit.opensees_model import ops_apply_combo_loads, ops_run_static
    from framekit.ec2_checks import ec2_MRd_rectangular

    def sec_of(eid):
        if element_sections and int(eid) in element_sections:
            return element_sections[int(eid)]
        return column_sec if elements[eid][2] == "column" else beam_sec

    Nmin = {}
    for cname in ("SEIS_E+", "SEIS_E-"):
        if cname not in combos:
            continue
        ops_build_model(nodes, elements, Ec_Pa, Es_Pa, column_sec, beam_sec, geom=geom, walls=walls,
                        wall_tw_mm=wall_tw_mm, wall_E_Pa=wall_E_Pa, element_sections=element_sections)
        ops_apply_combo_loads(nodes, elements, combos[cname]["loads"])
        if ops_run_static() != 0:
            continue
        for eid, (ni, nj, et) in elements.items():
            if et != "column":
                continue
            f = ops.eleResponse(eid, "localForce")
            for nid, Nc in ((ni, f[0] / 1000.0), (nj, -f[3] / 1000.0)):
                key = (eid, nid)
                Nmin[key] = min(Nmin.get(key, np.inf), Nc)

    rows = []
    for nid, (x, y) in nodes.items():
        st = int(round(y / geom.floor_height))
        if st < 1 or st >= geom.num_floors:
            continue
        cols = [e for e, (a, b, t) in elements.items() if t == "column" and nid in (a, b)]
        beams = [e for e, (a, b, t) in elements.items() if t == "beam" and nid in (a, b)]
        if not cols or not beams:
            continue
        MRc = sum(_column_MRd_at_N(sec_of(e), ec2mat, Nmin.get((e, nid), 0.0)) for e in cols)
        mr = []
        for e in beams:
            s = sec_of(e)
            left_side = elements[e][1] == nid    # beam ends at this joint -> beam on the left
            mr.append((left_side, ec2_MRd_rectangular(s, ec2mat, sagging=True),
                       ec2_MRd_rectangular(s, ec2mat, sagging=False)))
        if len(mr) == 1:
            MRb = max(mr[0][1], mr[0][2])
        else:
            l = [m for m in mr if m[0]][0]
            r = [m for m in mr if not m[0]][0]
            MRb = max(l[2] + r[1], l[1] + r[2])
        ratio = MRc / max(1.3 * MRb, 1e-9)
        rows.append({"Node": nid, "Story level": st, "ΣMRc [kNm]": MRc, "ΣMRb [kNm]": MRb,
                     "ΣMRc / (1.3 ΣMRb)": ratio, "Status": "OK" if ratio >= 1.0 else "FAIL"})
    return pd.DataFrame(rows)


def modal_mass_check(modes_table: pd.DataFrame, ratio_col="Mass_ratio"):
    """EN 1998-1 4.3.3.3.1(3): Σ effective modal mass ≥ 90%."""
    if modes_table is None or modes_table.empty or ratio_col not in modes_table:
        return np.nan, False
    tot = float(pd.to_numeric(modes_table[ratio_col], errors="coerce").sum())
    return tot, tot >= 0.90
