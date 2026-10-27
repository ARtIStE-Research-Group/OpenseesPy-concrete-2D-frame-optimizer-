import math
import numpy as np
import openseespy.opensees as ops

from framekit.geometry import node_id_at
from framekit.units import _mm_to_m, _MPa_to_Pa, _kN_to_N, _kNpm_to_Npm


# ============================================================
# SECTION FIELD COMPATIBILITY HELPERS
# ============================================================
def _get_sec_attr(sec, *names):
    for name in names:
        if hasattr(sec, name):
            return getattr(sec, name)
    raise AttributeError(f"Section object missing attributes. Tried: {names}")


# ============================================================
# SECTION PROPERTIES (elastic transformed)
# returns A [m^2], I [m^4]
# ============================================================
def section_properties_from_fiber(sec, Ec_Pa, Es_Pa):
    b = _mm_to_m(_get_sec_attr(sec, "b", "b_mm"))
    h = _mm_to_m(_get_sec_attr(sec, "h", "h_mm"))
    cover = _mm_to_m(_get_sec_attr(sec, "cover", "cover_mm"))

    top_bars = int(_get_sec_attr(sec, "top_bars", "n_top"))
    bottom_bars = int(_get_sec_attr(sec, "bottom_bars", "n_bot"))

    top_bar_d = _mm_to_m(_get_sec_attr(sec, "top_bar_d", "phi_top_mm"))
    bottom_bar_d = _mm_to_m(_get_sec_attr(sec, "bottom_bar_d", "phi_bot_mm"))

    Ac = b * h
    Ic = b * h**3 / 12.0

    As_top = top_bars * math.pi * (top_bar_d ** 2) / 4.0
    As_bot = bottom_bars * math.pi * (bottom_bar_d ** 2) / 4.0

    y_top = h / 2.0 - cover
    y_bot = -h / 2.0 + cover

    n = Es_Pa / Ec_Pa
    A_eq = Ac + n * (As_top + As_bot)
    I_eq = Ic + n * (As_top * y_top**2 + As_bot * y_bot**2)

    # jacket cage (if any)
    t_j = float(getattr(sec, "jacket_t_mm", 0.0) or 0.0)
    phi_j = float(getattr(sec, "jphi_mm", 0.0) or 0.0)
    if t_j > 0.0 and phi_j > 0.0:
        a_j = math.pi * _mm_to_m(phi_j) ** 2 / 4.0
        e_j = _mm_to_m(float(getattr(sec, "jacket_cover_mm", 30.0)) + float(getattr(sec, "jacket_stirrup_phi_mm", 8.0)) + 0.5 * phi_j)
        y_j = h / 2.0 - e_j
        n_tb = int(getattr(sec, "jn_top", 0)) + int(getattr(sec, "jn_bot", 0))
        n_sd = int(getattr(sec, "jn_left", 0)) + int(getattr(sec, "jn_right", 0))
        A_eq += n * a_j * (n_tb + n_sd)
        I_eq += n * a_j * n_tb * y_j ** 2
    return A_eq, I_eq


# ============================================================
# BASIC HELPERS
# ============================================================
def _elem_cos_sin(nodes, elements, eid):
    ni, nj, _ = elements[eid]
    x1, y1 = nodes[ni]
    x2, y2 = nodes[nj]
    L = math.hypot(x2 - x1, y2 - y1)

    if L < 1e-12:
        return 0.0, 1.0, 0.0

    c = (x2 - x1) / L
    s = (y2 - y1) / L
    return L, c, s


def _floor_nodes(geom, floor):
    nfp = geom.num_bays + 1
    return [node_id_at(geom, floor, ix) for ix in range(nfp)]


def _floor_x_center(nodes, floor_node_ids):
    xs = [nodes[nid][0] for nid in floor_node_ids]
    return float(sum(xs) / max(len(xs), 1))


# ============================================================
# SHEAR WALL EQUIVALENT BRACES
# ============================================================
def _shear_wall_brace_properties(h_story, L_bay, tw_mm, E_wall_Pa):
    """
    Compute equivalent brace area [m2] for two diagonal braces
    representing one wall panel.
    """
    tw_m = tw_mm / 1000.0
    nu = 0.20
    G_wall = E_wall_Pa / (2.0 * (1.0 + nu))

    # Shear stiffness of wall panel
    K_wall = G_wall * L_bay * tw_m / h_story

    Ld_m = math.hypot(h_story, L_bay)
    cos_th = L_bay / Ld_m

    # Two braces contribute
    Ab_m2 = K_wall * Ld_m / (2.0 * E_wall_Pa * cos_th**2)

    return Ab_m2, Ld_m


# ============================================================
# MODEL BUILD
# ============================================================
def ops_build_model(
    nodes,
    elements,
    Ec_Pa,
    Es_Pa,
    column_sec,
    beam_sec,
    geom=None,
    use_rigid_diaphragm=False,
    walls=None,
    wall_tw_mm=200.0,
    wall_E_Pa=30e9,
    element_sections=None,
    stiffness_factor=1.0,
):
    """Build the 2D OpenSees model.

    stiffness_factor multiplies the flexural stiffness EI of all members and the
    shear stiffness of the wall panels (EC8 4.3.1(7): 0.5 for cracked RC in the
    seismic analyses). Axial stiffness is not reduced.
    """
    kf = float(stiffness_factor) if stiffness_factor else 1.0
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    ops.geomTransf("Linear", 1)

    # Nodes and supports
    for nid, (x, y) in nodes.items():
        ops.node(nid, x, y)
        if abs(y) < 1e-12:
            ops.fix(nid, 1, 1, 1)

    # Frame elements
    for eid, (ni, nj, etype) in elements.items():
        if element_sections is not None and int(eid) in element_sections:
            sec = element_sections[int(eid)]
        else:
            sec = column_sec if etype == "column" else beam_sec
        A, I = section_properties_from_fiber(sec, Ec_Pa, Es_Pa)
        ops.element("elasticBeamColumn", eid, ni, nj, A, Ec_Pa, kf * I, 1)

    # Shear walls as equivalent diagonal braces
    if walls and geom is not None:
        ops.geomTransf("Linear", 2)

        next_brace_id = max(elements.keys()) + 1

        for w in walls:
            n1 = node_id_at(geom, w.floor_bot, w.bay_left - 1)   # bottom-left
            n2 = node_id_at(geom, w.floor_bot, w.bay_right - 1)  # bottom-right
            n3 = node_id_at(geom, w.floor_top, w.bay_right - 1)  # top-right
            n4 = node_id_at(geom, w.floor_top, w.bay_left - 1)   # top-left

            x1, y1 = nodes[n1]
            x2, _ = nodes[n2]
            _, y4 = nodes[n4]

            L_bay = abs(x2 - x1)
            h_story = abs(y4 - y1)

            tw_this = float(getattr(w, "tw_mm", None) or wall_tw_mm)
            Ab_m2, _ = _shear_wall_brace_properties(h_story, L_bay, tw_this, wall_E_Pa)
            Ab_m2 *= kf

            Ib_m4 = 1e-10

            ops.element("elasticBeamColumn", next_brace_id, n1, n3, Ab_m2, wall_E_Pa, Ib_m4, 2)
            next_brace_id += 1

            ops.element("elasticBeamColumn", next_brace_id, n2, n4, Ab_m2, wall_E_Pa, Ib_m4, 2)
            next_brace_id += 1

        print(
            f"✔ Shear walls modeled: {len(walls)} panel(s), "
            f"{2*len(walls)} equivalent diagonal brace elements added."
        )

    # Optional rigid diaphragm
    master_nodes = {}
    if use_rigid_diaphragm:
        if geom is None:
            raise ValueError("geom must be provided when use_rigid_diaphragm=True")

        next_id = max(nodes.keys()) + 1

        for st in range(1, geom.num_floors + 1):
            fnodes = _floor_nodes(geom, st)
            xM = _floor_x_center(nodes, fnodes)
            yM = st * geom.floor_height

            mid = next_id
            next_id += 1
            master_nodes[st] = mid

            ops.node(mid, xM, yM)
            ops.fix(mid, 0, 1, 1)

            for nid in fnodes:
                ops.equalDOF(mid, nid, 1)

    return master_nodes


# ============================================================
# LOAD APPLICATION
# ============================================================
def ops_apply_combo_loads(nodes, elements, loads_combo, pattern_tag=1, ts_tag=1):
    ops.timeSeries("Linear", ts_tag)
    ops.pattern("Plain", pattern_tag, ts_tag)

    for Ld in loads_combo:
        # Point load
        if Ld.load_type == "point" and Ld.node is not None:
            fx = _kN_to_N(Ld.value) if Ld.direction == "horizontal" else 0.0
            fy = _kN_to_N(Ld.value) if Ld.direction == "vertical" else 0.0
            ops.load(Ld.node, fx, fy, 0.0)
            continue

        if Ld.element is None:
            continue

        _, c, s = _elem_cos_sin(nodes, elements, Ld.element)

        # Uniform load
        if Ld.load_type == "uniform":
            if Ld.direction == "vertical":
                qx_g, qy_g = 0.0, _kNpm_to_Npm(Ld.value)
            else:
                qx_g, qy_g = _kNpm_to_Npm(Ld.value), 0.0

            wx = c * qx_g + s * qy_g
            wy = -s * qx_g + c * qy_g

            ops.eleLoad("-ele", Ld.element, "-type", "-beamUniform", wy, wx)

        # Triangular load
        elif Ld.load_type == "triangular":
            w1g = _kNpm_to_Npm(Ld.w1 if Ld.w1 is not None else 0.0)
            w2g = _kNpm_to_Npm(Ld.w2 if Ld.w2 is not None else 0.0)

            if Ld.direction == "vertical":
                qx1_g, qy1_g = 0.0, w1g
                qx2_g, qy2_g = 0.0, w2g
            else:
                qx1_g, qy1_g = w1g, 0.0
                qx2_g, qy2_g = w2g, 0.0

            wy1 = -s * qx1_g + c * qy1_g
            wy2 = -s * qx2_g + c * qy2_g
            wx1 = c * qx1_g + s * qy1_g
            wx2 = c * qx2_g + s * qy2_g

            # Linearly varying load over the full length (Beam2dPartialUniformLoad):
            # -beamUniform Wya Wxa aOverL bOverL Wyb Wxb
            ops.eleLoad("-ele", Ld.element, "-type", "-beamUniform",
                        wy1, wx1, 0.0, 1.0, wy2, wx2)


# ============================================================
# STATIC ANALYSIS
# ============================================================
def ops_run_static():
    ops.system("ProfileSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.test("NormDispIncr", 1e-10, 50, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    return ops.analyze(1)


def get_node_disps_from_ops(nodes):
    return {
        nid: (
            ops.nodeDisp(nid, 1),
            ops.nodeDisp(nid, 2),
            ops.nodeDisp(nid, 3)
        )
        for nid in nodes
    }


def ops_ele_end_forces_local(eid):
    return np.array(ops.eleResponse(eid, "localForce"), dtype=float)