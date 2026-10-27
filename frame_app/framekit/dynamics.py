import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openseespy.opensees as ops

from framekit.geometry import node_id_at
from framekit.constants import PSI_Q2, G_STD


# ============================================================
# FLOOR NODE HELPERS
# ============================================================
def _floor_nodes(geom, floor):
    nfp = geom.num_bays + 1
    return [node_id_at(geom, floor, ix) for ix in range(nfp)]


# ============================================================
# MASS LOADSET
# ============================================================
def build_mass_loadset_from_cases(cases, psi_mass_Q=PSI_Q2):
    import copy

    mass_loads = []

    for key in ("SW", "G"):
        mass_loads += cases.get(key, [])

    if abs(psi_mass_Q) > 1e-15 and "Q" in cases:
        for Ld in cases["Q"]:
            Lc = copy.deepcopy(Ld)
            if Lc.load_type == "triangular":
                if Lc.w1 is not None:
                    Lc.w1 *= psi_mass_Q
                if Lc.w2 is not None:
                    Lc.w2 *= psi_mass_Q
            else:
                Lc.value *= psi_mass_Q
            mass_loads.append(Lc)

    return mass_loads


# ============================================================
# ELEMENT LENGTH
# ============================================================
def _element_length_m(nodes, elements, eid):
    ni, nj, _ = elements[eid]
    x1, y1 = nodes[ni]
    x2, y2 = nodes[nj]
    return float(np.hypot(x2 - x1, y2 - y1))


# ============================================================
# BEAM IDS BY LEVEL
# ============================================================
def get_beam_element_ids_by_level(nodes, elements, geom, level):
    y_level = level * geom.floor_height
    out = []

    for eid, (ni, nj, etype) in elements.items():
        if etype != "beam":
            continue
        if abs(nodes[ni][1] - y_level) < 1e-9 and abs(nodes[nj][1] - y_level) < 1e-9:
            out.append(eid)

    return out


# ============================================================
# FLOOR MASSES
# ============================================================
def compute_floor_masses_kg_from_line_loads(
    geom,
    nodes,
    elements,
    loads_for_mass,
    add_lumped_mass_per_floor_kg=0.0
):
    """
    Lumped seismic mass per floor [kg] from the vertical gravity loads.

    Every distributed vertical load on ANY element (beams AND columns) is
    integrated over the element length and lumped 50/50 to its end nodes;
    each half is assigned to the floor of that node. Halves landing on the
    base (y = 0) go to the foundation and are not part of the seismic mass.
    So a storey-k column contributes half its weight to floor k-1 and half
    to floor k (top storey: half to the roof only).
    Vertical point loads are assigned to the floor of their node.
    """
    from framekit.units import _kN_to_N, _kNpm_to_Npm

    floor_masses = {st: 0.0 for st in range(1, geom.num_floors + 1)}
    tol = 1e-6 * max(geom.floor_height, 1.0)

    def _floor_of_y(y):
        st = int(round(y / geom.floor_height))
        if abs(y - st * geom.floor_height) > tol:
            return None
        return st if 1 <= st <= geom.num_floors else None

    for Ld in loads_for_mass:
        if Ld.direction != "vertical":
            continue

        if Ld.load_type == "point":
            if Ld.node is None:
                continue
            st = _floor_of_y(nodes[Ld.node][1])
            if st is not None:
                floor_masses[st] += abs(_kN_to_N(Ld.value)) / G_STD
            continue

        if Ld.element is None or Ld.element not in elements:
            continue

        L_m = _element_length_m(nodes, elements, Ld.element)

        if Ld.load_type == "uniform":
            W_N = abs(_kNpm_to_Npm(Ld.value)) * L_m
        elif Ld.load_type == "triangular":
            w1 = abs(_kNpm_to_Npm(Ld.w1 or 0.0))
            w2 = abs(_kNpm_to_Npm(Ld.w2 or 0.0))
            W_N = 0.5 * (w1 + w2) * L_m
        else:
            continue

        ni, nj, _ = elements[Ld.element]
        for nid in (ni, nj):
            st = _floor_of_y(nodes[nid][1])
            if st is not None:
                floor_masses[st] += 0.5 * W_N / G_STD

    if add_lumped_mass_per_floor_kg > 0.0:
        for st in floor_masses:
            floor_masses[st] += add_lumped_mass_per_floor_kg

    return floor_masses


def print_floor_mass_report(floor_masses_kg):
    tot = sum(floor_masses_kg.values())
    print("\n=== MASS REPORT (lumped by floor) ===")
    for st in sorted(floor_masses_kg.keys()):
        print(f"Floor {st:2d}: {floor_masses_kg[st]:.3f} kg")
    print(f"TOTAL MASS: {tot:.3f} kg")
    return tot


# ============================================================
# ASSIGN MASSES TO OPS MODEL
# ============================================================
def ops_assign_floor_masses(nodes, geom, floor_masses_kg, master_nodes=None, tiny_mass_ratio=1.0e-8):
    """
    Important:
    - Main seismic mass goes in X direction (DOF 1)
    - Tiny stabilizing masses are added in DOF 2 and DOF 3
      to avoid singular/undetermined higher modes in 2D frame eigen analysis
    """
    for st in range(1, geom.num_floors + 1):
        m_floor = float(floor_masses_kg.get(st, 0.0))
        m_tiny = max(m_floor * tiny_mass_ratio, 1.0e-9)

        if master_nodes:
            ops.mass(master_nodes[st], m_floor, m_tiny, m_tiny)
        else:
            ids = _floor_nodes(geom, st)
            n_ids = max(len(ids), 1)
            m_per = m_floor / n_ids
            m_tiny_per = max(m_per * tiny_mass_ratio, 1.0e-9)

            for nid in ids:
                ops.mass(nid, m_per, m_tiny_per, m_tiny_per)


# ============================================================
# MODE SHAPE PLOTS
# ============================================================
def _mode_deformed_lines(nodes, elements, mode_index, scale=None):
    U = {}
    umax = 0.0
    for nid in nodes.keys():
        ux = float(ops.nodeEigenvector(nid, mode_index, 1))
        uy = float(ops.nodeEigenvector(nid, mode_index, 2))
        try:
            rz = float(ops.nodeEigenvector(nid, mode_index, 3))
        except Exception:
            rz = 0.0
        U[nid] = (ux, uy, rz)
        umax = max(umax, abs(ux), abs(uy))

    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    Lref = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    if scale is None:
        scale = 0.12 * Lref / max(umax, 1e-12)

    lines = []
    for eid, (ni, nj, etype) in elements.items():
        x1, y1 = nodes[ni]
        x2, y2 = nodes[nj]
        ux1, uy1, rz1 = U[ni]
        ux2, uy2, rz2 = U[nj]
        dx, dy = x2 - x1, y2 - y1
        L = float(np.hypot(dx, dy))
        if L < 1e-12:
            continue
        c, s = dx / L, dy / L
        u1l, v1l = c * ux1 + s * uy1, -s * ux1 + c * uy1
        u2l, v2l = c * ux2 + s * uy2, -s * ux2 + c * uy2
        xi = np.linspace(0.0, L, 40)
        r = xi / L
        ul = (1.0 - r) * u1l + r * u2l
        N1 = 1.0 - 3.0 * r**2 + 2.0 * r**3
        N2 = L * (r - 2.0 * r**2 + r**3)
        N3 = 3.0 * r**2 - 2.0 * r**3
        N4 = L * (-r**2 + r**3)
        vl = N1 * v1l + N2 * rz1 + N3 * v2l + N4 * rz2
        lines.append((etype,
                      x1 + c * xi + scale * (c * ul - s * vl),
                      y1 + s * xi + scale * (s * ul + c * vl)))
    return lines, scale


def _draw_mode_on_ax(ax, nodes, elements, mode_index, period=None, scale=None, title=None):
    from framekit.plot_style import COLORS, draw_supports, finish_frame_axes
    for eid, (ni, nj, _) in elements.items():
        x1, y1 = nodes[ni]
        x2, y2 = nodes[nj]
        ax.plot([x1, x2], [y1, y2], color=COLORS["undeformed"], lw=1.0, ls=(0, (4, 3)), zorder=1)
    lines, sc = _mode_deformed_lines(nodes, elements, mode_index, scale)
    for etype, xd, yd in lines:
        ax.plot(xd, yd, color=COLORS["column"] if etype == "column" else COLORS["beam"],
                lw=2.4 if etype == "column" else 2.0, zorder=3)
    draw_supports(ax, nodes)
    finish_frame_axes(ax, nodes, pad=0.16)
    t = title or f"Mode {mode_index}"
    if period is not None and np.isfinite(period):
        t += f"\nT = {period:.3f} s   f = {1.0 / period:.2f} Hz"
    ax.set_title(t, loc="center", fontsize=11)
    return sc


def plot_mode_deformed_structure(nodes, elements, mode_index, scale=None, title_prefix="Mode Shape", period=None):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    _draw_mode_on_ax(ax, nodes, elements, mode_index, period=period, scale=scale,
                     title=f"{title_prefix} {mode_index}")
    plt.show()


def plot_modes_grid(nodes, elements, mode_ids, periods, max_cols=4):
    """All mode shapes side by side in one figure."""
    n = len(mode_ids)
    if n == 0:
        return
    ncol = min(max_cols, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 0.5, 3.9 * nrow + 0.6), squeeze=False)
    for k, (mid, T) in enumerate(zip(mode_ids, periods)):
        _draw_mode_on_ax(axes[k // ncol][k % ncol], nodes, elements, mid, period=T)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Mode shapes (eigen analysis)", x=0.01, ha="left")
    plt.show()


# ============================================================
# EIGEN ANALYSIS
# ============================================================
def ops_run_eigen_and_print_and_plot(
    geom,
    nodes,
    elements,
    num_modes=3,
    ref_grid_x=0,
    normalize=True,
    plot_floor_shapes=True,
    plot_structure_each_mode=True
):
    num_modes = int(num_modes)

    try:
        eig_raw = ops.eigen("-fullGenLapack", num_modes)
    except Exception:
        eig_raw = ops.eigen(num_modes)

    if eig_raw is None:
        raise RuntimeError("Eigen analysis failed: ops.eigen returned None")

    if np.isscalar(eig_raw):
        eig_raw = [float(eig_raw)]

    lambdas_raw = np.array(eig_raw, dtype=float)
    if lambdas_raw.size == 0:
        raise RuntimeError("Eigen analysis failed: empty eigenvalue list returned")

    print("\nRaw eigenvalues returned by OpenSees:")
    for i, lam in enumerate(lambdas_raw, start=1):
        print(f"  mode {i}: {lam}")

    valid_mask = np.isfinite(lambdas_raw) & (lambdas_raw > 0.0) & (lambdas_raw < 1.0e100)

    if not np.any(valid_mask):
        raise RuntimeError("Eigen analysis failed: no valid positive finite eigenvalues returned")

    lambdas = lambdas_raw[valid_mask]
    valid_mode_ids = [i + 1 for i, ok in enumerate(valid_mask) if ok]

    invalid_count = int(lambdas_raw.size - lambdas.size)
    if invalid_count > 0:
        print(f"⚠️ Ignored {invalid_count} invalid/undetermined eigenvalues.")

    omegas = np.sqrt(lambdas)
    periods = np.where(omegas > 0.0, 2.0 * np.pi / omegas, np.inf)

    try:
        ops.modalProperties()
    except Exception as e:
        print(f"⚠️ modalProperties() warning: {e}")

    floor_nodes = [node_id_at(geom, st, ref_grid_x) for st in range(1, geom.num_floors + 1)]
    floor_heights = np.array([st * geom.floor_height for st in range(1, geom.num_floors + 1)], dtype=float)

    print("\n=== EIGEN ANALYSIS ===")
    for i, mode_id in enumerate(valid_mode_ids):
        print(f"Mode {mode_id}: lambda={lambdas[i]:.6e}  omega={omegas[i]:.6e} rad/s  T={periods[i]:.6f} s")

    shapes = []
    kept_mode_ids = []
    kept_periods = []

    for i, mode_id in enumerate(valid_mode_ids):
        vec = np.array([ops.nodeEigenvector(nid, mode_id, 1) for nid in floor_nodes], dtype=float)

        if not np.all(np.isfinite(vec)):
            print(f"⚠️ Skipping mode {mode_id}: invalid eigenvector values.")
            continue

        if normalize:
            m = np.max(np.abs(vec))
            if m > 0.0:
                vec = vec / m

        shapes.append(vec)
        kept_mode_ids.append(mode_id)
        kept_periods.append(periods[i])

    print("\nMode shapes (X at one node per floor; normalized to max=1):")
    for mode_id, vec in zip(kept_mode_ids, shapes):
        s = "  ".join([f"{v:+.3f}" for v in vec])
        print(f"Mode {mode_id}: [{s}]")

    if plot_floor_shapes and shapes:
        fig, ax = plt.subplots(figsize=(6.2, 6.8))
        hh = np.r_[0.0, floor_heights]
        for mode_id, Tmode, vec in zip(kept_mode_ids, kept_periods, shapes):
            ax.plot(np.r_[0.0, vec], hh, marker="o", ms=5, mfc="white", mew=1.4,
                    label=f"Mode {mode_id}  (T = {Tmode:.3f} s)")
        ax.axvline(0.0, color="#888888", lw=0.9)
        for h in hh:
            ax.axhline(h, color="#EEEEEE", lw=0.8, zorder=0)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(0, hh[-1] * 1.04 if hh[-1] > 0 else 1)
        ax.set_yticks(hh)
        ax.set_yticklabels([f"{h:.1f}" for h in hh])
        ax.set_xlabel("Normalised lateral mode shape φ (max = 1)")
        ax.set_ylabel("Height [m]")
        ax.set_title("Lateral mode shapes by floor")
        ax.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95)
        plt.show()

    if plot_structure_each_mode:
        plot_modes_grid(nodes, elements, kept_mode_ids, kept_periods)

    return np.array(kept_periods, dtype=float), shapes


# ============================================================
# MODAL PARTICIPATION
# ============================================================
def modal_participation_table(geom, floor_masses_kg, num_modes):
    """Participation from OpenSees using the assembled nodal mass (all nodes, all DOFs)."""
    mp = ops.modalProperties("-return")
    gam = mp["partiFactorMX"]
    meff = mp["partiMassMX"]
    ratio = mp["partiMassRatiosMX"]
    rows = []
    for k in range(min(int(num_modes), len(gam))):
        if not np.isfinite(mp["eigenLambda"][k]) or mp["eigenLambda"][k] <= 0 or mp["eigenLambda"][k] >= 1e100:
            continue
        rows.append([k + 1, gam[k], meff[k], ratio[k] / 100.0])
    df = pd.DataFrame(rows, columns=["Mode", "Gamma", "M_eff_kg", "Mass_ratio"])
    df["Cum_mass_ratio"] = df["Mass_ratio"].cumsum()
    return df
