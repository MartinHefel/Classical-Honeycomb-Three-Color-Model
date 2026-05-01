
# General Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Arc
from copy import copy

# Internal Imports
from main.tools.utils import validate_type
from main.tools.visualization import (
    plt_lattice, setup_matplotlib_style, Colors, 
    DEFAULT_FIGSIZE, STYLE_QUASI_PARTICLES, STYLE_BONDS)
from main.model.unit_cell import HoneycombUnitCell, HoneycombRectangularUnitCell
from main.model.lattice import LatticeFromUnitCell
from main.honeycomb_specific.height_field import get_height_field
from main.honeycomb_specific.helpers import occupation_states_to_spin_states



###################################################################################################
# CONFIG / STYLES
###################################################################################################

PARTITION_COLORS = [Colors.BLUE, Colors.GREEN, Colors.RED]
PARTITION_ZORDER_OFFSET = [1, 0, -1]
STYLE_PLAQUETTES_PARTITIONED = dict(
    zorder=0, edgecolor=Colors.NONE, alpha=0.25, linewidth=0.3)
STYLE_HEIGHT_TEXT = dict(
    fontsize=10, ha="center", va="center")
STYLE_SIC_ARROW = dict(
    zorder=20, width=0.008, headwidth=4, headlength=4, headaxislength=4)
STYLE_LOOP_PATH = dict(
    zorder=20, linewidth=2.)
STYLE_LOOP_ARROW = dict(
    zorder=20, head_width=0.2, head_length=0.2)



###################################################################################################
# LATTICE
###################################################################################################

def plt_HC_lattice(
        lattice:LatticeFromUnitCell,
        scalar_for_sites:np.ndarray=None,
        scalar_for_sites_label:str=r"scalar value sites",
        scalar_for_plaquettes:np.ndarray=None,
        scalar_for_plaquettes_label:str=r"scalar value plaquettes",
        m_ref:int=None,
        ax:Axes=None,
        xlim:list[float, float]=None, ylim:list[float, float]=None,
        # shows
        show_bonds:bool=True, show_plaquettes:bool=True, show_sites:bool=True,
        show_site_ids:bool=False, show_plaquette_ids:bool=False,
        show_dual_lattice_bonds:bool=False, show_dual_lattice_sites:bool=False,
        show_scalar_for_sites:bool=True, show_scalar_for_plaquettes:bool=True,
        show_scalar_for_sites_cbar:bool=True, show_scalar_for_plaquettes_cbar:bool=True,
        # Honeycomb shows
        show_bond_partition_color:bool=False,
        show_quasi_particle_color:bool=False,
        partitions_to_show_loop_model_for:list[int]=None,
        partitions_to_show_pl_part_color:list[int]=None,
        HF_partitions_to_show_height_field_for:list[int]=None,
        HF_initial_ref_plaquette:int=0,
        SIC_partitions_to_show_configuration_for:list[int]=None,
        SIC_arrow_scaling:float=1., SIC_arrow_offset:float=0, SIC_scalar_is_occupation:bool=True,
        SIC_combine_arrows:bool=False, SIC_rescale_arrow_ratio_for_clarity:bool=False
) -> Axes:
    validate_type(lattice, LatticeFromUnitCell, "lattice")
    assert (
        isinstance(lattice.unit_cell, HoneycombUnitCell) or 
        isinstance(lattice.unit_cell, HoneycombRectangularUnitCell))

    setup_matplotlib_style()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FIGSIZE)
        plot_directly = True
    else:
        plot_directly = False
    
    if partitions_to_show_pl_part_color is not None:
        assert len(partitions_to_show_pl_part_color) <= 3
        assert all(x in [0,1,2] for x in partitions_to_show_pl_part_color)
        show_plaquettes = False
    if show_bond_partition_color:
        show_bonds = False
    if (partitions_to_show_loop_model_for is not None 
        or HF_partitions_to_show_height_field_for is not None):
        if partitions_to_show_loop_model_for is not None:
            assert len(partitions_to_show_loop_model_for) <= 3
            assert all(x in [0,1,2] for x in partitions_to_show_loop_model_for)
        if HF_partitions_to_show_height_field_for is not None:
            assert len(HF_partitions_to_show_height_field_for) <= 3
            assert all(x in [0,1,2] for x in HF_partitions_to_show_height_field_for)
        assert SIC_scalar_is_occupation
        assert scalar_for_sites is not None
    if show_quasi_particle_color:
        assert m_ref is not None
        assert scalar_for_sites is not None
        assert np.all((scalar_for_sites == 0) | (scalar_for_sites == 1))
        quasi_particle_state = np.zeros((lattice.plaquettes.N,), dtype=np.int8)
        for pl_id in range(lattice.plaquettes.N):
            site_ids = lattice.plaquettes.site_ids[pl_id].entries
            q = np.sum(scalar_for_sites[site_ids])-m_ref
            quasi_particle_state[pl_id] = q
    else:
        quasi_particle_state = None
    if SIC_partitions_to_show_configuration_for is not None:
        assert len(SIC_partitions_to_show_configuration_for) <= 3
        assert all(x in [0,1,2] for x in SIC_partitions_to_show_configuration_for)
        assert scalar_for_sites is not None
        if SIC_scalar_is_occupation:
            assert m_ref is not None
            assert m_ref in [1,2,3]

    eta = 1

    ax = plt_lattice(
        lattice, scalar_for_sites, scalar_for_sites_label, scalar_for_plaquettes, 
        scalar_for_plaquettes_label, m_ref, ax, xlim, ylim, show_bonds, show_plaquettes, 
        show_sites, show_site_ids, show_plaquette_ids, show_dual_lattice_bonds, 
        show_dual_lattice_sites, show_scalar_for_sites, show_scalar_for_plaquettes,
        show_scalar_for_sites_cbar, show_scalar_for_plaquettes_cbar)

    # Plaquette partition color -------------------------------------------------------------------
    if partitions_to_show_pl_part_color:
        partition_to_pl_ids = lattice.unit_cell.get_partition_to_pl_ids(
            lattice.plaquettes.N, lattice.L1)
        for part in range(3):
            if part in partitions_to_show_pl_part_color:
                for pl_id in partition_to_pl_ids[part]:
                    poly_pts = lattice.plaquettes.coords[pl_id]
                    ax.fill(
                        poly_pts[:, 0], poly_pts[:, 1], facecolor=PARTITION_COLORS[part], 
                        **STYLE_PLAQUETTES_PARTITIONED)

    # Bond partition color ------------------------------------------------------------------------
    if show_bond_partition_color:
        STYLE_BONDS_NO_COLOR = {k: v for k, v in STYLE_BONDS.items() if k != "color"}
        pl_id_to_partition = lattice.unit_cell.get_pl_id_to_partition(
            lattice.plaquettes.N, lattice.L1)
        _, connecting_site_ids, _ = lattice.get_monte_carlo_worm_network()
        for pl_id, conn_site_ids in enumerate(connecting_site_ids):
                part = pl_id_to_partition[pl_id]
                for conn_id in range(len(conn_site_ids)):
                    site_id_A, site_id_B = conn_site_ids[conn_id].entries
                    xA, yA = lattice.sites.coords[site_id_A]
                    xB, yB = lattice.sites.coords[site_id_B]
                    sx_A, sx_B = conn_site_ids[conn_id].x_shifts
                    sy_A, sy_B = conn_site_ids[conn_id].y_shifts
                    R_shift_A = lattice.sites.get_shift_vector(sx_A, sy_A)
                    R_shift_B = lattice.sites.get_shift_vector(sx_B, sy_B)
                    xA_shifted = xA + R_shift_A[0]
                    yA_shifted = yA + R_shift_A[1]
                    xB_shifted = xB + R_shift_B[0]
                    yB_shifted = yB + R_shift_B[1]
                    ax.plot(
                        [xA_shifted, xB_shifted], [yA_shifted, yB_shifted], 
                        color=PARTITION_COLORS[part], **STYLE_BONDS_NO_COLOR)

    # Color quasi particles on plaquettes ---------------------------------------------------------
    if quasi_particle_state is not None:
        zorder_facecolor = STYLE_QUASI_PARTICLES["zorder"]-1
        for pl_id, coord in enumerate(lattice.plaquettes.center_coords):
            x, y = coord
            part = lattice.unit_cell.get_plaquette_partition(pl_id, lattice.L1)
            q = quasi_particle_state[pl_id]
            if q == 0:
                continue
            num_of_rings = int(np.abs(q))
            j = num_of_rings-1
            style = copy(STYLE_QUASI_PARTICLES)
            style["radius"] *= 1+j/3
            style["edgecolor"] = "none"
            style["facecolor"] = PARTITION_COLORS[part]
            style["alpha"] = 0.5
            style["zorder"] = zorder_facecolor
            circle = plt.Circle((x, y), **style)
            ax.add_patch(circle)

    # Loop model ----------------------------------------------------------------------------------
    if partitions_to_show_loop_model_for is not None:
        part_to_other_part = [
            [2, 1],
            [0, 2],
            [1, 0]]
        style_loop_path = STYLE_LOOP_PATH.copy()
        zorder_loop_path = style_loop_path.pop("zorder")
        style_loop_arrow = STYLE_LOOP_ARROW.copy()
        zorder_loop_arrow = style_loop_arrow.pop("zorder")
        for pl_id, pl_obj in enumerate(lattice.plaquettes.site_ids):
            if quasi_particle_state is not None:
                q = quasi_particle_state[pl_id]
                draw_loops = (q == 0)
            else:
                draw_loops = True
            if draw_loops:
                pl_site_ids:list[int] = pl_obj.entries
                part = lattice.unit_cell.get_plaquette_partition(pl_id, lattice.L1)
                pl_coords = lattice.plaquettes.coords[pl_id]
                for i in range(6):
                    a1, a2 = (i)%6, (i+1)%6
                    b1, b2 = (i+2)%6, (i+3)%6
                    id_a1, id_a2 = pl_site_ids[a1], pl_site_ids[a2]
                    id_b1, id_b2 = pl_site_ids[b1], pl_site_ids[b2]
                    if (scalar_for_sites[id_a1] == scalar_for_sites[id_a2] and 
                        scalar_for_sites[id_b1] == scalar_for_sites[id_b2]):

                        state_a, state_b = scalar_for_sites[id_a1], scalar_for_sites[id_b1]
                        coord_a1, coord_a2 = pl_coords[a1], pl_coords[a2]
                        coord_b1, coord_b2 = pl_coords[b1], pl_coords[b2]
                        xy, phi1, phi2, alpha, r = get_arc(coord_a1, coord_a2, coord_b1, coord_b2)
                        other_part = part_to_other_part[part][i%2]
                        if other_part in partitions_to_show_loop_model_for:
                            color = PARTITION_COLORS[other_part]
                            zorder = zorder_loop_path+PARTITION_ZORDER_OFFSET[other_part]

                            # Draw Arc
                            arc = Arc(
                                xy, 2*r, 2*r, angle=0, theta1=phi1, theta2=phi2, color=color,
                                zorder=zorder,**style_loop_path)
                            ax.add_patch(arc)

                            # Draw Arrow
                            if state_a != state_b:
                                zorder = zorder_loop_arrow+PARTITION_ZORDER_OFFSET[other_part]
                                theta_mid = np.deg2rad(phi1 - alpha/2)
                                x_mid = xy[0] + r * np.cos(theta_mid)
                                y_mid = xy[1] + r * np.sin(theta_mid)
                                theta_arrow = theta_mid+np.pi/2+np.pi

                                if eta == 1:
                                    clockwise = False
                                else:
                                    clockwise = True
                                if state_a == 0: # arrows point unocc.-> occ. (occ.-> unocc.)
                                    clockwise = not clockwise
                                if i%2 == 1: # arrows should point consistently along the loop
                                    clockwise = not clockwise
                                if not clockwise:
                                    theta_arrow -= np.pi
                                dx = np.cos(theta_arrow)
                                dy = np.sin(theta_arrow)
                                arrow_length = 0.01
                                dx *= arrow_length
                                dy *= arrow_length
                                ax.arrow(
                                    x_mid, y_mid, dx, dy, fc=color, ec=color, zorder=zorder,
                                    **style_loop_arrow)
    
    # Height field --------------------------------------------------------------------------------
    if HF_partitions_to_show_height_field_for is not None:
        pl_id_to_height = get_height_field(
            lattice, scalar_for_sites, m_ref, 
            initial_ref_plaquette=HF_initial_ref_plaquette)
        for pl_id, coord in enumerate(lattice.plaquettes.center_coords):
            x, y = coord
            part = lattice.unit_cell.get_plaquette_partition(pl_id, lattice.L1)
            h=pl_id_to_height[pl_id]
            if part in HF_partitions_to_show_height_field_for:
                ax.text(
                    x, y, fr"${h}$", color=PARTITION_COLORS[part], 
                    **STYLE_HEIGHT_TEXT).set_clip_on(True)

    # Spin Ice configuration ----------------------------------------------------------------------
    if SIC_partitions_to_show_configuration_for is not None:
        if SIC_scalar_is_occupation:
            if SIC_rescale_arrow_ratio_for_clarity:
                scalar_for_arrows = occupation_states_to_spin_states(scalar_for_sites, m_ref=2)
            else:
                scalar_for_arrows = occupation_states_to_spin_states(scalar_for_sites, m_ref)
        else:
            scalar_for_arrows = scalar_for_sites
        partition_to_pl_ids = lattice.unit_cell.get_partition_to_pl_ids(
            lattice.plaquettes.N, lattice.L1)
        arrow_convention = [eta, eta, eta]
        arrow_angles = [-2*np.pi/3, 0, 2*np.pi/3]
        for part in SIC_partitions_to_show_configuration_for:
            f_conv = arrow_convention[part]
            part_for_arrow_reference = (part+1)%3
            for pl_id in partition_to_pl_ids[part_for_arrow_reference]:
                site_ids = lattice.plaquettes.site_ids[int(pl_id)].entries
                if SIC_combine_arrows:
                    num_of_is = 3
                else:
                    num_of_is = 6
                for i in range(num_of_is):
                    if SIC_combine_arrows:
                        site_id_A, site_id_B = site_ids[2*i], site_ids[2*i+1]
                        xA, yA = lattice.sites.coords[site_id_A]
                        xB, yB = lattice.sites.coords[site_id_B]
                        A_neighs = lattice.sites.nearest_neighbors[site_id_A]
                        B_ind = A_neighs.entries.index(site_id_B)
                        sx = A_neighs.x_shifts[B_ind]
                        sy = A_neighs.y_shifts[B_ind]
                        R_shift = lattice.sites.get_shift_vector(sx, sy)
                        xB = xB + R_shift[0]
                        yB = yB + R_shift[1]
                        x, y = (xA+xB)/2, (yA+yB)/2
                        sA, sB = scalar_for_arrows[site_id_A], scalar_for_arrows[site_id_B]
                        s = (sA+sB)*f_conv*SIC_arrow_scaling
                        angle = arrow_angles[i]
                    else:
                        site_id = site_ids[i]
                        x, y = lattice.sites.coords[site_id]
                        s = scalar_for_arrows[site_id]*f_conv*SIC_arrow_scaling
                        angle = arrow_angles[i//2]
                    x += SIC_arrow_offset*np.cos(angle+np.pi/2*(-1)**(i%2))
                    y += SIC_arrow_offset*np.sin(angle+np.pi/2*(-1)**(i%2))
                    dx = s * np.cos(angle)
                    dy = s * np.sin(angle)
                    x_start = x - dx / 2
                    y_start = y - dy / 2
                    if abs(s) > 1e-10:
                        ax.quiver(
                            x_start, y_start, dx, dy,
                            angles='xy', scale_units='xy', scale=1, color=PARTITION_COLORS[part],
                            **STYLE_SIC_ARROW)

    # Layout --------------------------------------------------------------------------------------
    plt.tight_layout()
    if plot_directly:
        plt.show()
    return ax



###################################################################################################
# HELPERS
###################################################################################################

def get_arc(coord_a1, coord_a2, coord_b1, coord_b2):
    """
    Compute intersection, starting angle (phi), and smaller angle (alpha)
    between lines (a1,a2) and (b1,b2).

    Returns
    -------
    coord_intersection : np.ndarray
        (x, y) coordinates of intersection point
    phi1, phi2 : float
        Start and end angles in degrees (counterclockwise from +x, [0, 360)).
        Always ordered so the arc corresponds to the smaller angle.
    alpha : float
        The smaller angle between the lines (in radians).
    """
    x1, y1 = coord_a1
    x2, y2 = coord_a2
    x3, y3 = coord_b1
    x4, y4 = coord_b2

    # Intersection
    a = np.array([[x2 - x1, x3 - x4],
                  [y2 - y1, y3 - y4]])
    b = np.array([x3 - x1, y3 - y1])
    t, u = np.linalg.solve(a, b)
    coord_intersection = np.array([x1 + t*(x2 - x1), y1 + t*(y2 - y1)])

    # Direction vectors
    v1 = (coord_a1+coord_a2)/2-coord_intersection
    v2 = (coord_b1+coord_b2)/2-coord_intersection
    r_1, r_2 = np.linalg.norm(v1), np.linalg.norm(v2)
    assert (r_1 - r_2) < 1e-12
    v1 /= r_1
    v2 /= r_2

    # Angles in degrees, counterclockwise from +x
    phi1 = np.degrees(np.arctan2(v1[1], v1[0]))%360
    phi2 = np.degrees(np.arctan2(v2[1], v2[0]))%360

    diff = (phi2 - phi1) % 360
    if diff > 180:
        diff -= 360
    alpha = diff

    if diff < 0:
        phi1, phi2 = phi2, phi1

    return coord_intersection, phi1, phi2, alpha, r_1
