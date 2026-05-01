
# General Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Arc
from copy import copy

# Internal Imports
from main.tools.utils import validate_type
from main.tools.visualization import plt_lattice, setup_matplotlib_style, Colors, DEFAULT_FIGSIZE
from main.model.unit_cell import SquareSpinIceUnitCell
from main.model.lattice import LatticeFromUnitCell
from main.spin_ice_specific.height_field import get_height_field



###################################################################################################
# CONFIG / STYLES
###################################################################################################
SI_COLOR = Colors.BLUE
STYLE_HEIGHT_TEXT = dict(
    fontsize=10, ha="center", va="center", color=SI_COLOR)
STYLE_VERTICIES = dict(
    zorder=9, linewidth=0.75, linestyle="--", color=Colors.BLACK)
STYLE_SIC_ARROW = dict(
    zorder=20, width=0.005, headwidth=4, headlength=4, headaxislength=4, color=SI_COLOR)
STYLE_LOOP_PATH_PRIMARY = dict(
    zorder=20, linewidth=3, linestyle="-", color=SI_COLOR)
STYLE_LOOP_PATH_SECONDARY = dict(
    zorder=20, linewidth=0.5, linestyle="--", color=SI_COLOR)



###################################################################################################
# LATTICE
###################################################################################################

def plt_SI_lattice(
        lattice:LatticeFromUnitCell,
        scalar_for_sites:np.ndarray=None,
        scalar_for_sites_label:str=r"scalar value sites",
        scalar_for_plaquettes:np.ndarray=None,
        scalar_for_plaquettes_label:str=r"scalar value plaquettes",
        m_ref:int=None,
        ax:Axes=None,
        xlim=None,
        ylim=None,
        # shows
        show_bonds:bool=True, show_plaquettes:bool=True, show_sites:bool=True,
        show_site_ids:bool=False, show_plaquette_ids:bool=False,
        show_dual_lattice_bonds:bool=False, show_dual_lattice_sites:bool=False,
        # Spin Ice shows
        show_verticies:bool=False,
        show_loop_model:bool=False,
        show_height_field:bool=False,
        SIC_show_configuration:bool=False,
        SIC_arrow_scaling:float=1., SIC_scalar_is_occupation:bool=True,
) -> Axes:
    validate_type(lattice, LatticeFromUnitCell, "lattice")
    assert isinstance(lattice.unit_cell, SquareSpinIceUnitCell)

    setup_matplotlib_style()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FIGSIZE)
        plot_directly = True
    else:
        plot_directly = False
    

    if show_loop_model or show_height_field:
        assert SIC_scalar_is_occupation
        assert scalar_for_sites is not None
    if SIC_show_configuration:
        assert scalar_for_sites is not None

    ax = plt_lattice(
        lattice, scalar_for_sites, scalar_for_sites_label, scalar_for_plaquettes, 
        scalar_for_plaquettes_label, m_ref, ax, xlim, ylim, show_bonds, show_plaquettes, show_sites, 
        show_site_ids, show_plaquette_ids, show_dual_lattice_bonds, show_dual_lattice_sites)

    # Verticies -----------------------------------------------------------------------------------
    if show_verticies:
        for pl_id, site_ids in enumerate(lattice.plaquettes.site_ids):
            pl_site_ids:list[int] = site_ids.entries
            xy_cen = lattice.plaquettes.center_coords[pl_id]
            for i in range(4):
                state = scalar_for_sites[pl_site_ids[i]]
                x, y = lattice.plaquettes.coords[pl_id][i]
                ax.plot( 
                    [xy_cen[0],x],[xy_cen[1],y], **STYLE_VERTICIES)

    # Loop model ----------------------------------------------------------------------------------
    if show_loop_model:
        for pl_id, site_ids in enumerate(lattice.plaquettes.site_ids):
            pl_site_ids:list[int] = site_ids.entries
            xy_cen = lattice.plaquettes.center_coords[pl_id]
            for i in range(4):
                state = scalar_for_sites[pl_site_ids[i]]
                x, y = lattice.plaquettes.coords[pl_id][i]
                if state == 1:
                    ax.plot( 
                        [xy_cen[0],x],[xy_cen[1],y], **STYLE_LOOP_PATH_PRIMARY)
                else:
                    ax.plot( 
                        [xy_cen[0],x],[xy_cen[1],y], **STYLE_LOOP_PATH_SECONDARY)
    
    # Height field --------------------------------------------------------------------------------
    if show_height_field:
        pl_id_B_to_height = get_height_field(lattice, scalar_for_sites)
        # go through each unit cell
        for uc_id in range(lattice.L1*lattice.L2):
            x, y = lattice.plaquettes.center_coords[2*uc_id] # pl_id_A1 = 2*uc_id
            pl_id_B1_coords = x, y+1
            pl_id_B2_coords = x+1, y
            h_pl_id_B1 =pl_id_B_to_height[2*uc_id]
            h_pl_id_B2 =pl_id_B_to_height[2*uc_id+1]
            ax.text(
                pl_id_B1_coords[0], pl_id_B1_coords[1], fr"${h_pl_id_B1}$",
                **STYLE_HEIGHT_TEXT).set_clip_on(True)
            ax.text(
                pl_id_B2_coords[0], pl_id_B2_coords[1], fr"${h_pl_id_B2}$",
                **STYLE_HEIGHT_TEXT).set_clip_on(True)

    # Spin Ice configuration ----------------------------------------------------------------------
    if SIC_show_configuration:
        if SIC_scalar_is_occupation:
            scalar_for_arrows = 2*scalar_for_sites-1
        else:
            scalar_for_arrows = scalar_for_sites
        f_conv = -1
        arrow_angles = [np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4]

        for uc_id in range(lattice.L1*lattice.L2):
            pl_id_A2 = 2*uc_id+1
            site_ids = lattice.plaquettes.site_ids[int(pl_id_A2)].entries
            for i in range(4):
                site_id = site_ids[i]
                x, y = lattice.sites.coords[site_id]
                s = scalar_for_arrows[site_id]*f_conv*SIC_arrow_scaling
                angle = arrow_angles[i]
                dx = s * np.cos(angle)
                dy = s * np.sin(angle)
                x_start = x - dx / 2
                y_start = y - dy / 2
                if abs(s) > 1e-10:
                    ax.quiver(
                        x_start, y_start, dx, dy,
                        angles='xy', scale_units='xy', scale=1, **STYLE_SIC_ARROW)

    # Layout --------------------------------------------------------------------------------------
    plt.tight_layout()
    if plot_directly:
        plt.show()
    return ax
