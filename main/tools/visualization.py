
# General Imports
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from enum import Enum
from copy import copy

# Internal Imports
from main.tools.utils import validate_type
from main.model.lattice import Lattice

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None



###################################################################################################
# CONFIG / STYLES
###################################################################################################

def setup_matplotlib_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "text.latex.preamble": r"\usepackage{amsmath}"})

class Colors(str, Enum):
    WHITE = "white"
    BLACK = "black"
    BLUE = "tab:blue"
    GREEN = "tab:green"
    PURPLE = "purple"
    RED = "tab:red"
    DARKGRAY = "gray"
    GRAY = "darkgray"
    LIGHTGRAY = "lightgray"
    NONE = "none"

ID_LABEL_OFFSET = 0.20
DEFAULT_FIGSIZE = (6,6)
DEFAULT_FIGSIZE_STRUCTURE_FACTOR = (5,4)

# Lattice plot styles
STYLE_PLAQUETTES = dict(
    zorder=0, facecolor=Colors.GRAY, edgecolor=Colors.NONE, alpha=0.25, linewidth=0.3)
STYLE_BONDS_DUAL_LATTICE = dict(
    zorder=1, color=Colors.GRAY, alpha=1.0, linewidth=1, linestyle="--", dashes=(4, 2), 
    solid_capstyle="round")
STYLE_SITES_DUAL_LATTICE = dict(
    zorder=2, color=Colors.LIGHTGRAY, edgecolor=Colors.DARKGRAY, alpha=1.0, s=30)
STYLE_BONDS = dict(
    zorder=3, color=Colors.BLACK, alpha=1.0, linewidth=0.75, solid_capstyle="round")
STYLE_SITES = dict(
    zorder=4, color=Colors.WHITE, edgecolor=Colors.BLACK, alpha=1.0, s=30)
STYLE_ID_TEXT = dict(
    fontsize=10, ha="center", va="center", color=Colors.BLACK)
STYLE_SCALAR_CMAP = ["binary", "viridis", "bwr"] # [binary, only pos. or neg., pos. & neg.]
STYLE_SCALAR_FOR_SITES = dict(
    zorder=10, edgecolors=Colors.BLACK, linewidths=0.5, s=30)
STYLE_SCALAR_FOR_PLAQUETTES = dict(
    zorder=10, edgecolors=Colors.BLACK, linewidths=0.5, s=30)
STYLE_QUASI_PARTICLES = dict(
    zorder=10, facecolor="none", edgecolor=Colors.BLACK, linewidth=1.0, radius=0.15)
QUASI_PARTICLE_SIGN_SCALE = 0.5

# Structure factor styles
STYLE_STRUCTURE_FACTOR_MESH = dict(shading="nearest", cmap="viridis")

# Scalar for lattice 3D (plotly) plot styles
STYLE_3D_MARKER = dict(size=6, colorscale="viridis", opacity=0.85)



###################################################################################################
# LATTICE
###################################################################################################

def plt_lattice(
        lattice:Lattice,
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
) -> Axes:
    r"""
    Visualize a lattice using Matplotlib.

    This function plots the geometric structure of a lattice including bonds, plaquettes, and 
    sites. Optionally, scalar values defined on sites and quasi-particle charges defined on 
    plaquettes can be displayed.

    Parameters
    ----------
    lattice : Lattice
        The lattice object.
    scalar_for_sites : np.ndarray, optional
        Array of shape (lattice.sites.N,) containing scalar values defined on lattice sites. 
        If provided, sites are colored according to these values.
    scalar_for_sites_label : str, optional
        Label for the colorbar when ``scalar_for_sites`` is provided.
        Defaults to ``"scalar value sites"``.
    scalar_for_plaquettes : np.ndarray, optional
        Array of shape (lattice.plaquettes.N,) containing scalar values defined on plaquettes.
        If provided, plaquette centers (dual lattice sites) are colored according to these values.
    scalar_for_plaquettes_label : str, optional
        Label for the colorbar when ``scalar_for_plaquettes`` is provided.
        Defaults to ``"scalar value plaquettes"``.
    m_ref : int, optional
        Reference filling per plaquette (proportional to effective chemical potential).
        When provided, the function computes quasi-particle charges on each plaquette as the sum of
        site occupations minus ``m_ref``. In this case, ``scalar_for_sites`` must be provided and 
        contain only 0/1 values (representing a state). The computed quasi-particle state overrides
        any ``quasi_particle_state`` input. Non-zero entries are visualized as charges with a 
        "+" or "-" sign.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axes to draw on. If ``None``, a new figure and axes are created and 
        displayed.
    xlim : list[float, float], optional
        Limits for the x-axis in the form ``[xmin, xmax]``. If provided, the plot is restricted to
        this horizontal range. If ``None``, the limits are determined automatically by Matplotlib.
    ylim : list[float, float], optional
        Limits for the y-axis in the form ``[ymin, ymax]``. If provided, the plot is restricted to
        this vertical range. If ``None``, the limits are determined automatically by Matplotlib.
    show_bonds : bool, default=True
        Whether to draw nearest-neighbor bonds between sites.
    show_plaquettes : bool, default=True
        Whether to draw plaquette polygons.
    show_sites : bool, default=True
        Whether to draw site markers (set to ``False`` if ``show_scalar_for_sites=True``).
    show_site_ids : bool, default=False
        Whether to display site indices next to each site.
    show_plaquette_ids : bool, default=False
        Whether to display plaquette indices at plaquette centers.
    show_dual_lattice_bonds : bool, default=False
        Whether to draw bonds of the dual lattice connecting neighboring plaquettes.
    show_dual_lattice_sites : bool, default=False
        Whether to draw markers at dual lattice sites (plaquette centers)
        (set to ``False`` if ``show_scalar_for_plaquettes=True``).
    show_scalar_for_sites : bool, default=True
        Whether to color the sites according to ``scalar_for_sites`` if provided.
        Overrides ``show_sites`` if ``True``.
    show_scalar_for_plaquettes : bool, default=True
        Whether to color the plaquette centers (dual lattice sites) according to 
        ``scalar_for_plaquettes`` if provided.
        Overrides ``show_dual_lattice_sites`` if ``True``.
    show_scalar_for_sites_cbar : bool, default=True
        Whether to display a colorbar for the ``scalar_for_sites`` when 
        ``show_scalar_for_sites=True``.
    show_scalar_for_plaquettes_cbar : bool, default=True
        Whether to display a colorbar for the ``scalar_for_plaquettes`` when 
        ``show_scalar_for_plaquettes=True``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Matplotlib axes containing the plot.
    """
    validate_type(lattice, Lattice, "lattice")

    if ax is None:
        setup_matplotlib_style()
        fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FIGSIZE)
        plot_directly = True
    else:
        plot_directly = False
    
    if scalar_for_sites is not None:
        assert scalar_for_sites.shape[0] == lattice.sites.N
    
    if show_scalar_for_sites:
        show_sites = False
    
    if scalar_for_plaquettes is not None:
        assert scalar_for_plaquettes.shape[0] == lattice.plaquettes.N
    
    if show_scalar_for_plaquettes:
        show_dual_lattice_sites = False
    
    if m_ref is not None:
        assert scalar_for_sites is not None
        assert np.all((scalar_for_sites == 0) | (scalar_for_sites == 1))
        quasi_particle_state = np.zeros((lattice.plaquettes.N,), dtype=np.int8)
        for pl_id in range(lattice.plaquettes.N):
            site_ids = lattice.plaquettes.site_ids[pl_id].entries
            q = np.sum(scalar_for_sites[site_ids])-m_ref
            quasi_particle_state[pl_id] = q
    else:
        quasi_particle_state = None
    
    # Lattice -------------------------------------------------------------------------------------
    # Plaquettes
    if show_plaquettes:
        for poly_pts in lattice.plaquettes.coords:
            ax.fill(poly_pts[:, 0], poly_pts[:, 1], **STYLE_PLAQUETTES)

    # Bonds
    if show_bonds:
        for i, nn_data in enumerate(lattice.sites.nearest_neighbors):
            xi, yi = lattice.sites.coords[i]
            entries = nn_data.entries
            shifts_x = nn_data.x_shifts
            shifts_y = nn_data.y_shifts
            for site_j, sx, sy in zip(entries, shifts_x, shifts_y):
                if site_j >= i: # avoid double-plotting
                    continue
                xj, yj = lattice.sites.coords[site_j]
                R_shift = lattice.sites.get_shift_vector(sx, sy)
                xj_shifted = xj + R_shift[0]
                yj_shifted = yj + R_shift[1]
                ax.plot([xi, xj_shifted], [yi, yj_shifted], **STYLE_BONDS)

    # Sites
    if show_sites:
        for coord in lattice.sites.coords:
            x, y = coord
            ax.scatter(x, y, **STYLE_SITES)

    # Scalar for sites
    if show_scalar_for_sites and scalar_for_sites is not None:
        is_binary = np.all((scalar_for_sites == 0) | (scalar_for_sites == 1))
        is_positive_and_negative = (np.any(scalar_for_sites > 0) and np.any(scalar_for_sites < 0))
        if is_binary:
            cmap = STYLE_SCALAR_CMAP[0]
        elif is_positive_and_negative:
            cmap = STYLE_SCALAR_CMAP[2]
        else:
            cmap = STYLE_SCALAR_CMAP[1]
        coords = np.array(lattice.sites.coords)
        x, y = coords[:, 0], coords[:, 1]
        sc_sites = ax.scatter(x, y, c=scalar_for_sites, cmap=cmap, **STYLE_SCALAR_FOR_SITES)
    
    # Dual Lattice --------------------------------------------------------------------------------
    # Dual lattice bonds
    if show_dual_lattice_bonds:
        for i, nn_data in enumerate(lattice.plaquettes.nearest_neighbors):
            xi, yi = lattice.plaquettes.center_coords[i]
            entries = nn_data.entries
            shifts_x = nn_data.x_shifts
            shifts_y = nn_data.y_shifts
            # Plot bonds
            for pl_j, sx, sy in zip(entries, shifts_x, shifts_y):
                if pl_j >= i:  # avoid double-plotting
                    continue
                xj, yj = lattice.plaquettes.center_coords[pl_j]
                R_shift = lattice.sites.get_shift_vector(sx, sy)
                dx = xj - xi + R_shift[0]
                dy = yj - yi + R_shift[1]
                ax.plot([xi, xi + dx], [yi, yi + dy], **STYLE_BONDS_DUAL_LATTICE)

    # Dual lattice sites
    if show_dual_lattice_sites:
        for coord in lattice.plaquettes.center_coords:
            x, y = coord
            ax.scatter(x, y, **STYLE_SITES_DUAL_LATTICE)

    # Scalar for plaquettes
    if show_scalar_for_plaquettes and scalar_for_plaquettes is not None:
        is_binary = np.all((scalar_for_plaquettes == 0) | (scalar_for_plaquettes == 1))
        is_positive_and_negative = (
            np.any(scalar_for_plaquettes > 0) and np.any(scalar_for_plaquettes < 0))
        if is_binary:
            cmap = STYLE_SCALAR_CMAP[0]
        elif is_positive_and_negative:
            cmap = STYLE_SCALAR_CMAP[2]
        else:
            cmap = STYLE_SCALAR_CMAP[1]
        coords = np.array(lattice.plaquettes.center_coords)
        x, y = coords[:, 0], coords[:, 1]
        sc_plaquettes = ax.scatter(
            x, y, c=scalar_for_plaquettes, cmap=cmap, **STYLE_SCALAR_FOR_PLAQUETTES)
    
    # Quasi particles on plaquettes ---------------------------------------------------------------
    if quasi_particle_state is not None:
        sign_size = STYLE_QUASI_PARTICLES["radius"]*QUASI_PARTICLE_SIGN_SCALE
        lw = STYLE_QUASI_PARTICLES["linewidth"]
        color = STYLE_QUASI_PARTICLES["edgecolor"]
        zorder = STYLE_QUASI_PARTICLES["zorder"]+1
        for i, coord in enumerate(lattice.plaquettes.center_coords):
            x, y = coord
            q = quasi_particle_state[i]
            if q == 0:
                continue
            num_of_rings = int(np.abs(q))
            for j in range(num_of_rings):
                style = copy(STYLE_QUASI_PARTICLES)
                style["radius"] *= 1+j/3
                circle = plt.Circle((x, y), **style)
                ax.add_patch(circle)
            ax.plot(
                [x-sign_size, x+sign_size], [y, y], zorder=zorder, color=color, lw=lw,
                solid_capstyle="round")
            if q > 0:
                ax.plot(
                    [x, x], [y-sign_size, y+sign_size], zorder=zorder, color=color, lw=lw,
                    solid_capstyle="round")

    # Numbering -----------------------------------------------------------------------------------
    # Plaquette Numbering
    if show_plaquette_ids:
        for i, coord in enumerate(lattice.plaquettes.center_coords):
            x, y = coord
            ax.text(x, y, str(i),**STYLE_ID_TEXT).set_clip_on(True)
    
    # Site Numbering
    if show_site_ids:
        for i, coord in enumerate(lattice.sites.coords):
            x, y = coord
            ax.text(
                x+ID_LABEL_OFFSET, y+ID_LABEL_OFFSET, str(i), **STYLE_ID_TEXT).set_clip_on(True)

    # Layout --------------------------------------------------------------------------------------
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.invert_yaxis()

    if show_scalar_for_sites_cbar and show_scalar_for_sites and scalar_for_sites is not None:
        cbar = plt.colorbar(sc_sites, ax=ax, shrink=0.8, location='left')
        cbar.set_label(scalar_for_sites_label, labelpad=15)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    
    if (show_scalar_for_plaquettes_cbar and show_scalar_for_plaquettes and 
        scalar_for_plaquettes is not None):
        cbar = plt.colorbar(sc_plaquettes, ax=ax, shrink=0.8, location='left')
        cbar.set_label(scalar_for_plaquettes_label, labelpad=15)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
            
    plt.tight_layout()
    if plot_directly:
        plt.show()
    return ax



###################################################################################################
# STRUCTURE FACTOR
###################################################################################################

def plt_structure_factor(
        Sq_grid:np.ndarray, qx:np.ndarray, qy:np.ndarray, cbar_label:str=r"$S(\mathbf{q})$",
        vmax_cbar:float=None, kx_lim:list[float, float]=None, ky_lim:list[float, float]=None, 
        xlabel=r"$\pi k_x$", ylabel=r"$\pi k_y$", tick_spacing:float=np.pi, add_colorbar:bool=True,
        ax:Axes=None, 
) -> Axes:
    r"""
    Plot the static structure factor ``S(q)`` on a 2D momentum grid.

    Parameters
    ----------
    Sq_grid : np.ndarray
        2D array of the static structure factor evaluated on the grid, shape ``(Nq, Nq)``.
        Each element corresponds to the value at ``(qx[i,j], qy[i,j])``.
    qx : np.ndarray
        2D array of x-components of the wave vectors, shape ``(Nq, Nq)``.
    qy : np.ndarray
        2D array of y-components of the wave vectors, shape ``(Nq, Nq)``.
    cbar_label : str, optional
        Label for the colorbar. Defaults to ``r"$S(\mathbf{q})$"``.
    vmax_cbar : float, optional
        Upper limit of the color scale. If ``None``, the maximum value of
        ``Sq_grid`` is used. The lower limit is fixed to zero.
    kx_lim : list[float, float], optional
        Limits for the x-axis in Cartesian momentum units.
    ky_lim : list[float, float], optional
        Limits for the y-axis in Cartesian momentum units.
    xlabel : str, optional
        Label for the x-axis. If ``None``, no x-axis label is set. Defaults to ``r"$\pi k_x$"``.
    ylabel : str, optional
        Label for the y-axis. If ``None``, no y-axis label is set. Defaults to ``r"$\pi k_y$"``.
    tick_spacing : float, optional
        Spacing between major ticks in Cartesian momentum units. The tick labels are formatted 
        as multiples of ``pi``. Default is ``pi``.
        If ``None``, no manual tick placement or formatting is applied, and Matplotlib's default
        tick locator and formatter are used.
    add_colorbar : bool, default=True
        Whether to display the colorbar.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib axes to draw on. If ``None``, a new figure and axes are created and 
        displayed.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Matplotlib axes containing the plot.

    Notes
    -----
    To generate the grid data, use:
    >>> qx, qy, Sq_grid = methods.observables.compute_static_structure_factor_grid(C_ij, lattice)
    """
    validate_type(Sq_grid, np.ndarray, "Sq_grid")
    validate_type(qx, np.ndarray, "qx")
    validate_type(qy, np.ndarray, "qy")
    
    if ax is None:
        setup_matplotlib_style()
        fig, ax = plt.subplots(1, 1, figsize=DEFAULT_FIGSIZE_STRUCTURE_FACTOR)
        plot_directly = True
    else:
        plot_directly = False
    
    # Plot mesh
    if vmax_cbar is None:
        vmax_cbar = Sq_grid.max()
    pcm = ax.pcolormesh(qx, qy, Sq_grid, vmin=0, vmax=vmax_cbar, **STYLE_STRUCTURE_FACTOR_MESH)
    if add_colorbar:
        plt.colorbar(pcm, label=cbar_label)

    # Ticks
    def pi_fraction_formatter(val, pos):
        coeff = val/np.pi
        coeff = np.round(coeff,1)
        if np.isclose(coeff, int(coeff)):
            coeff = int(coeff)
        return rf"${coeff}$"
    
    if tick_spacing is not None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xticks = np.arange(
            np.floor(xmin/tick_spacing)*tick_spacing,
            np.ceil(xmax/tick_spacing)*tick_spacing+tick_spacing, tick_spacing)
        yticks = np.arange(
            np.floor(ymin/tick_spacing)*tick_spacing,
            np.ceil(ymax/tick_spacing)*tick_spacing+tick_spacing, tick_spacing)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.xaxis.set_major_formatter(FuncFormatter(pi_fraction_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(pi_fraction_formatter))

    # Axes
    ax.set_aspect("equal", adjustable="box")

    if kx_lim is not None:
        ax.set_xlim(kx_lim)
    if ky_lim is not None:
        ax.set_ylim(ky_lim)
    
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if plot_directly:
        plt.tight_layout()
        plt.show()
    return ax



###################################################################################################
# SCALAR FOR LATTICE 3D (PLOTLY)
###################################################################################################

def plt_scalar_for_lattice_3D(
        lattice:Lattice,
        scalar_for_sites:np.ndarray=None,
        scalar_for_plaquettes:np.ndarray=None,
        scalar_value_label:str=None,
        zlim:list[float, float]=None,
        log_scale:bool=False,
        ignore_warning:bool=False,
):
    r"""
    Visualize scalar values on lattice sites or plaquettes in 3D using Plotly.

    A 3D scatter plot is generated where the x and y coordinates correspond
    to lattice positions and the z coordinate represents the scalar value.
    Coloring follows the scalar magnitude (optionally in logarithmic scale).

    Parameters
    ----------
    lattice : Lattice
        The lattice object.
    scalar_for_sites : np.ndarray, optional
        Array of shape (lattice.sites.N,) containing scalar values defined on lattice sites.
    scalar_for_plaquettes : np.ndarray, optional
        Array of shape (lattice.plaquettes.N,) containing scalar values defined on plaquette 
        centers.
    scalar_value_label : str, optional
        Label for the z-axis and colorbar. Defaults to ``"scalar value"``.
    zlim : tuple[float, float], optional
        Explicit limits for the z-axis (and color scale). If ``None``, limits are determined 
        from the data.
    log_scale : bool, default=False
        If ``True``, the scalar values are displayed on a base-10 logarithmic scale. 
        Non-positive values are converted to absolute values.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The Plotly figure object containing the 3D visualization.

    Raises
    ------
    ImportError
        If Plotly is not available.
    ValueError
        If neither or both of `scalar_for_sites` and
        `scalar_for_plaquettes` are provided.

    Notes
    -----
    - Plot styling is controlled via ``STYLE_3D_MARKER`` in ``main.config``.
    """
    validate_type(lattice, Lattice, "lattice")
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("`Plotly` is required for 3D visualization.")
    
    if scalar_for_sites is not None:
        assert scalar_for_plaquettes is None
        assert scalar_for_sites.shape[0] == lattice.sites.N
        scalar_values = scalar_for_sites
        coords = lattice.sites.coords
    elif scalar_for_plaquettes is not None:
        assert scalar_for_sites is None
        assert scalar_for_plaquettes.shape[0] == lattice.plaquettes.N
        scalar_values = scalar_for_plaquettes
        coords = lattice.plaquettes.center_coords
    else:
        raise ValueError(f"Either provide `scalar_for_sites` or `scalar_for_plaquettes`.")

    if scalar_value_label is None:
        scalar_value_label = "scalar value"

    xs, ys = coords[:, 0], coords[:, 1]

    # Prepare values
    scalar_values = np.array(scalar_values, dtype=float)
    if log_scale:
        if np.any(scalar_values <= 0):
            if not ignore_warning:
                warnings.warn(
                    "Non-positive values encountered in scalar values while using "
                    "`log_scale=True`. Taking absolute values. Zeros will be masked.")
        scalar_values = np.abs(scalar_values)
        scalar_values = np.clip(scalar_values, 1e-15, None)
        log_values = np.log10(scalar_values)
        zscale = "log"
    else:
        log_values = scalar_values
        zscale = "linear"

    if zlim is None:
        cmin, cmax = np.min(log_values), np.max(log_values)
    else:
        cmin, cmax = zlim

    # Build 3D scatter
    fig = go.Figure(
        data=[
            go.Scatter3d(x=xs, y=ys, z=scalar_values, mode="markers",
                marker=dict(
                    color=log_values, colorbar=dict(
                        title=scalar_value_label), cmin=cmin, cmax=cmax, **STYLE_3D_MARKER))])

    # Layout
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    xrange = xmax-xmin
    yrange = ymax-ymin
    max_range = max(xrange, yrange)
    xmid = (xmin+xmax)/2
    ymid = (ymin+ymax)/2
    xlim = [xmid-max_range/2, xmid+max_range/2]
    ylim = [ymid-max_range/2, ymid+max_range/2]
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(title="x", range=xlim),
            yaxis=dict(title="y", range=ylim),
            zaxis=dict(title=scalar_value_label, type=zscale, range=zlim),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(projection=dict(type="orthographic"))))
    fig.show()
    return fig
