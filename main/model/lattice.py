
# General Imports
import numpy as np
import warnings

# Internal Imports
from main.tools.printers import make_table, make_table_data
from main.tools.utils import nested_list_to_padded_array, validate_type



###################################################################################################
# PERIODIC NODE LIST
###################################################################################################

class ListWithShifts():
    r"""
    Parameters
    ----------
    entries : list[int]
        Integer value with associated shifts.
        Example: the site IDs of a lattice
    x_shifts : list[int]
        Periodicity shifts in the x-direction.  
        Example: the rightmost site of a unit cell connects to the leftmost site of the adjacent 
        cell in the positive x-direction -> ``x_shifts[i] = 1``.
    y_shifts : list[int]
        Periodicity shifts in the y-direction.  
        Example: a site in the first row connects to a site in the last row of the unit cell in the
        negative y-direction -> ``y_shifts[i] = -1``.
    
    Attributes
    ----------
    entries : list[int]
        Same as parameter.
    x_shifts : list[int]
        Same as parameter.
    y_shifts : list[int]
        Same as parameter.
    """
    def __init__(self, entries:list[int], x_shifts:list[int], y_shifts:list[int]):
        validate_type(entries, [list, int], "entries")
        validate_type(x_shifts, [list, int], "x_shifts")
        validate_type(y_shifts, [list, int], "y_shifts")
        assert len(entries) == len(x_shifts)
        assert len(entries) == len(y_shifts)
        
        self.entries:list[int] = entries
        self.x_shifts:list[int] = x_shifts
        self.y_shifts:list[int] = y_shifts
    
    @classmethod
    def from_list(cls, entries: list[int]):
        """
        Construct a ListWithShifts from a list of node IDs, assuming zero periodic shifts.
        """
        validate_type(entries, list, "entries")
        n = len(entries)
        return cls(entries=entries, x_shifts=[0]*n, y_shifts=[0]*n)
    
    @property
    def number_of_entries(self) -> int:
        return len(self.entries)
    
    def __len__(self):
        return self.number_of_entries

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ListWithShifts(
                self.entries[idx], self.x_shifts[idx], self.y_shifts[idx])
        return self.entries[idx]
    
    def __str__(self):
        headers = ["Index", "Entry", "Shift x", "Shift y"]
        rows = [[i, entry, sx, sy] for i, (entry, sx, sy) in enumerate(
            zip(self.entries, self.x_shifts, self.y_shifts) )]
        return make_table(headers, make_table_data(rows))

    def __repr__(self):
        return self.__str__()



###################################################################################################
# BUILDING BLOCKS OF LATTICE
###################################################################################################

class Sites():
    r"""
    Parameters
    ----------
    a1 : np.ndarray of shape (2,)
        Primitive lattice vector ``a1`` in Cartesian coordinates.
    a2 : np.ndarray of shape (2,)
        Primitive lattice vector ``a2`` in Cartesian coordinates.
    d1 : float
        Scaling factor along the primitive vector ``a1`` to define the site cell.
        The corresponding Cartesian edge of the site cell is ``D1 = d1 * a1``.
    d2 : float
        Scaling factor along the primitive vector ``a2`` to define the site cell.
        The corresponding Cartesian edge of the site cell is ``D2 = d2 * a2``.
    us : np.ndarray of shape (N, 2)
        Unitless fractional coordinates of the sites within the site cell.
        Each row is (u1, u2) such that the real-space position ``r[site_id]`` is
        ``r[site_id] = us[site_id, 0] * a1 + us[site_id, 1] * a2``.
    nearest_neighbors : list[list[int]] | list[ListWithShifts]
        List of length ``N``. Each element contains the indices of nearest neighbors of the 
        corresponding site.

    Attributes
    ----------
    N : int
        Total number of sites.
    is_periodic : bool
        Whether the lattice has periodic boundary conditions. (True if ``nearest_neighbors`` are
        ``ListWithShifts`` objects).
    D1 : np.ndarray of shape (2,)
        Cartesian vector forming the edge of the site cell along a1: ``D1 = d1 * a1``.
    D2 : np.ndarray of shape (2,)
        Cartesian vector forming the edge of the site cell along a2: ``D2 = d2 * a2``.
    coords : np.ndarray of shape (N, 2)
        Cartesian coordinates of the sites.
    a1 : np.ndarray of shape (2,)
        Same as parameter.
    a2 : np.ndarray of shape (2,)
        Same as parameter.
    d1 : float
        Same as parameter.
    d2 : float
        Same as parameter.
    us : np.ndarray of shape (N, 2)
        Same as parameter.
    nearest_neighbors : list[ListWithShifts]
        Same as parameter.
    """
    def __init__(self, 
            a1:np.ndarray, a2:np.ndarray, d1:float, d2:float, us:np.ndarray, 
            nearest_neighbors:list[list[int]]|list[ListWithShifts]
    ):
        validate_type(a1, np.ndarray, "a1")
        validate_type(a2, np.ndarray, "a2")
        validate_type(us, np.ndarray, "us")
        validate_type(d1, float, "d1")
        validate_type(d2, float, "d2")
        validate_type(nearest_neighbors, [list, (ListWithShifts, list)], "nearest_neighbors")
        assert a1.ndim == 1 and a2.ndim == 1
        assert a1.shape[0] == 2 and a2.shape[0] == 2
        assert us.ndim == 2
        N, spatial_dim = us.shape
        assert spatial_dim == 2
        assert len(nearest_neighbors) == N

        # Convert list[list[int]] to ListWithShifts if necessary
        if isinstance(nearest_neighbors[0], list):
            is_periodic = False
            validate_type(nearest_neighbors, [list, list, int], "nearest_neighbors")
            nearest_neighbors:list[ListWithShifts] = (
                nested_list_to_nested_ListWithShifts(nearest_neighbors) )
        else:
            is_periodic = True
        
        # Check if every site has a neighbor
        warns = []
        for site_id, nn_i in enumerate(nearest_neighbors):
            if len(nn_i.entries) == 0:
                warns.append(str(site_id))
        if len(warns) != 0:
            warnings.warn(
                f"The following site ids have no nearest neighbor:\n"
                + ", ".join(warns))
        
        # Initialize
        self.is_periodic:bool = is_periodic
        self.N:int = N

        self.a1:np.ndarray = a1
        self.a2:np.ndarray = a2
        self.d1:float = d1
        self.d2:float = d2
        self.D1:np.ndarray = d1*a1
        self.D2:np.ndarray = d2*a2

        self.us:np.ndarray = us # ns.shape = (N, 2)
        self.coords:np.ndarray = self._coords_from_us() # coords.shape = (N, 2)

        self.nearest_neighbors:list[ListWithShifts] = nearest_neighbors

    @classmethod
    def from_cartesian_coordinates(
            cls, Dx:float, Dy:float, coords:np.ndarray, nearest_neighbors:list
    ):
        r"""
        Parameters
        ----------
        Dx : float
            Real-space length of the rectangular site cell in the x-direction.
        Dy : float
            Real-space length of the rectangular site cell in the y-direction.
        coords : np.ndarray of shape (N, 2)
            Cartesian coordinates of the sites.
        nearest_neighbors : list[list[int]] | list[ListWithShifts]
            List of length ``N``. Each element contains the indices of nearest neighbors of the 
            corresponding site.
        """
        validate_type(Dx, float, "Dx")
        validate_type(Dy, float, "Dy")
        validate_type(coords, np.ndarray, "coords")
        assert coords.ndim == 2
        _, spatial_dim = coords.shape
        assert spatial_dim == 2
        a1 = np.array([1,0])
        a2 = np.array([0,1])
        d1, d2 = Dx, Dy
        us = coords
        return cls(a1, a2, d1, d2, us, nearest_neighbors)
    
    def _coords_from_us(self):
        return self.us[:, 0:1] * self.a1 + self.us[:, 1:2] * self.a2
    
    def get_shift_vector(self, s1_shift:int, s2_shift:int):
        return s1_shift*self.D1 + s2_shift*self.D2
    
    def get_bonds(self) -> list[list[int]]:
        nearest_neighbors_list = nested_ListWithShifts_to_nested_list(self.nearest_neighbors)
        bonds = []
        seen = set()
        for i, neighbors in enumerate(nearest_neighbors_list):
            for j in neighbors:
                if i < j and (i, j) not in seen:
                    bonds.append((i, j))
                    seen.add((i, j))
                elif j < i and (j, i) not in seen:
                    bonds.append((j, i))
                    seen.add((j, i))
        return [list(bond) for bond in bonds]

    def get_coordination_numbers(self) -> list[int]:
        list_of_neighs = nested_ListWithShifts_to_nested_list(self.nearest_neighbors) 
        coordination_numbers = []
        for site_id, neigh_site_ids in enumerate(list_of_neighs):
            coordination_numbers.append(len(neigh_site_ids))
        return coordination_numbers
    
    def __str__(self):
        headers = ["Site ID", "x", "y", "Neighbors (with (sx,sy) if periodic)"]
        data = [
            [i, x, y, nn]
            for i, (coord, nn) in enumerate(zip(self.coords, self.nearest_neighbors))
            for x, y in [coord]]
        return make_table(headers, make_table_data(data))
    
    def __repr__(self):
        return self.__str__()



class Plaquettes():
    r"""

    Parameters
    ----------
    site_ids : list[list[int]] | list[ListWithShifts]
        List of length ``N``. Each element represents the site IDs forming a plaquette.
        The first site ID is used as a reference for the coordinates of the plaquette.
    nearest_neighbors : list[list[int]] | list[ListWithShifts] | None
        List of length ``N``. Each element contains the nearest neighbors of a plaquette.
        If ``nearest_neighbors=None``, ``nearest_neighbors`` will be determined automatically,
        however, without information about periodicity.
    sites : Sites
        Sites of the plaquettes. Only needed to compute the coordinates of the plaquettes.
    
    Attributes
    ----------
    N : int
        Total number of plaquettes.
    is_periodic : bool
        Whether the underlying sites have periodic boundary conditions.
    coords : list[list[float]]
        Cartesian coordinates of each plaquette's sites.
        Shape is (N, num_of_sites_in_pl, 2).
    center_coords : np.ndarray of shape (N, 2)
        Cartesian coordinates of the plaquette centers (site coordinates of the dual-lattice).
    site_ids : list[ListWithShifts]
        Same as parameter.
    nearest_neighbors : list[ListWithShifts]
        Same as parameter.
    """
    def __init__(
            self, site_ids:list[list[int]]|list[ListWithShifts], 
            nearest_neighbors:list[list[int]]|list[ListWithShifts]|None, sites:Sites
    ):
        validate_type(site_ids, [list, (list, ListWithShifts)], "site_ids")
        validate_type(sites, Sites, "sites")
        N = len(site_ids)
        if nearest_neighbors is not None:
            validate_type(nearest_neighbors, [list, (list, ListWithShifts)], "nearest_neighbors")
            assert len(nearest_neighbors) == N

        is_periodic = sites.is_periodic
        
        # Convert list[list[int]] to ListWithShifts if necessary
        if isinstance(site_ids[0], list):
            validate_type(site_ids, [list, list, int], "site_ids")
            site_ids:list[ListWithShifts] = nested_list_to_nested_ListWithShifts(site_ids)
            if is_periodic:
                warnings.warn(
                    "Expected type `ListWithShifts` for `site_ids`, since `sites` are "
                    "periodic, but got `list[list[int]]`."
                    "This may restrict the use of certain functions.")
                
        if nearest_neighbors is None:
            list_of_pls = nested_ListWithShifts_to_nested_list(site_ids)
            nearest_neighbors:list[ListWithShifts] = (
                get_adjacent_loops_for_local_changes(
                    sites.N, list_of_pls, list_of_pls, filter_self=True) )
            if is_periodic:
                warnings.warn(
                    "Expected type `ListWithShifts` for `nearest_neighbors`, since `sites` are "
                    "periodic, but got `None`. `nearest_neighbors` will be determined "
                    "automatically without information about periodicity. "
                    "This may restrict the use of certain functions.")
        elif isinstance(nearest_neighbors[0], list):
            if is_periodic:
                warnings.warn(
                    "Expected type `ListWithShifts` for `nearest_neighbors`, since `sites` are "
                    "periodic, but got `list[list[int]]`. "
                    "This may restrict the use of certain functions.")
        
        if isinstance(nearest_neighbors[0], list):
            validate_type(nearest_neighbors, [list, list, int], "nearest_neighbors")
            nearest_neighbors:list[ListWithShifts] = (
                nested_list_to_nested_ListWithShifts(nearest_neighbors) )

        # Check if every plaquette is defined by at least one site
        warns = []
        for pl_id, site_ids_i in enumerate(site_ids):
            if site_ids_i.entries == 0:
                warns.append(str(pl_id))
        if len(warns) != 0:
            warnings.warn(
                f"The following plaquette ids are defined by zero sites:\n"
                + ", ".join(warns))

        # Check if every plaquette has a neighbor
        warns = []
        for pl_id, nn_i in enumerate(nearest_neighbors):
            if len(nn_i.entries) == 0:
                warns.append(str(pl_id))
        if len(warns) != 0:
            warnings.warn(
                f"The following plaquette ids have no nearest neighbor:\n"
                + ", ".join(warns))

        # Initialize
        self.is_periodic:bool = is_periodic
        self.N:int = N
        self.site_ids:list[ListWithShifts] = site_ids
        coords, center_coords = self._get_plaquette_coords(sites)
        # coords are list, because might be nested
        self.coords:list[list[float]] = coords # coords.shape ~= (N, num_of_sites_in_pl, 2)
        self.center_coords:np.ndarray = center_coords # center_coords.shape = (N, 2)
        self.nearest_neighbors:list[ListWithShifts] = nearest_neighbors

    def _get_plaquette_coords(self, sites:Sites)-> tuple[list[list[float]], np.ndarray]:
        coords = []
        center_coords = []
        for pl_id, pl in enumerate(self.site_ids):
            poly_pts = []
            for site_id, s1, s2 in zip(pl.entries, pl.x_shifts, pl.y_shifts):
                x, y = sites.coords[site_id]
                R_shift = sites.get_shift_vector(s1, s2)
                x += R_shift[0]
                y += R_shift[1]
                poly_pts.append((x, y))
            poly_pts = np.array(poly_pts)
            center = poly_pts.mean(axis=0)
            coords.append(poly_pts)
            center_coords.append(center)
        center_coords = np.array(center_coords)
        return coords, center_coords
    
    def __str__(self):
        headers = [
            "Plaq. ID", "Center x", "Center y", "Sites (with (sx,sy) if periodic)",
            "Neighbors (with (sx,sy) if periodic)"]
        data = [[i, cx, cy, pl_sites, nn] for i, (pl_sites, (cx, cy), nn) in 
                enumerate(zip(self.site_ids, self.center_coords, self.nearest_neighbors))]
        return make_table(headers, make_table_data(data))

    def __repr__(self):
        return self.__str__()



###################################################################################################
# LATTICE
###################################################################################################

class Lattice:
    r"""
    Parameters
    ----------
    sites : Sites
        Sites of the lattice.
    plaquettes : Plaquettes
        Plaquettes of the lattice.

    Attributes
    ----------
    sites : Sites
        Same as parameter.
    plaquettes : Plaquettes
        Same as parameter.
    """
    _required_methods = [
        "get_monte_carlo_worm_network",
        "get_monte_carlo_local_loops",
        "get_RK_loops",
        "get_A_transfer"]
    
    def __init__(self, sites:Sites, plaquettes:Plaquettes):
        validate_type(sites, Sites, "sites")
        validate_type(plaquettes, Plaquettes, "plaquettes")
        self.sites:Sites = sites
        self.plaquettes:Plaquettes = plaquettes
        #self._check_required_methods()

    def get_adjacent_plaquettes_for_sites(self) -> list[list[int]]:
        list_of_pl = nested_ListWithShifts_to_nested_list(self.plaquettes.site_ids) 
        return get_adjacent_loops_for_sites(self.sites.N, list_of_pl)

    def get_plaquette_coordination_number_of_sites(self) -> list[int]:
        site_to_adjacent_pl = self.get_adjacent_plaquettes_for_sites()
        coordination_numbers = np.zeros((self.sites.N), dtype=np.int64)
        for site_id, pl_ids in enumerate(site_to_adjacent_pl):
            coordination_numbers[site_id] = len(pl_ids)
        return coordination_numbers

    def get_uniform_plaquette_coordination_number(self) -> int:
        pl_coordination_numbers = self.get_plaquette_coordination_number_of_sites()
        if not np.all(pl_coordination_numbers == pl_coordination_numbers[0]):
            raise ValueError(
                "Plaquette coordination number of sites is not uniform.")
        return int(pl_coordination_numbers[0])

    def get_m_ref(self, N_filling: int) -> int:
        pl_coord_num = self.get_uniform_plaquette_coordination_number()
        m_ref = (N_filling*pl_coord_num)/self.plaquettes.N
        m_ref_rounded = int(np.round(m_ref))
        if not np.isclose(m_ref, m_ref_rounded, atol=1e-10):
            raise ValueError(
                f"Computed plaquette filling n_pl={m_ref} is not an integer.")
        return m_ref_rounded

    def get_N_filling(self, m_ref: int) -> int:
        pl_coord_num = self.get_uniform_plaquette_coordination_number()
        N_filling = (m_ref * self.plaquettes.N) / pl_coord_num
        N_filling_rounded = int(np.round(N_filling))
        if not np.isclose(N_filling, N_filling_rounded, atol=1e-10):
            raise ValueError(
                f"Computed N_filling={N_filling} for m_ref={m_ref} is not an integer.")
        return N_filling_rounded

    # Functions for methods
    #######################

    def get_monte_carlo_worm_network(self, **kwargs):
        r""" See `methods.monte_carlo.sample_states` for details on how the parameters are defined 
        and used."""
        func_name = "get_monte_carlo_worm_network"
        limitations = ["'worm' method in `methods.monte_carlo`"]
        self._throw_error(func_name, limitations)
        nearest_neighbors:list[ListWithShifts] = None
        connecting_site_ids:list[list[ListWithShifts]] = None
        check_site_ids_of_path:bool = None
        return nearest_neighbors, connecting_site_ids, check_site_ids_of_path
    
    def get_monte_carlo_local_loops(self, **kwargs):
        r""" See `methods.monte_carlo.sample_states` for details on how the parameters are defined 
        and used."""
        func_name = "get_monte_carlo_local_loops"
        limitations = ["'local loop' method in `methods.monte_carlo`"]
        self._throw_error(func_name, limitations)
        local_loops:list[ListWithShifts] = None
        return local_loops

    def get_RK_loops(self, **kwargs):
        r""" See `methods.hamiltonian.get_RK_H` for details on how the parameters are defined 
        and used."""
        func_name = "get_RK_loops"
        limitations = [
            "`methods.hamiltonian.get_RK_ground_state`",
            "`methods.hamiltonian.get_RK_H`"]
        self._throw_error(func_name, limitations)
        RK_loops:list[ListWithShifts] = None
        return RK_loops
    
    def get_A_transfer(self, **kwargs):
        r""" See `methods.tensor_network.get_alpha_from_transfer_matrix` for details on how the 
        parameters are defined and used."""
        func_name = "get_A_transfer"
        limitations = ["`methods.tensor_network.get_alpha_from_transfer_matrix`"]
        self._throw_error(func_name, limitations)
        A_transfer:np.ndarray = None
        num_of_sites_per_A_transfer:int = None
        return A_transfer, num_of_sites_per_A_transfer
    
    # Helpers
    #########

    def _throw_error(self, func_name: str, limitations: list[str]):
        msg = (
            f"`{func_name}` needs to be implemented in the subclass. "
            f"Cannot use {', '.join(limitations)}.")
        raise NotImplementedError(msg)

    def _check_required_methods(self):
        missing = []
        for method_name in self._required_methods:
            base_method = getattr(Lattice, method_name)
            subclass_method = getattr(self.__class__, method_name)
            if subclass_method is base_method:
                missing.append(method_name)
        if len(missing) != 0:
            warnings.warn(
                f"The subclass `{self.__class__.__name__}` does not implement "
                f"the following methods:\n"
                + ", ".join(missing))
    
    def __str__(self):
        return f"Lattice: {self.sites.N} sites, {self.plaquettes.N} plaquettes"

    def __repr__(self):
        return self.__str__()



###################################################################################################
# UNIT CELL
###################################################################################################

class UnitCell(Lattice):
    r"""
    Parameters
    ----------
    sites : Sites
        Sites of the unit cell. ``Sites.nearest_neighbors`` must be a list of 
        :class:`ListWithShifts`.
    plaquettes : Plaquettes
        Plaquettes of the unit cell. ``Plaquettes.site_ids`` must be a list of 
        :class:`ListWithShifts`.
    
    Attributes
    ----------
    sites : Sites
        Same as parameter.
    plaquettes : Plaquettes
        Same as parameter.
    """
    def __init__(self, sites:Sites, plaquettes:Plaquettes):
        validate_type(sites.nearest_neighbors, [list,ListWithShifts], "sites.nearest_neighbors")
        validate_type(plaquettes.site_ids, [list,ListWithShifts], "plaquettes.site_ids")
        super().__init__(sites, plaquettes)
    
    def __str__(self):
        return f"UnitCell: {self.sites.N} sites, {self.plaquettes.N} plaquettes"

    def __repr__(self):
        return self.__str__()



###################################################################################################
# LATTICE MADE FROM UNIT CELL
###################################################################################################

class LatticeFromUnitCell(Lattice):
    r"""
    Construct a larger lattice by tiling a unit cell.

    This class generates a full lattice by repeating the given ``unit_cell`` in two directions.
    The resulting lattice has ``L1 * L2`` unit cells, preserving the connectivity and periodicity
    defined in the unit cell.

    Parameters
    ----------
    L1 : int
        Number of unit cells in direction 1.
    L2 : int
        Number of unit cells in direction 2.
    unit_cell : UnitCell
        Unit cell used to construct the lattice.

    Attributes
    ----------
    L1 : int
        Same as parameter.
    L2 : int
        Same as parameter.
    unit_cell : UnitCell
        Same as parameter.
    sites : Sites
        Sites of the full lattice.
    plaquettes : Plaquettes
        Plaquettes of the full lattice.
    """
    def __init__(self, L1:int, L2:int, unit_cell:UnitCell):
        validate_type(L1, int, "L1")
        validate_type(L2, int, "L2")
        validate_type(unit_cell, UnitCell, "unit_cell")

        self.L1:int = L1
        self.L2:int = L2
        self.unit_cell:UnitCell = unit_cell

        sites, plaquettes = self._build_lattice()

        super().__init__(sites, plaquettes)


    # Functions for methods
    #######################

    def get_monte_carlo_worm_network(self, **unit_cell_kwargs):
        uc_nearest_neighbors, uc_connecting_site_ids, check_site_ids_of_path = (
            self.unit_cell.get_monte_carlo_worm_network(**unit_cell_kwargs))
        nearest_neighbors = self.wrap_local_nested_ListWithShifts_around_torus(
            uc_nearest_neighbors, len(uc_nearest_neighbors))
        connecting_site_ids = self.wrap_local_nested_ListWithShifts_around_torus(
            uc_connecting_site_ids, self.unit_cell.sites.N)
        return nearest_neighbors, connecting_site_ids, check_site_ids_of_path
    
    def get_monte_carlo_local_loops(self, **unit_cell_kwargs):
        uc_local_loops = self.unit_cell.get_monte_carlo_local_loops(**unit_cell_kwargs)
        local_loops = self.wrap_local_nested_ListWithShifts_around_torus(
            uc_local_loops, self.unit_cell.sites.N)
        return local_loops
    
    def get_RK_loops(self, **unit_cell_kwargs):
        RK_loops_unit_cell = self.unit_cell.get_RK_loops(**unit_cell_kwargs)
        RK_loops = self.wrap_local_nested_ListWithShifts_around_torus(
                RK_loops_unit_cell, self.unit_cell.sites.N)
        return RK_loops


    # Build Lattice
    ###############

    def _build_lattice(self):
        sites = self._build_sites()
        plaquettes = self._build_plaquettes(sites)
        return sites, plaquettes

    def _build_sites(self):
        unit_cell = self.unit_cell
        L1, L2 = self.L1, self.L2
        N = L1*L2*unit_cell.sites.N

        # Wrap us
        us = np.zeros((N, 2), dtype=np.float64)
        for idy in range(L2): 
            for idx in range(L1):
                id_uc = self.get_id_unit_cell(idx, idy)
                ref = id_uc * unit_cell.sites.N
                u_shift = np.array([idx*unit_cell.sites.d1, idy*unit_cell.sites.d2])
                us[ref:ref+unit_cell.sites.N] = unit_cell.sites.us + u_shift
        
        # Wrap nearest neighbors
        nearest_neighbors = self.wrap_local_nested_ListWithShifts_around_torus(
            self.unit_cell.sites.nearest_neighbors, self.unit_cell.sites.N)
        
        # Create Sites object
        a1 = unit_cell.sites.a1
        a2 = unit_cell.sites.a2
        d1 = L1*unit_cell.sites.d1
        d2 = L2*unit_cell.sites.d2
        sites = Sites(a1, a2, d1, d2, us, nearest_neighbors)
        return sites

    def _build_plaquettes(self, sites:Sites):
        # Wrap plaquettes (defined over the site_ids which form them)
        sites_ids = self.wrap_local_nested_ListWithShifts_around_torus(
            self.unit_cell.plaquettes.site_ids, self.unit_cell.sites.N)
        # Wrap nearest neighbors
        if isinstance(self.unit_cell.plaquettes.nearest_neighbors[0], ListWithShifts):
            nearest_neighbors = self.wrap_local_nested_ListWithShifts_around_torus(
                self.unit_cell.plaquettes.nearest_neighbors, self.unit_cell.plaquettes.N)
        else:
            nearest_neighbors = None

        # Create Plaquettes object
        plaquettes = Plaquettes(sites_ids, nearest_neighbors, sites)
        return plaquettes


    # Id referencing scheme
    #######################

    def get_id_unit_cell(self, idx_unit_cell:int, idy_unit_cell:int):
        idx_unit_cell, idy_unit_cell = idx_unit_cell%self.L1, idy_unit_cell%self.L2 # periodicity
        return idy_unit_cell*self.L1+idx_unit_cell
    
    def get_idx_idy_unit_cell(self, id_unit_cell:int):
        idx_unit_cell = id_unit_cell%(self.L1)
        idy_unit_cell = (id_unit_cell-idx_unit_cell)//self.L1
        return idx_unit_cell, idy_unit_cell

    def get_id_of_unit_cell(self, site_id:int):
        id_of_unit_cell = site_id//self.unit_cell.sites.N
        id_in_unit_cell = site_id-id_of_unit_cell*self.unit_cell.sites.N
        return id_of_unit_cell, id_in_unit_cell


    # Global Pattern
    ################

    def get_global_patterns(
            self, unit_cell_ids:ListWithShifts, sx_shift:int, sy_shift:int, 
            for_plaquettes:bool=False, ignore_ids:list[int]=None, r_max:int=None
    ) -> list[list[int]]:
        r"""
        Construct global pattern for a periodically repeated local pattern.

        This method takes a local pattern defined on the unit cell (``unit_cell_ids``) and 
        propagates it across the full lattice by repeatedly shifting the reference unit cell by
        ``(sx_shift, sy_shift)``. For each unit cell in the lattice, it generates the sequence of 
        global indices obtained by iteratively applying the shift until either:
        - the starting global index is reached again, or
        - ``r_max`` elements have been collected.

        Parameters
        ----------
        unit_cell_ids : ListWithShifts
            Defines the local pattern inside a unit cell. Contains:
            - ``entries``: local indices within the unit cell
            - ``x_shifts``, ``y_shifts``: relative unit cell offsets for each entry
        sx_shift : int
            Translation in x-direction applied after each full pattern cycle.
        sy_shift : int
            Translation in y-direction applied after each full pattern cycle.
        for_plaquettes : bool, optional
            If ``True``, construct pattern for plaquettes instead of sites.
        ignore_ids : list[int] or int, optional
            Global indices to exclude from the pattern.
        r_max : int, optional
            Distance (distance parameter ``r``) before terminating the construction of a sequence.
            If given, the sequence contains at most ``r_max+1`` elements.

        Returns
        -------
        global_ids : list[list[int]]
            For each unit cell, the ordered list of global indices generated by the periodic 
            propagation of the pattern.
            Note: The length of each sequence may differ if some IDs are ignored via 
            ``ignore_ids``.

        Notes
        -----
        For each lattice unit cell (``L1 * L2`` in total), the algorithm:

        1. Uses the current unit cell as reference.
        2. Applies the local pattern defined by ``unit_cell_ids``.
        3. Repeatedly shifts the reference unit cell by ``(sx_shift, sy_shift)`` after completing 
        one full pass through ``unit_cell_ids``.
        4. Stops when the starting global index is reached again or when ``r_max`` is exceeded.
        """
        validate_type(unit_cell_ids, ListWithShifts, "unit_cell_ids")
        validate_type(sx_shift, int, "sx_shift")
        validate_type(sy_shift, int, "sy_shift")
        validate_type(for_plaquettes, bool, "for_plaquettes")

        if for_plaquettes:
            num_of_ids = self.plaquettes.N
            ids_per_unit_cell = self.unit_cell.plaquettes.N
        else:
            num_of_ids = self.sites.N
            ids_per_unit_cell = self.unit_cell.sites.N

        if ignore_ids is not None:
            validate_type(ignore_ids, [list, int], "ignore_ids")
            id_to_ignore = np.zeros(num_of_ids, dtype=bool)
            id_to_ignore[ignore_ids] = True

        if r_max is not None:
            validate_type(r_max, int, "r_max")

        # Go through each unit cell of the lattice and find the global ids
        global_ids = []
        for idy_unit_cell in range(self.L2):
            for idx_unit_cell in range(self.L1):
                global_ids.append([])

                # go step by step further away from the initial reference point
                # until we are r_max steps away, or reach the starting point again
                r = 0 # distance between the wandering point and the origin point unit_cell_id
                k = 0 # counts the number of repeating patterns
                while True:
                    i = r%len(unit_cell_ids)

                    # reference point of the start point of the wandering unit cell
                    wandering_unit_cell_id = self.get_id_unit_cell(
                        idx_unit_cell+sx_shift*k+unit_cell_ids.x_shifts[i], 
                        idy_unit_cell+sy_shift*k+unit_cell_ids.y_shifts[i])
                    ref = wandering_unit_cell_id*ids_per_unit_cell
                    
                    # global id of the wandering point
                    global_id = ref + unit_cell_ids.entries[i]

                    # check if r_max is reached
                    if r_max is not None:
                        if r == r_max+1:
                            break
                    # check if starting id is reached
                    if r == 0:
                        id0 = global_id
                    else:
                        if id0 == global_id:
                            break
                    # check if should be ignored
                    if ignore_ids is not None:
                        add_to_pattern = not id_to_ignore[global_id]
                    else:
                        add_to_pattern = True
                    # collect id
                    if add_to_pattern:
                        global_ids[-1].append(global_id)
                    r += 1

                    # check if periodic structure is completed
                    if r%len(unit_cell_ids) == 0:
                        k+=1
        return global_ids


    # Node Wrapping
    ###############

    def wrap_local_nested_ListWithShifts_around_torus(
            self, nested_ListWithShifts: list, 
            N_nodes_of_ListWithShifts:int
    ):
        r""" 
        Transform a local list (or nested list) of :class:`ListWithShifts` into a global list,
        i.e., map the nodes of a unit cell onto a torus while handling boundary conditions.

        Parameters
        ----------
        nested_ListWithShifts : list[ListWithShifts] | list[list[ListWithShifts]]
            Local list or nested list of :class:`ListWithShifts` objects.  
            Example: list of sites forming a plaquette with 
            ``N_pl_local = len(list_node_list_local)``.
        N_nodes_of_ListWithShifts : int
            Number of local nodes. This may differ from ``len(list_node_list_local)``.  
            Example: number of sites in a unit cell, such that
            ``list_node_list_local.entries[i] < N_nodes_of_ListWithShifts``.
        """
        validate_type(
            nested_ListWithShifts, [list, (list, ListWithShifts)], "nested_ListWithShifts")
        validate_type(N_nodes_of_ListWithShifts, int, "N_nodes_of_ListWithShifts")

        N_nodes_local = len(nested_ListWithShifts)
        N_nodes = self.L1 * self.L2 * N_nodes_local
        nested_ListWithShifts_global = [[] for _ in range(N_nodes)]

        # Go through each unit cell of the lattice and fill it with the unit cell nodes
        for idy_unit_cell in range(self.L2):
            for idx_unit_cell in range(self.L1):
                # reference point of unit cell
                id_unit_cell = self.get_id_unit_cell(idx_unit_cell, idy_unit_cell)
                ref_node_id = id_unit_cell * N_nodes_local
                # go through each ListWithShifts (e.g. each list of sites forming the 
                # global_id-th plaquette)
                if isinstance(nested_ListWithShifts[0], ListWithShifts):
                    for local_node_id, local_ListWithShifts in enumerate(nested_ListWithShifts):
                        global_id = ref_node_id + local_node_id
                        ListWithShifts_global = self.local_to_global_ListWithShifts(
                            N_nodes_of_ListWithShifts, idx_unit_cell, idy_unit_cell, 
                            local_ListWithShifts)
                        nested_ListWithShifts_global[global_id] = ListWithShifts_global
                elif isinstance(nested_ListWithShifts[0][0], ListWithShifts):
                    for local_node_id in range(len(nested_ListWithShifts)):
                        global_id = ref_node_id + local_node_id
                        list_ListWithShifts_global = []
                        for local_ListWithShifts in nested_ListWithShifts[local_node_id]:
                            list_ListWithShifts_global.append( 
                                self.local_to_global_ListWithShifts(
                                    N_nodes_of_ListWithShifts, idx_unit_cell, idy_unit_cell,
                                    local_ListWithShifts))
                        nested_ListWithShifts_global[global_id] = list_ListWithShifts_global
                else:
                    raise TypeError(
                        "Unsupported parameter: list_node_list_local must be a list of "
                        "ListWithShifts with maximum nesting depth of 2. "
                        "Meaning one of the following must be ture: "
                        "isinstance(list_node_list_local[0], ListWithShifts) "
                        "or isinstance(list_node_list_local[0][0], ListWithShifts)")
        return nested_ListWithShifts_global

    def local_to_global_ListWithShifts(
            self, N_nodes_unit_cell:int, idx_unit_cell:int, idy_unit_cell:int, 
            local_node_list: ListWithShifts
    ):
        r"""
        Transform a local :class:`ListWithShifts` into a global :class:`ListWithShifts`.

        Parameters
        ----------
        N_nodes_unit_cell : int
            Number of nodes (e.g., sites or plaquettes) per unit cell.
        idx_unit_cell : int
            x-index of the unit cell.
        idy_unit_cell : int
            y-index of the unit cell.
        local_node_list : :class:`ListWithShifts`
            List of nodes to be mapped from local to global.
        """
        validate_type(N_nodes_unit_cell, int, "N_nodes_unit_cell")
        validate_type(idx_unit_cell, int, "idx_unit_cell")
        validate_type(idy_unit_cell, int, "idy_unit_cell")
        validate_type(local_node_list, ListWithShifts, "local_node_list")

        global_node_ids = []
        global_shifts_in_x = []
        global_shifts_in_y = []
        # Loop over all nodes
        for i in range(len(local_node_list.entries)):
            node_id_local = local_node_list.entries[i]
            idx_unit_cell_incr = local_node_list.x_shifts[i]
            idy_unit_cell_incr = local_node_list.y_shifts[i]

            idx_unit_cell_new = (idx_unit_cell + idx_unit_cell_incr) 
            idy_unit_cell_new = (idy_unit_cell + idy_unit_cell_incr)

            # Detect periodic wrapping
            if idx_unit_cell_new < 0:
                global_shifts_in_x.append(-1)
            elif idx_unit_cell_new >= self.L1:
                global_shifts_in_x.append(+1)
            else:
                global_shifts_in_x.append(0)

            if idy_unit_cell_new < 0:
                global_shifts_in_y.append(-1)
            elif idy_unit_cell_new >= self.L2:
                global_shifts_in_y.append(+1)
            else:
                global_shifts_in_y.append(0)

            idx_unit_cell_new = idx_unit_cell_new % self.L1
            idy_unit_cell_new = idy_unit_cell_new % self.L2

            id_unit_cell_new = self.get_id_unit_cell(idx_unit_cell_new, idy_unit_cell_new)
            node_id_global = id_unit_cell_new*N_nodes_unit_cell + node_id_local
            global_node_ids.append(node_id_global)
        return ListWithShifts(global_node_ids, global_shifts_in_x, global_shifts_in_y)

    def __str__(self):
            return (f"LatticeFromUnitCell: {self.L1} x {self.L2} unit cells "
            f"({self.sites.N} sites, {self.plaquettes.N} plaquettes), "
            f"unit cell type: {type(self.unit_cell).__name__}")

    def __repr__(self):
        return self.__str__()


###################################################################################################
# HELPER FUNCTIONS: ADJACENCY MAPPINGS
###################################################################################################

def get_adjacent_loops_for_sites(
        num_of_sites:int, list_of_loops:list[list[int]]) -> list[list[int]]:
    site_to_adjacent_loop = [[] for _ in range(num_of_sites)]
    for loop_id, site_ids in enumerate(list_of_loops):
        for site_id in site_ids:
            site_to_adjacent_loop[site_id].append(loop_id)
    return site_to_adjacent_loop


def get_adjacent_loops_for_local_changes(
        num_of_sites:int, list_of_loops:list[list[int]]|list[ListWithShifts], 
        list_of_local_changes:list[list[int]]|list[ListWithShifts], 
        filter_self:bool=False, filter_even:bool=False, as_padded_array:bool=False
) -> list[list[int]]:
    r""" 
    Parameters
    ----------
    num_of_sites : int
        Total number of sites in the system

    list_of_loops : list[list[int]]
        A list of loops, where each loop is specified by the site IDs it contains.
        For example:
        ``loop_0 = [5, 4, 7, 3]`` and
        ``list_of_loops = [loop_0, loop_1, ...]``.

    list_of_local_changes : list[list[int]]
        A list of local changes, where each change is specified by the site IDs it acts on. 
        For example:
        ``change_0 = [3, 5]`` and
        ``list_of_local_changes = [change_0, change_1, ...]``.

    filter_self : bool
        If ``True`` and ``list_of_loops`` is identical to ``list_of_local_changes``, the function 
        excludes the loop corresponding to the local change itself.
        For example for ``list_of_loops == list_of_local_changes = plaquettes.site_ids`` and 
        ``filter_self=True`` the function returns the nearest neighboring plaquettes of a 
        respective plaquettes.

    filter_even : bool
        If ``True``, a loop is considered affected by a local change only if it overlaps with that 
        local change on an **odd** number of sites. Loops that share an even (nonzero) number of 
        sites with the same local change are excluded. This is needed, for example, when applying 
        two local changes to the same loop leaves the loop effectively unaffected.

        If ``False``, any loop that shares at least one site with the local change is considered 
        affected.
    
    as_padded_array : bool
        Whether to convert the nested list into a fully padded NumPy array using ``-1`` as the 
        fill value.

    Returns
    -------
    local_change_to_adjacent_loop : list[list[int]]
        A list of length ``len(list_of_local_changes)``.
        Entry ``local_change_to_adjacent_loop[change_id]`` contains the IDs of all
        loops that are affected by performing the local change with index
        ``change_id``.
        A loop is considered affected if it shares at least one site ID with the
        corresponding local change.
    """
    validate_type(num_of_sites, int, "num_of_sites")
    validate_type(list_of_loops, [list, (list, ListWithShifts)], "list_of_loops")
    validate_type(list_of_local_changes, [list, (list, ListWithShifts)], "list_of_local_changes")
    validate_type(filter_self, bool, "filter_self")
    validate_type(filter_even, bool, "filter_even")
    validate_type(as_padded_array, bool, "as_padded_array")
    
    if isinstance(list_of_loops[0], ListWithShifts):
        list_of_loops = nested_ListWithShifts_to_nested_list(list_of_loops)
    if isinstance(list_of_local_changes[0], ListWithShifts):
        list_of_local_changes = nested_ListWithShifts_to_nested_list(list_of_local_changes)

    if filter_self:
        assert list_of_loops == list_of_local_changes
    site_id_to_adjacent_loop = get_adjacent_loops_for_sites(num_of_sites, list_of_loops)
    num_of_loops = len(list_of_loops)
    num_of_local_changes = len(list_of_local_changes)
    local_change_to_adjacent_loop = [[] for _ in range(num_of_local_changes)]

    loop_id_to_affected = np.zeros((num_of_loops,), dtype=np.int64)
    for local_change_id, sites in enumerate(list_of_local_changes):
        touched_loops = []
        for site_id in sites:
            for loop_id in site_id_to_adjacent_loop[site_id]:
                if filter_self and (local_change_id == loop_id):
                    continue
                if loop_id_to_affected[loop_id] == 0:
                    touched_loops.append(loop_id)
                loop_id_to_affected[loop_id] += 1
                    
        for loop_id in touched_loops:
            if filter_even:
                if loop_id_to_affected[loop_id]%2 != 0:
                    local_change_to_adjacent_loop[local_change_id].append(loop_id)
            elif loop_id_to_affected[loop_id] != 0:
                local_change_to_adjacent_loop[local_change_id].append(loop_id)
            loop_id_to_affected[loop_id] = 0
    if as_padded_array:
        local_change_to_adjacent_loop = nested_list_to_padded_array(local_change_to_adjacent_loop)
    return local_change_to_adjacent_loop



###################################################################################################
# HELPER FUNCTIONS: ListWithShifts CONVERTERS
###################################################################################################

def nested_list_to_nested_ListWithShifts(nested):
    """
    Recursively convert a nested structure of lists into a nested structure of 
    :class:`ListWithShifts` objects, assuming zero periodic shifts.

    Parameters
    ----------
    nested : list
        Nested list of integers. The innermost lists will be converted to ListWithShifts.

    Returns
    -------
    nested ListWithShifts structure
        Same nesting as input, but innermost lists are replaced with ListWithShifts.
    """

    # Base case: innermost list
    if isinstance(nested, list) and nested and all(isinstance(x, int) for x in nested):
        return ListWithShifts.from_list(nested)

    # Recursive case: list of lists
    elif isinstance(nested, list):
        return [nested_list_to_nested_ListWithShifts(sub) for sub in nested]
    else:
        raise TypeError(f"Unsupported type {type(nested)}. Expected nested list.")


def nested_ListWithShifts_to_nested_list(nested, attr:str="entries") -> list:
    """
    Recursively convert a attribute from a nested structure of :class:`ListWithShifts` objects
    into a nested structure of lists.

    Parameters
    ----------
    nested : ListWithShifts | list
        Nested list containing :class:`ListWithShifts` objects.
    attr : str
        Attribute name to extract (``"entries"``, ``"x_shifts"``, or ``"y_shifts"``).
    """

    # Base case: reached a ListWithShifts instance
    if isinstance(nested, ListWithShifts):
        return getattr(nested, attr)

    # Recursive case: a list (or tuple) of more nested elements
    elif isinstance(nested, (list, tuple)):
        return [nested_ListWithShifts_to_nested_list(sub, attr=attr) for sub in nested]
    else:
        raise TypeError(f"Unsupported type: {type(nested)} (expected list or ListWithShifts)")


def nested_ListWithShifts_to_padded_array(
        nested_ListWithShifts:list, attr:str="entries", fill_value=-1, dtype=int
) -> np.ndarray :
    validate_type(nested_ListWithShifts, list, "nested")
    validate_type(attr, str, "attr")

    nested_list = nested_ListWithShifts_to_nested_list(nested_ListWithShifts, attr=attr)
    padded_array = nested_list_to_padded_array(nested_list, fill_value=fill_value, dtype=dtype)
    return padded_array
