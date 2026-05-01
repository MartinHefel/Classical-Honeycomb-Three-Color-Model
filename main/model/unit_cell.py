
# General Imports
import numpy as np
import tensornetwork as tn

# Internal Imports
from main.config import TENSORNETWORK_BACKEND
from main.model.lattice import Sites, Plaquettes, UnitCell
from main.model.lattice import ListWithShifts as LWS
from main.methods.tensor_network import get_T_tensor, get_delta_tensor

# Set tensornetwork backend
tn.set_default_backend(TENSORNETWORK_BACKEND)



###################################################################################################
# HONEYCOMB UNIT CELL
###################################################################################################

class HoneycombUnitCell(UnitCell):
    r"""
    Honeycomb unit cell:
    .. code-block:: text
        | .--> x
        | |
        | V y
        |       | 
        |    -- 0 -- 1 --
        |            |  p0
    
    """
    def __init__(
            self, normalize_to_lattice_bonds:bool=False, normalize_to_dual_lattice_bonds:bool=True,
            a_custom:float=None
    ):
        # Determine lattice scaling ---------------------------------------------------------------
        if a_custom is not None:
            assert normalize_to_lattice_bonds is False
            assert normalize_to_dual_lattice_bonds is False
            a = a_custom
        else:
            assert normalize_to_lattice_bonds ^ normalize_to_dual_lattice_bonds
            if normalize_to_lattice_bonds:
                a = 1.
            elif normalize_to_dual_lattice_bonds:
                a = 1/np.sqrt(3)

        # Build unit cell -------------------------------------------------------------------------
        sites = self._init_sites(a)
        plaquettes = self._init_plaquettes(sites)
        super().__init__(sites, plaquettes)

    # Functions for building unit cell
    ##################################

    def _init_sites(self, a):
        a1 = np.array([1,0])*a
        a2 = np.array([1/2, np.sqrt(3)/2])*a
        c = 2/np.sqrt(12)
        u1 = c*np.array([1,1])
        u2 = 2*c*np.array([1,1])
        us = np.array([u1, u2])
        d1, d2 = 3*c, 3*c
        nearest_neighbors = [
            LWS([1,1,1], [0,0,-1], [-1,0,0]), LWS([0,0,0], [0,0,1], [1,0,0])]
        return Sites(a1, a2, d1, d2, us, nearest_neighbors)

    def _init_plaquettes(self, sites:Sites):
        site_ids = [
            LWS([1,0,1,0,1,0], [0,1,1,1,0,0], [0,0,0,1,1,1])]
        nearest_neighbors = [
            LWS([0,0,0,0,0,0], [1,1,0,-1,-1,0], [-1,0,1,1,0,-1])]
        return Plaquettes(site_ids, nearest_neighbors, sites)

    # Functions for methods
    #######################

    def get_monte_carlo_worm_network(self, test_para_A=None):
        if test_para_A is not None:
            print("print in worm",test_para_A)
        nearest_neighbors = [
            LWS([0,0,0,0,0,0], [1,2,1,-1,-2,-1], [-2,-1,1,2,1,-1])]
        connecting_site_ids = [
            [LWS([0,1], [1,1], [0,-1]), LWS([1,0], [1,2], [0,0]),
             LWS([0,1], [1,1], [1,1]), LWS([1,0], [0,0], [1,2]),
             LWS([0,1], [0,-1], [1,1]), LWS([1,0], [0,0], [0,0])]]
        check_site_ids_of_path = False
        return nearest_neighbors, connecting_site_ids, check_site_ids_of_path
    
    def get_monte_carlo_local_loops(self, m_ref:int):
        if m_ref == 3:
            local_loops = self.plaquettes.site_ids
        elif m_ref in [2, 4]:
            local_loops = [
                LWS([0,1,1,0,1,0,0,1], [0,0,1,2,2,2,1,0], [1,0,-1,-1,-1,0,1,1]),
                LWS([1,0,0,1,0,1,1,0], [0,1,2,2,2,1,0,0], [0,0,0,0,1,1,1,1]),
                LWS([0,1,1,0,1,0,0,1], [1,1,1,1,0,0,0,0], [0,0,1,2,2,2,1,0])]
        else:
            raise NotImplementedError(f"m_ref={m_ref} not implemented")
        return local_loops

    def get_RK_loops(self, m_ref:int):
        return self.get_monte_carlo_local_loops(m_ref)

    def get_A_transfer(self, m_ref:int):
        r"""
        .. code-block:: text
        
            |         |  \ /  |          
            |        (3) (4) (5)         T_l  del  del T_r
            |        del del del         tl,t,t,l  r,t,t,tr    t            r
            |   \ | /   \ | /   \ | /       \||\    /||/        \          /
            |   T_l(0)  T_m(1)  T_r(2)   ->  A_transfer      ->  A_transfer
            |   / | \   / | \   / | \       /||/    \||\        /          \
            |        del del del         bl,b,b,l  r,b,b,br    l            b
            |        (6) (7) (8)         T_l  del  del T_r
            |         |  / \  |
        
        """
        num_of_sites_per_A_transfer = 6

        # Build tensors
        tensors = []
        for pl_id in range(3):
            T = get_T_tensor(6, m_ref)
            tensors.append(tn.Node(T, name=f"T_{pl_id}"))
        for site_id in range(6):
            delta = get_delta_tensor(3)
            tensors.append(tn.Node(delta, name=f"delta_{site_id}"))
        
        # Open legs
        top    = [tensors[0][5], tensors[0][0], tensors[3][0], tensors[4][1]]
        right  = [tensors[4][2], tensors[5][0], tensors[2][0], tensors[2][1]]
        bottom = [tensors[7][1], tensors[8][0], tensors[2][3], tensors[2][2]]
        left   = [tensors[0][4], tensors[0][3], tensors[6][0], tensors[7][2]]

        # Connect legs
        tensors[0][1] ^ tensors[3][2]
        tensors[0][2] ^ tensors[6][1]
        tensors[1][0] ^ tensors[4][0]
        tensors[1][1] ^ tensors[5][2]
        tensors[1][2] ^ tensors[8][1]
        tensors[1][3] ^ tensors[7][0]
        tensors[1][4] ^ tensors[6][2]
        tensors[1][5] ^ tensors[3][1]
        tensors[2][4] ^ tensors[8][2]
        tensors[2][5] ^ tensors[5][1]

        # Contract connected legs
        A_transfer = tn.contractors.auto(tensors, output_edge_order=top+right+bottom+left)
        
        # Merge open legs
        tn.flatten_edges(top)
        tn.flatten_edges(right)
        tn.flatten_edges(bottom)
        tn.flatten_edges(left)

        return np.array(A_transfer.tensor), num_of_sites_per_A_transfer
    
    # Additional unit cell specific functions
    #########################################

    def get_plaquette_partition(self, pl_id:int, L1:int):
        assert L1%3 == 0
        i = pl_id // L1
        j = pl_id % L1
        return (j-i) % 3
    
    def get_pl_id_to_partition(self, num_of_pls:int, L1:int):
        pl_id_to_partition = np.empty((num_of_pls,), dtype=np.int64)
        for pl_id in range(num_of_pls):
            pl_id_to_partition[pl_id] = self.get_plaquette_partition(pl_id, L1)
        return pl_id_to_partition

    def get_partition_to_pl_ids(self, num_of_pls:int, L1:int):
        get_partition_to_pl_ids = np.empty((3, num_of_pls//3), dtype=np.int64)
        partition_counter = np.zeros((3,), dtype=np.int64)
        for pl_id in range(num_of_pls):
            part = self.get_plaquette_partition(pl_id, L1)
            pos = partition_counter[part]
            assert pos < num_of_pls//3
            get_partition_to_pl_ids[part, pos] = pl_id
            partition_counter[part] += 1
        for part in range(3):
            assert partition_counter[part] == num_of_pls//3
        return get_partition_to_pl_ids


###################################################################################################
# HONEYCOMB RECTANGULAR UNIT CELL
###################################################################################################

class HoneycombRectangularUnitCell(UnitCell):
    r"""
    Rectangular Honeycomb unit cell:
    .. code-block:: text
        | .--> x
        | |
        | V y
        |            |
        |    -- 0 -- 1 --
        |       |   p0
        |    -- 2 -- 3 --
        |            |   p1
    
    """
    def __init__(
            self, normalize_to_lattice_bonds:bool=False, normalize_to_dual_lattice_bonds:bool=True,
            a_custom:float=None
    ):
        # Determine lattice scaling ---------------------------------------------------------------
        if a_custom is not None:
            assert normalize_to_lattice_bonds is False
            assert normalize_to_dual_lattice_bonds is False
            a = a_custom
        else:
            assert normalize_to_lattice_bonds ^ normalize_to_dual_lattice_bonds
            if normalize_to_lattice_bonds:
                a = 1.
            elif normalize_to_dual_lattice_bonds:
                a = 1/np.sqrt(3)

        # Build unit cell -------------------------------------------------------------------------     
        sites = self._init_sites(a)
        plaquettes = self._init_plaquettes(sites)
        super().__init__(sites, plaquettes)
    
    # Functions for building unit cell
    ##################################

    def _init_sites(self, a):
        al = 2*np.pi/6
        r = a/(2*np.sin(al/2))
        Dx = 2*r*np.sin(al)
        Dy = 2*r+a
        coords = np.array([   
                [ 0,            r*(1-np.cos(al))   ],
                [ r*np.sin(al), 0                  ],
                [ 0,            r*(1-np.cos(al))+a ],
                [ r*np.sin(al), 2*r                ]],
            dtype=np.float64)
        nearest_neighbors = [
            LWS([1,2,1], [0,0,-1], [0,0,0]),
            LWS([0,0,3], [+1,0,0], [0,0,-1]),
            LWS([3,3,0], [0,-1,0], [0,0,0]),
            LWS([2,1,2], [+1,0,0], [0,+1,0])]
        return Sites.from_cartesian_coordinates(Dx, Dy, coords, nearest_neighbors)

    def _init_plaquettes(self, sites:Sites):
        site_ids = [
            LWS([0,1,0,2,3,2], [0,0,+1,+1,0,0],   [0,0,0,0,0,0]),
            LWS([3,2,3,1,0,1], [0,+1,+1,+1,+1,0], [0,0,0,+1,+1,+1])]
        nearest_neighbors = [
            LWS([1,0,1,1,0,1], [0,1,0,-1,-1,-1], [-1,0,0,0,0,-1]),
            LWS([0,1,0,0,1,0], [1,1,1,0,-1,0], [0,0,1,1,0,0])]
        return Plaquettes(site_ids, nearest_neighbors, sites)
    
    # Functions for methods
    #######################

    def get_monte_carlo_worm_network(self):
        nearest_neighbors = [
            LWS([0,1,1,0,1,1], [0,1,1,0,-2,-2], [-1,-1,0,1,0,-1]),
            LWS([1,0,0,1,0,0], [0,2,2,0,-1,-1], [-1,0,1,1,1,0])]
        connecting_site_ids = [
            [LWS([1,3], [0,0],  [0,-1]), LWS([0,1], [1,1],  [0,0]),
             LWS([2,3], [1,1],  [0,0]),  LWS([3,1], [0,0],  [0,1]),
             LWS([2,3], [0,-1], [0,0]),  LWS([0,1], [0,-1], [0,0])],
            [LWS([2,0], [1,1],  [0,0]),  LWS([3,2], [1,2],  [0,0]),
             LWS([1,0], [1,2],  [1,1]),  LWS([0,2], [1,1],  [1,1]),
             LWS([1,0], [0,0],  [1,1]),  LWS([3,2], [0,0],  [0,0])]]
        check_site_ids_of_path=False
        return nearest_neighbors, connecting_site_ids, check_site_ids_of_path
    
    def get_monte_carlo_local_loops(self, m_ref:int):
        if m_ref == 3:
            local_loops = self.plaquettes.site_ids
        elif m_ref in [2, 4]:
            local_loops = [
                LWS([3,2, 3,1, 2,3, 2,0], [0,1, 1,1, 1,0, 0,0], [-1,-1, -1,0, 0,0, 0,0]),
                LWS([1,0, 2,3, 3,2, 0,1], [1,2, 2,1, 0,0, 0,0], [0,0, 0,0, 0,0, 0,0]),
                LWS([3,1,0,1,2,0,1,0], [1,1, 1,0, 0,0, 0,1], [0,1, 1,1, 0,0, 0,0]),
                LWS([0,1,0,2,1,0,1,3], [1,1, 2,2, 1,1, 0,0], [0,0, 0,0, 1,1, 1,0]),
                LWS([2,3,1,0,0,1,3,2], [2,2, 2,2, 1,0, 0,1], [0,0, 1,1, 1,1, 0,0]),
                LWS([0,2,3,2,1,3,2,3], [2,2, 1,1, 0,0, 1,1], [1,1, 1,1, 1,0, 0,0])]
        else:
            raise NotImplementedError(f"m_ref={m_ref} not implemented")
        return local_loops
    
    def get_RK_loops(self, m_ref:int):
        return self.get_monte_carlo_local_loops(m_ref)

    # Additional unit cell specific functions
    #########################################

    def get_plaquette_partition(self, pl_id:int, L1:int):
        assert L1%3 == 0
        part = {0: 0, 1: 2, 2: 1}[pl_id%3]
        return part

    def get_pl_id_to_partition(self, num_of_pls:int, L1:int):
        pl_id_to_partition = np.empty((num_of_pls,), dtype=np.int64)
        for pl_id in range(num_of_pls):
            pl_id_to_partition[pl_id] = self.get_plaquette_partition(pl_id, L1)
        return pl_id_to_partition

    def get_partition_to_pl_ids(self, num_of_pls:int, L1:int):
        get_partition_to_pl_ids = np.empty((3, num_of_pls//3), dtype=np.int64)
        partition_counter = np.zeros((3,), dtype=np.int64)
        for pl_id in range(num_of_pls):
            part = self.get_plaquette_partition(pl_id, L1)
            pos = partition_counter[part]
            assert pos < num_of_pls//3
            get_partition_to_pl_ids[part, pos] = pl_id
            partition_counter[part] += 1
        for part in range(3):
            assert partition_counter[part] == num_of_pls//3
        return get_partition_to_pl_ids


###################################################################################################
# SQUARE-SPIN-ICE UNIT CELL
###################################################################################################

class SquareSpinIceUnitCell(UnitCell):
    r"""
    Spin Ice on square lattice unit cell:
    .. code-block:: text
        | .--> x
        | |
        | V y
        |      |      |
        |    - 0 ---- 1 ----
        |      |  p0  |  X
        |    - 2 ---- 3 ----
        |      |  X   |  p1

    
    """
    def __init__(self, a:float=1.):
        sites = self._init_sites(a)
        plaquettes = self._init_plaquettes(sites)
        super().__init__(sites, plaquettes)

    # Functions for building unit cell
    ##################################

    def _init_sites(self, a):
        Dx = 2*a
        Dy = 2*a
        coords = np.array([[0, 0], [a, 0], [0, a], [a, a]], dtype=np.float64)
        nearest_neighbors = [
            LWS([2,1,2,1], [0,0,0,-1], [-1,0,0,0]),
            LWS([3,0,3,0], [0,1,0,0], [-1,0,0,0]),
            LWS([0,3,0,3], [0,0,0,-1], [0,0,1,0]),
            LWS([1,2,1,2], [0,1,0,0], [0,0,1,0])]
        return Sites.from_cartesian_coordinates(Dx, Dy, coords, nearest_neighbors)

    def _init_plaquettes(self, sites:Sites):
        site_ids = [
            LWS([0,1,3,2], [0,0,0,0], [0,0,0,0]),
            LWS([3,2,0,1], [0,1,1,0], [0,0,1,1])]
        nearest_neighbors = [
            LWS([1,1,1,1], [0,0,-1,-1], [-1,0,0,-1]),
            LWS([0,0,0,0], [1,1,0,0], [0,1,1,0])]
        return Plaquettes(site_ids, nearest_neighbors, sites)
    
    # Functions for methods
    #######################

    def get_monte_carlo_worm_network(self):
        nearest_neighbors = [
            LWS([1,1,1,1], [0,0,-1,-1], [-1,0,0,-1]),
            LWS([0,0,0,0], [1,1,0,0], [0,1,1,0])]
        connecting_site_ids = [
            [LWS([1], [0], [0]), LWS([3], [0], [0]),
             LWS([2], [0], [0]), LWS([0], [0], [0])],
            [LWS([2], [1], [0]), LWS([0], [1], [1]),
             LWS([1], [0], [1]), LWS([3], [0], [0])]]
        check_site_ids_of_path = False
        return nearest_neighbors, connecting_site_ids, check_site_ids_of_path

    def get_RK_loops(self):
        RK_loops = [
            LWS([2,3,1,0], [0,0,0,0], [0,0,1,1]),
            LWS([1,0,2,3], [0,1,1,0], [0,0,0,0])]
        return RK_loops
    
    # Additional unit cell specific functions
    #########################################

    def get_plaquette_partition(self, pl_id:int, L1:int):
        return pl_id%2
    
    def get_pl_id_B(self, uc_id:int, l:int, L1:int, L2:int):
        r"""
        .. code-block:: text
            | .--> x
            | |
            | V y
            |      |      |
            |    - 0 ---- 1 ----
            |      |  A1  |  B2
            |      |      |  (1)
            |    - 2 ---- 3 ----
            |      |  B1  |  A2 
            |      |  (0) | (ref)
        """
        if l in [0,1]:
            pl_id_B = 2*uc_id+l
        elif l == 2:
            uc_id_right = (uc_id+1)
            if uc_id_right%L1 == 0:
                row = uc_id//L1
                uc_id_right = row*L1
            pl_id_B = 2*uc_id_right
        elif l == 3:
            uc_id_below = (uc_id+L1)%(L1*L2)
            pl_id_B = 2*uc_id_below+1
        else:
            pl_id_B = None
        return pl_id_B
