
# General Imports
import numpy as np
from copy import deepcopy

# Internal Imports
from main.model.lattice import (
    ListWithShifts, UnitCell, LatticeFromUnitCell, Sites, Plaquettes)
from main.model.unit_cell import HoneycombUnitCell, HoneycombRectangularUnitCell
from main.tools.utils import validate_type


###################################################################################################
# OCCUPATION STATE TO SPIN STATES
###################################################################################################

def occupation_states_to_spin_states(states:np.ndarray, m_ref:int, as_integer:bool=False):
    spins = states-m_ref/6
    if as_integer:
        smallest_magnitude = min(1-m_ref/6, m_ref/6)
        kappa = 1/smallest_magnitude
        spins *= kappa
        if not np.allclose(spins, np.round(spins)):
            raise ValueError("Spin values are not (close to) integers.")
        return np.round(spins).astype(int)
    else:
        return spins


###################################################################################################
# FULL HONEYCOMB TO SPIN ICE HONEYCOMB FOR SINGLE PARTITION
###################################################################################################

class to_HoneycombSpinIce(LatticeFromUnitCell):
    def __init__(self, honeycomb_lattice:LatticeFromUnitCell, partition_to_ignore:int):
        validate_type(honeycomb_lattice, LatticeFromUnitCell, "honeycomb_lattice")
        validate_type(partition_to_ignore, int, "partition_to_ignore")
        assert (
            isinstance(honeycomb_lattice.unit_cell, HoneycombUnitCell) or 
            isinstance(honeycomb_lattice.unit_cell, HoneycombRectangularUnitCell))
        assert partition_to_ignore in [0,1,2]

        honeycomb_spin_ice_lattice = deepcopy(honeycomb_lattice)
        self.L1:int = honeycomb_spin_ice_lattice.L1
        self.L2:int = honeycomb_spin_ice_lattice.L2
        self.unit_cell:UnitCell = honeycomb_spin_ice_lattice.unit_cell
        self.sites:Sites = honeycomb_spin_ice_lattice.sites
        self.plaquettes:Plaquettes = honeycomb_spin_ice_lattice.plaquettes

        self.partition_to_ignore = partition_to_ignore
        pl_ids_to_keep, pl_id_old_to_new = self._filter_plaquettes()
        pl_id_to_paris = self._get_pl_id_to_paris()
        worm_site_id_to_bonded_site_id = self._get_worm_site_id_to_bonded_site_id()

        self.pl_ids_to_keep:list[int] = pl_ids_to_keep
        self.pl_id_old_to_new:dict = pl_id_old_to_new
        self.pl_id_to_paris:list[list[list[int,int]]] = pl_id_to_paris
        self.worm_site_id_to_bonded_site_id = worm_site_id_to_bonded_site_id

    def _filter_plaquettes(self):
        num_of_pls = self.plaquettes.N
        L1 = self.L1
        part_to_pl_ids = self.unit_cell.get_partition_to_pl_ids(num_of_pls, L1)

        other_partitions = [(self.partition_to_ignore+1)%3, (self.partition_to_ignore+2)%3]
        part_A, part_B = min(other_partitions), max(other_partitions)
        pl_ids_to_keep = np.concatenate((part_to_pl_ids[part_A], part_to_pl_ids[part_B]), axis=0)
        
        # Create mapping from old to new sites
        pl_id_old_to_new = {old: new for new, old in enumerate(pl_ids_to_keep)}

        # Remove other two partitions
        pls = self.plaquettes
        pls.site_ids = [pls.site_ids[pl_id] for pl_id in pl_ids_to_keep]
        pls.coords = [pls.coords[pl_id] for pl_id in pl_ids_to_keep]
        pls.center_coords = pls.center_coords[pl_ids_to_keep]
        pls.nearest_neighbors = [
            [pl_id_old_to_new[n] for n in pls.nearest_neighbors[pl_id] if n in pl_id_old_to_new]
            for pl_id in pl_ids_to_keep]
        pls.N = len(pls.site_ids)

        return pl_ids_to_keep, pl_id_old_to_new

    def _get_pl_id_to_paris(self) -> list[list[list[int,int]]]:
        pl_id_to_paris = []
        for pl_id in range(self.plaquettes.N):
            pl_id_to_paris.append([])
            pl_sites = set(self.plaquettes.site_ids[pl_id])
            nn_pl_ids = self.plaquettes.nearest_neighbors[pl_id]
            
            for pl_id_nn in nn_pl_ids:
                nn_sites = set(self.plaquettes.site_ids[pl_id_nn])
                overlap = pl_sites & nn_sites
                pl_id_to_paris[-1].append(list(overlap))
        return pl_id_to_paris

    def _get_worm_site_id_to_bonded_site_id(self):
        pl_id_to_paris = self._get_pl_id_to_paris()
        num_sites = self.sites.N
        site_id_to_bonded_site_id = [None] * num_sites
        for pl_pairs in pl_id_to_paris:
            for pair in pl_pairs:
                assert len(pair) == 2
                for i in range(2):
                    site_id_current = pair[i]
                    site_id_bonded = pair[(i+1)%2]
                    if site_id_to_bonded_site_id[site_id_current] is None:
                        site_id_to_bonded_site_id[site_id_current] = site_id_bonded
                    else:
                        assert site_id_to_bonded_site_id[site_id_current] == site_id_bonded
        return site_id_to_bonded_site_id


    # Functions for methods
    #######################

    def get_monte_carlo_worm_network(self, **unit_cell_kwargs):
        check_site_ids_of_path = False

        pl_id_to_paris = self.pl_id_to_paris

        nearest_neighbors = []
        connecting_site_ids = []
        for pl_id in range(self.plaquettes.N):
            connecting_site_ids.append([])
            nn_pl_ids = self.plaquettes.nearest_neighbors[pl_id]
            nns = []
            for i, pl_id_nn in enumerate(nn_pl_ids):
                pair = pl_id_to_paris[pl_id][i]
                for site_id in pair:
                    nns.append(pl_id_nn)
                    connecting_site_ids[-1].append(ListWithShifts.from_list([site_id]))
            nearest_neighbors.append(ListWithShifts.from_list(nns))
        return nearest_neighbors, connecting_site_ids, check_site_ids_of_path
    
    def get_monte_carlo_local_loops(self, **unit_cell_kwargs):
        raise NotImplementedError(
            "`get_monte_carlo_local_loops` not implemented for HoneycombSpinIce.")
    
    def get_RK_loops(self, **unit_cell_kwargs):
        raise NotImplementedError(
            "`get_RK_loops` not implemented for HoneycombSpinIce.")



###################################################################################################
# PATTERN CREATOR
###################################################################################################

def get_patterns(lattice:LatticeFromUnitCell):
    assert isinstance(lattice.unit_cell, (HoneycombUnitCell, HoneycombRectangularUnitCell))
    num_of_pls = lattice.plaquettes.N
    pl_id_to_partition = lattice.unit_cell.get_pl_id_to_partition(num_of_pls, lattice.L1)
    
    # Create patterns for sites
    if isinstance(lattice.unit_cell, HoneycombUnitCell):
        unit_cell_site_ids =ListWithShifts([0,1,0,1], [0,0,1,1], [0,0,0,0])
        sx, sy = 1, 1
    else:
        unit_cell_site_ids =ListWithShifts(
            [1,0,1,0,2,3,2,3], [-1,0,0,1,1,1,2,2], [0,0,0,0,0,0,0,0])
        sx, sy = 3, 1
    pattern_site_ids = lattice.get_global_patterns(unit_cell_site_ids, sx, sy)
    pattern_length = None
    for site_ids in pattern_site_ids:
        if pattern_length is None:
            pattern_length = len(site_ids)
        else:
            assert pattern_length == len(site_ids)
    k_to_i_field = []
    for uc_id in range(len(pattern_site_ids)):
        k_to_i_field.append(pattern_site_ids[uc_id][0])
    # sort according to partitions (used for Honeycomp Spin Ice)
    part_to_k_to_i_field = [[],[],[]]
    for uc_id in range(len(pattern_site_ids)):
        ref_pl_id = uc_id*lattice.unit_cell.plaquettes.N
        part = pl_id_to_partition[ref_pl_id]
        for part_i in range(3):
            if part_i == part:
                part_to_k_to_i_field[part_i].append(pattern_site_ids[uc_id][0])
            else:
                part_to_k_to_i_field[part_i].append(None)

    # Create patterns for plaquettes
    if isinstance(lattice.unit_cell, HoneycombUnitCell):
        unit_cell_pl_ids = ListWithShifts([0,0], [0,1], [0,0])
        sx, sy = 1, 1
    else:
        unit_cell_pl_ids = ListWithShifts([0,0,1,1], [0,1,1,2], [0,0,0,0])
        sx, sy = 3, 1
    pattern_plaquette_ids = lattice.get_global_patterns(
        unit_cell_pl_ids, sx, sy, for_plaquettes=True)
    pattern_length = None
    for pl_ids in pattern_site_ids:
        if pattern_length is None:
            pattern_length = len(pl_ids)
        else:
            assert pattern_length == len(pl_ids)
    k_to_i_QP = []
    for uc_id in range(len(pattern_plaquette_ids)):
        k_to_i_QP.append(pattern_plaquette_ids[uc_id][0])
    # sort according to partitions (used for QP on only one partition)
    part_to_k_to_i_QP = [[],[],[]]
    for uc_id in range(len(pattern_plaquette_ids)):
        pl_id = pattern_plaquette_ids[uc_id][0]
        part = pl_id_to_partition[pl_id]
        for part_i in range(3):
            if part_i == part:
                part_to_k_to_i_QP[part_i].append(pattern_plaquette_ids[uc_id][0])
            else:
                part_to_k_to_i_QP[part_i].append(None)
    return (
        pattern_site_ids, k_to_i_field, part_to_k_to_i_field,
        pattern_plaquette_ids, k_to_i_QP, part_to_k_to_i_QP)



###################################################################################################
# FIND NONZERO PARTITION FOR C_ij_QP
###################################################################################################

def get_nonzero_partition(lattice:LatticeFromUnitCell, C_ij_QP:np.ndarray):
    assert isinstance(lattice.unit_cell, HoneycombUnitCell)
    C_min, C_max = np.min(C_ij_QP), np.max(C_ij_QP)
    C_threshold = C_max/10
    partition = None
    for pl_id in range(lattice.plaquettes.N):
        C_pl_id_max = C_ij_QP[pl_id, pl_id]
        if C_pl_id_max > C_threshold:
            partition = lattice.unit_cell.get_plaquette_partition(pl_id, lattice.L1)
            break
    assert partition is not None
    return partition
