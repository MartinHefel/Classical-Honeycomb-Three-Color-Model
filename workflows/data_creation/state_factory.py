
# General Imports
import numpy as np

# Internal Imports
from main import model



def get_initial_state_for_MC(m_ref:int, lattice:model.lattice.LatticeFromUnitCell, is_for_spin_ice:bool=False):
    assert m_ref in [1,2,3]
    num_of_sites = lattice.sites.N
    initial_state = np.zeros((num_of_sites,), dtype=bool)

    if m_ref == 1:
        if not is_for_spin_ice:
            for pl_id, site_ids in enumerate(lattice.plaquettes.site_ids):
                partition = lattice.unit_cell.get_plaquette_partition(pl_id, lattice.L1)
                if partition == 0:
                    initial_state[site_ids[0]] = 1
        else:
            for site_id in range(lattice.sites.N):
                idy = site_id//(2*lattice.L1)
                idx = site_id%(2*lattice.L1)
                if idy%2 == 0 and idx%3==0:
                    initial_state[site_id] = 1
    elif m_ref == 2:
        for pl_id, site_ids in enumerate(lattice.plaquettes.site_ids):
                partition = lattice.unit_cell.get_plaquette_partition(pl_id, lattice.L1)
                if partition == 0:
                    idy = pl_id//lattice.L1
                    if (idy)%3 == 0:
                        initial_state[site_ids[0]] = 1
                        initial_state[site_ids[2]] = 1
                    elif (idy)%3 == 1:
                        initial_state[site_ids[3]] = 1
                        initial_state[site_ids[5]] = 1
                    else:
                        initial_state[site_ids[1]] = 1
                        initial_state[site_ids[4]] = 1
    elif m_ref == 3:
        for pl_id, site_ids in enumerate(lattice.plaquettes.site_ids):
            partition = lattice.unit_cell.get_plaquette_partition(pl_id, lattice.L1)
            if partition == 0:
                initial_state[site_ids[1]] = 1
                initial_state[site_ids[3]] = 1
                initial_state[site_ids[5]] = 1
    return initial_state



def place_defects(
        state:np.ndarray, num_of_extra_particles:int, num_of_excitations:int, m_ref:int,
        lattice:model.lattice.LatticeFromUnitCell
):
    site_id_to_adj_pl_ids = lattice.get_adjacent_plaquettes_for_sites()

    # place/remove particles (create +,+,+/-,-,- QP triplets)
    if num_of_extra_particles != 0:
        occupation = (num_of_extra_particles < 0) # 0/1 for placing/removing
        num_of_modified_particles = 0
        while num_of_modified_particles != abs(num_of_extra_particles):
            site_id_rand = np.random.randint(lattice.sites.N)
            if state[site_id_rand] == occupation:
                QP_nearby = _is_quasi_particle_nearby(
                    site_id_rand, m_ref, state, site_id_to_adj_pl_ids, lattice)
                if not QP_nearby:
                    state[site_id_rand] = not state[site_id_rand]
                    num_of_modified_particles += 1

    # create excitations (create +,- QP pairs)
    if num_of_excitations!=0:
        num_of_created_excitations = 0
        bonds = lattice.sites.get_bonds()
        num_of_bonds = len(bonds)
        while num_of_created_excitations != num_of_excitations:
            bond_id_rand = np.random.randint(num_of_bonds)
            site_id_A, site_id_B = bonds[bond_id_rand]
            if state[site_id_A] != state[site_id_B]:
                QP_nearby_A = _is_quasi_particle_nearby(
                    site_id_A, m_ref, state, site_id_to_adj_pl_ids, lattice)
                QP_nearby_B = _is_quasi_particle_nearby(
                    site_id_B, m_ref, state, site_id_to_adj_pl_ids, lattice)
                QP_nearby = QP_nearby_A or QP_nearby_B
                if not QP_nearby:
                    state[site_id_A] = not state[site_id_A]
                    state[site_id_B] = not state[site_id_B]
                    num_of_created_excitations += 1
    return state



def _is_quasi_particle_nearby(
        site_id:int, m_ref:int, state:np.ndarray, site_id_to_adj_pl_ids:list[list[int]], 
        lattice:model.lattice.LatticeFromUnitCell
):
    quasi_particle_nearby = False
    pl_ids = site_id_to_adj_pl_ids[site_id]
    for pl_id in pl_ids:
        site_ids = lattice.plaquettes.site_ids[pl_id].entries
        q = np.sum(state[site_ids])-m_ref
        if q != 0:
            quasi_particle_nearby = True
            break
    return quasi_particle_nearby