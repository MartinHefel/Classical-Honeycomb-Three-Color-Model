
# General imports
import numpy as np
import pytest

# Internal imports
from main import model, methods



@pytest.mark.parametrize("UnitCell", model.ALL_UNIT_CELLS)
def test_commensurate_states_half_filling(UnitCell:model.lattice.UnitCell):
    lattice = _get_lattice(UnitCell)
    m_ref = lattice.get_m_ref(lattice.sites.N//2)
    _check_if_commensurate_states_can_be_found(lattice, m_ref)



@pytest.mark.parametrize(
"UnitCell, m_ref", [
    (model.unit_cell.HoneycombUnitCell, 1),
    (model.unit_cell.HoneycombUnitCell, 2),
    (model.unit_cell.HoneycombUnitCell, 4),
    (model.unit_cell.HoneycombUnitCell, 5),
    (model.unit_cell.HoneycombRectangularUnitCell, 1),
    (model.unit_cell.HoneycombRectangularUnitCell, 2),
    (model.unit_cell.HoneycombRectangularUnitCell, 4),
    (model.unit_cell.HoneycombRectangularUnitCell, 5),
    ])
def test_commensurate_states(
    UnitCell:model.lattice.UnitCell, m_ref:int
):
    lattice = _get_lattice(UnitCell)
    _check_if_commensurate_states_can_be_found(lattice, m_ref)



@pytest.mark.parametrize(
    "UnitCell, m_ref", [
        (model.unit_cell.HoneycombUnitCell, 1),
        (model.unit_cell.HoneycombUnitCell, 2),
        (model.unit_cell.HoneycombUnitCell, 3),
        (model.unit_cell.HoneycombRectangularUnitCell, 1),
        (model.unit_cell.HoneycombRectangularUnitCell, 2),
        (model.unit_cell.HoneycombRectangularUnitCell, 3)
        ])
def test_excited_states(
    UnitCell:model.lattice.UnitCell, m_ref:int
):
    inputs = [(1,0), (0,1), (0,2)]
    for func_input in inputs:
        level_index, num_of_extra_particles = func_input
        lattice = _get_lattice(UnitCell)
        _check_if_excited_states_can_be_found(
            lattice, m_ref, level_index, num_of_extra_particles, number_of_final_checks=4)



def _get_lattice(UnitCell:model.lattice.UnitCell):
    unit_cell = UnitCell()
    L1 = model.ALL_UNIT_CELLS_BRUTE_FORCE_SIZES[unit_cell.__class__.__name__]["L1"]
    L2 = model.ALL_UNIT_CELLS_BRUTE_FORCE_SIZES[unit_cell.__class__.__name__]["L2"]
    return model.lattice.LatticeFromUnitCell(L1, L2, unit_cell)



def _check_if_commensurate_states_can_be_found(
        lattice:model.lattice.LatticeFromUnitCell, m_ref:int, is_HC_SI:bool=False,
        number_of_final_checks:int=3
):
    UnitCell = lattice.unit_cell.__class__
    N_filling = lattice.get_N_filling(m_ref)

    # Brute Force
    ground_states_BF, _ = methods.hamiltonian.get_all_basis_states(
        m_ref, N_filling, ground_states_only=True, print_progress=False, lattice=lattice)
    number_of_states_BF = ground_states_BF.number_of_states
    
    # Monte Carlo
    if is_HC_SI:
        worm_site_id_to_bonded_site_id = lattice.worm_site_id_to_bonded_site_id
    else:
        worm_site_id_to_bonded_site_id = None
    initial_state = ground_states_BF.get_state(0) # use one state as initial state (could by any)
    ground_states_MC, _ = methods.monte_carlo.get_all_states_within_manifold(
        initial_state, 0, m_ref, 0., print_progress=False, lattice=lattice, 
        number_of_final_checks=number_of_final_checks, 
        worm_site_id_to_bonded_site_id=worm_site_id_to_bonded_site_id)
    number_of_states_MC = ground_states_MC.number_of_states
    
    # Tensor Network Contraction Method
    number_of_states_TNM, _ = methods.tensor_network.get_number_of_all_states_within_manifold(
        m_ref, lattice=lattice)

    # Ensure all Methods find the same number of ground states
    print(number_of_states_BF, number_of_states_MC, number_of_states_TNM)
    assert number_of_states_BF == number_of_states_MC == number_of_states_TNM, (
        f"{UnitCell.__name__} with m_ref={m_ref}: States have different #gs: "
        f"{number_of_states_BF} (BF), {number_of_states_MC} (MC), "
        f"{number_of_states_TNM} (TNM)")
    
    # Ensure that Monte Carlo and Brute Force obtain the same states
    _check_if_states_match(
        UnitCell, m_ref, ground_states_BF.as_array(), ground_states_MC.as_array())
    return ground_states_MC.as_array()


def _check_if_excited_states_can_be_found(
        lattice:model.lattice.LatticeFromUnitCell, m_ref:int,
        level_index:int=0, num_of_extra_particles:int=0,
        number_of_final_checks:int=3
):
    is_Honeycomb = (
        isinstance(lattice.unit_cell, model.unit_cell.HoneycombUnitCell) or
        isinstance(lattice.unit_cell, model.unit_cell.HoneycombRectangularUnitCell))
    assert is_Honeycomb, ("Only for HoneycombUnitCell or HoneycombRectangularUnitCell")
    UnitCell = lattice.unit_cell.__class__

    if level_index != 0:
        assert num_of_extra_particles == 0
    if num_of_extra_particles != 0:
        assert level_index == 0

    N_filling = lattice.get_N_filling(m_ref) + num_of_extra_particles
    field_update_method = 0 # worm method

    # Brute Force
    states_BF, _ = methods.hamiltonian.get_all_basis_states(
        m_ref, N_filling, print_progress=False, lattice=lattice)
    states_BF = states_BF.extract_energy_level(level_index)
    number_of_states_BF = states_BF.number_of_states
    states_BF_array = states_BF.as_array()
    
    # Monte Carlo
    if num_of_extra_particles == 0 and level_index != 0:
        # Sectors are disconnected since +,- QP always on the same partition
        initial_states = [None, None, None]
        unit_cell = lattice.unit_cell
        pl_id_to_site_ids = model.lattice.nested_ListWithShifts_to_nested_list(
            lattice.plaquettes.site_ids)
        for state_id in range(number_of_states_BF):
            for pl_id in range(lattice.plaquettes.N):
                site_ids = pl_id_to_site_ids[pl_id]
                n = np.sum(states_BF_array[state_id, site_ids])
                if n != m_ref:
                    part = unit_cell.get_plaquette_partition(pl_id, lattice.L1)
                    if initial_states[part] is None:
                        print(UnitCell, part, pl_id, state_id)
                        initial_states[part] = states_BF_array[state_id,:]
                    break
    else:
        initial_states = [states_BF.get_state(0)]
    state_MC_list = []

    for initial_state in initial_states:
        ground_states_MC, _ = methods.monte_carlo.get_all_states_within_manifold(
            initial_state, field_update_method, m_ref, p_quasi_particle_hop=0.5, lattice=lattice,
            number_of_final_checks=number_of_final_checks)
        state_MC_list.append(ground_states_MC.as_array())
    states_MC_array = np.concatenate(state_MC_list, axis=0)

    # Ensure that Monte Carlo and Brute Force obtain the same states
    extra_info = (f", level_index={level_index}, num_of_extra_particles={num_of_extra_particles}")
    _check_if_states_match(UnitCell, m_ref, states_BF_array, states_MC_array, extra_info)



def _check_if_states_match(
        UnitCell:model.lattice.UnitCell, m_ref, BF_states:np.ndarray, MC_states:np.ndarray,
        extra_info:str = ""
):
    v1 = MC_states.view([('', MC_states.dtype)] * MC_states.shape[1])
    v2 = BF_states.view([('', BF_states.dtype)] * BF_states.shape[1])
    u1 = np.unique(v1)
    u2 = np.unique(v2)
    assert u1.shape == u2.shape, (
        f"{UnitCell.__name__} with m_ref={m_ref}{extra_info}: Different number of unique "
        f"states: {u1.shape[0]} (MC) vs {u2.shape[0]} (BF)")
    assert np.all(u1 == u2), (
        f"{UnitCell.__name__} with m_ref={m_ref}{extra_info}: Ground states from Monte "
        f"Carlo and brute force do not match")
