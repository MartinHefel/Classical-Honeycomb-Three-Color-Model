
# General imports
import numpy as np
import pytest

# Internal imports
from main import model, methods, honeycomb_specific
from test_method_agreement_finding_states import _get_lattice


@pytest.mark.parametrize(
"UnitCell, m_ref", [
    (model.unit_cell.HoneycombUnitCell, 2),
    (model.unit_cell.HoneycombUnitCell, 3),
    (model.unit_cell.HoneycombUnitCell, 4),
    (model.unit_cell.HoneycombRectangularUnitCell, 2),
    (model.unit_cell.HoneycombRectangularUnitCell, 3),
    (model.unit_cell.HoneycombRectangularUnitCell, 4),
    ])
def test_ground_state_degeneracy_matches_connected_sector_count(
    UnitCell:model.lattice.UnitCell, m_ref:int
):
    lattice = _get_lattice(UnitCell)
    N_filling = lattice.get_N_filling(m_ref)

    # Get classical ground states as basis states
    states, _ = methods.hamiltonian.get_all_basis_states(
        m_ref, N_filling, lattice=lattice, ground_states_only=True)
    basis_state_StNrs = states.as_StNrs()

    # Get RK ground states
    V_RK = 1.
    t_RK = 1.
    ground_states, _ = methods.hamiltonian.get_RK_ground_state(
        V_RK, t_RK, m_ref, basis_state_StNrs=basis_state_StNrs, 
        lattice=lattice, RK_loops_kwargs={"m_ref": m_ref})
    RK_degeneracy = len(ground_states)

    # Get connectivity
    loop_list = lattice.get_RK_loops(m_ref=m_ref)
    loop_list = model.lattice.nested_ListWithShifts_to_nested_list(loop_list)
    sector_collections = honeycomb_specific.sectors.get_connected_sectors(
        states, loop_list)
    sector_degeneracy = len(sector_collections)

    # Check if degeneracy matches
    assert RK_degeneracy == sector_degeneracy, (
        f"RK degeneracy ({RK_degeneracy}) does not match "
        f"sector degeneracy ({sector_degeneracy}) "
        f"for UnitCell={UnitCell.__name__}, m_ref={m_ref}")



@pytest.mark.parametrize(
"UnitCell, m_ref", [
    (model.unit_cell.HoneycombUnitCell, 2),
    (model.unit_cell.HoneycombUnitCell, 3),
    (model.unit_cell.HoneycombUnitCell, 4),
    (model.unit_cell.HoneycombRectangularUnitCell, 2),
    (model.unit_cell.HoneycombRectangularUnitCell, 3),
    (model.unit_cell.HoneycombRectangularUnitCell, 4),
    ])
def test_RK_point_ground_states_form_equal_weight_superposition(UnitCell:model.lattice.UnitCell, m_ref:int
):
    lattice = _get_lattice(UnitCell)
    N_filling = lattice.get_N_filling(m_ref)

    # Get classical ground states as basis states
    states, _ = methods.hamiltonian.get_all_basis_states(
        m_ref, N_filling, lattice=lattice, ground_states_only=True)

    # Filter winding sector
    states, minimal_winding = honeycomb_specific.sectors.get_minimal_winding_sector(
        m_ref, states, lattice)
    basis_state_StNrs = states.as_StNrs()
    
    # Get RK ground states
    V_RK = 1.
    t_RK = 1.
    ground_states, _ = methods.hamiltonian.get_RK_ground_state(
        V_RK, t_RK, m_ref, basis_state_StNrs=basis_state_StNrs, 
        lattice=lattice, RK_loops_kwargs={"m_ref": m_ref})
    degeneracy = len(ground_states)
    
    # Check equal weight superposition
    total_non_zero = np.zeros((ground_states[0].number_of_basis_states), dtype=np.int64)
    for i in range(degeneracy):
        amplitudes = ground_states[i].amplitudes_as_array()
        non_zero_mask = (np.abs(amplitudes) > 1e-5).astype(int)
        non_zero_indices = np.where(non_zero_mask)[0]
        expected_amp = 1 / np.sqrt(np.sum(non_zero_mask))
        assert (np.allclose(amplitudes[non_zero_indices], expected_amp, atol=1e-5) or
                np.allclose(amplitudes[non_zero_indices], -expected_amp, atol=1e-5)), (
            "State is not equal weight superposition")
        total_non_zero += non_zero_mask
    assert np.all(total_non_zero == 1), (
        "Each basis state should appear exactly once across all RK ground states.")
