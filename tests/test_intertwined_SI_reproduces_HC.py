
# General imports
import pytest

# Internal imports
from main import model, honeycomb_specific
from test_method_agreement_finding_states import (
    _get_lattice, _check_if_commensurate_states_can_be_found)


@pytest.mark.parametrize(
"UnitCell, m_ref", [
    (model.unit_cell.HoneycombUnitCell, 1),
    (model.unit_cell.HoneycombUnitCell, 2),
    (model.unit_cell.HoneycombUnitCell, 3),
    (model.unit_cell.HoneycombUnitCell, 4),
    (model.unit_cell.HoneycombUnitCell, 5),
    (model.unit_cell.HoneycombRectangularUnitCell, 1),
    (model.unit_cell.HoneycombRectangularUnitCell, 2),
    (model.unit_cell.HoneycombRectangularUnitCell, 3),
    (model.unit_cell.HoneycombRectangularUnitCell, 4),
    (model.unit_cell.HoneycombRectangularUnitCell, 5),
    ])
def test_intertwined_SI_partitions_reproduce_full_HC_states(
    UnitCell:model.lattice.UnitCell, m_ref:int
):
    lattice = _get_lattice(UnitCell)
    states_full_HC = _check_if_commensurate_states_can_be_found(
        lattice, m_ref, number_of_final_checks=3)

    states_collection = []
    for part in [0,1,2]:
        lattice_SI = honeycomb_specific.helpers.to_HoneycombSpinIce(lattice, part)
        states = _check_if_commensurate_states_can_be_found(
            lattice_SI, m_ref, is_HC_SI=True, number_of_final_checks=6)
        states_collection.append(states)
    
    num_states = [s.shape[0] for s in states_collection]
    assert all(n == num_states[0] for n in num_states), (
        f"Partitions have different number of states: {num_states}")

    assert num_states[0] > states_full_HC.shape[0], (
        f"Number of states in each partition ({num_states[0]}) must be bigger than full Honeycomb "
        f"({states_full_HC.shape[0]})")

    states_sets = [set(map(tuple, s)) for s in states_collection]
    common_states_set = set.intersection(*states_sets)
    states_full_HC_set = set(map(tuple, states_full_HC))
    assert common_states_set == states_full_HC_set, (
        "Three intertwined Spin Ice models do not match the full Honeycomb model")

