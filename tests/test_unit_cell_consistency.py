
# General imports
from copy import deepcopy
import pytest

# Internal imports
import main.model as model



@pytest.mark.parametrize("UnitCell", model.ALL_UNIT_CELLS, ids=lambda uc: uc.__name__)
def test_site_reciprocity(UnitCell):
    unit_cell: model.lattice.UnitCell = UnitCell()
    for (name, nearest_neighbors) in [
        ["site",unit_cell.sites.nearest_neighbors],
        ["plaquette",unit_cell.plaquettes.nearest_neighbors]]:
        if isinstance(nearest_neighbors, model.lattice.ListWithShifts):
            errors = _check_neighbors_reciprocity(nearest_neighbors)
            assert not errors, (
                f"Inconsistent {name} nearest_neighbors reciprocity in {UnitCell.__name__}:\n"
                + "\n".join(errors))

def _check_neighbors_reciprocity(nearest_neighbors:model.lattice.ListWithShifts):
    """
    Ensures that for every neighbor relation i->j with shifts (sx, sy),
    there exists exactly one reverse relation j->i with shifts (-sx, -sy).
    """
    nearest_neighbors = deepcopy(nearest_neighbors)
    N_nodes = len(nearest_neighbors)

    # Keep track of which connections were used
    used_pairs = {j: set() for j in range(N_nodes)}
    errors = []

    for node_i, nn_i in enumerate(nearest_neighbors):
        for idx, neigh in enumerate(nn_i.entries):
            sx_i = nn_i.x_shifts[idx]
            sy_i = nn_i.y_shifts[idx]

            found_reversed = False

            # Look for reverse connection (j -> i) with reversed shifts
            for node_j, nn_j in enumerate(nearest_neighbors):
                for j_idx, neigh_j in enumerate(nn_j.entries):
                    if j_idx in used_pairs[node_j]:
                        # already used this entry: skip it
                        continue

                    sx_j = nn_j.x_shifts[j_idx]
                    sy_j = nn_j.y_shifts[j_idx]

                    # Check reciprocity
                    if neigh_j == node_i and sx_j == -sx_i and sy_j == -sy_i:
                        found_reversed = True
                        used_pairs[node_j].add(j_idx)
                        break
                if found_reversed:
                    break

            if not found_reversed:
                errors.append(
                    f"Missing reverse connection for site {node_i} -> {neigh} "
                    f"with shifts ({sx_i}, {sy_i})")
    return errors



@pytest.mark.parametrize("UnitCell", model.ALL_UNIT_CELLS, ids=lambda uc: uc.__name__)
def test_monte_carlo_worm_network_reciprocity(UnitCell):
    """
    Ensure that every Monte Carlo neighbor relation node_i -> node_j
    has a reciprocal node_j -> node_i with reversed shifts (-sx, -sy),
    and that the corresponding connecting_site_ids are reversed as well.
    Each reverse relation is only used once.
    """
    unit_cell: model.lattice.UnitCell = UnitCell()
    nearest_neighbors, connecting_site_ids, check_site_ids_of_path = (
        unit_cell.get_monte_carlo_worm_network())
    
    N_nodes = len(nearest_neighbors)

    used_pairs = {j: set() for j in range(N_nodes)}
    errors = []

    for node_i, nn_i in enumerate(nearest_neighbors):
        conn_i = connecting_site_ids[node_i]

        for n_idx, node_j in enumerate(nn_i.entries):
            sx_i = nn_i.x_shifts[n_idx]
            sy_i = nn_i.y_shifts[n_idx]

            conn_sites_ij = conn_i[n_idx]
            found_reversed = False

            nn_j = nearest_neighbors[node_j]
            conn_j = connecting_site_ids[node_j]

            for m_idx, neigh_back in enumerate(nn_j.entries):
                if m_idx in used_pairs[node_j]:
                    continue
                if neigh_back != node_i:
                    continue

                sx_j = nn_j.x_shifts[m_idx]
                sy_j = nn_j.y_shifts[m_idx]

                if sx_j == -sx_i and sy_j == -sy_i:
                    conn_sites_ji = conn_j[m_idx]

                    # both are flat integer lists, e.g. [a,b]
                    sites_ij = conn_sites_ij.entries
                    sites_ji = conn_sites_ji.entries

                    if len(sites_ij) != len(sites_ji):
                        errors.append(
                            f"Site connection count mismatch between "
                            f"{node_i}->{node_j} and {node_j}->{node_i}")
                        continue

                    # check reversed list equality: [a,b] ↔ [b,a]
                    if sites_ij != list(reversed(sites_ji)):
                        errors.append(
                            f"Site connection mismatch between "
                            f"{node_i}->{node_j} {sites_ij} "
                            f"and {node_j}->{node_i} {sites_ji}")

                    used_pairs[node_j].add(m_idx)
                    found_reversed = True
                    break

            if not found_reversed:
                errors.append(
                    f"Missing reverse connection for node {node_i} -> {node_j} "
                    f"with shifts ({sx_i},{sy_i})")

    assert not errors, (
        f"Inconsistent Monte Carlo network in {UnitCell.__name__}:\n"
        + "\n".join(errors))
