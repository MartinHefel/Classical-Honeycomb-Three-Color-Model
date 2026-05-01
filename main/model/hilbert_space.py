
# General Imports
import numpy as np
from scipy.sparse import diags, csr_matrix

# Internal Imports
from main.tools.utils import validate_type
from main.model.states import HardcoreBosonMBS



def braket(psi_a:HardcoreBosonMBS, psi_b:HardcoreBosonMBS) -> complex:
    """
    Compute the inner product ``<psi_a|psi_b>``.

    Parameters
    ----------
    psi_a : HardcoreBosonMBS
        state a
    psi_b : HardcoreBosonMBS
        state b

    Returns
    -------
    complex
        The complex inner product ``<psi_a|psi_b>``.
    """
    validate_type(psi_a, HardcoreBosonMBS, "psi_a")
    validate_type(psi_b, HardcoreBosonMBS, "psi_b")

    if psi_a.number_of_basis_states != psi_b.number_of_basis_states:
        raise ValueError(
            f"Cannot compute braket: basis sizes (number_of_basis_states) differ "
            f"({psi_a.number_of_basis_states} vs {psi_b.number_of_basis_states}).")
    if psi_a.number_of_sites != psi_b.number_of_sites:
        raise ValueError(
            f"Cannot compute braket: basis dimensionality (number_of_sites) differs "
            f"({psi_a.number_of_sites} sites vs {psi_b.number_of_sites} sites).")
    if not np.array_equal(psi_a._basis, psi_b._basis):
        raise ValueError("Cannot compute braket: basis states do not match.")

    return complex(np.vdot(psi_a._psi, psi_b._psi))


def apply_operator(operator:np.ndarray|csr_matrix, psi:HardcoreBosonMBS) -> HardcoreBosonMBS:
    """
    Apply a ``operator`` to a state ``|psi>``.

    Parameters
    ----------
    operator : np.ndarray
        Operator matrix to be applied. Must have shape ``(N_basis, N_basis)``,
        where ``N_basis = psi.number_of_basis_states``.
    psi : HardcoreBosonMBS
        State to which the operator is applied.

    Returns
    -------
    HardcoreBosonMBS
        New state ``|psi'> = operator |psi>``.
    """
    validate_type(operator, (np.ndarray, csr_matrix), "operator")
    validate_type(psi, HardcoreBosonMBS, "psi")

    N_basis = psi.number_of_basis_states
    if operator.shape != (N_basis, N_basis):
        raise ValueError(
            f"Operator shape {operator.shape} does not match "
            f"basis size {N_basis} of psi.")
    new_psi_vector = operator @ psi._psi
    new_psi = HardcoreBosonMBS(psi=new_psi_vector, basis=psi._basis.copy())
    return new_psi


def expectation_value(operator:np.ndarray|csr_matrix, psi:HardcoreBosonMBS) -> complex:
    """
    Compute the expectation value ``<psi|operator|psi>``.

    Parameters
    ----------
    operator : np.ndarray
        Operator matrix to be applied. Must have shape ``(N_basis, N_basis)``,
        where ``N_basis = psi.number_of_basis_states``.
    psi : HardcoreBosonMBS
        State for which to compute the expectation value.

    Returns
    -------
    complex
        The expectation value ``<psi|operator|psi>``.
    """
    validate_type(operator, (np.ndarray, csr_matrix), "operator")
    validate_type(psi, HardcoreBosonMBS, "psi")

    psi_prime = apply_operator(operator, psi)
    return braket(psi, psi_prime)


def norm(psi:HardcoreBosonMBS) -> float:
    """
    Compute the norm ||psi|| = sqrt(<psi|psi>).

    Parameters
    ----------
    psi : HardcoreBosonMBS
        state
    """
    validate_type(psi, HardcoreBosonMBS, "psi")
    return float(np.sqrt(braket(psi, psi)))


def normalize_state(psi:HardcoreBosonMBS, flip_sign_if_negative:bool=False) -> HardcoreBosonMBS:
    """
    Normalize a ``HardcoreBosonMBS``.

    Parameters
    ----------
    psi : HardcoreBosonMBS
        State to normalize.
    flip_sign_if_negative : bool, default=False
        If ``True``, flip the sign of all amplitudes if they are all ``<= 0``.
        If ``False``, preserves the phase.
    """
    validate_type(psi, HardcoreBosonMBS, "psi")
    validate_type(flip_sign_if_negative, bool, "flip_sign_if_negative")

    N = norm(psi)
    if flip_sign_if_negative and np.all(psi._psi <= 0):
        psi._psi = -psi._psi
    psi._psi *= 1 / N
    return psi


def projector_matrix(
        list_of_site_ids:list[int], occupation_scheme:list[int], basis_states:np.ndarray
) -> csr_matrix:
    """
    Build a projector matrix ``P`` such that 

        ``|psi'> = P |psi>``

    retains only the components of ``|psi>`` where the occupations at the specified 
    sites match the given ``occupation_scheme``. If multiple sets of sites are provided
    (i.e. ``len(list_of_site_ids) > 1``), the resulting projector is the sum of projectors
    for each set.

    Parameters
    ----------
    list_of_site_ids : list[list[int]]
        Each element is a list of site IDs in the basis to check.
    occupation_scheme : list[int]
        Desired occupations (0 or 1) corresponding to the sites in each list of 
        ``list_of_site_ids``.
    basis_states : np.ndarray
        The basis in which the projector is construct.

    Returns
    -------
    P : csr_matrix
        Diagonal projector matrix with entries ``1`` for basis states matching 
        ``occupation_scheme`` and ``0`` otherwise.
    """
    validate_type(list_of_site_ids, [list, list, int], "list_of_site_ids")
    validate_type(occupation_scheme, [list, int], "occupation_scheme")
    validate_type(basis_states, np.ndarray, "basis_states")
    assert basis_states.ndim == 2

    P = None
    for site_ids in list_of_site_ids:
        if len(site_ids) != len(occupation_scheme):
            raise ValueError(
                "site_ids and occupation_scheme must have the same number of entries, but have "
                f"{len(site_ids)} and {len(occupation_scheme)}.")
        occupations_in_basis = basis_states[:, site_ids]
        preserved_states_mask = np.all(occupations_in_basis == occupation_scheme, axis=1)
        diag = preserved_states_mask.astype(np.float64)
        if P is None:
            P = diags(diag, offsets=0, format='csr')
        else:
            P += diags(diag, offsets=0, format='csr')
    return P


def measure_two_point_correlation_function(psi: HardcoreBosonMBS) -> np.ndarray:
    """
    Compute the correlation matrix ``<n_i n_j>``.

    Parameters
    ----------
    psi : HardcoreBosonMBS
        State on which the expectation value is evaluated.

    Returns
    -------
    C : np.ndarray
        Matrix with entries ``C[i,j] = <n_i n_j>``.
    """
    validate_type(psi, HardcoreBosonMBS, "psi")

    basis = psi.basis_states_as_array()
    amp = np.abs(psi._psi) ** 2
    weighted_basis = basis * amp[:, None]
    C = weighted_basis.T @ basis
    return C
