
# General Imports
import numpy as np

# Internal Imports
from main.tools.optional_njit import optional_njit
from main.tools.printers import make_table, make_table_data
from main.tools.utils import validate_type



###################################################################################################
# STATE COLLECTION (CLASSICAL)
###################################################################################################

class StateCollection():
    """
    Container for a collection of many-body states and their associated energies.

    Parameters
    ----------
    states : np.ndarray
        2D array of shape ``(number_of_states, number_of_sites)``
        representing the collection of states.
    Es : np.ndarray, optional
        1D array of length ``number_of_states`` with the energies of the states.
        If not provided, energies can be set later using ``init_Es(Es)``.

    Attributes
    ----------
    number_of_states : int
        Total number of states in the collection.
    number_of_sites : int
        Number of sites per state.
    _array : np.ndarray
        Internal storage of the state array.
    _Es : np.ndarray or None
        Energies corresponding to the states, or ``None`` if not assigned.
    """
    def __init__(self, states:np.ndarray, Es:np.ndarray=None):
        validate_type(states, np.ndarray, "states")
        if states.ndim != 2:
            raise ValueError(
                f"states invalid, expected `np.ndarray` of dimension 2 and shape "
                f"number_of_states x number_of_sites but got array with "
                f"dimension `{states.ndim}`.")

        self._array = states
        self.init_Es(Es)
    
    @property
    def number_of_states(self) -> int:
        return self._array.shape[0]
    
    @property
    def number_of_sites(self) -> int:
        return self._array.shape[1]
    
    def init_Es(self, Es:np.ndarray):
        if Es is not None:
            validate_type(Es, (np.ndarray, list), "Es")
            if isinstance(Es, list):
                Es = np.array(Es)
            if Es.ndim != 1:
                raise ValueError(
                    f"Es invalid, expected `np.ndarray` of dimension 1 and shape "
                    f"number_of_states but got array with "
                    f"dimension `{Es.ndim}`.")
            if Es.shape[0] != self.number_of_states:
                raise ValueError(
                    f"Number of energies ({Es.shape[0]}) must match number of states "
                    f"{self.number_of_states}.")
        self._Es: np.ndarray = Es


    def as_array(self, dtype=None) -> np.ndarray:
        if dtype is not None:
            return np.asarray(self._array, dtype=dtype)
        else:
            return np.asarray(self._array)
    
    def get_E(self, state_id:int) -> float:
        return float(self._Es[state_id])
        
    def get_state(self, state_id:int, dtype=None, as_copy:bool=False) -> np.ndarray:
        if dtype is not None:
            return np.asarray(self._array[state_id,:], dtype=dtype, copy=as_copy)
        else:
            return np.asarray(self._array[state_id,:], copy=as_copy)
    
    def extract_energy_level(self, level_index:int, decimals:int=12) -> "StateCollection":
        """
        Extract all states corresponding to the i-th distinct energy level.

        Parameters
        ----------
        level_index : int
            Index of the distinct energy level to extract:
            0 = ground state, 1 = first excited state, etc.
        decimals : int, default 12
            Number of decimals used when rounding energies for numerical comparison.

        Returns
        -------
        StateCollection
            A new state collection containing all states with the specified energy level.
        """
        if self._Es is None:
            raise ValueError(
                f"self._Es is `None`. Use `.add_Es(Es)` to associate energies to the states.")

        # batch Es up to decimals
        unique_Es = np.unique(np.round(self._Es, decimals=decimals))
        if level_index >= len(unique_Es):
            raise IndexError(
                f"Requested level_index={level_index}, but only {len(unique_Es)} distinct levels "
                f"exist.")

        # extract states
        target_E = unique_Es[level_index]
        mask = np.isclose(self._Es, target_E)
        extracted_Es = [E for E, m in zip(self._Es, mask) if m]
        extracted_states = self._array[mask]
        return self.__class__(extracted_states, Es=extracted_Es)

    def split_by_energy(self, decimals:int=12) -> list["StateCollection"]:
        """
        Split the collection into separate StateCollections grouped by distinct energy levels, 
        sorted by increasing energy.

        Parameters
        ----------
        decimals : int, default 12
            Number of decimals used when rounding energies for numerical comparison.

        Returns
        -------
        list[StateCollection]
            A list of StateCollections, ordered by increasing energy.
            The first entry corresponds to the ground state sector.
        """
        if self._Es is None:
            raise ValueError(
                f"self._Es is `None`. Use `.add_Es(Es)` to associate energies to the states.")

        rounded_Es = np.round(self._Es, decimals=decimals)
        unique_Es = np.unique(rounded_Es)
        collections = []
        for E in unique_Es:
            mask = np.isclose(self._Es, E)
            extracted_states = self._array[mask]
            extracted_Es = self._Es[mask]
            collections.append(self.__class__(extracted_states, Es=extracted_Es))
        return collections

    def __str__(self):
        first_line = "\nNumber of states in collection: "+str(self.number_of_states)
        if self._Es is not None:
            headers = ["state_id", "Energy", "State"]
            rows = [[i, E, state] 
                    for i, (E, state) in enumerate(zip(self._Es, self._array))]
        else:
            headers = ["state_id", "State"]
            rows = [[i, state] 
                    for i, (state) in enumerate(zip(self._array))]
        return first_line+make_table(headers, make_table_data(rows))

    def __repr__(self):
        return self.__str__()



###################################################################################################
# HARDCORE BOSON STATE COLLECTION (CLASSICAL)
###################################################################################################

class HardcoreBosonStateCollection(StateCollection):
    """
    Collection of hardcore boson states.

    Parameters
    ----------
    states : np.ndarray
        Array of states (shape: number_of_states x number_of_sites)
    Es : np.ndarray, optional
        Energies of the states. Can also be assigned later using 
        ``StateCollection.init_Es(Es)``.
    
    Attributes
    ----------
    Same as :class:`StateCollection`
    """
    def __init__(self, states:np.ndarray, Es:np.ndarray=None):
        # Ensure all states are hardcore bosons: only 0 or 1 entries
        if not np.all(np.logical_or(states == 0, states == 1)):
            raise ValueError("All entries in `states` must be 0 or 1 for hardcore bosons.")
        super().__init__(states=states, Es=Es)
    
    @classmethod
    def from_StNrs(cls, StNrs:np.ndarray, number_of_sites:int, Es:list[float]=None):
        validate_type(StNrs, np.ndarray, "StNrs")
        validate_type(number_of_sites, int, "number_of_sites")
        assert StNrs.ndim == 1
        number_of_states = StNrs.shape[0]
        states = np.empty((number_of_states, number_of_sites), dtype=np.bool)
        for state_id, StNr in enumerate(StNrs):
            states[state_id, :] = StNr_to_basis_state(StNr, number_of_sites)
        return cls(states=states, Es=Es)

    def as_StNrs(self) -> np.ndarray:
        StNrs = np.empty((self.number_of_states), dtype=np.int64)
        for state_id in range(self.number_of_states):
            StNrs[state_id] = basis_state_to_StNr(self.get_state(state_id))
        return StNrs

    def extract_particle_filling(self, N_filling: int) -> "HardcoreBosonStateCollection":
        """
        Extract all states with a given particle filling.

        Parameters
        ----------
        N_filling : int
            Desired particle filling (number of occupied sites)

        Returns
        -------
        HardcoreBosonStateCollection
            New collection containing only states with the specified filling.
        """
        filtered_Es = np.empty((self.number_of_states))
        filtered_states = np.empty((self.number_of_states, self.number_of_sites))
        found_states = 0

        for state_id in range(self.number_of_states):
            E, state = self.get_E(state_id), self.get_state(state_id)
            filling = int(np.sum(state))

            if filling == N_filling:
                filtered_Es[found_states] = E
                filtered_states[found_states] = state
                found_states += 1

        if found_states == 0:
            raise IndexError(
                f"No states found with filling N_filling={N_filling}.")
        filtered_Es = filtered_Es[:found_states]
        filtered_states = filtered_states[:found_states, :]
        return HardcoreBosonStateCollection(filtered_Es, filtered_states)

    def split_by_particle_filling(self) -> list["HardcoreBosonStateCollection"]:
        """
        Split the collection into separate HardcoreBosonStateCollections grouped by particle 
        filling, sorted by increasing filling.

        Returns
        -------
        list[HardcoreBosonStateCollection]
            A list of collections ordered from smallest to largest filling.
        """
        fillings = np.sum(self._array, axis=1)
        unique_fillings = np.unique(fillings)
        collections = []
        for N in unique_fillings:
            mask = fillings == N
            extracted_states = self._array[mask]
            extracted_Es = self._Es[mask] if self._Es is not None else None
            collections.append(
                self.__class__(extracted_states, Es=extracted_Es))
        return collections



###################################################################################################
# HARDCORE BOSON MANY BODY STATE
###################################################################################################

class HardcoreBosonMBS():
    r"""
    Representation of a hardcore boson quantum many-body state.

    Parameters
    ----------
    psi : np.ndarray
        Complex amplitudes of the many-body wavefunction in the chosen basis
        (shape: number_of_basis_states,).
    basis : np.ndarray
        Basis states corresponding to the amplitudes ``psi`` 
        (shape: number_of_basis_states x number_of_sites).

    Attributes
    ----------
    _psi : np.ndarray
        Stored amplitudes of the many-body state.
    _basis : np.ndarray
        Stored basis states corresponding to `_psi`.
    """
    def __init__(self, psi:np.ndarray, basis:np.ndarray):
        validate_type(psi, np.ndarray, "psi")
        validate_type(basis, np.ndarray, "basis")
        if psi.ndim != 1:
            raise ValueError(
                f"psi invalid, expected `np.ndarray` of dimension 1 and shape "
                f"number_of_basis_states but got array with "
                f"dimension `{psi.ndim}`.")
        if basis.ndim != 2:
            raise ValueError(
                f"psi invalid, expected `np.ndarray` of dimension 2 and shape "
                f"number_of_basis_states x number_of_sites but got array with "
                f"dimension `{basis.ndim}`.")
        num_of_basis_states = psi.shape[0]
        assert num_of_basis_states == basis.shape[0]
        
        self._psi = psi
        self._basis = basis

    @classmethod
    def from_StNrs(cls, psi:np.ndarray, basis_StNrs:np.ndarray, number_of_sites:int):
        validate_type(psi, np.ndarray, "psi")
        validate_type(basis_StNrs, np.ndarray, "basis_StNrs")
        validate_type(number_of_sites, int, "number_of_sites")
        assert basis_StNrs.ndim == 1
        number_of_basis_states = basis_StNrs.shape[0]
        basis = np.empty((number_of_basis_states, number_of_sites), dtype=np.bool)
        for state_id, StNr in enumerate(basis_StNrs):
            basis[state_id, :] = StNr_to_basis_state(StNr, number_of_sites)
        return cls(psi=psi, basis=basis)
    
    @property
    def number_of_basis_states(self) -> int:
        return self._basis.shape[0]
    
    @property
    def number_of_sites(self) -> int:
        return self._basis.shape[1]
    
    def get_amplitude(self, basis_state_id:int):
        return self._psi[basis_state_id]
    
    def get_basis_state(self, basis_state_id:int, dtype=None) -> np.ndarray:
        if dtype is not None:
            return np.asarray(self._basis[basis_state_id,:], dtype=dtype)
        else:
            return np.asarray(self._basis[basis_state_id,:])
    
    def amplitudes_as_array(self) -> np.ndarray:
        return self._psi

    def basis_states_as_array(self, dtype=None) -> np.ndarray:
        if dtype is not None:
            return np.asarray(self._basis, dtype=dtype)
        else:
            return np.asarray(self._basis)
    
    def __str__(self):
        first_line = (
            f"\nState with {self.number_of_basis_states} basis states of a system with "+
            f"{self.number_of_sites} sites.")
        headers = ["id", "Amplitude" ,"Basis State"]
        rows = [[i, psi, np.array(state, dtype=np.int64)] 
                for i, (psi, state) in enumerate(zip(self._psi, self._basis))]
        return first_line+make_table(headers, make_table_data(rows))

    def __repr__(self):
        return self.__str__()



###################################################################################################
# BITWISE OPERATIONS FOR SATES
###################################################################################################
"""
**Note for bitwise operations for basis states:**

We number all possible ``2**N`` basis states by a basis state number ``StNr`` and represent each 
basis state by the binary representation of its basis state number ``StNr``, hence
    
    ``basis_state = bin(StNr)``

Additional info:
The bitwise operation ``StNr&1`` checks if the rightmost bit is ``1``, if yes it returns ``1``, 
else ``0``. Since the bitwise. The operator ``StNr>>j`` removes the ``j`` rightmost bits of 
``bin(StNr)``. Hence the occupation of ``site_j`` (referenced from the right end) is give by 
``i`` by ``(StNr>>i)&1``.
"""


@optional_njit()
def basis_state_to_StNr(basis_state:np.ndarray) -> int:
    """
    Parameters
    ----------
    basis_state : np.ndarray
        The basis state as 1D array

    Returns
    -------
    StNr : int
        Basis state number of ``basis_state``
    """
    assert basis_state.ndim == 1
    StNr = 0
    n = len(basis_state)
    for idx, bit in enumerate(basis_state):
        StNr += bit << (n - idx - 1)
    return StNr


@optional_njit()
def StNr_to_basis_state(StNr: int, number_of_sites: int) -> np.ndarray:
    """
    Parameters
    ----------
    StNr : int
        Basis state number
    number_of_sites : int
        The number of sites of the state


    Returns
    -------
    basis_state : np.ndarray
        The basis state as 1D array with ``dtype=np.bool_``
    """
    basis_state = np.zeros((number_of_sites,), dtype=np.bool_)
    for i in range(number_of_sites):
        bit_pos = number_of_sites - i - 1
        bit = (StNr >> bit_pos) & 1
        basis_state[i] = bit
    return basis_state


@optional_njit()
def _get_basis_StNr(N:int, N_filling:int) -> np.ndarray:
    """Determine the basis states for a system of size `N` with particle filling `N_filling`.
    
    Parameters
    ----------
    N: int
        The number of sites
    N_filling: int
        The number of particles in the system

    Returns
    -------
    basis_StNrs: list
        Array containing the basis states numbers for filling 'N_filling'
    """
    max_basis_states = 2 ** N
    basis_StNrs = np.empty(max_basis_states, dtype=np.int64)
    number_of_basis_states = 0

    for StNr in range(max_basis_states):
        # count bits
        N_state = 0
        x = StNr
        while x:
            N_state += x & 1
            x >>= 1
        # check if basis state is in wanted sector
        if N_state == N_filling:
            basis_StNrs[number_of_basis_states] = StNr
            number_of_basis_states += 1
    return basis_StNrs[:number_of_basis_states]


@optional_njit()
def _flip_sites_of_StNr(StNr: int, sites_to_flip: np.ndarray, n_bits: int) -> int:
    """
    Flip the bits of `sites_to_flip` (from the left, hence ``i=0`` is the leftmost site) of a
    StNr.

    Parameters
    ----------
    StNr : int
        The basis state number whose bits should be flipped
    sites_to_flip : np.ndarray
        Array of bit indices to flip
    n_bits : int
        Total number of bits (system size)

    Returns
    -------
    StNr_flipped : int
        New basis state number with specified bits flipped
    """
    flipped = StNr
    for site_id in sites_to_flip:
        if site_id != -1: # padding guard
            bit_pos = n_bits - site_id - 1
            flipped ^= (1 << bit_pos)
    return flipped


@optional_njit()
def _get_n(StNr: int, i: int, n_bits: int) -> int:
    ''' Get the occupation of site `i` (from the left, hence ``i=0`` is the leftmost site) in a 
    basis state with basis state number `StNr`.

    Parameters
    ----------
    StNr : int
        The basis state number
    i : int
        The site number of which the occupation should be returned.
    n_bits : int
        Total number of bits (system size)

    Returns
    -------
    n : int
        Occupation of site `i` (either 0 or 1)
    '''
    n = (StNr >> (n_bits - i - 1)) & 1
    return n


@optional_njit()
def _get_N(StNr: int, n_bits: int) -> int:
    ''' Get the total number of particles for a given state with state number `StNr`.

    Parameters
    ----------
    StNr : int
        The basis state number
    n_bits : int
        Total number of bits (system size)

    Returns
    -------
    N : int
        The total number of particles
    '''
    N = 0
    for i in range(n_bits):
        N += (StNr >> i) & 1
    return N
