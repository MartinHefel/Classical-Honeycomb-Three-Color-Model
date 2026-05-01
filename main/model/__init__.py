"""
Models and dataclasses used as inputs for the algorithms.

Includes:
- **lattice**: lattice geometry and connectivity
- **unit_cell**: definition of different periodically defined unit cells
- **states**: definition of states on lattices
- **hilbert_space**: operations on quantum states
"""

from . import lattice
from . import states
from . import unit_cell # imports from lattice
from . import hilbert_space # imports from states

ALL_UNIT_CELLS:list[lattice.UnitCell] = [
    unit_cell.HoneycombUnitCell,
    unit_cell.HoneycombRectangularUnitCell,
    unit_cell.SquareSpinIceUnitCell,
]

ALL_UNIT_CELLS_BRUTE_FORCE_SIZES: dict = {
    unit_cell.HoneycombUnitCell.__name__ : {"L1":3, "L2":3},
    unit_cell.HoneycombRectangularUnitCell.__name__ : {"L1":3, "L2":2},
    unit_cell.SquareSpinIceUnitCell.__name__ : {"L1":3, "L2":2},
}