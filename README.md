This repository contains the code developed during the course of my [masters's thesis](https://mediatum.ub.tum.de/1865856):
## Emergent $U(1)\times U(1)$ Gauge Theory in the Classical Honeycomb Three Color Model

### Thesis Information
Strongly correlated systems provide a fertile ground for emergent phenomena that cannot be understood in terms of independent particles. Prominent examples include fractionalization and emergent gauge fields in systems such as spin ice and quantum dimer models, which motivate the study of similarly constrained lattice models. We study a classical model of hardcore bosons on the honeycomb lattice, where the Hamiltonian enforces ground states with exactly $m$ particles per plaquette. At the commensurate filling fractions $f_m = m/6$ with $m \in \{1,2,3\}$, the system hosts an extensive ground-state degeneracy. Upon doping, a single boson fractionalizes into three individually colored but collectively color-neutral quasiparticles. Using a Monte Carlo worm algorithm, we compute real-space two-point correlation functions of the bosons and various quasiparticle configurations. The bosonic correlations decay algebraically, and the associated structure factor exhibits pinch points in momentum space, identifying the phase as a classical Coulomb liquid. The quasiparticle correlations reveal a color and charge-dependent interaction structure. These observations are supported by an effective field theory obtained by mapping the model onto three coupled spin-ice models. This mapping yields three coupled divergence-free fields, with the key result that the low-energy physics is governed by two independent emergent $U(1)\times U(1)$ gauge fields. Within this framework, the quasiparticles appear as composite objects carrying gauge charges under both fields. Further, we briefly discuss the quantum extension of the model. In the strong-coupling limit, the system maps onto a Rokhsar-Kivelson-type Hamiltonian, with quantum fluctuations arising from ring-exchange processes within the constrained manifold.

### Library Information
The library (in `main`) provides an object-oriented framework for defining and simulating lattice models with classical ground-state degeneracy. Its core feature is a **Monte Carlo Worm Algorithm** for efficient ergodic sampling of hardcore boson configurations within a degenerate manifold. Planar models are defined via a 2D unit cell in `main/model/unit_cell.py`.

Test functions are located in `tests/` and can be run from the project root:
```bash
pytest -v
```
Version compatibility can be checked via:
```bash
python3 version_compatibility.py
```

### Data Creation
The data used in the thesis is generated and processed using the functions in `workflows/data_creation/`, and plotted using those in `workflows/plotting/`.

### Disclosure of Generative AI Use

ChatGPT 5 was used for debugging, performance improvements, generating docstrings, and writing simple helper and test functions.
