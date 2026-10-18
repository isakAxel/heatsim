# heatsim

heatsim is a Python project for simulating transient heat conduction in solid targets exposed to pulsed proton beams, developed in the context of the European Spallation Source (ESS) in Lund, Sweden.

> **Note:** This repository is a GitHub mirror of the project's GitLab repository. The GitLab repository is the primary source.

The main goal is to evaluate time-splitting schemes for the heat equation, in particular Peaceman-Rachford (PR) splitting, and to obtain accurate and efficient temperature estimates that support beam diagnostics at ESS. Accurate temperature fields help interpret beam-induced heating, validate beam parameters, and assess the thermal behavior of target materials under operating conditions.

> **Status:** This project is under active development. Interfaces, file layout, and numerical methods may change, and results should be treated as preliminary until they have been validated against reference solutions.

## Features

- Finite-difference discretization of the 2D heat equation with Neumann and radiative/convective boundary conditions
- Gaussian-profile proton beam heating with a Bragg-peak depth dose
- Several time-integration schemes for comparison, including:
  - Peaceman-Rachford splitting (boundary-term and dimensional splitting)
  - Implicit trapezoidal rule (reference solution)
  - Explicit trapezoidal (Heun) method
- Material configurations for different target media
- Profiling and benchmarking utilities

## Project structure
heatsim/
├──benchmarks
│ ├── data/ # data directory that contains benchmark measurables
│ └── erranalysis.py # simple scripts for running testing
├── Sim/ # simulation package
│ ├── init.py
│ ├── config.py # material and simulation parameters
│ ├── operators.py # discrete Laplacians and boundary terms
│ ├── solvers.py # tridiagonal and linear solvers
│ ├── visual.py # plotting helpers
│ ├── heatsim2d.py # 2D simulation
│ ├── heatsim2dMU.py # 2D simulation (MU variant)
│ ├── heatsim3d.py # 3D simulation
│ └── utils/
│ ├── init.py
│ ├── profiling.py # timing and profiling helpers
│ └── utils.py # general helpers
├── heatsim2.m / # MATLAB reference implementation (heatsim2.m)
├── notebooks/ # exploratory notebooks
├── pyproject.toml # package metadata and dependencies
├── requirements.txt # pinned dependency versions
├── setup.py
└── README.md


## Installation
## Contact

heatsim is developed in connection with a Master's thesis project at ESS and LTH. The people involved are:

- Axel Isakson, MSc in Engineering Physics, LTH
- Cyrille Thomas, Supervisor, ESS
- Eskil Hansen, Supervisor, LTH

## Installation

Clone the repository and install the package in editable mode from the repository root:

```bash
git clone <repository-url>
cd heatsim
pip install -e .
```

Dependencies are declared in `pyproject.toml`. To reproduce the exact versions used for the reported results, install from `requirements.txt` instead:

```bash
pip install -r requirements.txt
```

## Usage

Run a simulation as a module from the repository root:

```bash
python -m Sim.heatsim2dMU
```

Default simulation parameters are set within the config.py module but a custom SimulationSetup object can be created with constructor. Plots are shown on screen by default.

## Development notes

- Debug output and intermediate files are written to the working directory as of now. Be weary when commiting
- Generated files (`data/`, `figures/`, `*.prof`, `*.log`) are ignored by Git.
- Solver choices should be checked against the reference implementation on a small grid (e.g. `M = 50`) before running larger cases.

