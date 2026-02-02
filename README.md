# Engine Simulator

A zero-dimensional (0D) engine cycle simulator with detailed chemical kinetics and an interactive GUI. Built as a learning tool for understanding HCCI combustion, heat transfer, and multi-zone modeling.

## Features

- **Single-zone and multi-zone HCCI models** (up to 50 zones)
- Detailed chemical kinetics via Cantera (33-species iso-octane mechanism)
- Woschni heat transfer correlation
- EGR handling with equilibrium composition
- Temperature stratification modeling (multi-zone with wall heat transfer)
- **Interactive Streamlit GUI** with Plotly plots
- **Run comparison mode** — overlay up to 4 simulations with different configurations
- **CVODE solver** (SUNDIALS) for robust handling of stiff chemistry
- Jupyter tutorial notebooks
- pytest test suite (50 tests)

## Installation

```bash
# Clone and install as editable package
git clone <repo-url>
cd 0D_engine_simulator
pip install -e .

# Install CVODE solver (recommended)
conda install -c conda-forge scikits.odes sundials

# Install GUI dependencies
pip install -e ".[gui]"

# Install development dependencies (tests, notebooks)
pip install -e ".[dev]"
```

Requires Python 3.9+ and Cantera 2.6+.

## Quick Start

### Interactive GUI

```bash
streamlit run engine_sim/gui/app.py
```

The GUI lets you configure equivalence ratio, EGR fraction, engine geometry, operating conditions, and model type (Single Zone HCCI or Multi Zone HCCI). Run simulations and compare results interactively.

### Command Line

```bash
python scripts/run_simulation.py
```

### Jupyter Notebooks

```
notebooks/01_quickstart.ipynb              # Run your first simulation
notebooks/02_engine_geometry.ipynb         # Slider-crank kinematics and geometry
notebooks/03_multizone_and_heat_transfer.ipynb  # Multi-zone model and Woschni correlation
```

## Project Structure

```
.
├── engine_sim/            # Python package
│   ├── config/            # Default config YAML and path utilities
│   ├── engine/            # Geometry and heat transfer
│   ├── models/            # Chemistry (Cantera wrapper)
│   ├── simulation/        # ODE solver, engine driver, results
│   ├── gui/               # Streamlit app and Plotly plotting
│   └── visualization/     # (reserved)
├── scripts/               # Standalone runnable scripts
├── notebooks/             # Jupyter tutorial notebooks
├── tests/                 # pytest test suite
├── data/
│   ├── mechanisms/        # Cantera reaction mechanisms
│   └── output/            # Simulation output files
└── pyproject.toml         # Package metadata and dependencies
```

## Configuration

Engine and simulation parameters can be set via the GUI sidebar, or modified in `engine_sim/config/default_config.yaml`:

```yaml
engine:
  geometry:
    bore: 0.086          # Bore diameter [m]
    stroke: 0.086        # Stroke length [m]
    con_rod: 0.1455      # Connecting rod length [m]
    comp_ratio: 12.5     # Compression ratio [-]

  operating_conditions:
    speed: 2000          # Engine speed [rpm]
    wall_temp: 400       # Wall temperature [K]

chemistry:
  mechanism: "mechanisms/Nissan_chem.yaml"
  fuel: "C8H18"         # Fuel species (iso-octane)
  phi: 0.7              # Equivalence ratio [-]
  egr: 0.3              # EGR fraction [-]

initial_conditions:
  pressure: 1.0e5        # Initial pressure [Pa]
  temperature: 450       # Initial temperature [K]
```

## Testing

```bash
# Run all tests
pytest

# Skip slow tests (chemistry initialization)
pytest -m "not slow"

# Skip tests requiring CVODE
pytest -m "not cvode"
```

## Model Formulation

### Governing Equations

The simulator solves conservation equations for a reacting variable-volume system:

1. **Energy conservation** — temperature evolution from compression work, chemical heat release, and wall heat transfer
2. **Species conservation** — mass fraction evolution from chemical kinetics
3. **Pressure evolution** — derived from the ideal gas equation of state
4. **Volume kinematics** — slider-crank mechanism

**Single-zone state vector:** `[T, V, P, m, Y₁...Y₃₃]`

**Multi-zone state vector:** `[T_bulk, P, V, T₁, Y₁...Y₃₃, ..., Tₙ, Y₁...Y₃₃, Y_bulk]`

### Multi-Zone Model

The cylinder is divided into concentric zones from the hot core to the wall-adjacent region. Each zone has its own temperature and composition but shares a common pressure. Wall-adjacent zones lose more heat, creating thermal stratification that affects combustion phasing.

### Heat Transfer

Wall heat transfer uses the Woschni correlation with coefficients C₁ = 2.28 (compression/expansion) and C₂ = 0.00324 (combustion).

### Numerical Solution

All cases use the CVODE solver (SUNDIALS BDF method) when available, with scipy LSODA as fallback. CVODE provides superior stability for the stiff ODE systems arising from detailed chemistry.

## Engine Specifications (Default)

| Parameter | Value |
|-----------|-------|
| Bore | 86 mm |
| Stroke | 86 mm |
| Connecting rod | 145.5 mm |
| Compression ratio | 12.5:1 |
| Displacement | ~0.5 L |

## Sample Results

![HCCI Engine Simulation Results](data/output/interactive_plots.png)

### Multizone Temperature Stratification

![Multizone Stratification](data/output/10zone_cvode_stratification.png)

## Limitations

- Closed cycle only (IVC to EVO)
- No direct injection modeling
- No turbulence modeling
- No inter-zone mass transfer
- No crevice or blow-by losses

## References

[1] T. Tsurushima, "A new skeletal PRF kinetic model for HCCI combustion," *Proceedings of the Combustion Institute*, vol. 32, no. 2, pp. 2835-2841, 2009. [DOI: 10.1016/j.proci.2008.06.018](https://doi.org/10.1016/j.proci.2008.06.018)
