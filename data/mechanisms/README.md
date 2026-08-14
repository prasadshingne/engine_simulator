# Reaction Mechanisms

This directory contains Cantera YAML mechanism files used by the engine simulator.
The files are not committed to the repository due to size and licensing — generate
them locally using the instructions below.

---

## Nissan PRF (33 species) — `Nissan_chem.yaml`

**Source:** Tsurushima, A. "A new skeletal PRF kinetic model for HCCI combustion."
*Proceedings of the Combustion Institute* 32 (2009) 2835–2841.

The mechanism ships as a Cantera CTML/XML file (`nissan_chem.xml`). Convert it to
the YAML format required by Cantera 3.x:

```bash
python -c "import cantera; cantera.ctml2yaml('nissan_chem.xml', 'Nissan_chem.yaml')"
```

Or with the CLI (Cantera 2.6+):

```bash
ctml2yaml nissan_chem.xml Nissan_chem.yaml
```

Place the output as `data/mechanisms/Nissan_chem.yaml`.

---

## LLNL Gasoline Surrogate (312 species) — `gasoline_surrogate_323.yaml`

**Source:** Mehl, M., Pitz, W. J., Westbrook, C. K., Curran, H. J.
"Kinetic Modeling of Gasoline Surrogate Components and Mixtures Under Engine Conditions."
*Proceedings of the Combustion Institute* 33 (2011) 193–200.

Download the three Chemkin-format source files from the LLNL combustion chemistry
repository (supplementary materials of the paper above or direct from
[https://combustion.llnl.gov](https://combustion.llnl.gov)):

| File | Description |
|------|-------------|
| `Chem323.inp.txt` | Reaction mechanism (323 reactions) |
| `gasoline_surrogate_therm.dat.txt` | Thermodynamic data |
| `gasoline_surrogate_transport.txt` | Transport properties |

Convert to Cantera YAML using `ck2yaml` (tested with Cantera 3.1.0):

```bash
ck2yaml \
  --input=Chem323.inp.txt \
  --thermo=gasoline_surrogate_therm.dat.txt \
  --transport=gasoline_surrogate_transport.txt \
  --output=gasoline_surrogate_323.yaml
```

Place the output as `data/mechanisms/gasoline_surrogate_323.yaml`.

---

## Fuel compositions

These are set in [`engine_sim/config/mechanisms.py`](../../engine_sim/config/mechanisms.py)
and do not need to match the filenames.

| Mechanism | Species | Composition (mass fraction) |
|-----------|---------|----------------------------|
| Nissan PRF | `C8H18`, `C7H16` | 85% iso-octane, 15% n-heptane (PRF85) |
| LLNL Surrogate | `IC8H18`, `NC7H16`, `C6H5CH3`, `C5H10-2` | 54.13% iso-octane, 14.88% n-heptane, 27.38% toluene, 3.61% 2-pentene |

---

## Verifying the conversion

After converting, run a quick Cantera load check:

```python
import cantera as ct
gas = ct.Solution("data/mechanisms/gasoline_surrogate_323.yaml")
print(gas.n_species, "species,", gas.n_reactions, "reactions")
```

Expected output: `312 species, 2469 reactions`

For the Nissan PRF: `33 species, 48 reactions`
