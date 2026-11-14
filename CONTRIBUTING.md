# Contributing to engine_simulator

Thanks for your interest. Bug reports, questions, and pull requests are all welcome.

## Reporting a bug or asking a question

Open an issue at <https://github.com/prasadshingne/engine_simulator/issues>. For a bug, please include:

- what you ran (a config file, a script, or the GUI settings)
- what you expected and what happened, with the full error message if there is one
- your Python version, Cantera version (`python -c "import cantera; print(cantera.__version__)"`), and operating system

For questions about the physics or the model formulation, an issue is fine too. The accompanying paper ([doi:10.5281/zenodo.24205162](https://doi.org/10.5281/zenodo.24205162)) describes the equations and the assumptions behind them.

## Setting up a development environment

```bash
git clone https://github.com/prasadshingne/engine_simulator.git
cd engine_simulator
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,gui]"
python scripts/fetch_mechanisms.py
```

Optional: the manual multi-zone solver uses CVODE through `scikits.odes`, which is easiest to install with conda:

```bash
conda install -c conda-forge scikits.odes sundials
```

## Running the tests

```bash
pytest
```

The suite needs the 33-species Nissan mechanism (`python scripts/fetch_mechanisms.py --nissan`). Tests marked `cvode` are skipped when `scikits.odes` is not installed. The same tests run on every push and pull request through GitHub Actions.

## Submitting a change

1. Fork the repository and create a branch from `main`.
2. Make your change. Keep pull requests focused on one thing.
3. Add or update tests in `tests/` for any change to the physics or numerics. Where possible, set expected values from a physical limit or an analytical result (for example, an isentropic compression, or the multi-zone model reducing to the single-zone model with wall heat transfer switched off), not from the code's own output.
4. Run `pytest` and make sure it passes.
5. Update `CHANGELOG.md` under "Unreleased".
6. Open a pull request describing what changed and why.

Scripts in `scripts/dev/` are exploratory and are not part of the test suite. Please do not add new tests there.

## Support expectations

This project is maintained by one person in their spare time. I aim to respond to issues and pull requests within about two weeks, but cannot promise a fixed response time. The software is provided as is, under the MIT license; it has not been validated against engine data, and results should be checked before they are relied on.

## Code of conduct

Be respectful and constructive. Harassment or personal attacks are not acceptable in issues, pull requests, or any other project space.
