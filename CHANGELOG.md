# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- GitHub Actions workflow that runs the test suite on every push and pull request.
- `scripts/fetch_mechanisms.py` to download the reaction mechanism files and check that they load.
- `LICENSE` file (MIT), `CONTRIBUTING.md` with development setup and support expectations, and this changelog.

### Changed
- Cantera (3.1 or newer) is now a declared dependency, so `pip install` brings it in.
- Package version in `pyproject.toml` set to 1.0.0 to match the release.
- Exploratory development scripts moved from the repository root into `scripts/dev/`.
- Citation metadata points to the paper's preprint, doi:10.5281/zenodo.24205162.

## [1.0.0] - 2026-08-14

First public release.

### Added
- Zero-dimensional single-zone and quasi-dimensional multi-zone HCCI models, with 10, 20 and 40 zone presets following Kodavasal et al. (2013).
- Cantera `ReactorNet` solver for large mechanisms, using coupled `IdealGasMoleReactor` instances with GMRES and an adaptive preconditioner.
- Two reaction mechanisms: the 33-species Nissan PRF mechanism and the 312-species LLNL gasoline surrogate.
- Woschni wall heat transfer correlation and EGR handling.
- Adiabatic Core phenomenological model, fitted to the multi-zone results.
- Streamlit GUI with Plotly plots and a run comparison mode.
- Tutorial notebooks and a pytest suite.
- Live deployment on Hugging Face Spaces.

[Unreleased]: https://github.com/prasadshingne/engine_simulator/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/prasadshingne/engine_simulator/releases/tag/v1.0.0
