# Development scripts

These are exploratory scripts kept from development of the multi-zone and
Cantera solvers: solver comparisons, coupling-coefficient sweeps, and
mechanism comparisons. They are not part of the automated test suite (that is
in `tests/` and runs with `pytest`), and they print or plot results rather than
assert them.

Run them from the repository root, after installing the package and the
mechanism files (see the main README), for example:

```bash
python scripts/dev/test_multizone_validation.py
```

Some need the LLNL gasoline surrogate mechanism, and `test_cvode_multizone.py`
needs the optional `scikits.odes` CVODE solver.
