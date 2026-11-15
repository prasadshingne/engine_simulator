"""Download the converted Cantera mechanism files into data/mechanisms/.

The YAML mechanisms are not stored in this repository. This script fetches the
copies used by the live Hugging Face Space deployment. To build them yourself
from the original sources instead, see data/mechanisms/README.md.

Usage (from the repository root):
    python scripts/fetch_mechanisms.py            # both mechanisms
    python scripts/fetch_mechanisms.py --nissan   # only the 33-species PRF mechanism
"""

import argparse
import sys
import urllib.request
import warnings
from pathlib import Path

BASE_URL = (
    "https://huggingface.co/spaces/prasadshingne/engine-simulator/"
    "resolve/main/data/mechanisms/"
)

# file name -> (expected species, expected reactions)
MECHANISMS = {
    "Nissan_chem.yaml": (33, 48),
    "gasoline_surrogate_323.yaml": (312, 2469),
}

DEST = Path(__file__).resolve().parent.parent / "data" / "mechanisms"


def fetch(name: str, force: bool = False) -> Path:
    path = DEST / name
    if path.exists() and not force:
        print(f"{name}: already present, skipping")
        return path
    print(f"{name}: downloading")
    with urllib.request.urlopen(BASE_URL + name, timeout=120) as resp:
        path.write_bytes(resp.read())
    return path


def check(path: Path, n_species: int, n_reactions: int) -> None:
    try:
        import cantera as ct
    except ImportError:
        print("  cantera not installed, skipping load check")
        return
    # The Nissan mechanism's NASA polynomials have small, known discontinuities
    # at Tmid; Cantera warns about each one. They do not affect loading.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gas = ct.Solution(str(path))
    ok = gas.n_species == n_species and gas.n_reactions == n_reactions
    print(f"  loaded: {gas.n_species} species, {gas.n_reactions} reactions"
          + ("" if ok else f" (expected {n_species}, {n_reactions})"))
    if not ok:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nissan", action="store_true",
                        help="fetch only the 33-species Nissan PRF mechanism")
    parser.add_argument("--force", action="store_true",
                        help="download even if the file already exists")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    names = ["Nissan_chem.yaml"] if args.nissan else list(MECHANISMS)
    for name in names:
        path = fetch(name, force=args.force)
        check(path, *MECHANISMS[name])


if __name__ == "__main__":
    main()
