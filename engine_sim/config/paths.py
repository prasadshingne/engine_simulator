"""Path resolution utilities for data files."""

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory (where pyproject.toml lives)."""
    return Path(__file__).parent.parent.parent


def get_default_config() -> Path:
    """Get path to default configuration YAML."""
    return Path(__file__).parent / "default_config.yaml"


def get_mechanism_dir() -> Path:
    """Get path to chemical mechanism files directory."""
    return get_project_root() / "data" / "mechanisms"


def resolve_mechanism_path(mechanism: str) -> str:
    """Resolve a mechanism path, checking relative to project root if needed.

    Tries in order:
    1. Absolute path (if exists)
    2. Relative to project root
    3. In the mechanisms directory by filename
    4. Returns as-is (let Cantera handle it)
    """
    p = Path(mechanism)
    if p.is_absolute() and p.exists():
        return str(p)
    # Try relative to project root
    project_path = get_project_root() / mechanism
    if project_path.exists():
        return str(project_path)
    # Try in mechanisms directory
    mech_path = get_mechanism_dir() / p.name
    if mech_path.exists():
        return str(mech_path)
    return mechanism
