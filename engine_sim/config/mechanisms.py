"""Mechanism presets for the engine simulator."""

MECHANISM_PRESETS = {
    "Nissan PRF (33 species)": {
        "file": "data/mechanisms/Nissan_chem.yaml",
        "fuel": "C8H18",
        "fuel_label": "iso-Octane (C8H18)",
        "major_species": ["C8H18", "O2", "CO2", "H2O"],
    },
    "LLNL Gasoline Surrogate (312 species)": {
        "file": "data/mechanisms/gasoline_surrogate_323.yaml",
        "fuel": {"IC8H18": 0.5413, "NC7H16": 0.1488, "C6H5CH3": 0.2738, "C5H10-2": 0.0361},
        "fuel_label": "4-Component Surrogate (54.13% iso-octane, 14.88% n-heptane, 27.38% toluene, 3.61% 2-pentene)",
        "major_species": ["IC8H18", "NC7H16", "C6H5CH3", "C5H10-2", "O2", "CO2", "H2O"],
    },
}

DEFAULT_MECHANISM = "Nissan PRF (33 species)"
