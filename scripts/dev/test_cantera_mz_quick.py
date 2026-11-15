"""Quick test of Cantera multizone — tune coupling coefficient."""
import numpy as np
import time
import cantera as ct

from engine_sim.engine.geometry import GeometryParams
from engine_sim.models.chemistry import Chemistry, ChemistryParams

geom = GeometryParams(bore=0.086, stroke=0.086, con_rod=0.143, comp_ratio=14.0)
T_init, P_init = 450.0, 1.5e5
rpm = 1200
nzones = 10

nissan_params = ChemistryParams(
    mechanism="data/mechanisms/Nissan_chem.yaml",
    fuel="C8H18", phi=0.4, egr=0.0
)

chem = Chemistry(nissan_params)
Y0 = chem.setup_initial_mixture(T=T_init, P=P_init)
nsp = len(Y0)
V0 = geom.cylinder_volume(np.deg2rad(-180))

omega = rpm * 2 * np.pi / 60
theta_start = np.deg2rad(-180)

def piston_velocity_per_zone(t):
    theta = omega * t + theta_start
    return -geom.volume_rate(theta, rpm) / nzones

# Test different configurations
configs = [
    ("Distributed piston only (no coupling)", 0.0),
    ("Distributed piston + K=1e-4", 1e-4),
    ("Distributed piston + K=1e-2", 1e-2),
    ("Distributed piston + K=1.0", 1.0),
]

for name, K in configs:
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"{'='*60}")

    # Create N reactors
    gases = []
    reactors = []
    for i in range(nzones):
        gas_i = ct.Solution("data/mechanisms/Nissan_chem.yaml")
        gas_i.TPY = T_init, P_init, Y0
        r = ct.IdealGasReactor(gas_i)
        r.volume = V0 / nzones
        gases.append(gas_i)
        reactors.append(r)

    # Distributed piston walls
    piston_envs = []
    for i in range(nzones):
        env_i = ct.Reservoir(ct.Solution('air.yaml'))
        piston_envs.append(env_i)
        pw = ct.Wall(reactors[i], env_i, velocity=piston_velocity_per_zone)

    # Coupling walls (if K > 0)
    if K > 0:
        for i in range(nzones - 1):
            cw = ct.Wall(reactors[i], reactors[i+1])
            cw.expansion_rate_coeff = K

    net = ct.ReactorNet(reactors)
    net.rtol = 1e-6
    net.atol = 1e-12
    net.max_steps = 500000

    # Compression only: -180 to -30 deg
    t_end = (np.deg2rad(-30) - theta_start) / omega
    n_points = 100
    t_eval = np.linspace(0, t_end, n_points)

    t0 = time.time()
    success = True
    try:
        for t_target in t_eval[1:]:
            net.advance(t_target)
    except Exception as e:
        success = False
        print(f"  FAILED at t={net.time*1000:.3f} ms: {e}")

    dt = time.time() - t0
    print(f"  Time: {dt:.1f}s, Success: {success}")

    if success:
        zone_temps = [r.T for r in reactors]
        zone_pressures = [r.thermo.P for r in reactors]
        zone_volumes = [r.volume for r in reactors]
        V_total = sum(zone_volumes)
        V_expected = geom.cylinder_volume(np.deg2rad(-30))

        print(f"  Zone temperatures: {[f'{T:.1f}' for T in zone_temps]}")
        print(f"  Zone pressures:    {[f'{P/1e5:.2f}' for P in zone_pressures]} bar")
        print(f"  V_total = {V_total*1e6:.2f} cm³ (expected: {V_expected*1e6:.2f})")
        print(f"  P spread: {(max(zone_pressures)-min(zone_pressures))/1e5:.4f} bar")
        print(f"  T spread: {max(zone_temps)-min(zone_temps):.4f} K")
