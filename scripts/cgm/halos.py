import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


# ==================== #
# HALO CATALOG LOADING #
# ==================== #

# Load Subfind halo catalog from IllustrisTNG/CAMEL simulation.
def load_subfind_catalog(snapshot_path, group_path=None):
    snapshot_path = Path(snapshot_path)

    # Load cosmology from snapshot header (needed for virial radius calculations)
    with h5py.File(snapshot_path, 'r') as f:
        header = f['Header']
        omega_m = float(header.attrs['Omega0'])
        omega_lambda = float(header.attrs['OmegaLambda'])

    print(f"Cosmology: Omega_m={omega_m:.4f}, Omega_Lambda={omega_lambda:.4f}")

    # Attempt to find group catalog if not provided
    if group_path is None:
        group_path = _find_group_catalog(snapshot_path)
        if group_path is None:
            raise FileNotFoundError(
                f"Could not find group catalog for {snapshot_path}\n"
            )

    if not os.path.exists(group_path):
        raise FileNotFoundError(f"Group catalog not found: {group_path}")

    print(f"Loading halo catalog from: {os.path.basename(group_path)}")

    with h5py.File(group_path, 'r') as f:
        # Load header information
        header = f['Header']
        redshift = float(header.attrs['Redshift'])
        hubble = float(header.attrs['HubbleParam'])
        boxsize = float(header.attrs['BoxSize'])  # ckpc/h

        # Check if Subhalo group exists
        if 'Subhalo' not in f:
            raise ValueError(f"No Subhalo data in group catalog: {group_path}")

        subhalo = f['Subhalo']
        n_subhalos = len(subhalo['SubhaloMass'])

        print(f"Found {n_subhalos:,} subhalos at z={redshift:.3f}")

        # Load subhalo properties
        # Mass fields (in 10^10 M_sun/h)
        mass_total = np.array(subhalo['SubhaloMass']) * 1e10 / hubble  # M_sun

        # Try to load different mass definitions (may not all exist)
        # M200c: Mass within R200c (critical density definition)
        if 'SubhaloMassType' in subhalo:
            mass_gas = np.array(
                subhalo['SubhaloMassType'][:, 0]) * 1e10 / hubble
            mass_dm = np.array(
                subhalo['SubhaloMassType'][:, 1]) * 1e10 / hubble
            mass_stars = np.array(
                subhalo['SubhaloMassType'][:, 4]) * 1e10 / hubble
        else:
            mass_gas = np.zeros(n_subhalos)
            mass_dm = np.zeros(n_subhalos)
            mass_stars = np.zeros(n_subhalos)

        # Positions (ckpc/h, comoving)
        positions = np.array(subhalo['SubhaloPos'])  # Shape: (N, 3)

        # Velocities (km/s)
        velocities = np.array(subhalo['SubhaloVel'])  # Shape: (N, 3)

        # Radii - Try multiple fields for compatibility
        # SubhaloHalfmassRad is typically available
        if 'SubhaloHalfmassRad' in subhalo:
            half_mass_radius = np.array(
                subhalo['SubhaloHalfmassRad'])  # ckpc/h
        else:
            half_mass_radius = np.zeros(n_subhalos)

        # Build catalog dataframe
        catalog_data = {
            'halo_id': np.arange(n_subhalos),
            'mass_total': mass_total,
            'mass_gas': mass_gas,
            'mass_dm': mass_dm,
            'mass_stars': mass_stars,
            'position_x': positions[:, 0],
            'position_y': positions[:, 1],
            'position_z': positions[:, 2],
            'velocity_x': velocities[:, 0],
            'velocity_y': velocities[:, 1],
            'velocity_z': velocities[:, 2],
            'half_mass_radius': half_mass_radius,
            'redshift': np.full(n_subhalos, redshift),
            'hubble': np.full(n_subhalos, hubble),
            'boxsize': np.full(n_subhalos, boxsize),
            'omega_m': np.full(n_subhalos, omega_m),
            'omega_lambda': np.full(n_subhalos, omega_lambda),
        }

        # Compute virial radii for all three mass definitions
        catalog_data['radius_200c'] = _compute_r200c(
            mass_total, redshift, hubble, omega_m, omega_lambda)
        catalog_data['radius_vir'] = _compute_rvir(
            mass_total, redshift, hubble, omega_m, omega_lambda)

        # Also store masses in different units for flexibility
        catalog_data['mass_200c'] = mass_total  # For now, use total mass
        catalog_data['mass_vir'] = mass_total   # User can modify if needed

    catalog = pd.DataFrame(catalog_data)

    # Filter out subhalos with zero or very low mass
    catalog = catalog[catalog['mass_total'] > 1e8].copy()  # > 10^8 M_sun

    print(f"Catalog loaded: {len(catalog):,} subhalos with M > 10^8 M_sun")

    return catalog


# Attempt to find group catalog file corresponding to snapshot.
def _find_group_catalog(snapshot_path):
    snapshot_path = Path(snapshot_path)
    snapshot_dir = snapshot_path.parent

    # Extract snapshot number
    snapshot_name = snapshot_path.stem
    snap_num = None
    if 'snap_' in snapshot_name or 'snap-' in snapshot_name:
        parts = snapshot_name.replace('-', '_').split('_')
        for i, part in enumerate(parts):
            if part == 'snap' and i + 1 < len(parts):
                snap_num = parts[i + 1]
                break

    if snap_num is None:
        return None

    # Try various patterns (prioritize CAMEL format first)
    patterns = [
        # CAMEL format
        snapshot_dir / f"groups_{snap_num}.hdf5",
        # Original TNG format
        snapshot_dir / f"fof_subhalo_tab_{snap_num}.hdf5",
        snapshot_dir / f"groups_{snap_num}" /
        f"fof_subhalo_tab_{snap_num}.0.hdf5",
        snapshot_dir / "groups" / f"fof_subhalo_tab_{snap_num}.hdf5",
        snapshot_dir / f"fof_subhalo_tab_{snap_num}.0.hdf5",
    ]

    for pattern in patterns:
        if pattern.exists():
            return str(pattern)

    return None


# Radius within which mean density is 200x critical density.
def _compute_r200c(mass, redshift, hubble, omega_m, omega_lambda):
    # Critical density at z=0: rho_crit,0 = 3H0^2 / (8 pi G)
    # rho_crit(z) = rho_crit,0 * E(z)^2, where E(z) = sqrt(Omega_m (1+z)^3 + Omega_Lambda)

    # Hubble parameter H0 = 100 h km/s/Mpc
    H0 = 100.0 * hubble  # km/s/Mpc

    # Critical density at z=0 (M_sun / kpc^3)
    G_kpc = 4.302e-6  # kpc (km/s)^2 / M_sun
    rho_crit_0 = 3 * (H0 / 1000)**2 / (8 * np.pi * G_kpc)  # M_sun / kpc^3

    # Evolution function
    E_z = np.sqrt(omega_m * (1 + redshift)**3 + omega_lambda)
    rho_crit_z = rho_crit_0 * E_z**2

    # R200c: M = (4/3) pi R^3 * 200 * rho_crit
    # R = (3M / (800 pi rho_crit))^(1/3)
    r200c = (3 * mass / (800 * np.pi * rho_crit_z))**(1./3.)  # physical kpc

    # Convert to comoving kpc/h
    r200c_comoving = r200c * (1 + redshift) * hubble  # ckpc/h

    return r200c_comoving


# Compute virial radius
def _compute_rvir(mass, redshift, hubble, omega_m, omega_lambda):
    # https://arxiv.org/pdf/astro-ph/9710107
    # For flat Lambda-CDM: Delta_vir(z) = 18 pi^2 + 82 x - 39 x^2
    # where x = Omega_m(z) - 1

    # Omega_m at redshift z
    omega_m_z = omega_m * (1 + redshift)**3 / \
        (omega_m * (1 + redshift)**3 + omega_lambda)

    x = omega_m_z - 1
    delta_vir = 18 * np.pi**2 + 82 * x - 39 * x**2

    # Similar to R200c but with delta_vir instead of 200
    H0 = 100.0 * hubble
    G_kpc = 4.302e-6
    rho_crit_0 = 3 * (H0 / 1000)**2 / (8 * np.pi * G_kpc)
    E_z = np.sqrt(omega_m * (1 + redshift)**3 + omega_lambda)
    rho_crit_z = rho_crit_0 * E_z**2

    r_vir = (3 * mass / (4 * np.pi * delta_vir *
             rho_crit_z))**(1./3.)  # physical kpc
    r_vir_comoving = r_vir * (1 + redshift) * hubble  # ckpc/h

    return r_vir_comoving


# ========================= #
# HALO FILTERING            #
# ========================= #

def filter_halos_by_mass(catalog, mass_range, mass_type='mass_total'):
    if mass_type not in catalog.columns:
        raise ValueError(f"Mass type '{mass_type}' not in catalog. "
                         f"Available: {[c for c in catalog.columns if 'mass' in c]}")

    log_mass = np.log10(catalog[mass_type])
    mask = (log_mass >= mass_range[0]) & (log_mass <= mass_range[1])

    filtered = catalog[mask].copy()

    print(f"Mass filter [{mass_range[0]:.1f}, {mass_range[1]:.1f}] (log M_sun): "
          f"{len(filtered):,} / {len(catalog):,} halos")

    return filtered


def filter_isolated_halos(catalog, isolation_factor=3.0, radius_type='radius_vir'):
    if len(catalog) == 0:
        return catalog

    positions = catalog[['position_x', 'position_y', 'position_z']].values
    masses = catalog['mass_total'].values
    radii = catalog[radius_type].values

    isolated_mask = np.ones(len(catalog), dtype=bool)

    for i in range(len(catalog)):
        # Check all other halos
        for j in range(len(catalog)):
            if i == j:
                continue

            # Distance between halos
            dx = positions[j] - positions[i]

            # Handle periodic boundary conditions
            boxsize = catalog.iloc[i]['boxsize']
            dx = np.where(dx > boxsize/2, dx - boxsize, dx)
            dx = np.where(dx < -boxsize/2, dx + boxsize, dx)

            distance = np.sqrt(np.sum(dx**2))

            # Check if neighbor is massive and too close
            isolation_radius = isolation_factor * radii[i]
            if distance < isolation_radius and masses[j] > 0.5 * masses[i]:
                isolated_mask[i] = False
                break

    filtered = catalog[isolated_mask].copy()

    print(f"Isolation filter ({isolation_factor:.1f} x R_vir): "
          f"{len(filtered):,} / {len(catalog):,} isolated halos")

    return filtered


# Compute gas temperature from internal energy and electron abundance.
def _compute_temperature(internal_energy, electron_abundance, helium_mass_fraction=0.24):
    # Uses ideal gas law: T = (gamma - 1) * u * mu * m_p / k_B
    # where mu is mean molecular weight per particle.

    # Physical constants
    gamma = 5./3.  # Adiabatic index for monoatomic gas
    k_B = 1.38064852e-16  # Boltzmann constant in erg/K
    m_p = 1.67262171e-24  # Proton mass in g

    # Convert internal energy from (km/s)^2 to erg/g
    # 1 (km/s)^2 = 1e10 (cm/s)^2 = 1e10 erg/g
    u_cgs = internal_energy * 1e10  # erg/g

    # Compute mean molecular weight
    # mu = 1 / (X * (1 + ne) + Y/4 + Z/16)
    # For fully ionized gas: X = hydrogen mass fraction, Y = helium, Z = metals (small)
    # Approximation: mu ≈ 1 / (2 * X + 3/4 * Y) for ionized, mu ≈ 4/(3+5*X) for neutral
    X_hydrogen = 1.0 - helium_mass_fraction  # Hydrogen mass fraction
    Y_helium = helium_mass_fraction

    # Mean molecular weight per particle (accounting for ionization)
    mu = (1.0 + 4.0 * Y_helium / X_hydrogen) / \
        (1.0 + Y_helium / X_hydrogen + electron_abundance)

    # Temperature: T = (gamma - 1) * u * mu * m_p / k_B
    temperature = (gamma - 1.0) * u_cgs * mu * m_p / k_B

    return temperature


# Extract gas particles within a sphere around halo center.
def get_gas_in_halo(snapshot_path, halo_position, radius,
                    fields=None, max_particles=None):
    # Temperature is computed from InternalEnergy and ElectronAbundance if not directly available in the snapshot.
    if fields is None:
        fields = ['Coordinates', 'Density', 'Temperature',
                  'NeutralHydrogenAbundance', 'Masses', 'Velocities']

    with h5py.File(snapshot_path, 'r') as f:
        if 'PartType0' not in f:
            raise ValueError("No gas particles in snapshot")

        gas = f['PartType0']

        # Load coordinates first to select particles
        coords = np.array(gas['Coordinates'])  # (N, 3) in ckpc/h

        # Get box size for periodic wrapping
        header = f['Header']
        boxsize = float(header.attrs['BoxSize'])

        # Compute distance from halo center (with periodic boundaries)
        dx = coords - halo_position
        dx = np.where(dx > boxsize/2, dx - boxsize, dx)
        dx = np.where(dx < -boxsize/2, dx + boxsize, dx)
        distance = np.sqrt(np.sum(dx**2, axis=1))

        # Select particles within radius
        mask = distance <= radius
        n_selected = np.sum(mask)

        if n_selected == 0:
            print(f"Warning: No gas particles found within {
                  radius:.1f} ckpc/h")
            return {'n_particles': 0}

        # Apply max_particles limit if needed
        if max_particles is not None and n_selected > max_particles:
            # Randomly subsample
            selected_indices = np.where(mask)[0]
            selected_indices = np.random.choice(selected_indices, max_particles,
                                                replace=False)
            mask = np.zeros(len(coords), dtype=bool)
            mask[selected_indices] = True
            n_selected = max_particles

        # Load requested fields
        data = {'n_particles': n_selected, 'distance': distance[mask]}

        # Track if we need to compute temperature
        compute_temp = False
        if 'Temperature' in fields and 'Temperature' not in gas:
            compute_temp = True
            # Add required fields for temperature computation
            fields_to_load = list(fields)
            if 'InternalEnergy' not in fields_to_load:
                fields_to_load.append('InternalEnergy')
            if 'ElectronAbundance' not in fields_to_load:
                fields_to_load.append('ElectronAbundance')
        else:
            fields_to_load = fields

        for field in fields_to_load:
            if field == 'Coordinates':
                data[field] = coords[mask]
            elif field == 'Temperature':
                if field in gas:
                    data[field] = np.array(gas[field])[mask]
                # else: will be computed below
            elif field in gas:
                data[field] = np.array(gas[field])[mask]
            else:
                # Don't warn for temp computation fields
                if field not in ['InternalEnergy', 'ElectronAbundance']:
                    print(f"Warning: Field '{field}' not found in snapshot")

        # Compute temperature if needed
        if compute_temp:
            if 'InternalEnergy' in data and 'ElectronAbundance' in data:
                data['Temperature'] = _compute_temperature(
                    data['InternalEnergy'],
                    data['ElectronAbundance']
                )
                # Clean up intermediate fields if they weren't originally requested
                if 'InternalEnergy' not in fields:
                    del data['InternalEnergy']
                if 'ElectronAbundance' not in fields:
                    del data['ElectronAbundance']
            else:
                print(
                    "Warning: Cannot compute temperature - InternalEnergy or ElectronAbundance missing")

    return data


def compute_virial_radius(halo_mass, redshift, hubble=0.6711,
                          definition='vir', omega_m=None, omega_lambda=None):
    if omega_m is None or omega_lambda is None:
        raise ValueError(
            "omega_m and omega_lambda are required for CAMEL simulations.\n"
            "Load from snapshot header: header.attrs['Omega0'] and ['OmegaLambda']"
        )

    if definition == 'vir':
        r = _compute_rvir(halo_mass, redshift, hubble, omega_m, omega_lambda)
    elif definition == '200c':
        r = _compute_r200c(halo_mass, redshift, hubble, omega_m, omega_lambda)
    else:
        raise ValueError(f"Unknown definition: {
                         definition}. Use 'vir' or '200c'")

    # Convert from comoving kpc/h to physical kpc
    r_physical = r / ((1 + redshift) * hubble)

    return r_physical
