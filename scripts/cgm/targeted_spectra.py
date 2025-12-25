import numpy as np
import h5py
import os


# ================================= #
# CGM SIGHTLINE POSITION GENERATION #
# ================================= #

# Generate sightline positions at specific impact parameters around halos.
def generate_cgm_sightlines(halo_catalog, impact_params=[0.25, 0.5, 0.75, 1.0, 1.25],
                            n_per_bin=100, azimuthal_samples=8, 
                            axis_direction='z', radius_type='radius_vir'):
    # For each halo and each impact parameter:
    #   - Sample azimuthal_samples angles around the halo
    #   - For each angle, generate n_per_bin/azimuthal_samples sightlines
    #   - Distribute sightlines along the line-of-sight direction
    print("\n" + "=" * 70)
    print("GENERATING CGM-TARGETED SIGHTLINES")
    print("=" * 70)
    print(f"Number of halos:       {len(halo_catalog)}")
    print(f"Impact parameters:     {impact_params} (in R_vir)")
    print(f"Sightlines per bin:    {n_per_bin}")
    print(f"Azimuthal samples:     {azimuthal_samples}")
    print(f"Axis direction:        {axis_direction}")
    print(f"Radius type:           {radius_type}")
    
    # Map axis direction to integer
    axis_map = {'x': 1, 'y': 2, 'z': 3}
    axis_int = axis_map[axis_direction.lower()]
    
    # Storage for all sightlines
    all_positions = []
    all_axes = []
    all_impact_rvir = []
    all_impact_kpc = []
    all_halo_ids = []
    
    # Loop through halos
    for idx, halo in halo_catalog.iterrows():
        halo_pos = np.array([halo['position_x'], halo['position_y'], halo['position_z']])
        halo_radius = halo[radius_type]
        halo_id = halo['halo_id']
        boxsize = halo['boxsize']
        
        # Loop through impact parameters
        for b_rvir in impact_params:
            b_physical = b_rvir * halo_radius  # Convert to ckpc/h
            
            # Sample azimuthal angles
            angles = np.linspace(0, 2*np.pi, azimuthal_samples, endpoint=False)
            
            # Number of sightlines per angle
            n_per_angle = max(1, n_per_bin // azimuthal_samples)
            
            for angle in angles:
                # Generate positions at this impact parameter and angle
                positions = sample_impact_parameter_positions(
                    halo_pos, b_physical, angle, n_per_angle,
                    axis_direction, boxsize
                )
                
                # Store
                all_positions.append(positions)
                all_axes.extend([axis_int] * n_per_angle)
                all_impact_rvir.extend([b_rvir] * n_per_angle)
                all_impact_kpc.extend([b_physical] * n_per_angle)
                all_halo_ids.extend([halo_id] * n_per_angle)
    
    # Convert to arrays
    cofm = np.vstack(all_positions)
    axis = np.array(all_axes)
    impact_rvir = np.array(all_impact_rvir)
    impact_kpc = np.array(all_impact_kpc)
    halo_ids = np.array(all_halo_ids)
    
    # Extract halo information
    halo_positions = halo_catalog[['position_x', 'position_y', 'position_z']].values
    halo_radii = halo_catalog[radius_type].values
    
    print(f"\nGenerated {len(cofm):,} sightlines total")
    print(f"Sightlines per halo:   {len(cofm) // len(halo_catalog)}")
    
    return {
        'cofm': cofm,
        'axis': axis,
        'impact_params_rvir': impact_rvir,
        'impact_params_kpc': impact_kpc,
        'halo_ids': halo_ids,
        'halo_positions': halo_positions,
        'halo_radii': halo_radii,
        'n_halos': len(halo_catalog),
        'n_sightlines': len(cofm),
    }


# Generate sightline positions at a specific impact parameter and azimuthal angle.
def sample_impact_parameter_positions(halo_position, impact_param, 
                                      azimuthal_angle, n_sightlines,
                                      axis_direction='z', boxsize=None):
    # Offset in the plane perpendicular to axis_direction
    if axis_direction.lower() == 'z':
        # Impact parameter in xy-plane
        offset_x = impact_param * np.cos(azimuthal_angle)
        offset_y = impact_param * np.sin(azimuthal_angle)
        
        # Generate positions distributed along z-axis
        # Sample uniformly within ±3 Mpc around halo (typical for CGM studies)
        z_range = 6000  # ckpc/h (6 Mpc/h)
        z_offsets = np.random.uniform(-z_range/2, z_range/2, n_sightlines)
        
        positions = np.zeros((n_sightlines, 3))
        positions[:, 0] = halo_position[0] + offset_x
        positions[:, 1] = halo_position[1] + offset_y
        positions[:, 2] = halo_position[2] + z_offsets
        
    elif axis_direction.lower() == 'y':
        # Impact parameter in xz-plane
        offset_x = impact_param * np.cos(azimuthal_angle)
        offset_z = impact_param * np.sin(azimuthal_angle)
        
        y_range = 6000
        y_offsets = np.random.uniform(-y_range/2, y_range/2, n_sightlines)
        
        positions = np.zeros((n_sightlines, 3))
        positions[:, 0] = halo_position[0] + offset_x
        positions[:, 1] = halo_position[1] + y_offsets
        positions[:, 2] = halo_position[2] + offset_z
        
    elif axis_direction.lower() == 'x':
        # Impact parameter in yz-plane
        offset_y = impact_param * np.cos(azimuthal_angle)
        offset_z = impact_param * np.sin(azimuthal_angle)
        
        x_range = 6000
        x_offsets = np.random.uniform(-x_range/2, x_range/2, n_sightlines)
        
        positions = np.zeros((n_sightlines, 3))
        positions[:, 0] = halo_position[0] + x_offsets
        positions[:, 1] = halo_position[1] + offset_y
        positions[:, 2] = halo_position[2] + offset_z
    else:
        raise ValueError(f"Invalid axis_direction: {axis_direction}. Use 'x', 'y', or 'z'")
    
    # Apply periodic boundary conditions if boxsize provided
    if boxsize is not None:
        positions = np.mod(positions, boxsize)
    
    return positions


# Compute impact parameters for sightlines relative to nearby halos.
def compute_impact_parameters(sightline_positions, halo_catalog, 
                              axis_direction='z', radius_type='radius_vir'):
    halo_positions = halo_catalog[['position_x', 'position_y', 'position_z']].values
    halo_radii = halo_catalog[radius_type].values
    halo_ids = halo_catalog['halo_id'].values
    halo_masses = halo_catalog['mass_total'].values
    boxsize = halo_catalog.iloc[0]['boxsize']
    
    n_sightlines = len(sightline_positions)
    impact_kpc = np.zeros(n_sightlines)
    impact_rvir = np.zeros(n_sightlines)
    nearest_ids = np.zeros(n_sightlines, dtype=int)
    nearest_masses = np.zeros(n_sightlines)
    
    # Determine which coordinates to use for impact parameter
    # (perpendicular to line-of-sight)
    if axis_direction.lower() == 'z':
        coords_indices = [0, 1]  # x, y
    elif axis_direction.lower() == 'y':
        coords_indices = [0, 2]  # x, z
    elif axis_direction.lower() == 'x':
        coords_indices = [1, 2]  # y, z
    else:
        raise ValueError(f"Invalid axis_direction: {axis_direction}")
    
    # For each sightline, find nearest halo
    for i in range(n_sightlines):
        sight_pos = sightline_positions[i]
        
        # Compute 2D distance to all halos (in plane perpendicular to LOS)
        dx = halo_positions[:, coords_indices[0]] - sight_pos[coords_indices[0]]
        dy = halo_positions[:, coords_indices[1]] - sight_pos[coords_indices[1]]
        
        # Handle periodic boundaries
        dx = np.where(dx > boxsize/2, dx - boxsize, dx)
        dx = np.where(dx < -boxsize/2, dx + boxsize, dx)
        dy = np.where(dy > boxsize/2, dy - boxsize, dy)
        dy = np.where(dy < -boxsize/2, dy + boxsize, dy)
        
        distances_2d = np.sqrt(dx**2 + dy**2)
        
        # Find nearest halo
        nearest_idx = np.argmin(distances_2d)
        
        impact_kpc[i] = distances_2d[nearest_idx]
        impact_rvir[i] = impact_kpc[i] / halo_radii[nearest_idx]
        nearest_ids[i] = halo_ids[nearest_idx]
        nearest_masses[i] = halo_masses[nearest_idx]
    
    return {
        'impact_params_kpc': impact_kpc,
        'impact_params_rvir': impact_rvir,
        'nearest_halo_ids': nearest_ids,
        'nearest_halo_masses': nearest_masses,
    }


# Append CGM-specific metadata to spectra HDF5 file.
def save_cgm_metadata(output_file, cgm_data, halo_catalog):
    if not os.path.exists(output_file):
        print(f"Error: Output file not found: {output_file}")
        return 1
    
    print(f"\n[CGM Metadata] Appending CGM metadata to {os.path.basename(output_file)}")
    
    with h5py.File(output_file, 'a') as f:
        # Create CGM_metadata group if it doesn't exist
        if 'CGM_metadata' in f:
            del f['CGM_metadata']  # Remove old metadata
        
        cgm_group = f.create_group('CGM_metadata')
        
        # Store sightline-level information
        cgm_group.create_dataset('halo_ids', data=cgm_data['halo_ids'])
        cgm_group.create_dataset('impact_params_rvir', data=cgm_data['impact_params_rvir'])
        cgm_group.create_dataset('impact_params_kpc', data=cgm_data['impact_params_kpc'])
        cgm_group.create_dataset('sightline_positions', data=cgm_data['cofm'])
        cgm_group.create_dataset('sightline_axes', data=cgm_data['axis'])
        
        # Store halo-level information
        cgm_group.create_dataset('halo_positions', data=cgm_data['halo_positions'])
        cgm_group.create_dataset('halo_radii', data=cgm_data['halo_radii'])
        
        # Store halo catalog summary
        cgm_group.create_dataset('halo_catalog/halo_ids', 
                                data=halo_catalog['halo_id'].values)
        cgm_group.create_dataset('halo_catalog/masses', 
                                data=halo_catalog['mass_total'].values)
        cgm_group.create_dataset('halo_catalog/positions', 
                                data=halo_catalog[['position_x', 'position_y', 
                                                   'position_z']].values)
        cgm_group.create_dataset('halo_catalog/redshift', 
                                data=halo_catalog['redshift'].values[0])
        
        # Add attributes
        cgm_group.attrs['n_halos'] = cgm_data['n_halos']
        cgm_group.attrs['n_sightlines'] = cgm_data['n_sightlines']
        cgm_group.attrs['description'] = 'CGM-targeted sightlines at specific impact parameters'
    
    print(f"Saved metadata for {cgm_data['n_sightlines']} sightlines around "
          f"{cgm_data['n_halos']} halos")
    
    return 0


# Analyze CGM spectra and create impact parameter binned statistics.
def analyze_cgm_spectra(spectra_file, output_dir=None):
    if not os.path.exists(spectra_file):
        print(f"Error: Spectra file not found: {spectra_file}")
        return 1
    
    print("\n" + "=" * 70)
    print("ANALYZING CGM SPECTRA")
    print("=" * 70)
    
    with h5py.File(spectra_file, 'r') as f:
        # Check for CGM metadata
        if 'CGM_metadata' not in f:
            print("Error: No CGM_metadata found. Is this a CGM-targeted spectra file?")
            return 1
        
        cgm = f['CGM_metadata']
        
        # Load metadata
        impact_rvir = np.array(cgm['impact_params_rvir'])
        impact_kpc = np.array(cgm['impact_params_kpc'])
        halo_ids = np.array(cgm['halo_ids'])
        
        print(f"Loaded CGM metadata:")
        print(f"  Sightlines: {len(impact_rvir):,}")
        print(f"  Unique halos: {len(np.unique(halo_ids))}")
        print(f"  Impact parameter range: {impact_rvir.min():.2f} - {impact_rvir.max():.2f} R_vir")
        
        # Load tau data (assuming Lyman-alpha for now)
        # TODO: Make this flexible for different lines
        if 'tau' in f and 'H' in f['tau'] and '1' in f['tau/H']:
            tau_group = f['tau/H/1']
            # Find the wavelength dataset (e.g., '1215')
            wavelengths = list(tau_group.keys())
            if wavelengths:
                tau_data = np.array(tau_group[wavelengths[0]])
                print(f"  Loaded tau data: {tau_data.shape}")
                
                # Compute flux
                flux = np.exp(-tau_data)
                
                # Compute mean flux per sightline
                mean_flux = np.mean(flux, axis=1)
                
                # Print statistics per impact parameter bin
                unique_impacts = np.unique(impact_rvir)
                print(f"\n  Mean flux vs impact parameter:")
                for b in sorted(unique_impacts):
                    mask = np.abs(impact_rvir - b) < 0.01
                    if np.sum(mask) > 0:
                        mean_f = np.mean(mean_flux[mask])
                        std_f = np.std(mean_flux[mask])
                        print(f"    b = {b:.2f} R_vir: <F> = {mean_f:.3f} ± {std_f:.3f} "
                              f"({np.sum(mask)} sightlines)")
            else:
                print("  Warning: No tau datasets found")
        else:
            print("  Warning: No Lyman-alpha tau data found")
    
    print("=" * 70)
    
    return 0
