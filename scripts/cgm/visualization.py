import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py

import scripts.utils as utils


# ===================== #
# HALO PROJECTION PLOTS #
# ===================== #

def plot_halo_projection(snapshot_path, halo, output_path, 
                         slice_thickness=1000, 
                         properties=['density', 'temperature'],
                         show_virial_circle=True,
                         axes_to_plot=['xy', 'xz', 'yz']):
    from .halos import get_gas_in_halo
    
    print("\n[Projection] Creating gas projection plots...")
    
    # Extract halo info
    halo_pos = np.array([halo['position_x'], halo['position_y'], halo['position_z']])
    r_vir = halo['radius_vir']
    redshift = halo['redshift']
    
    # Load gas within 3 x R_vir (to show extended CGM)
    extraction_radius = 3.0 * r_vir
    gas_data = get_gas_in_halo(
        snapshot_path, halo_pos, extraction_radius,
        fields=['Coordinates', 'Density', 'Temperature', 
                'NeutralHydrogenAbundance', 'Masses'],
        max_particles=500000  # Limit for memory
    )
    
    if gas_data['n_particles'] == 0:
        print("Error: No gas particles found")
        return 1
    
    coords = gas_data['Coordinates']
    density = gas_data.get('Density', None)
    temperature = gas_data.get('Temperature', None)
    neutral_frac = gas_data.get('NeutralHydrogenAbundance', None)
    masses = gas_data.get('Masses', None)
    
    print(f"Loaded {gas_data['n_particles']:,} particles within {extraction_radius:.1f} ckpc/h")
    
    # Center coordinates on halo
    coords_centered = coords - halo_pos
    
    # Setup figure
    n_props = len(properties)
    n_axes = len(axes_to_plot)
    fig, axes = plt.subplots(n_props, n_axes, figsize=(5*n_axes, 4*n_props))
    
    if n_props == 1 and n_axes == 1:
        axes = np.array([[axes]])
    elif n_props == 1:
        axes = axes.reshape(1, -1)
    elif n_axes == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each property and projection
    for i, prop in enumerate(properties):
        # Get property values
        if prop == 'density' and density is not None:
            values = np.log10(density * 1e10)  # Convert to log10(n_H / cm^-3)
            cmap = 'viridis'
            label = r'$\log_{10}(n_{\rm H} / {\rm cm}^{-3})$'
            vmin, vmax = -6, -1
        elif prop == 'temperature' and temperature is not None:
            values = np.log10(temperature)
            cmap = 'hot'
            label = r'$\log_{10}(T / {\rm K})$'
            vmin, vmax = 3, 7
        elif prop == 'neutral_fraction' and neutral_frac is not None:
            values = np.log10(neutral_frac + 1e-10)
            cmap = 'Blues'
            label = r'$\log_{10}(n_{\rm HI} / n_{\rm H})$'
            vmin, vmax = -4, 0
        else:
            print(f"Warning: Property '{prop}' not available, skipping")
            continue
        
        for j, axis_name in enumerate(axes_to_plot):
            ax = axes[i, j]
            
            # Select coordinates for projection
            if axis_name == 'xy':
                x_coord, y_coord = coords_centered[:, 0], coords_centered[:, 1]
                z_coord = coords_centered[:, 2]
                xlabel, ylabel = 'X [ckpc/h]', 'Y [ckpc/h]'
            elif axis_name == 'xz':
                x_coord, y_coord = coords_centered[:, 0], coords_centered[:, 2]
                z_coord = coords_centered[:, 1]
                xlabel, ylabel = 'X [ckpc/h]', 'Z [ckpc/h]'
            elif axis_name == 'yz':
                x_coord, y_coord = coords_centered[:, 1], coords_centered[:, 2]
                z_coord = coords_centered[:, 0]
                xlabel, ylabel = 'Y [ckpc/h]', 'Z [ckpc/h]'
            else:
                continue
            
            # Apply slice thickness filter
            slice_mask = np.abs(z_coord) < slice_thickness / 2
            
            # Create 2D histogram (mass-weighted for better visualization)
            extent = [-extraction_radius, extraction_radius, 
                     -extraction_radius, extraction_radius]
            bins = 200
            
            if masses is not None:
                weights = masses[slice_mask] * values[slice_mask]
            else:
                weights = values[slice_mask]
            
            h, xedges, yedges = np.histogram2d(
                x_coord[slice_mask], y_coord[slice_mask],
                bins=bins, range=[extent[:2], extent[2:]],
                weights=weights
            )
            
            # Normalize by mass
            if masses is not None:
                h_mass, _, _ = np.histogram2d(
                    x_coord[slice_mask], y_coord[slice_mask],
                    bins=bins, range=[extent[:2], extent[2:]],
                    weights=masses[slice_mask]
                )
                h = np.divide(h, h_mass, where=h_mass>0, out=np.zeros_like(h))
            
            # Plot
            im = ax.imshow(h.T, origin='lower', extent=extent, 
                          cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            
            # Overlay virial radius circle
            if show_virial_circle:
                circle = plt.Circle((0, 0), r_vir, fill=False, 
                                   edgecolor='white', linestyle='--', 
                                   linewidth=2, label=r'$R_{\rm vir}$')
                ax.add_patch(circle)
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{prop.replace('_', ' ').title()} ({axis_name.upper()} plane)")
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(label)
    
    plt.suptitle(f"Halo {halo['halo_id']} (z={redshift:.2f}, "
                f"M={halo['mass_total']:.2e} M$_\\odot$)", fontsize=14)
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved projection plot to {output_path}")
    plt.close()
    
    return 0


def plot_temperature_slices(snapshot_path, halo, output_path,
                            temp_bins=[(3.5, 4), (4, 4.5), (4.5, 7)],
                            slice_thickness=1000):
    from .halos import get_gas_in_halo
    
    print("\n[Temperature Slices] Creating temperature-filtered maps...")
    
    # Extract halo info
    halo_pos = np.array([halo['position_x'], halo['position_y'], halo['position_z']])
    r_vir = halo['radius_vir']
    
    # Load gas
    extraction_radius = 3.0 * r_vir
    gas_data = get_gas_in_halo(
        snapshot_path, halo_pos, extraction_radius,
        fields=['Coordinates', 'Density', 'Temperature', 'Masses'],
        max_particles=500000
    )
    
    if gas_data['n_particles'] == 0:
        return 1
    
    coords = gas_data['Coordinates'] - halo_pos
    density = gas_data['Density']
    temperature = gas_data['Temperature']
    masses = gas_data.get('Masses', None)
    log_temp = np.log10(temperature)
    
    # Filter to xy-plane slice
    slice_mask = np.abs(coords[:, 2]) < slice_thickness / 2
    
    # Setup figure
    fig, axes = plt.subplots(1, len(temp_bins), figsize=(15, 5))
    if len(temp_bins) == 1:
        axes = [axes]
    
    extent = [-extraction_radius, extraction_radius, 
             -extraction_radius, extraction_radius]
    bins = 200
    
    for i, (t_min, t_max) in enumerate(temp_bins):
        ax = axes[i]
        
        # Filter by temperature and slice
        temp_mask = (log_temp >= t_min) & (log_temp < t_max) & slice_mask
        
        if np.sum(temp_mask) == 0:
            print(f"Warning: No particles in temperature bin {t_min}-{t_max}")
            continue
        
        # Create histogram (mass-weighted density)
        weights = density[temp_mask]
        if masses is not None:
            weights = weights * masses[temp_mask]
        
        h, xedges, yedges = np.histogram2d(
            coords[temp_mask, 0], coords[temp_mask, 1],
            bins=bins, range=[extent[:2], extent[2:]],
            weights=weights
        )
        
        # Plot
        im = ax.imshow(h.T, origin='lower', extent=extent,
                      cmap='viridis', norm=LogNorm(vmin=1e-8, vmax=1e-2),
                      aspect='auto')
        
        # Virial radius circle
        circle = plt.Circle((0, 0), r_vir, fill=False, 
                           edgecolor='white', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_xlabel('X [ckpc/h]')
        ax.set_ylabel('Y [ckpc/h]')
        ax.set_title(f"$10^{{{t_min:.1f}}}$ K < T < $10^{{{t_max:.1f}}}$ K")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Density (arbitrary units)')
    
    plt.suptitle(f"Temperature-Filtered Density (Halo {halo['halo_id']})", fontsize=14)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved temperature slices to {output_path}")
    plt.close()
    
    return 0


def plot_radial_profiles(snapshot_path, halo, output_path,
                        properties=['density', 'temperature', 'neutral_fraction'],
                        r_bins=None):
    from .halos import get_gas_in_halo
    
    print("\n[Radial Profiles] Computing radial profiles...")
    
    # Extract halo info
    halo_pos = np.array([halo['position_x'], halo['position_y'], halo['position_z']])
    r_vir = halo['radius_vir']
    
    # Load gas
    extraction_radius = 3.0 * r_vir
    gas_data = get_gas_in_halo(
        snapshot_path, halo_pos, extraction_radius,
        fields=['Coordinates', 'Density', 'Temperature', 
                'NeutralHydrogenAbundance', 'Masses'],
        max_particles=500000
    )
    
    if gas_data['n_particles'] == 0:
        return 1
    
    # Compute radial distance
    distance = gas_data['distance']  # Already computed in get_gas_in_halo
    r_normalized = distance / r_vir  # In units of R_vir
    
    # Define radial bins if not provided
    if r_bins is None:
        r_bins = np.logspace(-2, np.log10(3), 20)  # 0.01 to 3 R_vir
    
    # Setup figure
    n_props = len(properties)
    fig, axes = plt.subplots(n_props, 1, figsize=(10, 4*n_props))
    if n_props == 1:
        axes = [axes]
    
    # Compute profiles for each property
    for i, prop in enumerate(properties):
        ax = axes[i]
        
        if prop == 'density':
            values = gas_data.get('Density', None)
            if values is None:
                continue
            ylabel = r'$\log_{10}(\rho / {\rm g \, cm}^{-3})$'
            values = np.log10(values)
        elif prop == 'temperature':
            values = gas_data.get('Temperature', None)
            if values is None:
                continue
            ylabel = r'$\log_{10}(T / {\rm K})$'
            values = np.log10(values)
        elif prop == 'neutral_fraction':
            values = gas_data.get('NeutralHydrogenAbundance', None)
            if values is None:
                continue
            ylabel = r'$\log_{10}(n_{\rm HI} / n_{\rm H})$'
            values = np.log10(values + 1e-10)
        else:
            continue
        
        # Compute median and percentiles in each radial bin
        r_centers = []
        medians = []
        p16 = []
        p84 = []
        
        for j in range(len(r_bins) - 1):
            mask = (r_normalized >= r_bins[j]) & (r_normalized < r_bins[j+1])
            if np.sum(mask) > 10:  # At least 10 particles
                r_centers.append((r_bins[j] + r_bins[j+1]) / 2)
                medians.append(np.median(values[mask]))
                p16.append(np.percentile(values[mask], 16))
                p84.append(np.percentile(values[mask], 84))
        
        r_centers = np.array(r_centers)
        medians = np.array(medians)
        p16 = np.array(p16)
        p84 = np.array(p84)
        
        # Plot
        ax.plot(r_centers, medians, 'o-', color='steelblue', linewidth=2)
        ax.fill_between(r_centers, p16, p84, alpha=0.3, color='steelblue')
        
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, 
                  label=r'$R_{\rm vir}$')
        
        ax.set_xlabel(r'$r / R_{\rm vir}$')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.set_xlim(r_bins[0], r_bins[-1])
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(f"{prop.replace('_', ' ').title()} Profile")
    
    plt.suptitle(f"Radial Profiles (Halo {halo['halo_id']})", fontsize=14)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved radial profiles to {output_path}")
    plt.close()
    
    return 0


def plot_halo_summary(snapshot_path, halo, output_path):
    from .halos import get_gas_in_halo
    
    print("\n[Summary] Creating comprehensive halo diagnostic plot...")
    
    # Extract halo info
    halo_pos = np.array([halo['position_x'], halo['position_y'], halo['position_z']])
    r_vir = halo['radius_vir']
    redshift = halo['redshift']
    
    # Load gas
    extraction_radius = 3.0 * r_vir
    gas_data = get_gas_in_halo(
        snapshot_path, halo_pos, extraction_radius,
        fields=['Coordinates', 'Density', 'Temperature', 
                'NeutralHydrogenAbundance', 'Masses'],
        max_particles=500000
    )
    
    if gas_data['n_particles'] == 0:
        return 1
    
    coords = gas_data['Coordinates'] - halo_pos
    density = gas_data['Density']
    temperature = gas_data['Temperature']
    neutral_frac = gas_data.get('NeutralHydrogenAbundance', None)
    distance = gas_data['distance']
    
    # Setup figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Density projection
    ax1 = fig.add_subplot(gs[0, 0])
    extent = [-extraction_radius, extraction_radius, 
             -extraction_radius, extraction_radius]
    slice_mask = np.abs(coords[:, 2]) < 500  # 500 ckpc/h slice
    
    print(f"  Density projection: {np.sum(slice_mask):,} particles in slice")
    print(f"  Density range: {density[slice_mask].min():.2e} to {density[slice_mask].max():.2e}")
    
    # Create mass-weighted density projection
    masses = gas_data.get('Masses', np.ones(len(density)))
    h_mass, xedges, yedges = np.histogram2d(
        coords[slice_mask, 0], coords[slice_mask, 1],
        bins=150, range=[extent[:2], extent[2:]],
        weights=masses[slice_mask]
    )
    
    h_dens, _, _ = np.histogram2d(
        coords[slice_mask, 0], coords[slice_mask, 1],
        bins=150, range=[extent[:2], extent[2:]],
        weights=density[slice_mask] * masses[slice_mask]
    )
    
    # Average density per bin (mass-weighted)
    h = np.divide(h_dens, h_mass, where=h_mass>0, out=np.zeros_like(h_dens))
    h_log = np.log10(h * 1e10, where=h>0, out=np.full_like(h, -10))
    
    # Use percentile-based color scaling
    h_valid = h_log[h > 0]
    if len(h_valid) > 0:
        vmin = np.percentile(h_valid, 5)
        vmax = np.percentile(h_valid, 95)
        print(f"  Density projection range (5-95th percentile): {vmin:.2f} to {vmax:.2f}")
    else:
        vmin, vmax = -6, -1
        print("  Warning: No valid density values in projection")
    
    im1 = ax1.imshow(h_log.T, origin='lower', extent=extent, cmap='viridis',
                    vmin=vmin, vmax=vmax, aspect='auto')
    circle = plt.Circle((0, 0), r_vir, fill=False, edgecolor='white', 
                       linestyle='--', linewidth=2)
    ax1.add_patch(circle)
    ax1.set_xlabel('X [ckpc/h]')
    ax1.set_ylabel('Y [ckpc/h]')
    ax1.set_title('Density Projection')
    plt.colorbar(im1, ax=ax1, label=r'$\log_{10}(n_{\rm H}/{\rm cm}^{-3})$')
    
    # Panel 2: Temperature projection
    ax2 = fig.add_subplot(gs[0, 1])
    print(f"  Temperature range: {temperature[slice_mask].min():.2e} to {temperature[slice_mask].max():.2e}")
    
    # Create mass-weighted temperature projection
    h_temp, _, _ = np.histogram2d(
        coords[slice_mask, 0], coords[slice_mask, 1],
        bins=150, range=[extent[:2], extent[2:]],
        weights=temperature[slice_mask] * masses[slice_mask]
    )
    
    # Average temperature per bin (mass-weighted)
    h = np.divide(h_temp, h_mass, where=h_mass>0, out=np.zeros_like(h_temp))
    h_log = np.log10(h, where=h>0, out=np.full_like(h, 3))
    
    # Use percentile-based color scaling
    h_valid = h_log[h > 0]
    if len(h_valid) > 0:
        vmin_t = np.percentile(h_valid, 5)
        vmax_t = np.percentile(h_valid, 95)
        print(f"  Temperature projection range (5-95th percentile): {vmin_t:.2f} to {vmax_t:.2f}")
    else:
        vmin_t, vmax_t = 3, 7
        print("  Warning: No valid temperature values in projection")
    
    im2 = ax2.imshow(h_log.T, origin='lower', extent=extent, cmap='hot',
                    vmin=vmin_t, vmax=vmax_t, aspect='auto')
    circle = plt.Circle((0, 0), r_vir, fill=False, edgecolor='white', 
                       linestyle='--', linewidth=2)
    ax2.add_patch(circle)
    ax2.set_xlabel('X [ckpc/h]')
    ax2.set_ylabel('Y [ckpc/h]')
    ax2.set_title('Temperature Projection')
    plt.colorbar(im2, ax=ax2, label=r'$\log_{10}(T/{\rm K})$')
    
    # Panel 3: Radial density profile
    ax3 = fig.add_subplot(gs[0, 2])
    r_normalized = distance / r_vir
    r_bins = np.logspace(-2, np.log10(3), 15)
    r_centers = []
    density_medians = []
    for j in range(len(r_bins) - 1):
        mask = (r_normalized >= r_bins[j]) & (r_normalized < r_bins[j+1])
        if np.sum(mask) > 10:
            r_centers.append((r_bins[j] + r_bins[j+1]) / 2)
            density_medians.append(np.median(np.log10(density[mask] * 1e10)))
    ax3.plot(r_centers, density_medians, 'o-', color='steelblue', linewidth=2)
    ax3.axvline(1.0, color='red', linestyle='--', label=r'$R_{\rm vir}$')
    ax3.set_xlabel(r'$r / R_{\rm vir}$')
    ax3.set_ylabel(r'$\log_{10}(n_{\rm H}/{\rm cm}^{-3})$')
    ax3.set_xscale('log')
    ax3.grid(alpha=0.3)
    ax3.legend()
    ax3.set_title('Radial Density Profile')
    
    # Panel 4: Radial temperature profile
    ax4 = fig.add_subplot(gs[1, 0])
    temp_medians = []
    for j in range(len(r_bins) - 1):
        mask = (r_normalized >= r_bins[j]) & (r_normalized < r_bins[j+1])
        if np.sum(mask) > 10:
            temp_medians.append(np.median(np.log10(temperature[mask])))
    ax4.plot(r_centers, temp_medians, 'o-', color='orangered', linewidth=2)
    ax4.axvline(1.0, color='red', linestyle='--', label=r'$R_{\rm vir}$')
    ax4.set_xlabel(r'$r / R_{\rm vir}$')
    ax4.set_ylabel(r'$\log_{10}(T/{\rm K})$')
    ax4.set_xscale('log')
    ax4.grid(alpha=0.3)
    ax4.legend()
    ax4.set_title('Radial Temperature Profile')
    
    # Panel 5: Temperature-density phase diagram
    ax5 = fig.add_subplot(gs[1, 1])
    subsample = np.random.choice(len(density), min(50000, len(density)), replace=False)
    log_dens = np.log10(density[subsample] * 1e10)
    log_temp = np.log10(temperature[subsample])
    ax5.hexbin(log_dens, log_temp, gridsize=100, cmap='Blues', 
              mincnt=1, bins='log')
    ax5.set_xlabel(r'$\log_{10}(n_{\rm H}/{\rm cm}^{-3})$')
    ax5.set_ylabel(r'$\log_{10}(T/{\rm K})$')
    ax5.set_title('Temperature-Density Phase Diagram')
    ax5.grid(alpha=0.3)
    
    # Panel 6: Neutral hydrogen distribution
    ax6 = fig.add_subplot(gs[1, 2])
    if neutral_frac is not None:
        ax6.hist(np.log10(neutral_frac + 1e-10), bins=50, 
                color='steelblue', alpha=0.7, edgecolor='black')
        ax6.set_xlabel(r'$\log_{10}(n_{\rm HI}/n_{\rm H})$')
        ax6.set_ylabel('Count')
        ax6.set_title('Neutral Hydrogen Distribution')
        ax6.grid(alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Neutral fraction\nnot available', 
                ha='center', va='center', fontsize=12)
        ax6.set_title('Neutral Hydrogen Distribution')
    
    plt.suptitle(f"Halo {halo['halo_id']} Summary (z={redshift:.2f}, "
                f"M={halo['mass_total']:.2e} M$_\\odot$)", fontsize=16)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary plot to {output_path}")
    plt.close()
    
    return 0
