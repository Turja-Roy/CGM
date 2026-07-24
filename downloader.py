import os
import sys
import argparse
import urllib.request
import ssl
import subprocess

BASE_URL = "https://users.flatironinstitute.org/~camels/Sims"


def show_progress(block_num, block_size, total_size):
    """Progress bar for downloads."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 ** 2)
        mb_total = total_size / (1024 ** 2)
        print(f"\r{percent:.1f}% | {mb_downloaded:.1f}/{mb_total:.1f} MB", end='', flush=True)


def download_with_wget(url, dest):
    try:
        # Use wget with:
        # --no-check-certificate: Skip SSL verification (trusted source)
        # --progress=bar:force: Show progress bar
        # -O: Output file
        # -c: Continue partial downloads
        cmd = ['wget', '--no-check-certificate', '--progress=bar:force', 
               '-O', dest, '-c', url]
        
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"\nwget failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        # wget not available
        return False


def download(suite, sim_set, sim_name, snapshot, dest, file_type='snapshot'):
    if file_type == 'snapshot':
        url = f"{BASE_URL}/{suite}/{sim_set}/{sim_name}/snapshot_{snapshot:03d}.hdf5"
    elif file_type == 'groups':
        # Group catalogs for CAMEL simulations
        url = f"{BASE_URL}/{suite}/{sim_set}/{sim_name}/groups_{snapshot:03d}.hdf5"
    else:
        raise ValueError(f"Unknown file_type: {file_type}")
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    print(f"Downloading: {url}")
    print(f"Saving to: {dest}")
    
    # Try wget first (works best on HPC clusters with SSL issues)
    print("\nTrying wget...")
    if download_with_wget(url, dest):
        print("\nDownload complete!")
        return True
    
    # Fallback to urllib with relaxed SSL verification
    print("\nwget failed or not available, falling back to urllib...")
    try:
        # Create SSL context that doesn't verify certificates
        # This is acceptable for trusted public research data
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Install the opener globally
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_context)
        )
        urllib.request.install_opener(opener)
        
        # Download with progress bar
        urllib.request.urlretrieve(url, dest, reporthook=show_progress)
        print("\nDownload complete!")
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_groups_for_snapshot(snapshot_path):
    from pathlib import Path
    
    snapshot_path = Path(snapshot_path)
    
    if not snapshot_path.exists():
        print(f"Error: Snapshot not found: {snapshot_path}")
        return False
    
    # Parse path to extract simulation info
    # Expected: data/<suite>/<set>/<sim_name>/snap_XXX.hdf5
    parts = snapshot_path.parts
    
    if len(parts) < 4:
        print(f"Error: Cannot parse simulation info from path: {snapshot_path}")
        print("Expected format: data/<suite>/<set>/<sim_name>/snap_XXX.hdf5")
        return False
    
    suite = parts[-4]  # e.g., IllustrisTNG
    sim_set = parts[-3]  # e.g., LH
    sim_name = parts[-2]  # e.g., LH_80
    
    # Extract snapshot number
    snap_file = snapshot_path.name
    try:
        snap_num = int(snap_file.split('_')[1].split('.')[0])
    except (IndexError, ValueError):
        print(f"Error: Cannot parse snapshot number from: {snap_file}")
        return False
    
    # Destination for group catalog
    dest = snapshot_path.parent / f"groups_{snap_num:03d}.hdf5"
    
    if dest.exists():
        print(f"Group catalog already exists: {dest}")
        return True
    
    print(f"\nDownloading group catalog for {sim_name} snap {snap_num}...")
    success = download(suite, sim_set, sim_name, snap_num, str(dest), file_type='groups')
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download CAMELS simulation data",
        epilog="""
Examples:
  # Download LH simulation snapshot
  python downloader.py --suite IllustrisTNG --set LH --sim 80 --snapshot 80
  
  # Download 1P baseline simulation
  python downloader.py --suite IllustrisTNG --set 1P --sim 0 --snapshot 82
  
  # Download 1P parameter variation
  python downloader.py --suite IllustrisTNG --set 1P --sim p11_2 --snapshot 82
  python downloader.py --suite IllustrisTNG --set 1P --sim p15_n1 --snapshot 80
  
  # Download group catalog for existing snapshot
  python downloader.py --groups data/IllustrisTNG/LH/LH_80/snap_080.hdf5
        """
    )
    parser.add_argument('--suite', default='IllustrisTNG', choices=['IllustrisTNG', 'SIMBA'])
    parser.add_argument('--set', dest='sim_set', default='LH', choices=['LH', '1P', 'CV', 'EX'])
    parser.add_argument('--sim', type=str, default='0',
                       help='Simulation identifier (e.g., "80" for LH_80, "p11_2" for 1P_p11_2, "0" for baseline)')
    parser.add_argument('--snapshot', type=int, default=14)
    parser.add_argument('--output', '-o', help='Output path')
    parser.add_argument('--groups', help='Download group catalog for existing snapshot file')
    parser.add_argument('--type', choices=['snapshot', 'groups'], default='snapshot',
                       help='Type of file to download')
    
    args = parser.parse_args()
    
    # If --groups provided, use helper function
    if args.groups:
        success = download_groups_for_snapshot(args.groups)
        return 0 if success else 1
    
    # Otherwise, download specified file
    # Smart construction based on sim_set to handle different naming conventions
    # For LH/CV: "80" -> "LH_80" or "CV_80"
    # For 1P: "0" -> "1P_0", "p11_2" -> "1P_p11_2"
    if args.sim_set == '1P':
        sim_name = f"1P_{args.sim}"
    else:
        sim_name = f"{args.sim_set}_{args.sim}"
    
    if args.type == 'snapshot':
        dest = args.output or f"data/{args.suite}/{args.sim_set}/{sim_name}/snap_{args.snapshot:03d}.hdf5"
    else:  # groups
        dest = args.output or f"data/{args.suite}/{args.sim_set}/{sim_name}/groups_{args.snapshot:03d}.hdf5"
    
    if os.path.exists(dest):
        print(f"File exists: {dest}")
        return 0
    
    success = download(args.suite, args.sim_set, sim_name, args.snapshot, dest, file_type=args.type)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
