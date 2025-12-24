import os
import sys
import argparse
import urllib.request

BASE_URL = "https://users.flatironinstitute.org/~camels/Sims"


def download(suite, sim_set, sim_name, snapshot, dest):
    """Download CAMELS snapshot via HTTP with progress bar."""
    url = f"{BASE_URL}/{suite}/{sim_set}/{sim_name}/snapshot_{snapshot:03d}.hdf5"
    
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    print(f"Downloading: {url}")
    print(f"Saving to: {dest}")
    
    try:
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 ** 2)
                mb_total = total_size / (1024 ** 2)
                print(f"\r{percent:.1f}% | {mb_downloaded:.1f}/{mb_total:.1f} MB", end='', flush=True)
        
        urllib.request.urlretrieve(url, dest, reporthook=show_progress)
        print("\nDownload complete!")
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download CAMELS simulation data")
    parser.add_argument('--suite', default='IllustrisTNG', choices=['IllustrisTNG', 'SIMBA'])
    parser.add_argument('--set', dest='sim_set', default='LH', choices=['LH', '1P', 'CV'])
    parser.add_argument('--sim', type=int, default=0)
    parser.add_argument('--snapshot', type=int, default=14)
    parser.add_argument('--output', '-o', help='Output path')
    
    args = parser.parse_args()
    
    sim_name = f"{args.sim_set}_{args.sim}"
    dest = args.output or f"data/{args.suite}/{args.sim_set}/{sim_name}/snap_{args.snapshot:03d}.hdf5"
    
    if os.path.exists(dest):
        print(f"File exists: {dest}")
        return 0
    
    success = download(args.suite, args.sim_set, sim_name, args.snapshot, dest)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
