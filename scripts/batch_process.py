#!/usr/bin/env python3
"""
Batch processing module for analyzing multiple snapshots

This module provides functionality to:
- Process multiple snapshots in parallel or sequentially
- Aggregate results across redshifts
- Generate comparative plots across simulations
- Track progress and handle errors gracefully
"""

import os
import sys
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import time

import config


def find_snapshots_in_directory(directory, pattern='snap_*.hdf5'):
    """
    Find all snapshot files in a directory.
    
    Args:
        directory: Path to search
        pattern: Glob pattern for snapshot files
        
    Returns:
        list: Paths to snapshot files
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    snapshots = sorted(directory.glob(pattern))
    return [str(s) for s in snapshots]


def find_spectra_files_in_directory(directory, pattern='camel_*_spectra_*.hdf5'):
    """
    Find all spectra files in a directory tree.
    
    Args:
        directory: Path to search
        pattern: Glob pattern for spectra files
        
    Returns:
        list: Paths to spectra files
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    spectra = sorted(directory.rglob(pattern))
    return [str(s) for s in spectra]


def run_generate_command(snapshot_path, num_sightlines=100, lines='lya', output_dir=None):
    """
    Run generate command for a single snapshot.
    
    Args:
        snapshot_path: Path to snapshot file
        num_sightlines: Number of sightlines to generate
        lines: Spectral lines to compute
        output_dir: Optional output directory
        
    Returns:
        dict: Result dictionary with status, output path, time elapsed
    """
    start_time = time.time()
    
    cmd = [
        'python3', 'analyze_spectra.py', 'generate',
        snapshot_path,
        '-n', str(num_sightlines),
        '--line', lines
    ]
    
    if output_dir:
        # Construct output filename
        snapshot_name = Path(snapshot_path).stem
        output_file = Path(output_dir) / f"camel_{lines}_spectra_{snapshot_name}.hdf5"
        cmd.extend(['-o', str(output_file)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.time() - start_time
        
        return {
            'snapshot': snapshot_path,
            'status': 'success' if result.returncode == 0 else 'failed',
            'returncode': result.returncode,
            'elapsed_time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            'snapshot': snapshot_path,
            'status': 'timeout',
            'elapsed_time': elapsed,
            'error': 'Command timed out after 3600s'
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'snapshot': snapshot_path,
            'status': 'error',
            'elapsed_time': elapsed,
            'error': str(e)
        }


def run_analyze_command(spectra_path, line=None):
    """
    Run comprehensive analysis for a single spectra file.
    
    Args:
        spectra_path: Path to spectra file
        line: Optional line to analyze
        
    Returns:
        dict: Result dictionary with status, time elapsed
    """
    start_time = time.time()
    
    cmd = ['python3', 'analyze_spectra.py', 'analyze', spectra_path]
    
    if line:
        cmd.extend(['--line', line])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        elapsed = time.time() - start_time
        
        return {
            'spectra_file': spectra_path,
            'status': 'success' if result.returncode == 0 else 'failed',
            'returncode': result.returncode,
            'elapsed_time': elapsed,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            'spectra_file': spectra_path,
            'status': 'timeout',
            'elapsed_time': elapsed,
            'error': 'Command timed out after 7200s'
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'spectra_file': spectra_path,
            'status': 'error',
            'elapsed_time': elapsed,
            'error': str(e)
        }


def batch_generate(snapshot_dir, num_sightlines=10000, lines='lya', max_workers=1, output_dir=None):
    """
    Generate spectra for all snapshots in a directory.
    
    Args:
        snapshot_dir: Directory containing snapshots
        num_sightlines: Number of sightlines per snapshot
        lines: Spectral lines to compute
        max_workers: Number of parallel workers (1 = sequential)
        output_dir: Optional output directory
        
    Returns:
        list: Results for each snapshot
    """
    snapshots = find_snapshots_in_directory(snapshot_dir)
    
    if not snapshots:
        print(f"No snapshots found in {snapshot_dir}")
        return []
    
    print(f"Found {len(snapshots)} snapshots")
    print(f"Sightlines per snapshot: {num_sightlines}")
    print(f"Lines to compute: {lines}")
    print(f"Parallel workers: {max_workers}")
    print()
    
    results = []
    
    if max_workers == 1:
        # Sequential processing
        for i, snapshot in enumerate(snapshots, 1):
            print(f"[{i}/{len(snapshots)}] Processing {Path(snapshot).name}...")
            result = run_generate_command(snapshot, num_sightlines, lines, output_dir)
            results.append(result)
            
            if result['status'] == 'success':
                print(f"  OK Success ({result['elapsed_time']:.1f}s)")
            else:
                print(f"  Failed {result['status'].capitalize()}: {result.get('error', 'See stderr')}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for snapshot in snapshots:
                future = executor.submit(run_generate_command, snapshot, num_sightlines, lines, output_dir)
                futures[future] = snapshot
            
            for i, future in enumerate(as_completed(futures), 1):
                snapshot = futures[future]
                print(f"[{i}/{len(snapshots)}] Completed {Path(snapshot).name}")
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"  ✓ Success ({result['elapsed_time']:.1f}s)")
                else:
                    print(f"  ✗ {result['status'].capitalize()}")
    
    return results


def batch_analyze(spectra_dir, line=None, max_workers=1):
    """
    Analyze all spectra files in a directory.
    
    Args:
        spectra_dir: Directory containing spectra files
        line: Optional line to analyze
        max_workers: Number of parallel workers
        
    Returns:
        list: Results for each spectra file
    """
    spectra_files = find_spectra_files_in_directory(spectra_dir)
    
    if not spectra_files:
        print(f"No spectra files found in {spectra_dir}")
        return []
    
    print(f"Found {len(spectra_files)} spectra files")
    print(f"Parallel workers: {max_workers}")
    print()
    
    results = []
    
    if max_workers == 1:
        # Sequential processing
        for i, spectra_file in enumerate(spectra_files, 1):
            print(f"[{i}/{len(spectra_files)}] Analyzing {Path(spectra_file).name}...")
            result = run_analyze_command(spectra_file, line)
            results.append(result)
            
            if result['status'] == 'success':
                print(f"  OK Success ({result['elapsed_time']:.1f}s)")
            else:
                print(f"  Failed {result['status'].capitalize()}")
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for spectra_file in spectra_files:
                future = executor.submit(run_analyze_command, spectra_file, line)
                futures[future] = spectra_file
            
            for i, future in enumerate(as_completed(futures), 1):
                spectra_file = futures[future]
                print(f"[{i}/{len(spectra_files)}] Completed {Path(spectra_file).name}")
                result = future.result()
                results.append(result)
                
                if result['status'] == 'success':
                    print(f"  ✓ Success ({result['elapsed_time']:.1f}s)")
                else:
                    print(f"  ✗ {result['status'].capitalize()}")
    
    return results


def save_batch_results(results, output_file):
    """
    Save batch processing results to JSON file.
    
    Args:
        results: List of result dictionaries
        output_file: Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def print_batch_summary(results):
    """
    Print summary of batch processing results.
    
    Args:
        results: List of result dictionaries
    """
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    timeouts = sum(1 for r in results if r['status'] == 'timeout')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    total_time = sum(r['elapsed_time'] for r in results)
    avg_time = total_time / total if total > 0 else 0
    
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total jobs:       {total}")
    print(f"Successful:       {successful} ({100*successful/total:.1f}%)")
    print(f"Failed:           {failed}")
    print(f"Timeouts:         {timeouts}")
    print(f"Errors:           {errors}")
    print(f"Total time:       {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"Average per job:  {avg_time:.1f}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Batch processing for CAMEL spectra analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate spectra for all snapshots in a directory
  python batch_process.py generate data/IllustrisTNG/LH/LH_0/ -n 10000
  
  # Analyze all spectra files in a directory
  python batch_process.py analyze data/IllustrisTNG/LH/LH_0/
  
  # Generate with parallel processing
  python batch_process.py generate data/IllustrisTNG/LH/LH_0/ -n 10000 --workers 4
  
  # Full pipeline for specific LH set
  python batch_process.py pipeline data/IllustrisTNG/LH/LH_80/ -n 10000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Batch command')
    subparsers.required = True
    
    # Generate subcommand
    parser_gen = subparsers.add_parser('generate', help='Batch generate spectra')
    parser_gen.add_argument('directory', help='Directory containing snapshots')
    parser_gen.add_argument('-n', '--sightlines', type=int, default=10000,
                           help='Number of sightlines per snapshot (default: 10000)')
    parser_gen.add_argument('--line', type=str, default='lya',
                           help='Spectral lines to compute (default: lya)')
    parser_gen.add_argument('--workers', type=int, default=1,
                           help='Number of parallel workers (default: 1)')
    parser_gen.add_argument('-o', '--output', type=str, default=None,
                           help='Output directory for spectra files')
    
    # Analyze subcommand
    parser_analyze = subparsers.add_parser('analyze', help='Batch analyze spectra')
    parser_analyze.add_argument('directory', help='Directory containing spectra files')
    parser_analyze.add_argument('--line', type=str, default=None,
                               help='Spectral line to analyze')
    parser_analyze.add_argument('--workers', type=int, default=1,
                               help='Number of parallel workers (default: 1)')
    
    # Pipeline subcommand
    parser_pipeline = subparsers.add_parser('pipeline', help='Full generate + analyze pipeline')
    parser_pipeline.add_argument('directory', help='Directory containing snapshots')
    parser_pipeline.add_argument('-n', '--sightlines', type=int, default=10000,
                                help='Number of sightlines per snapshot')
    parser_pipeline.add_argument('--line', type=str, default='lya',
                                help='Spectral lines to compute')
    parser_pipeline.add_argument('--workers', type=int, default=1,
                                help='Number of parallel workers')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        results = batch_generate(
            args.directory,
            num_sightlines=args.sightlines,
            lines=args.line,
            max_workers=args.workers,
            output_dir=args.output
        )
        print_batch_summary(results)
        save_batch_results(results, 'batch_generate_results.json')
    
    elif args.command == 'analyze':
        results = batch_analyze(
            args.directory,
            line=args.line,
            max_workers=args.workers
        )
        print_batch_summary(results)
        save_batch_results(results, 'batch_analyze_results.json')
    
    elif args.command == 'pipeline':
        print("STAGE 1/2: GENERATING SPECTRA")
        print("=" * 70)
        gen_results = batch_generate(
            args.directory,
            num_sightlines=args.sightlines,
            lines=args.line,
            max_workers=args.workers
        )
        print_batch_summary(gen_results)
        
        print("\n\nSTAGE 2/2: ANALYZING SPECTRA")
        print("=" * 70)
        analyze_results = batch_analyze(
            args.directory,
            line=args.line,
            max_workers=args.workers
        )
        print_batch_summary(analyze_results)
        
        # Save combined results
        all_results = {
            'generate': gen_results,
            'analyze': analyze_results
        }
        save_batch_results(all_results, 'batch_pipeline_results.json')
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
