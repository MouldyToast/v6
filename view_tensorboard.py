#!/usr/bin/env python3
"""
TensorBoard Viewer for V4 Training Runs

Utility script to launch TensorBoard and compare training sessions.

Usage:
    python view_tensorboard.py                    # View all runs
    python view_tensorboard.py --list             # List available runs
    python view_tensorboard.py --runs exp1 exp2   # Compare specific runs
    python view_tensorboard.py --port 6007        # Use different port
    python view_tensorboard.py --latest 3         # View only 3 most recent runs

Features:
    - Lists all available training runs with their metrics
    - Launches TensorBoard with proper configuration
    - Supports filtering runs for comparison
    - Shows run metadata and hyperparameters
"""

import os
import sys
import argparse
import subprocess
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Default directories
DEFAULT_TENSORBOARD_DIR = 'outputs_v4/tensorboard'
DEFAULT_OUTPUT_DIR = 'outputs_v4'


def find_runs(tensorboard_dir: str) -> List[Dict[str, Any]]:
    """
    Find all training runs in the tensorboard directory.

    Returns list of dicts with run info: name, path, timestamp, etc.
    """
    runs = []
    tb_path = Path(tensorboard_dir)

    if not tb_path.exists():
        return runs

    for run_dir in tb_path.iterdir():
        if run_dir.is_dir():
            run_info = {
                'name': run_dir.name,
                'path': str(run_dir),
                'created': None,
                'events_file': None,
                'size_mb': 0,
            }

            # Get creation time from directory
            try:
                run_info['created'] = datetime.fromtimestamp(run_dir.stat().st_mtime)
            except:
                pass

            # Find event files and calculate size
            total_size = 0
            for f in run_dir.rglob('*'):
                if f.is_file():
                    total_size += f.stat().st_size
                    if 'events.out.tfevents' in f.name:
                        run_info['events_file'] = str(f)

            run_info['size_mb'] = total_size / (1024 * 1024)

            # Try to extract experiment name from directory name
            parts = run_info['name'].split('_')
            if len(parts) >= 2:
                # Format: experimentname_YYYYMMDD_HHMMSS_comment
                run_info['experiment'] = parts[0]

            runs.append(run_info)

    # Sort by creation time (newest first)
    runs.sort(key=lambda x: x['created'] or datetime.min, reverse=True)

    return runs


def list_runs(tensorboard_dir: str, detailed: bool = False):
    """Print a formatted list of available runs."""
    runs = find_runs(tensorboard_dir)

    if not runs:
        print(f"\nNo runs found in {tensorboard_dir}")
        print(f"Run training first: python train_v4.py --experiment_name my_experiment")
        return

    print("\n" + "=" * 80)
    print("AVAILABLE TRAINING RUNS")
    print("=" * 80)
    print(f"\nTensorBoard directory: {tensorboard_dir}")
    print(f"Total runs: {len(runs)}")
    print()

    # Table header
    print(f"{'#':<3} {'Run Name':<45} {'Date':<20} {'Size':<10}")
    print("-" * 80)

    for i, run in enumerate(runs, 1):
        date_str = run['created'].strftime('%Y-%m-%d %H:%M') if run['created'] else 'Unknown'
        size_str = f"{run['size_mb']:.1f} MB"

        # Truncate long names
        name = run['name']
        if len(name) > 44:
            name = name[:41] + '...'

        print(f"{i:<3} {name:<45} {date_str:<20} {size_str:<10}")

    print("-" * 80)
    print()

    if detailed:
        print("Run Details:")
        print("-" * 40)
        for run in runs[:5]:  # Show details for 5 most recent
            print(f"\n  {run['name']}:")
            print(f"    Path: {run['path']}")
            if run.get('experiment'):
                print(f"    Experiment: {run['experiment']}")

    print("\nTo view runs in TensorBoard:")
    print(f"  python view_tensorboard.py")
    print(f"  python view_tensorboard.py --runs {runs[0]['name']}")
    print()


def launch_tensorboard(
    tensorboard_dir: str,
    runs: Optional[List[str]] = None,
    port: int = 6006,
    host: str = 'localhost',
    open_browser: bool = True,
    reload_interval: int = 30
):
    """
    Launch TensorBoard server.

    Args:
        tensorboard_dir: Base directory containing runs
        runs: Optional list of specific run names to include
        port: Port to serve on
        host: Host to bind to
        open_browser: Whether to open browser automatically
        reload_interval: Seconds between data reloads
    """
    # Determine log directory
    if runs:
        # Filter to specific runs
        available = find_runs(tensorboard_dir)
        available_names = {r['name'] for r in available}

        # Build comma-separated logdir spec for multiple runs
        log_dirs = []
        for run_name in runs:
            # Check if it's a full name or partial match
            matches = [r for r in available if run_name in r['name']]
            if matches:
                for m in matches:
                    log_dirs.append(f"{m['name']}:{m['path']}")
            else:
                print(f"Warning: Run '{run_name}' not found")

        if not log_dirs:
            print("Error: No matching runs found")
            list_runs(tensorboard_dir)
            return

        logdir_arg = ','.join(log_dirs)
    else:
        logdir_arg = tensorboard_dir

    url = f'http://{host}:{port}'

    print("\n" + "=" * 70)
    print("TENSORBOARD VIEWER")
    print("=" * 70)
    print(f"\nLog directory: {logdir_arg}")
    print(f"URL: {url}")
    print()
    print("Useful tabs for comparing runs:")
    print("  - SCALARS:  Compare training curves (loss, metrics)")
    print("  - HPARAMS:  Compare hyperparameters and final metrics")
    print("  - IMAGES:   View trajectory comparisons (real vs generated)")
    print("  - HISTOGRAMS: Monitor weight/gradient distributions")
    print("  - TEXT:     View config and run metadata")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()

    # Open browser
    if open_browser:
        try:
            webbrowser.open(url, new=2)
        except:
            pass

    # Try multiple methods to launch TensorBoard
    launched = False

    # Method 1: Try 'tensorboard' command directly
    try:
        cmd = ['tensorboard', '--logdir', logdir_arg, '--port', str(port),
               '--host', host, '--reload_interval', str(reload_interval)]
        subprocess.run(cmd)
        launched = True
    except FileNotFoundError:
        pass
    except KeyboardInterrupt:
        print("\nTensorBoard server stopped")
        return

    # Method 2: Try Python tensorboard.main module
    if not launched:
        try:
            from tensorboard import main as tb_main
            import sys as _sys
            # Set up argv for tensorboard
            original_argv = _sys.argv
            _sys.argv = ['tensorboard', '--logdir', logdir_arg, '--port', str(port),
                        '--host', host, '--reload_interval', str(reload_interval)]
            try:
                tb_main.run_main()
                launched = True
            except SystemExit:
                launched = True  # TensorBoard exits normally this way
            finally:
                _sys.argv = original_argv
        except ImportError:
            pass
        except KeyboardInterrupt:
            print("\nTensorBoard server stopped")
            return

    # Method 3: Try tensorboard.program API
    if not launched:
        try:
            from tensorboard import program
            tb = program.TensorBoard()
            tb.configure(argv=[None, '--logdir', logdir_arg, '--port', str(port),
                              '--host', host, '--reload_interval', str(reload_interval)])
            url = tb.launch()
            print(f"TensorBoard started at {url}")
            # Keep running
            import time
            while True:
                time.sleep(1)
        except ImportError:
            pass
        except KeyboardInterrupt:
            print("\nTensorBoard server stopped")
            return

    if not launched:
        print("\nError: Could not launch TensorBoard.")
        print("\nTry running manually:")
        print(f"  tensorboard --logdir {logdir_arg}")
        print("\nOr install tensorboard:")
        print("  pip install tensorboard")


def compare_runs_summary(tensorboard_dir: str, run_names: List[str]):
    """
    Print a comparison summary of specified runs.

    This reads the event files and extracts final metrics for comparison.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("Note: Install tensorboard for detailed comparison: pip install tensorboard")
        return

    runs = find_runs(tensorboard_dir)

    print("\n" + "=" * 70)
    print("RUN COMPARISON SUMMARY")
    print("=" * 70)

    for run_name in run_names:
        matches = [r for r in runs if run_name in r['name']]

        for run in matches:
            print(f"\n{run['name']}:")
            print("-" * 50)

            if not run['events_file']:
                print("  No event data found")
                continue

            try:
                ea = event_accumulator.EventAccumulator(run['path'])
                ea.Reload()

                # Get available scalars
                scalar_tags = ea.Tags().get('scalars', [])

                # Key metrics to show
                key_metrics = [
                    'Phase3/v4_score', 'Global/v4_score',
                    'Phase3/val_recon', 'Phase3/val_var', 'Phase3/val_cond',
                    'Final/best_v4_score', 'Final/training_time_min'
                ]

                for tag in key_metrics:
                    if tag in scalar_tags:
                        events = ea.Scalars(tag)
                        if events:
                            last_value = events[-1].value
                            print(f"  {tag}: {last_value:.6f}")

            except Exception as e:
                print(f"  Error reading events: {e}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='View and compare TensorBoard training runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Launch TensorBoard with all runs
  %(prog)s --list                   # List available runs
  %(prog)s --list --detailed        # List with more details
  %(prog)s --runs baseline test     # Compare specific runs
  %(prog)s --latest 3               # View only 3 most recent runs
  %(prog)s --port 6007              # Use different port
  %(prog)s --no-browser             # Don't auto-open browser
        """
    )

    parser.add_argument(
        '--tensorboard_dir', '-d',
        default=DEFAULT_TENSORBOARD_DIR,
        help=f'TensorBoard logs directory (default: {DEFAULT_TENSORBOARD_DIR})'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available runs without launching TensorBoard'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed information when listing'
    )

    parser.add_argument(
        '--runs', '-r',
        nargs='+',
        help='Specific run names to include (partial match supported)'
    )

    parser.add_argument(
        '--latest', '-n',
        type=int,
        help='Only include N most recent runs'
    )

    parser.add_argument(
        '--compare', '-c',
        nargs='+',
        help='Print comparison summary for specified runs'
    )

    parser.add_argument(
        '--port', '-p',
        type=int,
        default=6006,
        help='Port for TensorBoard server (default: 6006)'
    )

    parser.add_argument(
        '--host',
        default='localhost',
        help='Host to bind TensorBoard to (default: localhost)'
    )

    parser.add_argument(
        '--no-browser',
        action='store_true',
        help="Don't automatically open browser"
    )

    parser.add_argument(
        '--reload-interval',
        type=int,
        default=30,
        help='Seconds between data reloads (default: 30)'
    )

    args = parser.parse_args()

    # Check if tensorboard directory exists
    if not os.path.exists(args.tensorboard_dir):
        print(f"\nTensorBoard directory not found: {args.tensorboard_dir}")
        print(f"\nTo create runs, train a model first:")
        print(f"  python train_v4.py --experiment_name my_experiment")
        print(f"\nOr specify a different directory:")
        print(f"  python view_tensorboard.py --tensorboard_dir /path/to/logs")
        return 1

    # List mode
    if args.list:
        list_runs(args.tensorboard_dir, detailed=args.detailed)
        return 0

    # Compare mode
    if args.compare:
        compare_runs_summary(args.tensorboard_dir, args.compare)
        return 0

    # Determine which runs to include
    runs_to_show = args.runs

    if args.latest:
        all_runs = find_runs(args.tensorboard_dir)
        runs_to_show = [r['name'] for r in all_runs[:args.latest]]
        if not runs_to_show:
            print("No runs found")
            return 1

    # Launch TensorBoard
    launch_tensorboard(
        tensorboard_dir=args.tensorboard_dir,
        runs=runs_to_show,
        port=args.port,
        host=args.host,
        open_browser=not args.no_browser,
        reload_interval=args.reload_interval
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
