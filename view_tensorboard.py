#!/usr/bin/env python3
"""
TensorBoard Viewer for TimeGAN V6 Training Runs

Utility script to launch TensorBoard and compare training sessions.

V6 Directory Structure:
    checkpoints/v6/
        run_20251204_143022/        # Auto-named run directory
            stage1_final.pt         # Checkpoints
            stage2_final.pt
            logs/                   # TensorBoard logs (or directly in run dir)
                events.out.tfevents.*

Usage:
    python view_tensorboard.py                    # View all runs
    python view_tensorboard.py --list             # List available runs
    python view_tensorboard.py --runs exp1 exp2   # Compare specific runs
    python view_tensorboard.py --port 6007        # Use different port
    python view_tensorboard.py --latest 3         # View only 3 most recent runs

Features:
    - Auto-discovers V6 run directories
    - Lists all available training runs with their metrics
    - Launches TensorBoard with proper configuration
    - Supports filtering runs for comparison
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# V6 default directories
DEFAULT_CHECKPOINT_DIR = './checkpoints/v6'


def find_runs(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """
    Find all V6 training runs in the checkpoint directory.

    V6 runs are stored as:
        checkpoint_dir/run_YYYYMMDD_HHMMSS/
            - *.pt files (checkpoints)
            - logs/ or events.out.tfevents.* (tensorboard)

    Returns list of dicts with run info: name, path, timestamp, etc.
    """
    runs = []
    base_path = Path(checkpoint_dir)

    if not base_path.exists():
        return runs

    for run_dir in base_path.iterdir():
        if not run_dir.is_dir():
            continue

        # Skip non-run directories
        if not run_dir.name.startswith('run_'):
            continue

        run_info = {
            'name': run_dir.name,
            'path': str(run_dir),
            'log_path': None,
            'created': None,
            'checkpoints': [],
            'has_stage1': False,
            'has_stage2': False,
            'size_mb': 0,
        }

        # Parse timestamp from directory name (run_YYYYMMDD_HHMMSS)
        try:
            timestamp_str = run_dir.name.replace('run_', '')
            run_info['created'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            # Fall back to directory modification time
            try:
                run_info['created'] = datetime.fromtimestamp(run_dir.stat().st_mtime)
            except:
                pass

        # Find tensorboard logs (check logs/ subdir first, then run dir itself)
        logs_dir = run_dir / 'logs'
        if logs_dir.exists():
            run_info['log_path'] = str(logs_dir)
        else:
            # Check if event files exist directly in run dir
            event_files = list(run_dir.glob('events.out.tfevents.*'))
            if event_files:
                run_info['log_path'] = str(run_dir)

        # Find checkpoints and calculate size
        total_size = 0
        for f in run_dir.rglob('*'):
            if f.is_file():
                total_size += f.stat().st_size
                if f.suffix == '.pt':
                    run_info['checkpoints'].append(f.name)
                    if 'stage1' in f.name:
                        run_info['has_stage1'] = True
                    if 'stage2' in f.name:
                        run_info['has_stage2'] = True

        run_info['size_mb'] = total_size / (1024 * 1024)

        runs.append(run_info)

    # Sort by creation time (newest first)
    runs.sort(key=lambda x: x['created'] or datetime.min, reverse=True)

    return runs


def list_runs(checkpoint_dir: str, detailed: bool = False):
    """Print a formatted list of available runs."""
    runs = find_runs(checkpoint_dir)

    if not runs:
        print(f"\nNo runs found in {checkpoint_dir}")
        print(f"\nRun training first:")
        print(f"  1. Edit run_v6.py: Set RUN_MODE = 'train'")
        print(f"  2. Run: python run_v6.py")
        return

    print("\n" + "=" * 90)
    print("TIMEGAN V6 TRAINING RUNS")
    print("=" * 90)
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    print(f"Total runs: {len(runs)}")
    print()

    # Table header
    print(f"{'#':<3} {'Run Name':<30} {'Date':<18} {'Stage':<10} {'Logs':<6} {'Size':<10}")
    print("-" * 90)

    for i, run in enumerate(runs, 1):
        date_str = run['created'].strftime('%Y-%m-%d %H:%M') if run['created'] else 'Unknown'

        # Stage status
        if run['has_stage2']:
            stage = 'Stage 2'
        elif run['has_stage1']:
            stage = 'Stage 1'
        else:
            stage = 'Training'

        # Logs status
        logs_status = 'Yes' if run['log_path'] else 'No'

        size_str = f"{run['size_mb']:.1f} MB"

        print(f"{i:<3} {run['name']:<30} {date_str:<18} {stage:<10} {logs_status:<6} {size_str:<10}")

    print("-" * 90)
    print()

    if detailed:
        print("Run Details:")
        print("-" * 40)
        for run in runs[:5]:  # Show details for 5 most recent
            print(f"\n  {run['name']}:")
            print(f"    Path: {run['path']}")
            print(f"    Logs: {run['log_path'] or 'None'}")
            if run['checkpoints']:
                print(f"    Checkpoints: {', '.join(run['checkpoints'][:3])}")

    print("\nTo view runs in TensorBoard:")
    print(f"  python view_tensorboard.py")
    print(f"  python view_tensorboard.py --runs {runs[0]['name']}" if runs else "")
    print()


def get_tensorboard_logdir(checkpoint_dir: str, run_names: Optional[List[str]] = None,
                           latest_n: Optional[int] = None) -> str:
    """
    Build the logdir argument for TensorBoard.

    Returns a comma-separated string of name:path pairs for multiple runs,
    or a single directory path.
    """
    runs = find_runs(checkpoint_dir)

    if not runs:
        return checkpoint_dir

    # Filter runs
    if run_names:
        # Filter to specific runs (partial match)
        filtered = []
        for run_name in run_names:
            matches = [r for r in runs if run_name in r['name'] and r['log_path']]
            filtered.extend(matches)
        runs = filtered
    elif latest_n:
        runs = [r for r in runs if r['log_path']][:latest_n]
    else:
        runs = [r for r in runs if r['log_path']]

    if not runs:
        return checkpoint_dir

    # If single run, just return log path
    if len(runs) == 1:
        return runs[0]['log_path']

    # Multiple runs: build comma-separated name:path format
    log_dirs = [f"{r['name']}:{r['log_path']}" for r in runs]
    return ','.join(log_dirs)


def launch_tensorboard(
    checkpoint_dir: str,
    runs: Optional[List[str]] = None,
    latest_n: Optional[int] = None,
    port: int = 6006,
    host: str = 'localhost',
    open_browser: bool = True,
    reload_interval: int = 30
):
    """
    Launch TensorBoard server.

    Args:
        checkpoint_dir: Base directory containing runs
        runs: Optional list of specific run names to include
        latest_n: Optional, only include N most recent runs
        port: Port to serve on
        host: Host to bind to
        open_browser: Whether to open browser automatically
        reload_interval: Seconds between data reloads
    """
    logdir_arg = get_tensorboard_logdir(checkpoint_dir, runs, latest_n)

    url = f'http://{host}:{port}'

    print("\n" + "=" * 70)
    print("TENSORBOARD VIEWER - TimeGAN V6")
    print("=" * 70)
    print(f"\nLog directory: {logdir_arg}")
    print(f"URL: {url}")
    print()
    print("Key metrics to watch:")
    print("  Stage 1:")
    print("    - stage1/loss_direct: Encoder quality (should -> 0.05)")
    print("    - stage1/loss_recon:  Bottleneck quality (follows direct)")
    print("    - stage1/loss_latent: Expander matching encoder")
    print("  Stage 2:")
    print("    - stage2/wasserstein: GAN training progress (should increase)")
    print("    - stage2/d_loss:      Discriminator loss")
    print("    - stage2/g_loss:      Generator loss")
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
            original_argv = _sys.argv
            _sys.argv = ['tensorboard', '--logdir', logdir_arg, '--port', str(port),
                        '--host', host, '--reload_interval', str(reload_interval)]
            try:
                tb_main.run_main()
                launched = True
            except SystemExit:
                launched = True
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


def main():
    parser = argparse.ArgumentParser(
        description='View and compare TimeGAN V6 TensorBoard training runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Launch TensorBoard with all runs
  %(prog)s --list                   # List available runs
  %(prog)s --list --detailed        # List with more details
  %(prog)s --runs run_20251204      # View specific run (partial match)
  %(prog)s --latest 3               # View only 3 most recent runs
  %(prog)s --port 6007              # Use different port
  %(prog)s --no-browser             # Don't auto-open browser

V6 Directory Structure:
  checkpoints/v6/
      run_20251204_143022/
          stage1_final.pt
          stage2_final.pt
          logs/                     # TensorBoard events
        """
    )

    parser.add_argument(
        '--checkpoint-dir', '-d',
        default=DEFAULT_CHECKPOINT_DIR,
        help=f'Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR})'
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

    # Check if checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        print(f"\nCheckpoint directory not found: {args.checkpoint_dir}")
        print(f"\nTo create runs, train a model first:")
        print(f"  1. Edit run_v6.py: Set RUN_MODE = 'train'")
        print(f"  2. Run: python run_v6.py")
        print(f"\nOr specify a different directory:")
        print(f"  python view_tensorboard.py --checkpoint-dir /path/to/checkpoints")
        return 1

    # List mode
    if args.list:
        list_runs(args.checkpoint_dir, detailed=args.detailed)
        return 0

    # Launch TensorBoard
    launch_tensorboard(
        checkpoint_dir=args.checkpoint_dir,
        runs=args.runs,
        latest_n=args.latest,
        port=args.port,
        host=args.host,
        open_browser=not args.no_browser,
        reload_interval=args.reload_interval
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
