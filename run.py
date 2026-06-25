#!/usr/bin/env python3
"""
NRL Sentinel - Entry Point (Python)

Usage:
    python run.py              # Launch Streamlit dashboard
    python run.py --data-only  # Generate/update data only
    python run.py --help       # Show help
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Resolve the directory where this script lives (nrl_sentinel/)
SCRIPT_DIR = Path(__file__).resolve().parent

# Always run from the nrl_sentinel/ directory so relative imports work
os.chdir(SCRIPT_DIR)

# Add nrl_sentinel/ to path for direct module imports
sys.path.insert(0, str(SCRIPT_DIR))


def print_banner():
    """Print project banner"""
    print("""
    +=============================================================+
    |                                                             |
    |  NRL SENTINEL v2.0 -- AI Decision Intelligence System      |
    |  Numaligarh Refinery Limited | NREP Risk Analytics         |
    |  5 Modules: GRM | Demand | Maintenance | Cost | Pipeline   |
    |                                                             |
    +=============================================================+
    """)


def generate_data_only():
    """Generate/update datasets without launching dashboard"""
    print("Generating datasets...\n")
    from data_collector import (
        fetch_crude_and_product_prices,
        generate_ne_india_demand,
        generate_sensor_data,
        generate_project_data,
        generate_pipeline_data,
    )

    print("=" * 60)
    print("  DATA COLLECTION PIPELINE")
    print("=" * 60 + "\n")

    try:
        fetch_crude_and_product_prices()
        generate_ne_india_demand()
        generate_sensor_data()
        generate_project_data()
        generate_pipeline_data()
        print("\nAll 5 datasets generated successfully!")
        print(f"   Location: {SCRIPT_DIR / 'data'}/")
    except Exception as e:
        print(f"\nError during data generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def launch_dashboard():
    """Launch Streamlit dashboard"""
    app_path = SCRIPT_DIR / "app.py"

    print("Launching NRL Sentinel Dashboard...\n")
    print("   Opening browser at: http://localhost:8501")
    print("   Press Ctrl+C to stop\n")

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--logger.level=info",
            "--client.showErrorDetails=true"
        ], check=True, cwd=str(SCRIPT_DIR), env=env)
    except FileNotFoundError:
        print("Error: Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="NRL Sentinel — AI Decision Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                 # Launch Streamlit dashboard
  python run.py --data-only     # Generate/update datasets
  python run.py --help          # Show this help message
        """
    )

    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Generate/update datasets without launching dashboard"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="NRL Sentinel v2.0.0"
    )

    args = parser.parse_args()

    print_banner()

    if args.data_only:
        generate_data_only()
    else:
        launch_dashboard()


if __name__ == "__main__":
    main()
