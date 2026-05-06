"""Compatibility entry point for the refactored cascading lane-change scenario.

Prefer running `python -m maneuver_coordination` for the package-native entry
point. This file is kept as a short wrapper for local workflows.
"""

from maneuver_coordination.cli import main


if __name__ == "__main__":
    main()
