"""Command-line entry points for the project."""

import argparse

from maneuver_coordination.scenarios.cascading_coordination import main as run_cascading_lane_change
from maneuver_coordination.scenarios.coordination_2_vehicles import main as run_two_vehicle_priority
from maneuver_coordination.scenarios.rejected_request_then_free_lane import (
    main as run_rejected_request_then_free_lane,
)
from maneuver_coordination.scenarios.three_vehicles_coordination import main as run_three_vehicle_coordination
from maneuver_coordination.scenarios.three_vehicles_coordination_4_messages import (
    main as run_three_vehicle_coordination_4_messages,
)

TWO_VEHICLES_COORDINATION = "two_vehicles_coordination"
CASCADING_COORDINATION = "cascading_coordination"
REJECTED_REQUEST_THEN_FREE_LANE = "rejected_request_then_free_lane"
THREE_VEHICLES_COORDINATION = "three_vehicles_coordination"
THREE_VEHICLES_COORDINATION_4_MESSAGES = "three_vehicles_coordination_4_messages"
LEGACY_PRIORITY = "priority"
LEGACY_CASCADING = "cascading"
LEGACY_COORDINATION_2_VEHICLES = "coordination_2_vehicles"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser with current and legacy scenario aliases."""
    parser = argparse.ArgumentParser(description="Run maneuver coordination scenarios.")
    parser.add_argument(
        "scenario",
        nargs="?",
        default=CASCADING_COORDINATION,
        choices=[
            TWO_VEHICLES_COORDINATION,
            CASCADING_COORDINATION,
            REJECTED_REQUEST_THEN_FREE_LANE,
            THREE_VEHICLES_COORDINATION,
            THREE_VEHICLES_COORDINATION_4_MESSAGES,
            LEGACY_PRIORITY,
            LEGACY_CASCADING,
            LEGACY_COORDINATION_2_VEHICLES,
        ],
        help="Scenario to run.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Run without the live matplotlib animation window.",
    )
    parser.add_argument(
        "--save-output-dir",
        default=None,
        help="Directory where summary PNG outputs should be saved.",
    )
    parser.add_argument(
        "--save-animation",
        action="store_true",
        help="When used with --save-output-dir, also save a GIF animation of the scenario.",
    )
    return parser


def main() -> None:
    """Run a maneuver coordination scenario."""
    args = build_parser().parse_args()
    show_animation = not args.no_animation
    save_output_dir = args.save_output_dir
    save_animation = args.save_animation

    if args.scenario in (TWO_VEHICLES_COORDINATION, LEGACY_PRIORITY, LEGACY_COORDINATION_2_VEHICLES):
        run_two_vehicle_priority(
            show_animation=show_animation,
            save_output_dir=save_output_dir,
            save_animation=save_animation,
        )
        return
    if args.scenario == REJECTED_REQUEST_THEN_FREE_LANE:
        run_rejected_request_then_free_lane(
            show_animation=show_animation,
            save_output_dir=save_output_dir,
            save_animation=save_animation,
        )
        return
    if args.scenario == THREE_VEHICLES_COORDINATION:
        run_three_vehicle_coordination(
            show_animation=show_animation,
            save_output_dir=save_output_dir,
            save_animation=save_animation,
        )
        return
    if args.scenario == THREE_VEHICLES_COORDINATION_4_MESSAGES:
        run_three_vehicle_coordination_4_messages(
            show_animation=show_animation,
            save_output_dir=save_output_dir,
            save_animation=save_animation,
        )
        return

    run_cascading_lane_change(
        show_animation=show_animation,
        save_output_dir=save_output_dir,
        save_animation=save_animation,
    )
