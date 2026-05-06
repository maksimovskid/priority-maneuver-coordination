"""Runnable cascading coordination scenario.

This scenario demonstrates a chain reaction: ID 1 requests a lane change from
ID 3, and ID 3 may become a secondary requester so another vehicle can create
space for the cascading maneuver.
"""

import matplotlib.pyplot as plt
from maneuver_coordination.motion_planning import reeds_shepp_path_planning as rs

from maneuver_coordination.scenarios.output import save_scenario_animation, save_summary_plots
from maneuver_coordination.simulation.runner import (
    build_default_cascading_vehicle_configs,
    run_cascading_scenario,
)
from maneuver_coordination.vehicle.plotting import plot_road_boundaries


def build_obstacles():
    """Build the three-lane road boundary points and planner obstacle list."""
    ox, oy = [], []
    for i in range(0, 80):
        ox.append(i)
        oy.append(0.0)
    for i in range(0, 80):
        ox.append(i)
        oy.append(12.0)
    for i in range(80):
        ox.append(i)
        oy.append(8.0)
    for i in range(0, 80):
        ox.append(i)
        oy.append(4.0)

    ox2, oy2 = [], []
    for i in range(0, 80):
        ox2.append(i)
        oy2.append(0.0)
    for i in range(80):
        ox2.append(i)
        oy2.append(12.0)

    obstacle_list = [(i, 0) for i in range(0, 71)] + [(i, 12) for i in range(0, 85)]
    return ox, oy, ox2, oy2, obstacle_list


def main(*, show_animation: bool = True, save_output_dir: str | None = None, save_animation: bool = False):
    """Set up and run the cascading lane-change simulation."""
    print("start simulation: cascading_coordination")

    ox, oy, ox2, oy2, obstacle_list = build_obstacles()
    vehicle_configs = build_default_cascading_vehicle_configs()

    if show_animation:
        plt.plot(ox, oy, ".k")
        for vehicle in vehicle_configs:
            rs.plot_arrow(vehicle.start[0], vehicle.start[1], vehicle.start[2], fc=vehicle.color)
            rs.plot_arrow(vehicle.goal[0], vehicle.goal[1], vehicle.goal[2])

        plt.grid(True)
        plt.axis("equal")

    result = run_cascading_scenario(
        vehicle_configs=vehicle_configs,
        ox=ox,
        oy=oy,
        obstacle_list=obstacle_list,
        show_animation=show_animation,
    )

    print("end simulation: cascading_coordination")

    if save_output_dir:
        save_summary_plots(
            "cascading_coordination",
            vehicle_configs,
            result,
            road_end_x=80.0,
            lane_lines=(0.0, 4.0, 8.0, 12.0),
            output_dir=save_output_dir,
        )
        if save_animation:
            save_scenario_animation(
                "cascading_coordination",
                vehicle_configs,
                result,
                road_end_x=80.0,
                lane_lines=(0.0, 4.0, 8.0, 12.0),
                output_dir=save_output_dir,
            )
    elif result["show_animation"]:
        plt.close()
        plt.subplots(1)
        plot_road_boundaries(main_color="k", middle_color="y", road_end_x=80.0, lane_lines=(0.0, 4.0, 8.0, 12.0))
        for config, history in zip(vehicle_configs, result["histories"]):
            vehicle_color = config.color if config.color.startswith("-") else config.color
            path = result["paths_by_role"][config.role]
            plt.plot(path.x, path.y, linestyle="--", color=vehicle_color, label=f"{config.name} path")
            plt.plot(history.x, history.y, linestyle="-", color=vehicle_color, label=f"{config.name} tracking")
        plt.grid(False)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots(1)
        for config, history in zip(vehicle_configs, result["histories"]):
            vehicle_color = config.color if config.color.startswith("-") else config.color
            plt.plot(history.t, [speed * 3.6 for speed in history.v], color=vehicle_color, label=config.name)
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(False)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
