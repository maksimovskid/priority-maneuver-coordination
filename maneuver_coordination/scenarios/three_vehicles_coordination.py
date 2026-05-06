"""Runnable three-vehicle coordination scenario.

The requester negotiates with two target-lane vehicles. One vehicle can
decelerate and the other can accelerate so the pair creates a safe gap.
"""

import matplotlib.pyplot as plt
from maneuver_coordination.motion_planning import reeds_shepp_path_planning as rs

from maneuver_coordination.scenarios.output import save_scenario_animation, save_summary_plots
from maneuver_coordination.simulation.runner import (
    build_default_three_vehicle_coordination_configs,
    run_three_vehicle_coordination_scenario,
)
from maneuver_coordination.vehicle.plotting import plot_road_boundaries


def build_obstacles():
    """Build the extended two-lane road and obstacle boundaries."""
    road_end_x = 100
    ox, oy = [], []
    for i in range(0, road_end_x + 1):
        ox.append(i)
        oy.append(0.0)
    for i in range(0, road_end_x + 1):
        ox.append(i)
        oy.append(8.0)
    for i in range(0, road_end_x + 1):
        ox.append(i)
        oy.append(4.0)

    ox2, oy2 = [], []
    for i in range(0, road_end_x + 1):
        ox2.append(i)
        oy2.append(0.0)
    for i in range(0, road_end_x + 1):
        ox2.append(i)
        oy2.append(8.0)

    obstacle_list = [(i, 0) for i in range(0, road_end_x + 1)] + [(i, 8) for i in range(0, road_end_x + 1)]
    return ox, oy, ox2, oy2, obstacle_list


def main(*, show_animation: bool = True, save_output_dir: str | None = None, save_animation: bool = False):
    """Set up and run the three-vehicle coordination simulation."""
    print("start simulation: three_vehicles_coordination")

    ox, oy, ox2, oy2, obstacle_list = build_obstacles()
    vehicle_configs = build_default_three_vehicle_coordination_configs()

    if show_animation:
        plt.plot(ox, oy, ".k")
        for vehicle in vehicle_configs:
            rs.plot_arrow(vehicle.start[0], vehicle.start[1], vehicle.start[2], fc=vehicle.color)
            rs.plot_arrow(vehicle.goal[0], vehicle.goal[1], vehicle.goal[2])

        plt.grid(True)
        plt.axis("equal")

    result = run_three_vehicle_coordination_scenario(
        vehicle_configs=vehicle_configs,
        ox=ox,
        oy=oy,
        obstacle_list=obstacle_list,
        show_animation=show_animation,
    )

    print("end simulation: three_vehicles_coordination")

    if save_output_dir:
        save_summary_plots(
            "three_vehicles_coordination",
            vehicle_configs,
            result,
            road_end_x=100.0,
            lane_lines=(0.0, 4.0, 8.0),
            output_dir=save_output_dir,
        )
        if save_animation:
            save_scenario_animation(
                "three_vehicles_coordination",
                vehicle_configs,
                result,
                road_end_x=100.0,
                lane_lines=(0.0, 4.0, 8.0),
                output_dir=save_output_dir,
            )
    elif result["show_animation"]:
        plt.close()
        plt.subplots(1)
        plot_road_boundaries(main_color="k", middle_color="y", road_end_x=100.0, lane_lines=(0.0, 4.0, 8.0))
        for config, history in zip(vehicle_configs, result["histories"]):
            vehicle_color = config.color if config.color.startswith("-") else config.color
            path = result["paths_by_role"][config.role]
            plt.plot(path.x, path.y, linestyle="--", color=vehicle_color, label=f"{config.vehicle_id} path")
            plt.plot(history.x, history.y, linestyle="-", color=vehicle_color, label=f"ID {config.vehicle_id}")
        plt.grid(False)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots(1)
        for config, history in zip(vehicle_configs, result["histories"]):
            vehicle_color = config.color if config.color.startswith("-") else config.color
            plt.plot(history.t, [speed * 3.6 for speed in history.v], color=vehicle_color, label=f"ID {config.vehicle_id}")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(False)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
