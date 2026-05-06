"""Output helpers shared by scenario entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from maneuver_coordination.simulation.runner import render_simulation_frame
from maneuver_coordination.simulation.core.types import VehicleConfig
from maneuver_coordination.vehicle.plotting import plot_road_boundaries


def save_summary_plots(
    scenario_name: str,
    vehicle_configs: Sequence[VehicleConfig],
    result,
    *,
    road_end_x: float,
    lane_lines: tuple[float, ...],
    output_dir: str,
) -> None:
    """Save final path-tracking and speed plots for a completed scenario."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.close("all")

    plt.figure()
    plot_road_boundaries(main_color="k", middle_color="y", road_end_x=road_end_x, lane_lines=lane_lines)
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
    plt.tight_layout()
    plt.savefig(output_path / f"{scenario_name}_trajectories.png", dpi=150)

    plt.figure()
    for config, history in zip(vehicle_configs, result["histories"]):
        vehicle_color = config.color if config.color.startswith("-") else config.color
        plt.plot(history.t, [speed * 3.6 for speed in history.v], color=vehicle_color, label=config.name)
    plt.xlabel("Time[s]")
    plt.ylabel("Speed[km/h]")
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{scenario_name}_speeds.png", dpi=150)

    plt.close("all")


def save_scenario_animation(
    scenario_name: str,
    vehicle_configs: Sequence[VehicleConfig],
    result,
    *,
    road_end_x: float,
    lane_lines: tuple[float, ...],
    output_dir: str,
    frame_stride: int = 2,
    fps: int = 10,
) -> None:
    """Save a GIF using the same frame renderer as the live simulation view."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame_log = result.get("frame_log", [])
    if not frame_log:
        return

    frame_indices = list(range(0, len(frame_log), max(1, frame_stride)))
    if frame_indices[-1] != len(frame_log) - 1:
        frame_indices.append(len(frame_log) - 1)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(11, 4.5))

    def update(frame_position: int):
        frame_data = frame_log[frame_indices[frame_position]]
        render_simulation_frame(
            ax,
            vehicle_configs,
            frame_data,
            road_end_x=road_end_x,
            lane_lines=lane_lines,
            main_color="k",
            middle_color="k",
        )
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=max(1, int(1000 / max(1, fps))),
        blit=False,
    )
    animation.save(output_path / f"{scenario_name}_animation.gif", writer=PillowWriter(fps=fps))
    plt.close(fig)
