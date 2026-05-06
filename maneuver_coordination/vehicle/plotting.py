"""Scenario-level plotting helpers for roads and vehicles."""

from typing import Sequence

import matplotlib.pyplot as plt

from maneuver_coordination.simulation.core.types import State
from maneuver_coordination.vehicle.plotters import (
    plot_car1,
    plot_car2,
    plot_car3,
    plot_car4,
    plot_car5,
)


PLOTTERS = [plot_car1, plot_car2, plot_car3, plot_car4, plot_car5]


def plot_road_boundaries(
    main_color: str = "k",
    middle_color: str = "k",
    road_end_x: float = 80.0,
    lane_lines: Sequence[float] = (0.0, 4.0, 8.0, 12.0),
) -> None:
    """Draw lane boundaries for two- and three-lane scenarios."""
    if not lane_lines:
        return

    plt.plot([0, road_end_x], [lane_lines[0], lane_lines[0]], linestyle="-", color=main_color, lw=2.5)
    for lane_y in lane_lines[1:-1]:
        plt.plot([0, road_end_x], [lane_y, lane_y], linestyle="--", color=middle_color, lw=1.5)
    if len(lane_lines) > 1:
        plt.plot([0, road_end_x], [lane_lines[-1], lane_lines[-1]], linestyle="-", color=main_color, lw=2.5)


def connect_escape_key() -> None:
    """Allow closing a live Matplotlib animation with the Escape key."""
    plt.gcf().canvas.mpl_connect(
        "key_release_event",
        lambda event: [exit(0) if event.key == "escape" else None],
    )


def plot_vehicle(index: int, state: State, steer: float, color: str | None = None) -> None:
    """Plot one vehicle using the indexed vehicle drawing helper."""
    plotter = PLOTTERS[index] if index < len(PLOTTERS) else PLOTTERS[-1]
    color = color or "-k"
    color = color if color.startswith("-") else f"-{color}"
    plotter(state.x, state.y, state.yaw, steer=steer, truckcolor=color)


def plot_all_vehicles(states: Sequence[State], steers: Sequence[float], colors: Sequence[str] | None = None) -> None:
    """Plot all vehicles for the current simulation frame."""
    for index, (state, steer) in enumerate(zip(states, steers)):
        color = colors[index] if colors and index < len(colors) else None
        plot_vehicle(index, state, steer, color)
