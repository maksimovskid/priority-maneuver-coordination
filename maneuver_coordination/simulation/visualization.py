"""Live plotting and saved-frame rendering helpers for scenarios."""

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination.constants import send_request
from maneuver_coordination.simulation.coordination.events import (
    format_accept_state_label,
    format_motion_state_label,
    format_priority_label,
    format_request_state_label,
    infer_vehicle_operation_label,
)
from maneuver_coordination.simulation.motion.reference import calc_ref_trajectory
from maneuver_coordination.simulation.core.types import PlannedPath, State, VehicleConfig
from maneuver_coordination.vehicle.plotting import plot_all_vehicles, plot_road_boundaries


def requested_xy_segment(xreq, current_x: float, current_y: float):
    """Extract valid requested-trajectory points and prepend current position."""
    valid_mask = np.any(np.abs(xreq[:2, :]) > 1e-9, axis=0)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        return None, None

    last_index = int(valid_indices[-1]) + 1
    x_points = [current_x] + list(xreq[0, :last_index])
    y_points = [current_y] + list(xreq[1, :last_index])
    return x_points, y_points


def should_plot_requested_trajectory(
    request_state: int | None,
    motion_state: int | None = None,
) -> bool:
    """Return whether the yellow cooperative trajectory should be visible."""
    _ = motion_state
    return request_state in (send_request, C.confirm_offers, C.execute_request)


def update_requested_plot_cache(
    plot_cache: Dict[str, Dict[str, object]],
    role: str,
    xref: np.ndarray,
    enabled: bool,
    request_state: int | None,
    motion_state: int | None,
) -> None:
    """Keep yellow requested trajectories visible only while negotiation/execution is active."""
    if enabled and should_plot_requested_trajectory(request_state, motion_state):
        plot_cache[role] = {
            "xref": xref.copy(),
        }
    elif not should_plot_requested_trajectory(request_state, motion_state):
        plot_cache.pop(role, None)


def plot_requested_trajectories(
    plot_requests: Sequence[Dict[str, object]],
) -> None:
    """Plot active requested cooperative trajectories as yellow points."""
    for plot_request in plot_requests:
        xref = plot_request.get("xref")
        state = plot_request.get("state")
        if xref is None or state is None:
            continue
        req_x_plot, req_y_plot = requested_xy_segment(xref, state.x, state.y)
        if req_x_plot is not None:
            plt.plot(req_x_plot, req_y_plot, ".", color="y", markersize=4, zorder=6)


def plot_future_planned_trajectories(
    vehicle_configs: Sequence[VehicleConfig],
    states: Sequence[State],
    paths_by_role: Dict[str, PlannedPath],
    role_specific_paths: Dict[str, tuple],
) -> None:
    """Plot each vehicle's short future planned trajectory as colored points."""
    for state, config in zip(states, vehicle_configs):
        path_info = role_specific_paths.get(config.role)
        if path_info is None:
            planned_path = paths_by_role[config.role]
            path_x = planned_path.x
            path_y = planned_path.y
            path_yaw = planned_path.yaw
            path_k = planned_path.curvature
            speed_profile = planned_path.speed_profile
        else:
            path_x = path_info[0]
            path_y = path_info[1]
            path_yaw = path_info[2]
            path_k = path_info[3]
            speed_profile = path_info[4]

        if not path_x or not path_y or not path_yaw or not path_k:
            continue

        xref, _, _ = calc_ref_trajectory(
            state,
            path_x,
            path_y,
            path_yaw,
            path_k,
            speed_profile,
            0.1,
        )
        horizon = C.PLANNER_PARAMS.horizon_steps
        plt.plot(xref[0, :horizon], xref[1, :horizon], ".", color=config.color, markersize=4)


def get_requested_plot_xref(
    live_xref: np.ndarray | None,
    live_enabled: bool,
    request_state: int | None,
    motion_state: int | None,
    plot_cache: Dict[str, Dict[str, object]],
    role: str,
) -> np.ndarray | None:
    """Return the live or cached requested trajectory for plotting."""
    if live_enabled and should_plot_requested_trajectory(request_state, motion_state):
        return live_xref
    return (plot_cache.get(role) or {}).get("xref")


def add_vehicle_speed_overlay(
    vehicle_configs: Sequence[VehicleConfig],
    states: Sequence[State],
    time: float,
    priority: int | None = None,
    requester_role: str = "ego",
    coordination_by_requester: Dict[str, Dict[str, int]] | None = None,
    active_receiver_roles: Sequence[str] | None = None,
    receiver_accept_states: Dict[str, int] | None = None,
    recent_events: Sequence[str] | None = None,
) -> None:
    """Draw the time, speed, role, and recent-event overlay above the frame."""
    fig = plt.gcf()
    fig.subplots_adjust(top=0.76)
    fig.texts.clear()
    fig.text(
        0.98,
        0.99,
        f"t = {round(time, 2)} s",
        fontsize=13,
        color="black",
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    text_box = {"facecolor": "white", "alpha": 0.75, "edgecolor": "none"}
    vehicle_step = 0.96 / max(1, len(vehicle_configs))
    base_x = 0.02

    for idx, (config, state) in enumerate(zip(vehicle_configs, states)):
        speed_kmh = state.v * 3.6
        vehicle_color = config.color if not config.color.startswith("-") else config.color[1:]
        operation_label = infer_vehicle_operation_label(
            config.role,
            coordination_by_requester,
            active_receiver_roles,
        )
        fig.text(
            base_x + idx * vehicle_step,
            0.955,
            f"ID {config.vehicle_id}: {speed_kmh:.1f} km/h | {operation_label}",
            fontsize=10,
            color=vehicle_color,
            verticalalignment="top",
            bbox=text_box,
        )

    priority_label = format_priority_label(priority)
    if priority_label is not None:
        requester_config = next(
            (config for config in vehicle_configs if config.role == requester_role),
            vehicle_configs[0] if vehicle_configs else None,
        )
        priority_color = "k" if requester_config is None else (
            requester_config.color if not requester_config.color.startswith("-") else requester_config.color[1:]
        )
        fig.text(
            0.02,
            0.915,
            f"priority: {priority_label}",
            fontsize=11,
            color=priority_color,
            verticalalignment="top",
            bbox=text_box,
        )

    if coordination_by_requester:
        requester_items = list(coordination_by_requester.items())
        requester_step = 0.96 / max(1, len(requester_items))
        for idx, (requester_role_key, decision_state) in enumerate(requester_items):
            requester_config = next(
                (config for config in vehicle_configs if config.role == requester_role_key),
                None,
            )
            if requester_config is None:
                continue
            decision_color = requester_config.color if not requester_config.color.startswith("-") else requester_config.color[1:]
            motion_label = format_motion_state_label(decision_state.get("motion_state"))
            request_label = format_request_state_label(decision_state.get("request_state"))
            accept_label = format_accept_state_label(decision_state.get("accept_state"))
            fig.text(
                base_x + idx * requester_step,
                0.88,
                f"ID {requester_config.vehicle_id}: {motion_label} | {request_label} | {accept_label}",
                fontsize=10,
                color=decision_color,
                verticalalignment="top",
                bbox=text_box,
            )

    if receiver_accept_states:
        receiver_parts = []
        for receiver_role, accept_state in receiver_accept_states.items():
            receiver_config = next(
                (config for config in vehicle_configs if config.role == receiver_role),
                None,
            )
            if receiver_config is None:
                continue
            accept_label = format_accept_state_label(accept_state)
            receiver_parts.append(f"ID {receiver_config.vehicle_id}: {accept_label}")

        if receiver_parts:
            fig.text(
                0.02,
                0.845,
                "acceptors: " + " | ".join(receiver_parts),
                fontsize=10,
                color="black",
                verticalalignment="top",
                bbox=text_box,
            )

    if recent_events:
        event_y = 0.22
        for event in recent_events[-3:]:
            plt.gca().text(
                0.02,
                event_y,
                event,
                transform=plt.gca().transAxes,
                fontsize=9,
                color="black",
                verticalalignment="top",
                bbox=text_box,
            )
            event_y -= 0.06


def annotate_vehicle_ids(vehicle_configs: Sequence[VehicleConfig], states: Sequence[State]) -> None:
    """Draw vehicle IDs next to their current positions."""
    for config, state in zip(vehicle_configs, states):
        vehicle_color = config.color if not config.color.startswith("-") else config.color[1:]
        plt.text(
            state.x + 0.4,
            state.y + 0.7,
            f"ID {config.vehicle_id}",
            fontsize=10,
            color=vehicle_color,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        )


def clone_state(state: State) -> State:
    """Create a detached copy of a mutable vehicle state."""
    return State(x=state.x, y=state.y, yaw=state.yaw, v=state.v, a=state.a)


def clone_coordination_state_map(
    coordination_by_requester: Dict[str, Dict[str, int]] | None,
) -> Dict[str, Dict[str, int]] | None:
    """Copy nested coordination state for saved animation frames."""
    if coordination_by_requester is None:
        return None
    return {
        role: dict(state_map)
        for role, state_map in coordination_by_requester.items()
    }


def capture_simulation_frame(
    *,
    time: float,
    states: Sequence[State],
    steers: Sequence[float],
    vehicle_configs: Sequence[VehicleConfig],
    paths_by_role: Dict[str, PlannedPath],
    role_specific_paths: Dict[str, tuple],
    plot_requests: Sequence[Dict[str, object]],
    priority: int | None,
    requester_role: str,
    coordination_by_requester: Dict[str, Dict[str, int]] | None,
    active_receiver_roles: Sequence[str] | None,
    receiver_accept_states: Dict[str, int] | None,
    recent_events: Sequence[str] | None,
) -> Dict[str, object]:
    """Snapshot mutable simulation state for later animation export."""
    _ = vehicle_configs
    return {
        "time": time,
        "states": [clone_state(state) for state in states],
        "steers": list(steers),
        "paths_by_role": paths_by_role,
        "role_specific_paths": dict(role_specific_paths),
        "plot_requests": [
            {
                "xref": plot_request.get("xref"),
                "state": clone_state(plot_request["state"]) if plot_request.get("state") is not None else None,
            }
            for plot_request in plot_requests
        ],
        "priority": priority,
        "requester_role": requester_role,
        "coordination_by_requester": clone_coordination_state_map(coordination_by_requester),
        "active_receiver_roles": list(active_receiver_roles or []),
        "receiver_accept_states": dict(receiver_accept_states or {}),
        "recent_events": list(recent_events[-3:] if recent_events else []),
    }


def render_simulation_frame(
    ax,
    vehicle_configs: Sequence[VehicleConfig],
    frame_data: Dict[str, object],
    *,
    road_end_x: float,
    lane_lines: Sequence[float],
    main_color: str = "k",
    middle_color: str = "k",
) -> None:
    """Render one saved/live simulation frame with the standard Matplotlib view."""
    ax.clear()
    ax.figure.sca(ax)
    plot_road_boundaries(
        main_color=main_color,
        middle_color=middle_color,
        road_end_x=road_end_x,
        lane_lines=lane_lines,
    )
    plot_future_planned_trajectories(
        vehicle_configs,
        frame_data["states"],
        frame_data["paths_by_role"],
        frame_data["role_specific_paths"],
    )
    plot_requested_trajectories(frame_data["plot_requests"])
    plot_all_vehicles(
        frame_data["states"],
        frame_data["steers"],
        [config.color for config in vehicle_configs],
    )
    annotate_vehicle_ids(vehicle_configs, frame_data["states"])
    plt.axis("equal")
    plt.grid(False)
    plt.xlabel("x [m]", fontsize=15)
    plt.ylabel("y [m]", fontsize=15)
    add_vehicle_speed_overlay(
        vehicle_configs,
        frame_data["states"],
        frame_data["time"],
        frame_data["priority"],
        frame_data["requester_role"],
        frame_data["coordination_by_requester"],
        frame_data["active_receiver_roles"],
        frame_data["receiver_accept_states"],
        frame_data["recent_events"],
    )
