"""Human-readable logging helpers for coordination state changes."""

from typing import Dict, List, Sequence

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination.constants import (
    emergency_maneuver,
    find_lane_change,
    follow_lane,
    high_priority,
    low_priority,
    medium_priority,
    no_priority_request,
    no_request_received,
    no_request_sent,
    send_request,
)
from maneuver_coordination.simulation.core.types import VehicleConfig


def format_priority_label(priority: int | None) -> str | None:
    """Convert a priority enum/int into a terminal-friendly label."""
    priority_labels = {
        no_priority_request: "no_priority_request",
        low_priority: "low_priority",
        medium_priority: "medium_priority",
        high_priority: "high_priority",
        emergency_maneuver: "emergency_maneuver",
    }
    if priority is None:
        return None
    return priority_labels.get(priority, str(priority))


def format_request_state_label(state: int | None) -> str | None:
    """Convert requester state into a terminal-friendly label."""
    labels = {
        no_request_sent: "no_request_sent",
        send_request: "send_request",
        C.execute_request: "execute_request",
        C.no_need_for_request: "no_need_for_request",
    }
    if state is None:
        return None
    return labels.get(state, str(state))


def format_accept_state_label(state: int | None) -> str | None:
    """Convert acceptor state into a terminal-friendly label."""
    labels = {
        no_request_received: "no_request_received",
        C.request_received: "request_received",
        C.request_rejected: "request_rejected",
        C.request_accepted: "request_accepted",
        C.cascading_wait: "cascading_wait",
    }
    if state is None:
        return None
    return labels.get(state, str(state))


def format_motion_state_label(state: int | None) -> str | None:
    """Convert motion state into a terminal-friendly label."""
    labels = {
        follow_lane: "follow_lane",
        find_lane_change: "find_lane_change",
        C.lane_change: "lane_change",
        C.follow_second_lane: "follow_second_lane",
    }
    if state is None:
        return None
    return labels.get(state, str(state))


def append_simulation_event(event_log: List[str], message: str, echo: bool = True) -> None:
    """Store an event and optionally print it immediately."""
    event_log.append(message)
    if echo:
        print(message)


def format_event_time(time_s: float) -> str:
    """Format simulation time consistently for logs."""
    return f"t = {time_s:.1f} s"


def log_requester_state_changes(
    vehicle_configs: Sequence[VehicleConfig],
    coordination_by_requester: Dict[str, Dict[str, int]],
    last_coordination_snapshot: Dict[str, Dict[str, int]],
    event_log: List[str],
    time_s: float,
    active_receiver_roles: Sequence[str] | None = None,
    echo: bool = True,
) -> None:
    """Log only changed requester status/motion/request/acceptance fields."""
    config_by_role = {config.role: config for config in vehicle_configs}
    for requester_role, current_state in coordination_by_requester.items():
        previous_state = last_coordination_snapshot.setdefault(requester_role, {})
        config = config_by_role.get(requester_role)
        if config is None:
            continue

        current_operation = infer_vehicle_operation_label(
            requester_role,
            coordination_by_requester,
            active_receiver_roles,
        )
        if previous_state.get("operation_label") != current_operation:
            append_simulation_event(
                event_log,
                f"{format_event_time(time_s)}: ID {config.vehicle_id} status -> {current_operation}",
                echo=echo,
            )

        current_motion = current_state.get("motion_state")
        if previous_state.get("motion_state") != current_motion:
            motion_label = format_motion_state_label(current_motion)
            append_simulation_event(
                event_log,
                f"{format_event_time(time_s)}: ID {config.vehicle_id} motion -> {motion_label}",
                echo=echo,
            )

        current_request = current_state.get("request_state")
        if previous_state.get("request_state") != current_request:
            request_label = format_request_state_label(current_request)
            if current_request == send_request:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time_s)}: ID {config.vehicle_id} request sent",
                    echo=echo,
                )
            elif current_request == C.execute_request:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time_s)}: ID {config.vehicle_id} request approved for execution",
                    echo=echo,
                )
            else:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time_s)}: ID {config.vehicle_id} request -> {request_label}",
                    echo=echo,
                )

        current_accept = current_state.get("accept_state")
        if previous_state.get("accept_state") != current_accept:
            accept_label = format_accept_state_label(current_accept)
            if current_accept == C.request_accepted:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time_s)}: ID {config.vehicle_id} request accepted",
                    echo=echo,
                )
            elif current_accept == C.request_rejected:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time_s)}: ID {config.vehicle_id} request rejected",
                    echo=echo,
                )
            else:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time_s)}: ID {config.vehicle_id} acceptance -> {accept_label}",
                    echo=echo,
                )

        previous_state["operation_label"] = current_operation
        previous_state["motion_state"] = current_motion
        previous_state["request_state"] = current_request
        previous_state["accept_state"] = current_accept


def log_receiver_state_changes(
    vehicle_configs: Sequence[VehicleConfig],
    accept_state_by_role: Dict[str, int],
    last_coordination_snapshot: Dict[str, Dict[str, int]],
    event_log: List[str],
    time_s: float,
    active_receiver_roles: Sequence[str] | None = None,
    echo: bool = True,
) -> None:
    """Log only changed acceptor status and acceptance fields."""
    config_by_role = {config.role: config for config in vehicle_configs}
    active_receiver_roles = active_receiver_roles or []

    for receiver_role, current_accept in accept_state_by_role.items():
        config = config_by_role.get(receiver_role)
        if config is None:
            continue

        previous_state = last_coordination_snapshot.setdefault(receiver_role, {})
        current_operation = "acceptor" if receiver_role in active_receiver_roles else "normal_operation"
        if previous_state.get("operation_label") != current_operation:
            append_simulation_event(
                event_log,
                f"{format_event_time(time_s)}: ID {config.vehicle_id} status -> {current_operation}",
                echo=echo,
            )

        if previous_state.get("accept_state") != current_accept:
            accept_label = format_accept_state_label(current_accept)
            if current_accept == C.request_accepted:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time_s)}: ID {config.vehicle_id} request accepted",
                    echo=echo,
                )
            elif current_accept == C.request_rejected:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time_s)}: ID {config.vehicle_id} request rejected",
                    echo=echo,
                )
            else:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time_s)}: ID {config.vehicle_id} acceptance -> {accept_label}",
                    echo=echo,
                )

        previous_state["operation_label"] = current_operation
        previous_state["accept_state"] = current_accept


def infer_vehicle_operation_label(
    role: str,
    coordination_by_requester: Dict[str, Dict[str, int]] | None,
    active_receiver_roles: Sequence[str] | None,
) -> str:
    """Infer normal/requester/acceptor status from runtime coordination state."""
    if coordination_by_requester and role in coordination_by_requester:
        requester_state = coordination_by_requester[role]
        if (
            requester_state.get("request_state") != no_request_sent
            or requester_state.get("motion_state") != follow_lane
            or requester_state.get("accept_state") != no_request_received
        ):
            return "requester"

    if active_receiver_roles and role in active_receiver_roles:
        return "acceptor"

    return "normal_operation"


def get_active_receiver_roles(
    message_stage_specs: Sequence[Dict[str, object]],
    coordination_by_requester: Dict[str, Dict[str, int]] | None,
) -> List[str]:
    """Return receiver roles currently involved in active request stages."""
    if coordination_by_requester is None:
        return []

    active_roles: List[str] = []
    for stage_spec in message_stage_specs:
        requester_role = str(stage_spec["requester_role"])
        receiver_role = str(stage_spec["receiver_role"])
        requester_state = coordination_by_requester.get(requester_role, {})
        if requester_state.get("request_state") in (send_request, C.execute_request):
            if receiver_role and receiver_role not in active_roles:
                active_roles.append(receiver_role)

    return active_roles
