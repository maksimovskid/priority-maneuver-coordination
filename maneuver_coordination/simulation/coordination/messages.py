"""Small V2X-style message helpers used by the simulation loops."""

from typing import Dict, List, Sequence

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination.constants import send_request
from maneuver_coordination.simulation.coordination.coordination_flow import build_requester_conflict_bundle
from maneuver_coordination.simulation.coordination.events import (
    append_simulation_event,
    format_event_time,
    format_priority_label,
)
from maneuver_coordination.simulation.core.types import V2XMessage, VehicleConfig


def get_config_by_role(vehicle_configs: Sequence[VehicleConfig]) -> Dict[str, VehicleConfig]:
    """Index vehicle configs by runtime role name."""
    return {config.role: config for config in vehicle_configs}


def build_request_message_candidates(
    requester_role: str,
    receiver_role: str,
    request_state: int,
    priority: int | None,
    stage_label: str,
):
    """Describe a potential maneuver-request message for later emission."""
    return {
        "requester_role": requester_role,
        "receiver_role": receiver_role,
        "request_state": request_state,
        "priority": priority,
        "stage_label": stage_label,
    }


def emit_request_messages(
    time_s: float,
    request_candidates: Sequence[Dict[str, object]],
    config_by_role: Dict[str, VehicleConfig],
) -> List[V2XMessage]:
    """Create maneuver-request messages for candidates currently in SEND_REQUEST."""
    messages: List[V2XMessage] = []
    for candidate in request_candidates:
        requester_role = str(candidate["requester_role"])
        receiver_role = str(candidate["receiver_role"])
        request_state = int(candidate["request_state"])
        if request_state != send_request:
            continue
        if requester_role not in config_by_role or receiver_role not in config_by_role:
            continue

        messages.append(
            emit_v2x_request_message(
                time_s,
                config_by_role[requester_role],
                config_by_role[receiver_role],
                candidate.get("priority"),
                "maneuver_request",
                {"stage": str(candidate["stage_label"])},
            )
        )
    return messages


def emit_v2x_message_event(
    time_s: float,
    sender_role: str,
    receiver_role: str,
    priority: int | None,
    message_type: str,
    config_by_role: Dict[str, VehicleConfig],
    payload: Dict[str, str] | None = None,
) -> V2XMessage:
    """Create a generic V2X-style message between two configured roles."""
    return emit_v2x_request_message(
        time_s,
        config_by_role[sender_role],
        config_by_role[receiver_role],
        priority,
        message_type,
        payload,
    )


def append_message_events(
    event_log: List[str],
    messages: Sequence[V2XMessage],
    *,
    echo: bool,
) -> None:
    """Add human-readable message events to the scenario event log."""
    label_by_type = {
        "maneuver_request": "request sent",
        "maneuver_offer": "offer sent",
        "confirm_offer": "offer confirmed",
    }
    for message in messages:
        label = label_by_type.get(message.message_type, message.message_type)
        append_simulation_event(
            event_log,
            f"{format_event_time(message.time_s)}: ID {message.sender_id} -> ID {message.receiver_id} {label}",
            echo=echo,
        )


def request_state_for_multi_acceptor(
    motion_state: int,
    needs_by_role: Dict[str, bool],
    accept_states_by_role: Dict[str, int],
    offer_states_by_role: Dict[str, int] | None = None,
) -> int:
    """Compute requester message state when multiple acceptors are involved."""
    offer_states_by_role = offer_states_by_role or {}
    if motion_state == C.MotionState.LANE_CHANGE:
        return C.execute_request
    if motion_state == C.MotionState.FOLLOW_SECOND_LANE:
        return C.no_request_sent
    if motion_state != C.MotionState.FIND_LANE_CHANGE:
        return C.no_request_sent

    needed_roles = [role for role, needed in needs_by_role.items() if needed]
    if not needed_roles:
        return C.no_need_for_request

    if needed_roles and all(accept_states_by_role.get(role) == C.request_accepted for role in needed_roles):
        return C.execute_request

    if offer_states_by_role and all(
        accept_states_by_role.get(role) in (C.offer, C.request_accepted)
        for role in needed_roles
    ):
        return C.confirm_offers

    return C.send_request


def emit_v2x_request_message(
    time_s: float,
    sender_config: VehicleConfig,
    receiver_config: VehicleConfig,
    priority: int | None,
    message_type: str,
    payload: Dict[str, str] | None = None,
) -> V2XMessage:
    """Create the concrete message payload exchanged by scenario runners."""
    payload_data = dict(payload or {})
    if priority is not None:
        payload_data.setdefault("priority_code", str(int(priority)))
    payload_data.setdefault("requester_role", sender_config.role)
    if sender_config.target_lane_id is not None:
        payload_data.setdefault("target_lane_id", str(sender_config.target_lane_id))

    return V2XMessage(
        time_s=round(time_s, 2),
        sender_id=sender_config.vehicle_id,
        receiver_id=receiver_config.vehicle_id,
        sender_role=sender_config.role,
        receiver_role=receiver_config.role,
        message_type=message_type,
        priority_label=format_priority_label(priority) or "unknown",
        payload=payload_data,
    )


def get_vehicle_inbox(messages: Sequence[V2XMessage], receiver_id: int) -> List[V2XMessage]:
    """Return messages addressed to one vehicle."""
    return [message for message in messages if message.receiver_id == receiver_id]


def has_v2x_request_message(
    messages: Sequence[V2XMessage],
    sender_id: int,
    receiver_id: int,
    message_type: str = "maneuver_request",
) -> bool:
    """Check whether a specific sender already addressed a receiver."""
    return any(
        message.sender_id == sender_id
        and message.receiver_id == receiver_id
        and message.message_type == message_type
        for message in messages
    )


def get_latest_v2x_request(
    messages: Sequence[V2XMessage],
    receiver_id: int,
    sender_id: int | None = None,
    message_type: str = "maneuver_request",
) -> V2XMessage | None:
    """Return the most recent matching message for a receiver."""
    candidates = [
        message
        for message in messages
        if message.receiver_id == receiver_id
        and message.message_type == message_type
        and (sender_id is None or message.sender_id == sender_id)
    ]
    return candidates[-1] if candidates else None


def get_message_priority(message: V2XMessage | None, fallback_priority: int | None = None) -> int | None:
    """Read priority from a message payload with an optional fallback."""
    if message is None:
        return fallback_priority

    priority_code = message.payload.get("priority_code")
    if priority_code is not None:
        return int(priority_code)
    return fallback_priority


def select_request_message_for_receiver(
    messages: Sequence[V2XMessage],
    receiver_id: int,
) -> V2XMessage | None:
    """Select the highest-priority/latest request from a receiver inbox."""
    inbox = get_vehicle_inbox(messages, receiver_id)
    if not inbox:
        return None

    return max(
        inbox,
        key=lambda message: (
            get_message_priority(message, 0) or 0,
            message.time_s,
        ),
    )


def build_received_request_context(
    messages: Sequence[V2XMessage],
    receiver_id: int,
    requester_conflict_sources: Dict[str, tuple[bool, bool, bool]],
    fallback_requester_role: str,
    fallback_priority: int | None = None,
):
    """Resolve which requester a receiver should evaluate this step."""
    selected_message = select_request_message_for_receiver(messages, receiver_id)
    selected_requester_role = (
        selected_message.sender_role
        if selected_message is not None and selected_message.sender_role in requester_conflict_sources
        else fallback_requester_role
    )
    delivered = (
        selected_message is not None
        and selected_message.sender_role == selected_requester_role
    )
    return {
        "message": selected_message,
        "requester_role": selected_requester_role,
        "requester_conflicts": build_requester_conflict_bundle(
            selected_requester_role,
            requester_conflict_sources,
        ),
        "priority": get_message_priority(selected_message, fallback_priority),
        "delivered": delivered,
    }
