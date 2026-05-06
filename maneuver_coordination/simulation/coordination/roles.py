"""Role, lane, neighbor, ACC, and collision helpers for simulation loops."""

import math
from typing import Dict, List, Sequence

import numpy as np

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.simulation.core.settings import (
    ACC_MIN_GAP,
    ACC_SPEED_MARGIN,
    ACC_TIME_GAP,
    BRAKING_SPEED_DELTA,
    VEHICLE_COLLISION_DISTANCE,
)
from maneuver_coordination.simulation.core.types import PlannedPath, State, VehicleConfig


def get_role_to_index(vehicle_configs: Sequence[VehicleConfig]) -> Dict[str, int]:
    """Map role names to their index in parallel state/history lists."""
    return {config.role: index for index, config in enumerate(vehicle_configs)}


def require_roles(role_to_index: Dict[str, int], required_roles: Sequence[str]) -> None:
    """Fail early when a runner is missing required scenario roles."""
    missing_roles = [role for role in required_roles if role not in role_to_index]
    if missing_roles:
        raise ValueError(f"Missing required roles for cascading scenario: {missing_roles}")


def get_role_item(items: Sequence, role_to_index: Dict[str, int], role: str):
    """Fetch an item from a role-aligned list."""
    return items[role_to_index[role]]


def lane_center_y(lane_id: int | None) -> float:
    """Convert a lane ID into the corresponding lane-center y coordinate."""
    if lane_id is None:
        return 0.0
    return 2.0 + 4.0 * lane_id


def infer_lane_id_from_y(y: float, vehicle_configs: Sequence[VehicleConfig]) -> int | None:
    """Infer the closest configured lane ID from a vehicle y position."""
    lane_ids = sorted(
        {
            config.lane_id
            for config in vehicle_configs
            if config.lane_id is not None
        }
    )
    if not lane_ids:
        return None

    lane_centers = {lane_id: lane_center_y(lane_id) for lane_id in lane_ids}
    return min(lane_centers, key=lambda lane_id: abs(y - lane_centers[lane_id]))


def find_vehicle_ahead_in_lane(
    subject_role: str,
    states_by_role: Dict[str, State],
    vehicle_configs: Sequence[VehicleConfig],
    lane_id: int | None = None,
) -> str | None:
    """Find the nearest vehicle ahead of a subject in the selected/current lane."""
    config_by_role = {config.role: config for config in vehicle_configs}
    subject_config = config_by_role.get(subject_role)
    subject_state = states_by_role.get(subject_role)
    if subject_config is None or subject_state is None:
        return None

    lane_to_check = (
        infer_lane_id_from_y(subject_state.y, vehicle_configs)
        if lane_id is None
        else lane_id
    )
    if lane_to_check is None:
        return None

    best_role = None
    best_dx = float("inf")
    for config in vehicle_configs:
        if config.role == subject_role:
            continue

        candidate_state = states_by_role.get(config.role)
        if candidate_state is None:
            continue
        candidate_lane_id = infer_lane_id_from_y(candidate_state.y, vehicle_configs)
        if candidate_lane_id != lane_to_check:
            continue

        dx = candidate_state.x - subject_state.x
        if dx > 0.0 and dx < best_dx:
            best_dx = dx
            best_role = config.role

    return best_role


def find_vehicle_behind_in_lane(
    subject_role: str,
    states_by_role: Dict[str, State],
    vehicle_configs: Sequence[VehicleConfig],
    lane_id: int | None = None,
) -> str | None:
    """Find the nearest vehicle behind a subject in the selected/current lane."""
    config_by_role = {config.role: config for config in vehicle_configs}
    subject_config = config_by_role.get(subject_role)
    subject_state = states_by_role.get(subject_role)
    if subject_config is None or subject_state is None:
        return None

    lane_to_check = (
        infer_lane_id_from_y(subject_state.y, vehicle_configs)
        if lane_id is None
        else lane_id
    )
    if lane_to_check is None:
        return None

    best_role = None
    best_dx = float("-inf")
    for config in vehicle_configs:
        if config.role == subject_role:
            continue

        candidate_state = states_by_role.get(config.role)
        if candidate_state is None:
            continue
        candidate_lane_id = infer_lane_id_from_y(candidate_state.y, vehicle_configs)
        if candidate_lane_id != lane_to_check:
            continue

        dx = candidate_state.x - subject_state.x
        if dx < 0.0 and dx > best_dx:
            best_dx = dx
            best_role = config.role

    return best_role


def find_target_lane_neighbors(
    subject_role: str,
    states_by_role: Dict[str, State],
    vehicle_configs: Sequence[VehicleConfig],
) -> Dict[str, str | None]:
    """Return nearest front/rear vehicles in the requester's target lane."""
    config_by_role = {config.role: config for config in vehicle_configs}
    subject_config = config_by_role.get(subject_role)
    if subject_config is None:
        return {"front": None, "rear": None}

    target_lane_id = subject_config.target_lane_id
    if target_lane_id is None:
        return {"front": None, "rear": None}

    return {
        "front": find_vehicle_ahead_in_lane(subject_role, states_by_role, vehicle_configs, lane_id=target_lane_id),
        "rear": find_vehicle_behind_in_lane(subject_role, states_by_role, vehicle_configs, lane_id=target_lane_id),
    }


def is_vehicle_braking(vehicle_history, speed_drop_threshold: float = BRAKING_SPEED_DELTA) -> bool:
    """Detect braking from the latest two recorded speed samples."""
    if vehicle_history is None or len(vehicle_history.v) < 2:
        return False
    return (vehicle_history.v[-2] - vehicle_history.v[-1]) > speed_drop_threshold


def should_trigger_maneuver_search(
    follower_state: State,
    lead_state: State,
    lead_history,
    existing_path_length: int = 0,
) -> bool:
    """Decide whether a follower should start searching for a lane change."""
    if existing_path_length >= 2 or lead_state.x <= follower_state.x:
        return False

    lead_car_distance = math.hypot(follower_state.x - lead_state.x, follower_state.y - lead_state.y) - 4.0
    distance_gap = C.PLANNER_PARAMS.safe_time_gap * follower_state.v
    lead_is_slowing = lead_state.v < C.PLANNER_PARAMS.lane_change_trigger_speed or is_vehicle_braking(lead_history)
    return lead_car_distance < distance_gap and lead_is_slowing


def build_acc_speed_profile(
    path: PlannedPath,
    state: State,
    target_speed: float,
    lead_state: State | None,
) -> List[float]:
    """Build a simple ACC-style speed profile for following a detected lead vehicle."""
    if lead_state is None or lead_state.x <= state.x:
        return [target_speed for _ in path.x]

    gap = max(0.0, lead_state.x - state.x - VEHICLE_COLLISION_DISTANCE)
    desired_gap = max(ACC_MIN_GAP, state.v * ACC_TIME_GAP)
    follow_speed = max(0.0, min(target_speed, lead_state.v - ACC_SPEED_MARGIN))
    recovery_gap = max(5.0, desired_gap * 0.5)

    if gap <= desired_gap:
        capped_speed = follow_speed
    elif gap >= desired_gap + recovery_gap:
        capped_speed = target_speed
    else:
        blend = (gap - desired_gap) / recovery_gap
        capped_speed = follow_speed + blend * (target_speed - follow_speed)

    return [capped_speed for _ in path.x]


def build_delayed_speed_profile(
    path: PlannedPath,
    cruise_speed: float,
    slow_speed: float,
    time_s: float,
    braking_start_time: float,
) -> List[float]:
    """Switch from cruise to slow speed after the configured braking time."""
    target_speed = slow_speed if time_s >= braking_start_time else cruise_speed
    return [target_speed for _ in path.x]


def build_target_lane_reference_candidates(
    requester_role: str,
    available_references: Dict[str, np.ndarray],
    states_by_role: Dict[str, State],
    vehicle_configs: Sequence[VehicleConfig],
) -> Dict[str, np.ndarray]:
    """Order target-lane references by rear/front relevance to the requester."""
    neighbors = find_target_lane_neighbors(requester_role, states_by_role, vehicle_configs)
    ordered_roles = [role for role in (neighbors["rear"], neighbors["front"]) if role in available_references]
    for role in available_references:
        if role not in ordered_roles:
            ordered_roles.append(role)
    return {role: available_references[role] for role in ordered_roles}


def get_requester_candidate_roles(vehicle_configs: Sequence[VehicleConfig]) -> List[str]:
    """Return vehicles configured to search for a different target lane."""
    return [
        config.role
        for config in vehicle_configs
        if config.target_lane_id is not None and config.lane_id is not None and config.target_lane_id != config.lane_id
    ]


def detect_vehicle_collisions(states: Sequence[State], vehicle_configs: Sequence[VehicleConfig], min_distance: float = VEHICLE_COLLISION_DISTANCE):
    """Report vehicle pairs closer than the configured collision distance."""
    collisions = []
    for first_idx in range(len(states)):
        for second_idx in range(first_idx + 1, len(states)):
            dx = states[first_idx].x - states[second_idx].x
            dy = states[first_idx].y - states[second_idx].y
            if math.hypot(dx, dy) < min_distance:
                collisions.append((vehicle_configs[first_idx].name, vehicle_configs[second_idx].name))
    return collisions
