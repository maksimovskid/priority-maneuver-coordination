"""Longitudinal speed-profile helpers for normal and cooperative motion."""

import math
from typing import List, Sequence

from maneuver_coordination.coordination.constants import (
    follow_planned,
    increase_speed_for_req_high,
    increase_speed_for_req_low,
    increase_speed_for_req_medium,
    reduce_speed_for_req_high,
    reduce_speed_for_req_low,
    reduce_speed_for_req_medium,
)
from maneuver_coordination.simulation.core.math_helpers import calc_distance, calc_time
from maneuver_coordination.simulation.core.types import State


def make_base_speed_profile(cx: Sequence[float], cyaw: Sequence[float], target_speed: float) -> List[float]:
    """Create a constant target-speed profile along a geometric path."""
    speed_profile = [target_speed] * len(cx)
    direction = 1.0

    for i in range(len(cx) - 1):
        dyaw = cyaw[i + 1] - cyaw[i]
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1

        speed_profile[i] = -target_speed if direction != 1.0 else target_speed

        if switch:
            speed_profile[i] = 0.0

    return speed_profile


def apply_speed_state(base_profile: Sequence[float], base_speed: float, speed_state: int) -> List[float]:
    """Apply a discrete cooperative speed mode to a constant profile."""
    state_to_offset = {
        follow_planned: 0.0,
        reduce_speed_for_req_low: -1.3,
        reduce_speed_for_req_medium: -2.6,
        reduce_speed_for_req_high: -5.0,
        increase_speed_for_req_low: +1.0,
        increase_speed_for_req_medium: +2.0,
        increase_speed_for_req_high: +4.0,
    }
    offset = state_to_offset.get(speed_state, 0.0)
    return [base_speed + offset for _ in base_profile]


def apply_speed_delta(base_profile: Sequence[float], base_speed: float, speed_delta: float) -> List[float]:
    """Apply a selected continuous speed delta to a constant profile."""
    adjusted_speed = max(0.0, base_speed + speed_delta)
    return [adjusted_speed for _ in base_profile]


def calc_speed_profile(cx: Sequence[float], cy: Sequence[float], cyaw: Sequence[float], target_speed: float) -> List[float]:
    """Build the default profile with legacy near-goal slowdown behavior."""
    _ = cy
    speed_profile = make_base_speed_profile(cx, cyaw, target_speed)

    a_max = 2.0
    _ = calc_distance(target_speed, 0.277, -a_max)
    _ = calc_time(target_speed, 0.277, -a_max)

    stop_index = len(cx) - 1
    for i in range(max(0, (len(cx) - 1) - 500), stop_index):
        speed_profile[i] = 0.277
    speed_profile[-1] = 25.0 / 3.6

    return speed_profile


def calc_speed_profile5(cx: Sequence[float], cy: Sequence[float], cyaw: Sequence[float], target_speed: float) -> List[float]:
    """Build a constant-speed profile for non-cooperating traffic."""
    _ = cy
    return make_base_speed_profile(cx, cyaw, target_speed)


def calc_speed_profile_ego(
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
    target_speed: float,
    state: State,
    state2: State,
    target_ind: int,
) -> List[float]:
    """Build the requester speed profile along its current local path."""
    _ = (cy, state, state2, target_ind)
    return make_base_speed_profile(cx, cyaw, target_speed)


def calc_speed_profile_accepting_vehicle(
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
    target_speed: float,
    state: State,
    current_state: int,
    speed_profile_state: int,
    target_ind: int,
    speed_delta_override: float | None = None,
) -> List[float]:
    """Build a cooperating vehicle profile, optionally using a selected speed delta."""
    _ = (cy, state, current_state, target_ind)
    base_profile = make_base_speed_profile(cx, cyaw, target_speed)
    if speed_delta_override is not None:
        return apply_speed_delta(base_profile, target_speed, speed_delta_override)
    return apply_speed_state(base_profile, target_speed, speed_profile_state)


def calc_speed_profile_downstream_accepting_vehicle(
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
    target_speed: float,
    state: State,
    current_state: int,
    speed_profile_state: int,
    target_ind: int,
    requester_target_speed: float,
    speed_delta_override: float | None = None,
) -> List[float]:
    """Build a cooperative speed profile for a downstream accepting vehicle."""
    _ = (cy, state, current_state, target_ind, requester_target_speed)
    base_profile = make_base_speed_profile(cx, cyaw, target_speed)
    if speed_delta_override is not None:
        return apply_speed_delta(base_profile, target_speed, speed_delta_override)
    return apply_speed_state(base_profile, target_speed, speed_profile_state)


# Backward-compatible alias for older scenario code that referred to vehicle 4.
calc_speed_profile_accepting_vehicle_4 = calc_speed_profile_downstream_accepting_vehicle
