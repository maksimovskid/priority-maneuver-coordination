"""Candidate speed/acceleration adaptation helpers for cooperating vehicles."""

from typing import Dict, List, Sequence

import numpy as np

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination.conflicts import (
    is_sampled_trajectory_conflict_free,
    sampled_clearances,
)
from maneuver_coordination.simulation.motion.reference import (
    build_reference_for_speed,
    build_reference_for_target_speed,
)
from maneuver_coordination.simulation.core.settings import (
    ADAPTATION_ACCEL_LIMIT_BY_PRIORITY,
    ADAPTATION_ACCEL_STEP,
    ADAPTATION_SPEED_LIMIT_RATIO_BY_PRIORITY,
    ADAPTATION_SPEED_STEP_RATIO,
)
from maneuver_coordination.simulation.core.types import PlannedPath, State


def build_candidate_refs_for_speed_offsets(
    state: State,
    path: PlannedPath,
    dl: float,
    speed_offsets: Sequence[float],
    use_abs_diff: bool,
) -> List[np.ndarray]:
    """Build candidate reference trajectories for fixed speed offsets."""
    refs: List[np.ndarray] = []
    for speed_offset in speed_offsets:
        speed_profile = [state.v - speed_offset] * len(path.x)
        xref, _, _ = build_reference_for_speed(
            state,
            path.x,
            path.y,
            path.yaw,
            speed_profile,
            dl,
            speed_offset,
            use_abs_diff,
        )
        refs.append(xref)
    return refs


def priority_speed_ratio_candidates(priority: int) -> List[float]:
    """Return 5-percent speed-change candidates allowed by priority."""
    limit = ADAPTATION_SPEED_LIMIT_RATIO_BY_PRIORITY.get(priority, ADAPTATION_SPEED_STEP_RATIO)
    steps = max(1, int(round(limit / ADAPTATION_SPEED_STEP_RATIO)))
    return [ADAPTATION_SPEED_STEP_RATIO * step for step in range(1, steps + 1)]


def acceleration_limit_for_ratio(priority: int, ratio: float, *, increase: bool) -> float:
    """Map a speed-change ratio to the matching acceleration limit."""
    accel_limit = ADAPTATION_ACCEL_LIMIT_BY_PRIORITY.get(priority, ADAPTATION_ACCEL_STEP)
    ratio_steps = max(1, int(round(ratio / ADAPTATION_SPEED_STEP_RATIO)))
    selected_limit = min(accel_limit, ratio_steps * ADAPTATION_ACCEL_STEP)
    return selected_limit if increase else -selected_limit


def build_candidate_refs_for_accelerations(
    state: State,
    path: PlannedPath,
    dl: float,
    priority: int,
    *,
    increase: bool,
) -> List[Dict[str, object]]:
    """Generate candidate trajectories by stepping speed change within priority limits."""
    candidates: List[Dict[str, object]] = []
    base_target_speed = path.speed_profile[0] if path.speed_profile else state.v
    for ratio in priority_speed_ratio_candidates(priority):
        speed_delta = base_target_speed * ratio * (1.0 if increase else -1.0)
        target_speed = max(0.0, base_target_speed + speed_delta)
        xref, _, _ = build_reference_for_target_speed(
            state,
            path.x,
            path.y,
            path.yaw,
            dl,
            target_speed,
        )
        candidates.append(
            {
                "xref": xref,
                "acceleration": acceleration_limit_for_ratio(priority, ratio, increase=increase),
                "final_speed": target_speed,
                "speed_delta": speed_delta,
            }
        )
    return candidates


def select_first_feasible_reduction_candidate(
    requester_state: State,
    xreq: np.ndarray,
    req_traj: bool,
    candidate_refs: Sequence[Dict[str, object]],
) -> Dict[str, object] | None:
    """Select the first deceleration candidate that avoids the requested trajectory."""
    if req_traj is not True:
        return None

    for candidate in candidate_refs:
        clearances = sampled_clearances(xreq, candidate["xref"], requester_state.v)
        if is_sampled_trajectory_conflict_free(clearances, max(0.0, float(candidate["final_speed"]))):
            return candidate
    return None


def select_first_feasible_front_candidate(
    responder_state: State,
    xreq: np.ndarray,
    req_traj: bool,
    candidate_refs: Sequence[Dict[str, object]],
) -> Dict[str, object] | None:
    """Select the first acceleration candidate that avoids the request."""
    if req_traj is not True:
        return None

    for candidate in candidate_refs:
        clearances = sampled_clearances(xreq, candidate["xref"], responder_state.v)
        if is_sampled_trajectory_conflict_free(clearances, max(0.0, float(candidate["final_speed"]))):
            return candidate
    return None


def select_priority_candidate_ref(priority: int, candidate_refs: Sequence[np.ndarray]) -> np.ndarray:
    """Pick one legacy low/medium/high candidate reference by priority."""
    index_by_priority = {
        C.low_priority: 0,
        C.medium_priority: 1,
        C.high_priority: 2,
    }
    return candidate_refs[index_by_priority.get(priority, 0)]


def speed_state_for_priority(priority: int, *, increase: bool) -> int:
    """Translate request priority into a discrete speed adaptation state."""
    if increase:
        mapping = {
            C.low_priority: C.increase_speed_for_req_low,
            C.medium_priority: C.increase_speed_for_req_medium,
            C.high_priority: C.increase_speed_for_req_high,
        }
    else:
        mapping = {
            C.low_priority: C.reduce_speed_for_req_low,
            C.medium_priority: C.reduce_speed_for_req_medium,
            C.high_priority: C.reduce_speed_for_req_high,
        }
    return mapping.get(priority, C.follow_planned)


def cooperative_acceleration_bounds(selected_acceleration: float | None) -> tuple[float, float]:
    """Constrain the controller to the selected cooperative acceleration side."""
    if selected_acceleration is None:
        return -4.0, 4.0
    if selected_acceleration < 0.0:
        return selected_acceleration, 4.0
    if selected_acceleration > 0.0:
        return -4.0, selected_acceleration
    return -4.0, 4.0


def latched_candidate_value(
    current_value: float | None,
    accept_state: int,
    selected_candidate: Dict[str, object] | None,
    key: str,
) -> float | None:
    """Keep a selected candidate value stable after a request is accepted."""
    if accept_state != C.request_accepted:
        return None
    if current_value is not None:
        return current_value
    if selected_candidate is None:
        return None
    return float(selected_candidate[key])
