"""Helpers that assemble reusable coordination inputs for scenario runners.

Runner loops should describe the scenario story; this module builds common data
structures such as requester stage specs, requested trajectories, speed
profiles, conflict bundles, and responder reference trajectories.
"""

from typing import Dict, Sequence

import numpy as np

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination.conflicts import (
    is_sampled_trajectory_conflict_free,
    sampled_clearances,
)
from maneuver_coordination.coordination.constants import follow_planned, no_request_received, send_request
from maneuver_coordination.coordination.planner import BehaviouralLocalPlanner
from maneuver_coordination.simulation.motion.reference import calc_ref_trajectory
from maneuver_coordination.simulation.coordination.roles import (
    find_target_lane_neighbors,
    infer_lane_id_from_y,
)
from maneuver_coordination.simulation.motion.speed_profiles import (
    calc_speed_profile_accepting_vehicle,
    calc_speed_profile_ego,
)
from maneuver_coordination.simulation.core.types import PlannedPath, State, VehicleConfig


def build_requester_stage_data(xref, enabled, state):
    """Package one requester's requested trajectory and active flag."""
    return {
        "xref": xref,
        "enabled": enabled,
        "state": state,
    }


def build_requester_conflict_data(request_clear, accept_clear, accept_clear_new):
    """Package conflict-check booleans used by decision logic."""
    return {
        "request_clear": request_clear,
        "accept_clear": accept_clear,
        "accept_clear_new": accept_clear_new,
    }


def build_requester_conflict_sources(
    requester_role: str,
    request_clear: bool,
    accept_clear: bool,
    accept_clear_new: bool,
) -> Dict[str, tuple[bool, bool, bool]]:
    """Create the conflict-source map for one requester."""
    return {
        requester_role: (
            request_clear,
            accept_clear,
            accept_clear_new,
        )
    }


def build_requester_conflict_bundle(
    requester_role: str,
    requester_conflict_sources: Dict[str, tuple[bool, bool, bool]],
):
    """Fetch one requester's conflict data from the shared source map."""
    request_clear, accept_clear, accept_clear_new = requester_conflict_sources[requester_role]
    return build_requester_conflict_data(
        request_clear,
        accept_clear,
        accept_clear_new,
    )


def build_requester_stage_inputs(
    planner: BehaviouralLocalPlanner,
    requester_state: State,
    local_path_data,
    speed_profile,
    dl: float,
    motion_state: int,
    secondary: bool = False,
):
    """Build the requested trajectory payload for one requester stage."""
    local_path_x, local_path_y, local_path_yaw, _ = local_path_data
    xreq, _, _, req_enabled = (
        planner.calc_ref_req_trajectory_secondary(
            requester_state,
            local_path_x,
            local_path_y,
            local_path_yaw,
            speed_profile,
            dl,
            motion_state,
        )
        if secondary
        else planner.calc_ref_req_trajectory(
            requester_state,
            local_path_x,
            local_path_y,
            local_path_yaw,
            speed_profile,
            dl,
            motion_state,
        )
    )
    if (
        not req_enabled
        and motion_state in (C.find_lane_change, C.lane_change)
        and len(local_path_x) > 0
    ):
        xreq, _, _, req_enabled = planner._build_requested_trajectory(
            requester_state,
            (local_path_x, local_path_y, local_path_yaw, []),
            speed_profile,
            dl,
        )
    return build_requester_stage_data(xreq, req_enabled, requester_state)


def process_requester_stage(
    planner: BehaviouralLocalPlanner,
    coordination_by_requester: Dict[str, Dict[str, int]],
    requester_role: str,
    motion_state: int,
    requester_conflicts: Dict[str, Dict[str, bool]],
    requesting_priority: int,
    peer_accept_state: int | None = None,
    secondary: bool = False,
):
    """Update requester request/accept/speed state in one combined step."""
    request_state = (
        planner.requesting_vehicle_states_secondary(
            motion_state,
            coordination_by_requester[requester_role]["accept_state"],
            requester_conflicts[requester_role]["request_clear"],
            requesting_priority,
        )
        if secondary
        else planner.requesting_vehicle_states(
            motion_state,
            coordination_by_requester[requester_role]["accept_state"],
            requester_conflicts[requester_role]["request_clear"],
        )
    )
    coordination_by_requester[requester_role]["request_state"] = request_state

    accept_state, speed_state = (
        planner.accepting_vehicle_states_downstream(
            request_state,
            requesting_priority,
            requester_conflicts[requester_role]["accept_clear"],
            requester_conflicts[requester_role]["accept_clear_new"],
        )
        if secondary
        else planner.accepting_vehicle_states(
            request_state,
            requesting_priority,
            requester_conflicts[requester_role]["accept_clear"],
            requester_conflicts[requester_role]["accept_clear_new"],
            peer_accept_state,
        )
    )
    coordination_by_requester[requester_role]["accept_state"] = accept_state
    coordination_by_requester[requester_role]["speed_state"] = speed_state


def update_requester_request_state(
    planner: BehaviouralLocalPlanner,
    coordination_by_requester: Dict[str, Dict[str, int]],
    requester_role: str,
    motion_state: int,
    requester_conflicts: Dict[str, Dict[str, bool]],
    requesting_priority: int,
    secondary: bool = False,
):
    """Update and return only the requester-side message state."""
    request_state = (
        planner.requesting_vehicle_states_secondary(
            motion_state,
            coordination_by_requester[requester_role]["accept_state"],
            requester_conflicts[requester_role]["request_clear"],
            requesting_priority,
        )
        if secondary
        else planner.requesting_vehicle_states(
            motion_state,
            coordination_by_requester[requester_role]["accept_state"],
            requester_conflicts[requester_role]["request_clear"],
        )
    )
    coordination_by_requester[requester_role]["request_state"] = request_state
    return request_state


def update_requester_acceptance_state(
    planner: BehaviouralLocalPlanner,
    coordination_by_requester: Dict[str, Dict[str, int]],
    requester_role: str,
    requester_conflicts: Dict[str, Dict[str, bool]],
    requesting_priority: int,
    request_delivered: bool,
    peer_accept_state: int | None = None,
    secondary: bool = False,
):
    """Update and return the requester acceptance and speed-adaptation state."""
    request_state = coordination_by_requester[requester_role]["request_state"]

    if request_state == send_request and not request_delivered:
        coordination_by_requester[requester_role]["accept_state"] = no_request_received
        coordination_by_requester[requester_role]["speed_state"] = follow_planned
        return no_request_received, follow_planned

    accept_state, speed_state = (
        planner.accepting_vehicle_states_downstream(
            request_state,
            requesting_priority,
            requester_conflicts[requester_role]["accept_clear"],
            requester_conflicts[requester_role]["accept_clear_new"],
        )
        if secondary
        else planner.accepting_vehicle_states(
            request_state,
            requesting_priority,
            requester_conflicts[requester_role]["accept_clear"],
            requester_conflicts[requester_role]["accept_clear_new"],
            peer_accept_state,
        )
    )
    coordination_by_requester[requester_role]["accept_state"] = accept_state
    coordination_by_requester[requester_role]["speed_state"] = speed_state
    return accept_state, speed_state


def find_conflicting_vehicle_roles(
    requester_role: str,
    xreq,
    reference_trajectories: Dict[str, np.ndarray],
    states_by_role: Dict[str, State],
    vehicle_configs: Sequence[VehicleConfig],
):
    """Find target-lane vehicles whose planned trajectories conflict with a request."""
    config_by_role = {config.role: config for config in vehicle_configs}
    requester_config = config_by_role.get(requester_role)
    conflicting_roles = []

    for role, reference in reference_trajectories.items():
        if role == requester_role or role not in states_by_role:
            continue

        candidate_config = config_by_role.get(role)
        if requester_config is not None and candidate_config is not None:
            candidate_lane_id = infer_lane_id_from_y(states_by_role[role].y, vehicle_configs)
            if requester_config.target_lane_id is not None and candidate_lane_id is not None:
                if requester_config.target_lane_id != candidate_lane_id:
                    continue

        if not np.any(np.abs(reference[:2, :]) > 1e-9):
            continue

        clearances = sampled_clearances(
            xreq,
            reference,
            states_by_role[role].v,
            first_offset=2.0,
            later_offset=4.0,
        )
        distance_gap = 1.0 * states_by_role[role].v
        if not is_sampled_trajectory_conflict_free(clearances, distance_gap, initial_gap=1.0):
            conflicting_roles.append(role)

    return conflicting_roles


def select_lowest_cost_role(conflicting_roles: Sequence[str], vehicle_configs: Sequence[VehicleConfig], fallback_role: str):
    """Choose the cheapest cooperating role from a conflict set."""
    if not conflicting_roles:
        return fallback_role

    config_by_role = {config.role: config for config in vehicle_configs}
    return min(
        conflicting_roles,
        key=lambda role: config_by_role[role].cooperation_cost if role in config_by_role else float("inf"),
    )


def choose_cooperating_role(
    requester_role: str,
    conflicting_roles: Sequence[str],
    states_by_role: Dict[str, State],
    vehicle_configs: Sequence[VehicleConfig],
    fallback_role: str,
) -> str:
    """Prefer target-lane neighbors, then select the lowest-cost cooperator."""
    if not conflicting_roles:
        return fallback_role

    neighbors = find_target_lane_neighbors(requester_role, states_by_role, vehicle_configs)
    ordered_candidates = [role for role in (neighbors["rear"], neighbors["front"]) if role in conflicting_roles]
    for role in conflicting_roles:
        if role not in ordered_candidates:
            ordered_candidates.append(role)

    return select_lowest_cost_role(ordered_candidates, vehicle_configs, fallback_role)


def select_secondary_requester_role(
    primary_requester_role: str,
    requester_candidate_roles: Sequence[str],
    states_by_role: Dict[str, State],
    vehicle_configs: Sequence[VehicleConfig],
    fallback_role: str,
) -> str:
    """Choose the target-lane vehicle that should become the cascade requester."""
    secondary_candidates = [
        role for role in requester_candidate_roles
        if role != primary_requester_role
    ]
    if not secondary_candidates:
        return fallback_role

    neighbors = find_target_lane_neighbors(primary_requester_role, states_by_role, vehicle_configs)
    for role in (neighbors["front"], neighbors["rear"]):
        if role in secondary_candidates:
            return role

    return secondary_candidates[0]


def build_requester_stage_spec(
    requester_role: str,
    motion_state: int,
    receiver_role: str,
    stage_label: str,
    secondary: bool = False,
):
    """Describe one requester-to-receiver coordination stage."""
    return {
        "requester_role": requester_role,
        "motion_state": motion_state,
        "receiver_role": receiver_role,
        "stage_label": stage_label,
        "secondary": secondary,
    }


def build_requester_requested_trajectories(
    planner: BehaviouralLocalPlanner,
    requester_stage_specs: Sequence[Dict[str, object]],
    requester_states: Dict[str, State],
    requester_local_paths: Dict[str, tuple],
    requester_speed_profiles: Dict[str, Sequence[float]],
    dl: float,
):
    """Build requested trajectories for all active requester stages."""
    requested_trajectories = {}
    for stage_spec in requester_stage_specs:
        requester_role = str(stage_spec["requester_role"])
        requested_trajectories[requester_role] = build_requester_stage_inputs(
            planner,
            requester_states[requester_role],
            requester_local_paths[requester_role],
            requester_speed_profiles[requester_role],
            dl,
            int(stage_spec["motion_state"]),
            secondary=bool(stage_spec["secondary"]),
        )
    return requested_trajectories


def build_requester_local_paths(
    planner: BehaviouralLocalPlanner,
    requester_stage_specs: Sequence[Dict[str, object]],
    requester_path_inputs: Dict[str, Dict[str, PlannedPath]],
    requester_states: Dict[str, State],
    obstacle_list,
    coordination_by_requester: Dict[str, Dict[str, int]],
):
    """Resolve local paths for all requester stages in a scenario step."""
    local_paths = {}
    for stage_spec in requester_stage_specs:
        requester_role = str(stage_spec["requester_role"])
        path_inputs = requester_path_inputs[requester_role]
        request_state = coordination_by_requester[requester_role]["accept_state"]
        state = requester_states[requester_role]
        motion_state = int(stage_spec["motion_state"])

        if bool(stage_spec["secondary"]):
            local_path_x, local_path_y, local_path_yaw, local_path_k, _ = planner.local_path_secondary(
                path_inputs["base"].x,
                path_inputs["base"].y,
                path_inputs["base"].yaw,
                path_inputs["base"].curvature,
                path_inputs["alt"].x,
                path_inputs["alt"].y,
                path_inputs["alt"].yaw,
                path_inputs["alt"].curvature,
                motion_state,
                state,
                obstacle_list,
                request_state,
            )
        else:
            local_path_x, local_path_y, local_path_yaw, local_path_k, _ = planner.local_path(
                path_inputs["base"].x,
                path_inputs["base"].y,
                path_inputs["base"].yaw,
                path_inputs["base"].curvature,
                path_inputs["alt"].x,
                path_inputs["alt"].y,
                path_inputs["alt"].yaw,
                path_inputs["alt"].curvature,
                motion_state,
                state,
                obstacle_list,
                request_state,
            )

        local_paths[requester_role] = (local_path_x, local_path_y, local_path_yaw, local_path_k)

    return local_paths


def build_requester_speed_profiles(
    requester_stage_specs: Sequence[Dict[str, object]],
    requester_local_paths: Dict[str, tuple],
    requester_states: Dict[str, State],
    requester_target_speeds: Dict[str, float],
    histories_by_role: Dict[str, object],
    coordination_by_requester: Dict[str, Dict[str, int]],
    lead_state: State,
):
    """Build speed profiles for primary and secondary requesters."""
    speed_profiles = {}
    for stage_spec in requester_stage_specs:
        requester_role = str(stage_spec["requester_role"])
        local_path_x, local_path_y, local_path_yaw, _ = requester_local_paths[requester_role]
        requester_state = requester_states[requester_role]
        target_speed = requester_target_speeds[requester_role]
        target_ind = histories_by_role[requester_role].target_ind

        if bool(stage_spec["secondary"]):
            speed_profiles[requester_role] = calc_speed_profile_accepting_vehicle(
                local_path_x,
                local_path_y,
                local_path_yaw,
                target_speed,
                requester_state,
                int(stage_spec["motion_state"]),
                coordination_by_requester[requester_role]["speed_state"],
                target_ind,
                coordination_by_requester[requester_role].get("speed_delta"),
            )
        else:
            speed_profiles[requester_role] = calc_speed_profile_ego(
                local_path_x,
                local_path_y,
                local_path_yaw,
                target_speed,
                requester_state,
                lead_state,
                target_ind,
            )

    return speed_profiles


def build_requester_path_inputs(
    requester_roles: Sequence[str],
    config_by_role: Dict[str, VehicleConfig],
    paths_by_role: Dict[str, PlannedPath],
) -> Dict[str, Dict[str, PlannedPath]]:
    """Resolve base and alternate paths for each requester role."""
    path_inputs: Dict[str, Dict[str, PlannedPath]] = {}
    for requester_role in requester_roles:
        base_path = paths_by_role[requester_role]
        config = config_by_role[requester_role]
        alt_path = paths_by_role.get(config.fallback_path_key, base_path)
        path_inputs[requester_role] = {
            "base": base_path,
            "alt": alt_path,
        }
    return path_inputs


def build_role_reference_trajectories(
    role_inputs: Dict[str, Dict[str, object]],
    histories_by_role: Dict[str, object],
    dl: float,
) -> Dict[str, np.ndarray]:
    """Generate planned reference trajectories for responder/conflict checks."""
    references: Dict[str, np.ndarray] = {}
    for role, inputs in role_inputs.items():
        reference, histories_by_role[role].target_ind, _ = calc_ref_trajectory(
            inputs["state"],
            inputs["path_x"],
            inputs["path_y"],
            inputs["path_yaw"],
            inputs["path_k"],
            inputs["speed_profile"],
            dl,
        )
        references[role] = reference
    return references


def build_acceptance_stage_context(
    requester_role: str,
    requester_conflicts: Dict[str, bool],
    priority: int | None,
    delivered: bool,
    secondary: bool = False,
    peer_accept_state: int | None = None,
):
    """Package the context needed to evaluate an acceptance stage."""
    return {
        "requester_role": requester_role,
        "requester_conflicts": requester_conflicts,
        "priority": priority,
        "delivered": delivered,
        "secondary": secondary,
        "peer_accept_state": peer_accept_state,
    }
