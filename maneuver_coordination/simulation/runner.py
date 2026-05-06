"""Public simulation facade.

Scenario entry points import from this module so they do not need to know the
internal package layout. Keep this file thin: implementation should live in the
core, motion, coordination, visualization, and runners subpackages.
"""

from maneuver_coordination.simulation.coordination.coordination_flow import (
    build_acceptance_stage_context,
    build_requester_conflict_bundle,
    build_requester_conflict_sources,
    build_requester_local_paths,
    build_requester_path_inputs,
    build_requester_requested_trajectories,
    build_requester_speed_profiles,
    build_requester_stage_spec,
    build_role_reference_trajectories,
    find_conflicting_vehicle_roles,
    select_lowest_cost_role,
    select_secondary_requester_role,
)
from maneuver_coordination.simulation.coordination.messages import (
    build_received_request_context,
    build_request_message_candidates,
    emit_request_messages,
    emit_v2x_request_message,
    get_latest_v2x_request,
    get_message_priority,
    get_vehicle_inbox,
    has_v2x_request_message,
    select_request_message_for_receiver,
)
from maneuver_coordination.simulation.motion.paths import _build_planned_path, build_fallback_paths
from maneuver_coordination.simulation.coordination.roles import (
    build_acc_speed_profile,
    build_target_lane_reference_candidates,
    find_target_lane_neighbors,
    find_vehicle_ahead_in_lane,
    find_vehicle_behind_in_lane,
    get_requester_candidate_roles,
    infer_lane_id_from_y,
    is_vehicle_braking,
    should_trigger_maneuver_search,
)
from maneuver_coordination.simulation.runners.scenario_configs import (
    build_default_cascading_vehicle_configs,
    build_default_rejected_request_then_free_lane_configs,
    build_default_three_vehicle_coordination_4_messages_configs,
    build_default_three_vehicle_coordination_configs,
    build_default_two_vehicle_coordination_configs,
)
from maneuver_coordination.simulation.runners.scenario_runners import (
    run_cascading_scenario,
    run_multi_acceptor_three_vehicle_coordination_scenario,
    run_rejected_request_then_free_lane_scenario,
    run_three_vehicle_coordination_4_messages_scenario,
    run_three_vehicle_coordination_scenario,
    run_two_vehicle_coordination_scenario,
)
from maneuver_coordination.simulation.core.settings import ROAD_END_X
from maneuver_coordination.simulation.visualization import render_simulation_frame

__all__ = [
    "ROAD_END_X",
    "_build_planned_path",
    "build_acc_speed_profile",
    "build_acceptance_stage_context",
    "build_default_cascading_vehicle_configs",
    "build_default_rejected_request_then_free_lane_configs",
    "build_default_three_vehicle_coordination_4_messages_configs",
    "build_default_three_vehicle_coordination_configs",
    "build_default_two_vehicle_coordination_configs",
    "build_fallback_paths",
    "build_received_request_context",
    "build_request_message_candidates",
    "build_requester_conflict_bundle",
    "build_requester_conflict_sources",
    "build_requester_local_paths",
    "build_requester_path_inputs",
    "build_requester_requested_trajectories",
    "build_requester_speed_profiles",
    "build_requester_stage_spec",
    "build_role_reference_trajectories",
    "build_target_lane_reference_candidates",
    "emit_request_messages",
    "emit_v2x_request_message",
    "find_conflicting_vehicle_roles",
    "find_target_lane_neighbors",
    "find_vehicle_ahead_in_lane",
    "find_vehicle_behind_in_lane",
    "get_latest_v2x_request",
    "get_message_priority",
    "get_requester_candidate_roles",
    "get_vehicle_inbox",
    "has_v2x_request_message",
    "infer_lane_id_from_y",
    "is_vehicle_braking",
    "render_simulation_frame",
    "run_cascading_scenario",
    "run_multi_acceptor_three_vehicle_coordination_scenario",
    "run_rejected_request_then_free_lane_scenario",
    "run_three_vehicle_coordination_4_messages_scenario",
    "run_three_vehicle_coordination_scenario",
    "run_two_vehicle_coordination_scenario",
    "select_lowest_cost_role",
    "select_request_message_for_receiver",
    "select_secondary_requester_role",
    "should_trigger_maneuver_search",
]
