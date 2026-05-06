"""Closed-loop runner for the cascading coordination scenario.

This scenario models a chain reaction: the ego vehicle requests help from a
target-lane vehicle, and that target-lane vehicle can become a secondary
requester to create space farther downstream.
"""

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt

from maneuver_coordination.coordination.constants import (
    find_lane_change,
    follow_lane,
    follow_planned,
    no_request_received,
    no_request_sent,
)
from maneuver_coordination.coordination.planner import BehaviouralLocalPlanner
from maneuver_coordination.simulation.motion.controllers import pid_control, rear_wheel_feedback_control
from maneuver_coordination.simulation.motion.vehicle_dynamics import update
from maneuver_coordination.simulation.core.history import append_history, init_vehicle_history
from maneuver_coordination.simulation.core.math_helpers import reached_goal
from maneuver_coordination.simulation.motion.speed_profiles import calc_speed_profile_downstream_accepting_vehicle
from maneuver_coordination.simulation.core.types import State, V2XMessage, VehicleConfig
from maneuver_coordination.vehicle.plotting import connect_escape_key

from maneuver_coordination.simulation.motion.adaptation import (
    build_candidate_refs_for_accelerations,
    cooperative_acceleration_bounds,
    latched_candidate_value,
    select_first_feasible_front_candidate,
    select_first_feasible_reduction_candidate,
)
from maneuver_coordination.simulation.coordination.coordination_flow import (
    build_acceptance_stage_context,
    build_requester_conflict_data,
    build_requester_conflict_sources,
    build_requester_local_paths,
    build_requester_path_inputs,
    build_requester_requested_trajectories,
    build_requester_speed_profiles,
    build_requester_stage_spec,
    build_role_reference_trajectories,
    choose_cooperating_role,
    find_conflicting_vehicle_roles,
    select_secondary_requester_role,
    update_requester_acceptance_state,
    update_requester_request_state,
)
from maneuver_coordination.simulation.coordination.events import (
    append_simulation_event,
    format_event_time,
    get_active_receiver_roles,
    log_requester_state_changes,
)
from maneuver_coordination.simulation.coordination.messages import (
    build_received_request_context,
    build_request_message_candidates,
    emit_request_messages,
    get_config_by_role,
    get_vehicle_inbox,
)
from maneuver_coordination.simulation.motion.paths import build_paths_for_configs
from maneuver_coordination.simulation.coordination.roles import (
    build_acc_speed_profile,
    build_target_lane_reference_candidates,
    detect_vehicle_collisions,
    find_vehicle_ahead_in_lane,
    get_requester_candidate_roles,
    get_role_item,
    get_role_to_index,
    require_roles,
    should_trigger_maneuver_search,
)
from maneuver_coordination.simulation.core.settings import EGO_STOP_MARGIN, ROAD_END_X
from maneuver_coordination.simulation.visualization import (
    capture_simulation_frame,
    get_requested_plot_xref,
    render_simulation_frame,
    update_requested_plot_cache,
)


def build_local_path_reference_input(state, local_path, speed_profile):
    """Package a role's local path and speed profile for reference generation."""
    path_x, path_y, path_yaw, path_k = local_path
    return {
        "state": state,
        "path_x": path_x,
        "path_y": path_y,
        "path_yaw": path_yaw,
        "path_k": path_k,
        "speed_profile": speed_profile,
    }


def run_cascading_scenario(
    vehicle_configs: Sequence[VehicleConfig],
    ox: Sequence[float],
    oy: Sequence[float],
    obstacle_list,
    show_animation: bool = True,
    verbose_events: bool = True,
):
    """Run the cascading scenario and return histories, messages, events, and frames."""
    dl = 0.1
    sim_t = 500.0
    goal_dis = 1.0
    stop_speed = 0.05

    paths_by_role = build_paths_for_configs(vehicle_configs, ox, oy)
    role_to_index = get_role_to_index(vehicle_configs)
    config_by_role = get_config_by_role(vehicle_configs)
    require_roles(role_to_index, ["ego", "lead", "acceptor_1", "acceptor_2", "non_cooperating_vehicle"])
    requester_roles = get_requester_candidate_roles(vehicle_configs)
    primary_requester_role = requester_roles[0]
    default_secondary_requester_role = requester_roles[1] if len(requester_roles) > 1 else "acceptor_1"

    states = [
        State(
            x=paths_by_role[config.role].x[0],
            y=paths_by_role[config.role].y[0],
            yaw=paths_by_role[config.role].yaw[0],
            v=config.initial_speed,
        )
        for config in vehicle_configs
    ]
    histories = [
        init_vehicle_history(state, paths_by_role[config.role].x, paths_by_role[config.role].y, paths_by_role[config.role].yaw)
        for state, config in zip(states, vehicle_configs)
    ]
    steers = [0.0 for _ in vehicle_configs]

    planner = BehaviouralLocalPlanner(
        paths_by_role["ego"].x,
        paths_by_role["ego"].y,
        paths_by_role["ego"].yaw,
        paths_by_role["ego"].curvature,
    )

    coordination_by_requester = {
        primary_requester_role: {
            "accept_state": no_request_received,
            "speed_state": follow_planned,
            "speed_delta": None,
            "selected_acceleration": None,
            "request_state": no_request_sent,
            "motion_state": follow_lane,
        },
        default_secondary_requester_role: {
            "accept_state": no_request_received,
            "speed_state": follow_planned,
            "speed_delta": None,
            "selected_acceleration": None,
            "request_state": no_request_sent,
            "motion_state": follow_lane,
        },
    }
    time = 0.0
    message_log: List[V2XMessage] = []
    event_log: List[str] = []
    vehicle_inboxes: Dict[int, List[V2XMessage]] = {config.vehicle_id: [] for config in vehicle_configs}
    last_coordination_snapshot: Dict[str, Dict[str, int]] = {}
    braking_trigger_logged = False
    active_request_roles: set[str] = set()
    requested_plot_cache: Dict[str, Dict[str, object]] = {}
    frame_log: List[Dict[str, object]] = []
    frame_log: List[Dict[str, object]] = []
    coordination_completed = False
    coordination_started = False
    downstream_speed_delta: float | None = None
    downstream_acceleration: float | None = None

    while sim_t >= time:
        states_by_role = {config.role: state for config, state in zip(vehicle_configs, states)}
        secondary_requester_role = select_secondary_requester_role(
            primary_requester_role,
            requester_roles,
            states_by_role,
            vehicle_configs,
            default_secondary_requester_role,
        )
        if secondary_requester_role not in coordination_by_requester:
            coordination_by_requester[secondary_requester_role] = {
                "accept_state": no_request_received,
                "speed_state": follow_planned,
                "speed_delta": None,
                "selected_acceleration": None,
                "request_state": no_request_sent,
                "motion_state": follow_lane,
            }
        ego_state = states_by_role["ego"]
        secondary_requester_state = states_by_role[secondary_requester_role]
        lead_role = find_vehicle_ahead_in_lane("ego", states_by_role, vehicle_configs)
        lead_state = states_by_role[lead_role] if lead_role is not None else states_by_role["ego"]
        lead_history = get_role_item(histories, role_to_index, lead_role) if lead_role is not None else None
        acceptor_2_state = get_role_item(states, role_to_index, "acceptor_2")

        primary_motion_state = planner.transition_state(
            ego_state,
            lead_state,
            coordination_by_requester[primary_requester_role]["request_state"],
        )
        if should_trigger_maneuver_search(
            ego_state,
            lead_state,
            lead_history,
            existing_path_length=planner.generated_path_length(),
        ):
            primary_motion_state = find_lane_change
            if not braking_trigger_logged and lead_role is not None:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time)}: coordination need detected, ID {config_by_role[primary_requester_role].vehicle_id} sees vehicle ahead ID {config_by_role[lead_role].vehicle_id} decelerating",
                    echo=verbose_events,
                )
                braking_trigger_logged = True
        secondary_motion_state = planner.transition_state_secondary(
            secondary_requester_state,
            coordination_by_requester[primary_requester_role]["request_state"],
            coordination_by_requester[secondary_requester_role]["request_state"],
        )
        coordination_by_requester[primary_requester_role]["motion_state"] = primary_motion_state
        coordination_by_requester[secondary_requester_role]["motion_state"] = secondary_motion_state

        ego_path = paths_by_role["ego"]
        secondary_requester_path = paths_by_role[secondary_requester_role]
        acceptor_2_path = paths_by_role["acceptor_2"]
        lead_path = paths_by_role["lead"]
        non_cooperating_path = paths_by_role["non_cooperating_vehicle"]
        requester_states = {
            primary_requester_role: ego_state,
            secondary_requester_role: secondary_requester_state,
        }
        requester_stage_specs = [
            build_requester_stage_spec(
                primary_requester_role,
                primary_motion_state,
                "",
                "primary",
                secondary=False,
            ),
            build_requester_stage_spec(
                secondary_requester_role,
                secondary_motion_state,
                "",
                "secondary",
                secondary=True,
            ),
        ]
        requester_path_inputs = build_requester_path_inputs(
            [primary_requester_role, secondary_requester_role],
            config_by_role,
            paths_by_role,
        )
        requester_local_paths = build_requester_local_paths(
            planner,
            requester_stage_specs,
            requester_path_inputs,
            requester_states,
            obstacle_list,
            coordination_by_requester,
        )
        primary_requester_local_path = requester_local_paths[primary_requester_role]
        secondary_requester_local_path = requester_local_paths[secondary_requester_role]
        histories_by_role = {
            primary_requester_role: get_role_item(histories, role_to_index, "ego"),
            secondary_requester_role: get_role_item(histories, role_to_index, secondary_requester_role),
        }
        requester_target_speeds = {
            primary_requester_role: get_role_item(vehicle_configs, role_to_index, "ego").target_speed,
            secondary_requester_role: get_role_item(vehicle_configs, role_to_index, secondary_requester_role).target_speed,
        }
        requester_speed_profiles = build_requester_speed_profiles(
            requester_stage_specs,
            requester_local_paths,
            requester_states,
            requester_target_speeds,
            histories_by_role,
            coordination_by_requester,
            lead_state,
        )
        speed_profile_ego = requester_speed_profiles[primary_requester_role]
        speed_profile_accepting_vehicle = requester_speed_profiles[secondary_requester_role]
        downstream_accepting_speed_profile = calc_speed_profile_downstream_accepting_vehicle(
            acceptor_2_path.x,
            acceptor_2_path.y,
            acceptor_2_path.yaw,
            get_role_item(vehicle_configs, role_to_index, "acceptor_2").target_speed,
            acceptor_2_state,
            primary_motion_state,
            coordination_by_requester[secondary_requester_role]["speed_state"],
            get_role_item(histories, role_to_index, "acceptor_2").target_ind,
            get_role_item(vehicle_configs, role_to_index, "ego").target_speed,
            downstream_speed_delta,
        )
        non_cooperating_state = get_role_item(states, role_to_index, "non_cooperating_vehicle")
        non_cooperating_lead_role = find_vehicle_ahead_in_lane(
            "non_cooperating_vehicle",
            states_by_role,
            vehicle_configs,
        )
        non_cooperating_lead_state = (
            states_by_role[non_cooperating_lead_role]
            if non_cooperating_lead_role is not None
            else None
        )
        non_cooperating_speed_profile = build_acc_speed_profile(
            non_cooperating_path,
            non_cooperating_state,
            get_role_item(vehicle_configs, role_to_index, "non_cooperating_vehicle").target_speed,
            non_cooperating_lead_state,
        )

        requester_requested_trajectories = build_requester_requested_trajectories(
            planner,
            requester_stage_specs,
            requester_states,
            requester_local_paths,
            requester_speed_profiles,
            dl,
        )
        xreq = requester_requested_trajectories[primary_requester_role]["xref"]
        req_traj = requester_requested_trajectories[primary_requester_role]["enabled"]
        xreq_cascading = requester_requested_trajectories[secondary_requester_role]["xref"]
        req_traj_cascading = requester_requested_trajectories[secondary_requester_role]["enabled"]
        update_requested_plot_cache(
            requested_plot_cache,
            primary_requester_role,
            xreq,
            req_traj,
            coordination_by_requester[primary_requester_role]["request_state"],
            coordination_by_requester[primary_requester_role]["motion_state"],
        )
        update_requested_plot_cache(
            requested_plot_cache,
            secondary_requester_role,
            xreq_cascading,
            req_traj_cascading,
            coordination_by_requester[secondary_requester_role]["request_state"],
            coordination_by_requester[secondary_requester_role]["motion_state"],
        )

        responder_histories = {
            secondary_requester_role: get_role_item(histories, role_to_index, secondary_requester_role),
            "acceptor_2": get_role_item(histories, role_to_index, "acceptor_2"),
            "non_cooperating_vehicle": get_role_item(histories, role_to_index, "non_cooperating_vehicle"),
        }
        responder_reference_inputs = {
            secondary_requester_role: build_local_path_reference_input(
                secondary_requester_state,
                secondary_requester_local_path,
                speed_profile_accepting_vehicle,
            ),
            "acceptor_2": {
                "state": acceptor_2_state,
                "path_x": acceptor_2_path.x,
                "path_y": acceptor_2_path.y,
                "path_yaw": acceptor_2_path.yaw,
                "path_k": acceptor_2_path.curvature,
                "speed_profile": downstream_accepting_speed_profile,
            },
            "non_cooperating_vehicle": {
                "state": non_cooperating_state,
                "path_x": non_cooperating_path.x,
                "path_y": non_cooperating_path.y,
                "path_yaw": non_cooperating_path.yaw,
                "path_k": non_cooperating_path.curvature,
                "speed_profile": non_cooperating_speed_profile,
            },
        }
        responder_references = build_role_reference_trajectories(
            responder_reference_inputs,
            responder_histories,
            dl,
        )
        secondary_requester_reference = responder_references[secondary_requester_role]
        xref_acceptor_2 = responder_references["acceptor_2"]
        xref_non_cooperating_vehicle = responder_references["non_cooperating_vehicle"]

        primary_reference_candidates = build_target_lane_reference_candidates(
            primary_requester_role,
            {secondary_requester_role: secondary_requester_reference},
            states_by_role,
            vehicle_configs,
        )
        ego_conflicting_roles = find_conflicting_vehicle_roles(
            primary_requester_role,
            xreq,
            primary_reference_candidates,
            states_by_role,
            vehicle_configs,
        )
        ego_cooperating_role = choose_cooperating_role(
            primary_requester_role,
            ego_conflicting_roles,
            states_by_role,
            vehicle_configs,
            secondary_requester_role,
        )
        first_cooperating_state = get_role_item(states, role_to_index, ego_cooperating_role)
        first_cooperating_reference = primary_reference_candidates[ego_cooperating_role]

        secondary_reference_candidates = build_target_lane_reference_candidates(
            secondary_requester_role,
            {"acceptor_2": xref_acceptor_2},
            states_by_role,
            vehicle_configs,
        )
        cascading_conflicting_roles = find_conflicting_vehicle_roles(
            secondary_requester_role,
            xreq_cascading,
            secondary_reference_candidates,
            states_by_role,
            vehicle_configs,
        )
        downstream_cooperating_role = choose_cooperating_role(
            secondary_requester_role,
            cascading_conflicting_roles,
            states_by_role,
            vehicle_configs,
            "acceptor_2",
        )
        downstream_cooperating_state = get_role_item(states, role_to_index, downstream_cooperating_role)
        downstream_cooperating_reference = secondary_reference_candidates[downstream_cooperating_role]

        requesting_priority = planner.calc_priority(ego_state, lead_state)
        primary_candidate_refs = build_candidate_refs_for_accelerations(
            first_cooperating_state,
            paths_by_role[ego_cooperating_role],
            dl,
            requesting_priority,
            increase=False,
        )
        downstream_candidate_refs = build_candidate_refs_for_accelerations(
            downstream_cooperating_state,
            paths_by_role[downstream_cooperating_role],
            dl,
            requesting_priority,
            increase=False,
        )
        conflict_free_req = planner.check_req_tr_conflict(
            ego_state,
            first_cooperating_state,
            xreq,
            first_cooperating_reference,
            req_traj,
        )
        secondary_conflict_free_request = planner.check_req_tr_conflict_4(
            downstream_cooperating_state,
            xreq_cascading,
            downstream_cooperating_reference,
            req_traj_cascading,
        )
        conflict_free = planner.check_req_tr_conflict(
            ego_state,
            downstream_cooperating_state,
            xreq,
            first_cooperating_reference,
            req_traj,
        )
        secondary_conflict_free = planner.check_req_tr_conflict_4(
            downstream_cooperating_state,
            xreq_cascading,
            downstream_cooperating_reference,
            req_traj_cascading,
        )
        selected_primary_candidate = select_first_feasible_reduction_candidate(
            ego_state,
            xreq,
            req_traj,
            primary_candidate_refs,
        )
        conflict_free_new = selected_primary_candidate is not None
        selected_downstream_candidate = select_first_feasible_front_candidate(
            downstream_cooperating_state,
            xreq_cascading,
            req_traj_cascading,
            downstream_candidate_refs,
        )
        secondary_conflict_free_new = selected_downstream_candidate is not None
        requester_conflicts = {
            primary_requester_role: build_requester_conflict_data(
                conflict_free_req,
                conflict_free,
                conflict_free_new,
            ),
            secondary_requester_role: build_requester_conflict_data(
                secondary_conflict_free_request,
                secondary_conflict_free,
                secondary_conflict_free_new,
            ),
        }
        requester_conflict_sources = {}
        requester_conflict_sources.update(
            build_requester_conflict_sources(
                primary_requester_role,
                conflict_free_req,
                conflict_free,
                conflict_free_new,
            )
        )
        requester_conflict_sources.update(
            build_requester_conflict_sources(
                secondary_requester_role,
                secondary_conflict_free_request,
                secondary_conflict_free,
                secondary_conflict_free_new,
            )
        )
        for stage_spec in requester_stage_specs:
            update_requester_request_state(
                planner,
                coordination_by_requester,
                stage_spec["requester_role"],
                stage_spec["motion_state"],
                requester_conflicts,
                requesting_priority,
                secondary=bool(stage_spec["secondary"]),
            )

        message_stage_specs = [
            build_requester_stage_spec(
                primary_requester_role,
                primary_motion_state,
                ego_cooperating_role,
                "primary",
                secondary=False,
            ),
            build_requester_stage_spec(
                secondary_requester_role,
                secondary_motion_state,
                downstream_cooperating_role,
                "secondary",
                secondary=True,
            ),
        ]
        active_receiver_roles = get_active_receiver_roles(
            message_stage_specs,
            coordination_by_requester,
        )

        request_candidates = [
            build_request_message_candidates(
                stage_spec["requester_role"],
                stage_spec["receiver_role"],
                coordination_by_requester[stage_spec["requester_role"]]["request_state"],
                requesting_priority,
                stage_spec["stage_label"],
            )
            for stage_spec in message_stage_specs
        ]
        step_messages = emit_request_messages(time, request_candidates, config_by_role)
        message_log.extend(step_messages)
        for message in step_messages:
            append_simulation_event(
                event_log,
                f"{format_event_time(time)}: ID {message.sender_id} -> ID {message.receiver_id} request sent",
                echo=verbose_events,
            )
        vehicle_inboxes = {
            config.vehicle_id: get_vehicle_inbox(step_messages, config.vehicle_id)
            for config in vehicle_configs
        }
        primary_receiver_id = config_by_role[ego_cooperating_role].vehicle_id
        secondary_receiver_id = config_by_role[downstream_cooperating_role].vehicle_id
        primary_request_context = build_received_request_context(
            step_messages,
            primary_receiver_id,
            requester_conflict_sources,
            primary_requester_role,
            requesting_priority,
        )
        secondary_request_context = build_received_request_context(
            step_messages,
            secondary_receiver_id,
            requester_conflict_sources,
            secondary_requester_role,
            requesting_priority,
        )

        acceptance_stage_contexts = [
            build_acceptance_stage_context(
                primary_requester_role,
                primary_request_context["requester_conflicts"],
                primary_request_context["priority"],
                primary_request_context["delivered"],
                secondary=False,
                peer_accept_state=coordination_by_requester[secondary_requester_role]["accept_state"],
            ),
            build_acceptance_stage_context(
                secondary_requester_role,
                secondary_request_context["requester_conflicts"],
                secondary_request_context["priority"],
                secondary_request_context["delivered"],
                secondary=True,
            ),
        ]

        for stage_context in acceptance_stage_contexts:
            requester_role = stage_context["requester_role"]
            update_requester_acceptance_state(
                planner,
                coordination_by_requester,
                requester_role,
                {requester_role: stage_context["requester_conflicts"]},
                stage_context["priority"],
                stage_context["delivered"],
                peer_accept_state=stage_context["peer_accept_state"],
                secondary=bool(stage_context["secondary"]),
            )

        coordination_by_requester[secondary_requester_role]["speed_delta"] = (
            latched_candidate_value(
                coordination_by_requester[secondary_requester_role].get("speed_delta"),
                coordination_by_requester[primary_requester_role]["accept_state"],
                selected_primary_candidate,
                "speed_delta",
            )
        )
        coordination_by_requester[secondary_requester_role]["selected_acceleration"] = (
            latched_candidate_value(
                coordination_by_requester[secondary_requester_role].get("selected_acceleration"),
                coordination_by_requester[primary_requester_role]["accept_state"],
                selected_primary_candidate,
                "acceleration",
            )
        )
        downstream_speed_delta = (
            latched_candidate_value(
                downstream_speed_delta,
                coordination_by_requester[secondary_requester_role]["accept_state"],
                selected_downstream_candidate,
                "speed_delta",
            )
        )
        downstream_acceleration = (
            latched_candidate_value(
                downstream_acceleration,
                coordination_by_requester[secondary_requester_role]["accept_state"],
                selected_downstream_candidate,
                "acceleration",
            )
        )

        log_requester_state_changes(
            vehicle_configs,
            coordination_by_requester,
            last_coordination_snapshot,
            event_log,
            time,
            active_receiver_roles,
            echo=verbose_events,
        )

        role_specific_paths = {
            "ego": (*primary_requester_local_path, speed_profile_ego),
            "lead": (lead_path.x, lead_path.y, lead_path.yaw, lead_path.curvature, lead_path.speed_profile),
            secondary_requester_role: (*secondary_requester_local_path, speed_profile_accepting_vehicle),
            "acceptor_2": (
                acceptor_2_path.x,
                acceptor_2_path.y,
                acceptor_2_path.yaw,
                acceptor_2_path.curvature,
                downstream_accepting_speed_profile,
            ),
            "non_cooperating_vehicle": (
                non_cooperating_path.x,
                non_cooperating_path.y,
                non_cooperating_path.yaw,
                non_cooperating_path.curvature,
                non_cooperating_speed_profile,
            ),
        }

        for idx, (state, history, config) in enumerate(zip(states, histories, vehicle_configs)):
            default_path = paths_by_role[config.role]
            path_data = role_specific_paths.get(
                config.role,
                (
                    default_path.x,
                    default_path.y,
                    default_path.yaw,
                    default_path.curvature,
                    default_path.speed_profile,
                ),
            )
            px, py, pyaw, pk, sp = path_data
            steer, history.target_ind = rear_wheel_feedback_control(state, px, py, pyaw, pk, history.target_ind)
            steers[idx] = steer
            selected_acceleration = None
            if config.role == secondary_requester_role:
                selected_acceleration = coordination_by_requester[secondary_requester_role].get("selected_acceleration")
            elif config.role == "acceptor_2":
                selected_acceleration = downstream_acceleration
            min_accel, max_accel = cooperative_acceleration_bounds(selected_acceleration)
            accel = pid_control(sp[history.target_ind], state.v, min_accel, max_accel)
            states[idx] = update(state, accel, steer)
            if abs(states[idx].v) <= stop_speed:
                histories[idx].target_ind += 1

        time += 0.1

        for idx, (history, state, config) in enumerate(zip(histories, states, vehicle_configs)):
            append_history(history, state, steers[idx], time)
            if reached_goal(state, config.goal, goal_dis):
                history.goal_flag = True

        if any(history.goal_flag for history in histories):
            break

        collisions = detect_vehicle_collisions(states, vehicle_configs, min_distance=2.0)
        if collisions:
            collision_text = ", ".join(f"{first} vs {second}" for first, second in collisions)
            raise RuntimeError(f"Vehicle collision detected: {collision_text}")

        ego_reached_road_end = get_role_item(states, role_to_index, "ego").x >= (ROAD_END_X - EGO_STOP_MARGIN)
        if ego_reached_road_end:
            get_role_item(histories, role_to_index, "ego").goal_flag = True
            break

        if get_role_item(histories, role_to_index, "ego").goal_flag:
            break

        plot_requests = [
            {
                "xref": get_requested_plot_xref(
                    requester_requested_trajectories[primary_requester_role]["xref"],
                    requester_requested_trajectories[primary_requester_role]["enabled"],
                    coordination_by_requester[primary_requester_role]["request_state"],
                    coordination_by_requester[primary_requester_role]["motion_state"],
                    requested_plot_cache,
                    primary_requester_role,
                ),
                "state": requester_requested_trajectories[primary_requester_role]["state"],
            },
            {
                "xref": get_requested_plot_xref(
                    requester_requested_trajectories[secondary_requester_role]["xref"],
                    requester_requested_trajectories[secondary_requester_role]["enabled"],
                    coordination_by_requester[secondary_requester_role]["request_state"],
                    coordination_by_requester[secondary_requester_role]["motion_state"],
                    requested_plot_cache,
                    secondary_requester_role,
                ),
                "state": requester_requested_trajectories[secondary_requester_role]["state"],
            },
        ]
        frame_data = capture_simulation_frame(
            time=time,
            states=states,
            steers=steers,
            vehicle_configs=vehicle_configs,
            paths_by_role=paths_by_role,
            role_specific_paths=role_specific_paths,
            plot_requests=plot_requests,
            priority=requesting_priority,
            requester_role=primary_requester_role,
            coordination_by_requester=coordination_by_requester,
            active_receiver_roles=active_receiver_roles,
            receiver_accept_states={
                ego_cooperating_role: coordination_by_requester[primary_requester_role]["accept_state"],
                downstream_cooperating_role: coordination_by_requester[secondary_requester_role]["accept_state"],
            },
            recent_events=event_log,
        )
        frame_log.append(frame_data)

        if show_animation:
            connect_escape_key()
            render_simulation_frame(
                plt.gca(),
                vehicle_configs,
                frame_data,
                road_end_x=80.0,
                lane_lines=(0.0, 4.0, 8.0, 12.0),
            )
            plt.pause(0.1)

    return {
        "histories": histories,
        "states": states,
        "paths_by_role": paths_by_role,
        "messages": message_log,
        "events": event_log,
        "vehicle_inboxes": vehicle_inboxes,
        "show_animation": show_animation,
        "frame_log": frame_log,
    }
