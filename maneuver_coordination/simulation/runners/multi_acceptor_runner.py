"""Closed-loop runner for target-lane gap creation with multiple acceptors.

The ego vehicle requests a lane change into a gap shaped by a rear and a front
target-lane vehicle. The same runner supports the simple flow and the
request/offer/confirm/accept four-message flow.
"""

import math
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination.constants import (
    find_lane_change,
    send_request,
)
from maneuver_coordination.coordination.planner import BehaviouralLocalPlanner
from maneuver_coordination.simulation.motion.controllers import pid_control, rear_wheel_feedback_control
from maneuver_coordination.simulation.motion.vehicle_dynamics import update
from maneuver_coordination.simulation.core.history import append_history, init_vehicle_history
from maneuver_coordination.simulation.core.math_helpers import reached_goal
from maneuver_coordination.simulation.motion.reference import (
    calc_ref_trajectory,
)
from maneuver_coordination.simulation.motion.speed_profiles import (
    calc_speed_profile_accepting_vehicle,
    calc_speed_profile_downstream_accepting_vehicle,
)
from maneuver_coordination.simulation.core.types import State, V2XMessage, VehicleConfig
from maneuver_coordination.vehicle.plotting import connect_escape_key

from maneuver_coordination.simulation.motion.adaptation import (
    build_candidate_refs_for_accelerations,
    cooperative_acceleration_bounds,
    latched_candidate_value,
    select_first_feasible_front_candidate,
    select_first_feasible_reduction_candidate,
    speed_state_for_priority,
)
from maneuver_coordination.simulation.coordination.coordination_flow import (
    build_requester_local_paths,
    build_requester_path_inputs,
    build_requester_requested_trajectories,
    build_requester_speed_profiles,
    build_requester_stage_spec,
)
from maneuver_coordination.simulation.coordination.events import (
    append_simulation_event,
    format_event_time,
    get_active_receiver_roles,
    log_receiver_state_changes,
    log_requester_state_changes,
)
from maneuver_coordination.simulation.coordination.messages import (
    append_message_events,
    build_request_message_candidates,
    emit_request_messages,
    emit_v2x_message_event,
    get_config_by_role,
    get_vehicle_inbox,
    request_state_for_multi_acceptor,
)
from maneuver_coordination.simulation.motion.paths import build_paths_for_two_vehicle_coordination_configs
from maneuver_coordination.simulation.coordination.roles import (
    build_delayed_speed_profile,
    detect_vehicle_collisions,
    find_target_lane_neighbors,
    find_vehicle_ahead_in_lane,
    get_role_item,
    get_role_to_index,
    infer_lane_id_from_y,
    lane_center_y,
    require_roles,
    should_trigger_maneuver_search,
)
from maneuver_coordination.simulation.visualization import (
    capture_simulation_frame,
    get_requested_plot_xref,
    render_simulation_frame,
    update_requested_plot_cache,
)


def run_multi_acceptor_three_vehicle_coordination_scenario(
    vehicle_configs: Sequence[VehicleConfig],
    ox: Sequence[float],
    oy: Sequence[float],
    obstacle_list,
    *,
    use_offer_confirmation: bool,
    show_animation: bool = True,
    verbose_events: bool = True,
):
    """Run the multi-acceptor scenario with optional offer-confirmation messages."""
    dl = 0.1
    sim_t = 500.0
    goal_dis = 1.0
    stop_speed = 0.05
    priority_road_end_x = next(config.goal[0] for config in vehicle_configs if config.role == "ego")

    paths_by_role = build_paths_for_two_vehicle_coordination_configs(vehicle_configs, ox, oy)
    role_to_index = get_role_to_index(vehicle_configs)
    config_by_role = get_config_by_role(vehicle_configs)
    require_roles(role_to_index, ["ego", "lead", "target_lane_rear", "target_lane_front"])

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

    requester_role = "ego"
    responder_roles = ["target_lane_rear", "target_lane_front"]
    accept_state_by_role = {role: C.no_request_received for role in responder_roles}
    speed_state_by_role = {role: C.follow_planned for role in responder_roles}
    speed_delta_by_role: Dict[str, float | None] = {role: None for role in responder_roles}
    acceleration_by_role: Dict[str, float | None] = {role: None for role in responder_roles}
    request_state = C.no_request_sent
    motion_state = C.follow_lane
    selected_offer_ref_by_role: Dict[str, np.ndarray | None] = {role: None for role in responder_roles}
    time = 0.0
    message_log: List[V2XMessage] = []
    event_log: List[str] = []
    vehicle_inboxes: Dict[int, List[V2XMessage]] = {config.vehicle_id: [] for config in vehicle_configs}
    last_coordination_snapshot: Dict[str, Dict[str, int]] = {}
    braking_trigger_logged = False
    active_request_roles: set[str] = set()
    requested_plot_cache: Dict[str, Dict[str, object]] = {}
    frame_log: List[Dict[str, object]] = []
    coordination_completed = False
    coordination_started = False

    while sim_t >= time:
        states_by_role = {config.role: state for config, state in zip(vehicle_configs, states)}
        ego_state = states_by_role["ego"]
        lead_config = config_by_role["lead"]
        lead_braking_start_time = float(lead_config.metadata.get("braking_start_time", "-1.0"))
        lead_braking_target_speed = float(lead_config.metadata.get("braking_target_speed", str(lead_config.target_speed)))
        lead_role = find_vehicle_ahead_in_lane("ego", states_by_role, vehicle_configs)
        lead_state = states_by_role[lead_role] if lead_role is not None else states_by_role["ego"]
        lead_history = get_role_item(histories, role_to_index, lead_role) if lead_role is not None else None

        current_lane_id = infer_lane_id_from_y(ego_state.y, vehicle_configs)
        if request_state in (send_request, C.confirm_offers, C.execute_request, C.no_need_for_request):
            coordination_started = True
        if (
            coordination_started
            and request_state == C.no_request_sent
            and current_lane_id == config_by_role[requester_role].target_lane_id
        ):
            coordination_completed = True
        motion_state = planner.transition_state(ego_state, lead_state, request_state)
        if request_state == C.confirm_offers and motion_state == C.follow_lane:
            motion_state = C.find_lane_change
        elif request_state in (C.execute_request, C.no_need_for_request):
            motion_state = C.lane_change
        elif motion_state == C.follow_second_lane and (request_state != C.no_request_sent or active_request_roles):
            coordination_completed = True
            request_state = C.no_request_sent
            active_request_roles.clear()
            for responder_role in responder_roles:
                accept_state_by_role[responder_role] = C.no_request_received
                speed_state_by_role[responder_role] = C.follow_planned
            requested_plot_cache.pop(requester_role, None)
        if coordination_completed:
            motion_state = C.follow_second_lane
            request_state = C.no_request_sent
            active_request_roles.clear()
        lead_dx = ego_state.x - lead_state.x
        lead_dy = ego_state.y - lead_state.y
        lead_gap = math.hypot(lead_dx, lead_dy) - 4.0
        if (
            not coordination_completed
            and request_state == C.no_request_sent
            and current_lane_id == config_by_role[requester_role].lane_id
            and should_trigger_maneuver_search(
            ego_state,
            lead_state,
            lead_history,
            existing_path_length=planner.generated_path_length(),
            )
        ):
            motion_state = find_lane_change
            if not braking_trigger_logged and lead_role is not None:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time)}: coordination need detected, ID {config_by_role[requester_role].vehicle_id} sees vehicle ahead ID {config_by_role[lead_role].vehicle_id} decelerating",
                    echo=verbose_events,
                )
                braking_trigger_logged = True
        elif (
            not coordination_completed
            and request_state == C.no_request_sent
            and current_lane_id == config_by_role[requester_role].lane_id
            and lead_role is not None
            and time >= lead_braking_start_time
            and lead_gap < max(C.PLANNER_PARAMS.safe_time_gap * ego_state.v + 2.0, 40.0)
        ):
            motion_state = find_lane_change
            if not braking_trigger_logged:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time)}: coordination need detected, ID {config_by_role[requester_role].vehicle_id} sees vehicle ahead ID {config_by_role[lead_role].vehicle_id} decelerating",
                    echo=verbose_events,
                )
                braking_trigger_logged = True

        requester_stage_specs = [
            build_requester_stage_spec(
                requester_role,
                motion_state,
                "",
                "primary",
                secondary=False,
            )
        ]
        target_lane_neighbors = find_target_lane_neighbors(requester_role, states_by_role, vehicle_configs)
        target_lane_front_role = target_lane_neighbors.get("front")
        speed_lead_state = (
            states_by_role[target_lane_front_role]
            if motion_state != C.follow_lane and target_lane_front_role is not None
            else lead_state
        )
        requester_path_inputs = build_requester_path_inputs([requester_role], config_by_role, paths_by_role)
        aggregate_accept_state = C.no_request_received
        if any(state == C.request_accepted for state in accept_state_by_role.values()):
            aggregate_accept_state = C.request_accepted
        elif any(state == C.offer for state in accept_state_by_role.values()):
            aggregate_accept_state = C.offer
        elif any(state == C.request_rejected for state in accept_state_by_role.values()):
            aggregate_accept_state = C.request_rejected

        requester_local_paths = build_requester_local_paths(
            planner,
            requester_stage_specs,
            requester_path_inputs,
            {requester_role: ego_state},
            obstacle_list,
            {
                requester_role: {
                    "accept_state": aggregate_accept_state,
                }
            },
        )
        local_path_x, local_path_y, local_path_yaw, local_path_k = requester_local_paths[requester_role]
        requester_speed_profiles = build_requester_speed_profiles(
            requester_stage_specs,
            requester_local_paths,
            {requester_role: ego_state},
            {requester_role: config_by_role[requester_role].target_speed},
            {requester_role: get_role_item(histories, role_to_index, requester_role)},
            {requester_role: {"speed_state": C.follow_planned}},
            speed_lead_state,
        )
        speed_profile_ego = requester_speed_profiles[requester_role]

        requester_requested_trajectories = build_requester_requested_trajectories(
            planner,
            requester_stage_specs,
            {requester_role: ego_state},
            requester_local_paths,
            requester_speed_profiles,
            dl,
        )
        xreq = requester_requested_trajectories[requester_role]["xref"]
        req_traj = requester_requested_trajectories[requester_role]["enabled"]
        update_requested_plot_cache(
            requested_plot_cache,
            requester_role,
            xreq,
            req_traj,
            request_state,
            motion_state,
        )

        lead_path = paths_by_role["lead"]
        lead_speed_profile = build_delayed_speed_profile(
            lead_path,
            lead_config.target_speed,
            lead_braking_target_speed,
            time,
            lead_braking_start_time,
        )
        role_specific_paths = {
            "ego": (local_path_x, local_path_y, local_path_yaw, local_path_k, speed_profile_ego),
            "lead": (lead_path.x, lead_path.y, lead_path.yaw, lead_path.curvature, lead_speed_profile),
        }

        raw_need_by_role: Dict[str, bool] = {}
        responder_references: Dict[str, np.ndarray] = {}
        offer_states_for_request: Dict[str, int] = {}
        response_messages: List[V2XMessage] = []
        confirm_messages: List[V2XMessage] = []

        requesting_priority = planner.calc_priority(ego_state, lead_state)

        for responder_role in responder_roles:
            responder_state = states_by_role[responder_role]
            responder_history = get_role_item(histories, role_to_index, responder_role)
            responder_path = paths_by_role[responder_role]
            if responder_role == "target_lane_rear":
                speed_profile = calc_speed_profile_accepting_vehicle(
                    responder_path.x,
                    responder_path.y,
                    responder_path.yaw,
                    config_by_role[responder_role].target_speed,
                    responder_state,
                    motion_state,
                    speed_state_by_role[responder_role],
                    responder_history.target_ind,
                    speed_delta_by_role[responder_role],
                )
                direct_ref, responder_history.target_ind, _ = calc_ref_trajectory(
                    responder_state,
                    responder_path.x,
                    responder_path.y,
                    responder_path.yaw,
                    responder_path.curvature,
                    speed_profile,
                    dl,
                )
                direct_clear = planner.check_req_tr_conflict(
                    ego_state,
                    responder_state,
                    xreq,
                    direct_ref,
                    req_traj,
                )
                candidate_refs = build_candidate_refs_for_accelerations(
                    responder_state,
                    responder_path,
                    dl,
                    requesting_priority,
                    increase=False,
                )
                selected_candidate = select_first_feasible_reduction_candidate(
                    ego_state,
                    xreq,
                    req_traj,
                    candidate_refs,
                )
                adaptive_clear = selected_candidate is not None
                adaptive_speed_state = speed_state_for_priority(requesting_priority, increase=False)
            else:
                speed_profile = calc_speed_profile_downstream_accepting_vehicle(
                    responder_path.x,
                    responder_path.y,
                    responder_path.yaw,
                    config_by_role[responder_role].target_speed,
                    responder_state,
                    motion_state,
                    speed_state_by_role[responder_role],
                    responder_history.target_ind,
                    config_by_role[requester_role].target_speed,
                    speed_delta_by_role[responder_role],
                )
                direct_ref, responder_history.target_ind, _ = calc_ref_trajectory(
                    responder_state,
                    responder_path.x,
                    responder_path.y,
                    responder_path.yaw,
                    responder_path.curvature,
                    speed_profile,
                    dl,
                )
                direct_clear = planner.check_req_tr_conflict_4(
                    responder_state,
                    xreq,
                    direct_ref,
                    req_traj,
                )
                candidate_refs = build_candidate_refs_for_accelerations(
                    responder_state,
                    responder_path,
                    dl,
                    requesting_priority,
                    increase=True,
                )
                selected_candidate = select_first_feasible_front_candidate(
                    responder_state,
                    xreq,
                    req_traj,
                    candidate_refs,
                )
                adaptive_clear = selected_candidate is not None
                adaptive_speed_state = speed_state_for_priority(requesting_priority, increase=True)

            raw_need_by_role[responder_role] = not direct_clear
            responder_references[responder_role] = direct_ref
            selected_offer_ref_by_role[responder_role] = (
                selected_candidate["xref"] if selected_candidate is not None else None
            )

            if request_state == C.send_request and not use_offer_confirmation:
                if direct_clear:
                    accept_state_by_role[responder_role] = C.request_accepted
                    speed_state_by_role[responder_role] = C.follow_planned
                    speed_delta_by_role[responder_role] = None
                    acceleration_by_role[responder_role] = None
                elif adaptive_clear:
                    accept_state_by_role[responder_role] = C.request_accepted
                    speed_state_by_role[responder_role] = adaptive_speed_state
                    speed_delta_by_role[responder_role] = latched_candidate_value(
                        speed_delta_by_role[responder_role],
                        accept_state_by_role[responder_role],
                        selected_candidate,
                        "speed_delta",
                    )
                    acceleration_by_role[responder_role] = latched_candidate_value(
                        acceleration_by_role[responder_role],
                        accept_state_by_role[responder_role],
                        selected_candidate,
                        "acceleration",
                    )
                    responder_references[responder_role] = selected_offer_ref_by_role[responder_role]
                else:
                    accept_state_by_role[responder_role] = C.request_rejected
                    speed_state_by_role[responder_role] = C.follow_planned
                    speed_delta_by_role[responder_role] = None
                    acceleration_by_role[responder_role] = None
            elif request_state == C.confirm_offers:
                if accept_state_by_role[responder_role] == C.offer:
                    accept_state_by_role[responder_role] = C.request_accepted
                    speed_state_by_role[responder_role] = adaptive_speed_state
                    speed_delta_by_role[responder_role] = latched_candidate_value(
                        speed_delta_by_role[responder_role],
                        accept_state_by_role[responder_role],
                        selected_candidate,
                        "speed_delta",
                    )
                    acceleration_by_role[responder_role] = latched_candidate_value(
                        acceleration_by_role[responder_role],
                        accept_state_by_role[responder_role],
                        selected_candidate,
                        "acceleration",
                    )
                    responder_references[responder_role] = selected_offer_ref_by_role[responder_role]
                    confirm_messages.append(
                        emit_v2x_message_event(
                            time,
                            requester_role,
                            responder_role,
                            requesting_priority,
                            "confirm_offer",
                            config_by_role,
                            {"stage": responder_role},
                        )
                    )
            elif request_state in (C.execute_request, C.no_need_for_request):
                if accept_state_by_role[responder_role] == C.offer:
                    accept_state_by_role[responder_role] = C.request_accepted
                    speed_state_by_role[responder_role] = adaptive_speed_state
                    speed_delta_by_role[responder_role] = latched_candidate_value(
                        speed_delta_by_role[responder_role],
                        accept_state_by_role[responder_role],
                        selected_candidate,
                        "speed_delta",
                    )
                    acceleration_by_role[responder_role] = latched_candidate_value(
                        acceleration_by_role[responder_role],
                        accept_state_by_role[responder_role],
                        selected_candidate,
                        "acceleration",
                    )
                    responder_references[responder_role] = selected_offer_ref_by_role[responder_role]
            elif request_state == C.no_request_sent:
                accept_state_by_role[responder_role] = C.no_request_received
                speed_state_by_role[responder_role] = C.follow_planned
                speed_delta_by_role[responder_role] = None
                acceleration_by_role[responder_role] = None

            offer_states_for_request[responder_role] = accept_state_by_role[responder_role]
            role_specific_paths[responder_role] = (
                responder_path.x,
                responder_path.y,
                responder_path.yaw,
                responder_path.curvature,
                speed_profile,
            )

        if not active_request_roles and any(raw_need_by_role.values()):
            active_request_roles = {role for role, needed in raw_need_by_role.items() if needed}
        if motion_state == C.follow_second_lane:
            active_request_roles.clear()
            for responder_role in responder_roles:
                speed_delta_by_role[responder_role] = None
                acceleration_by_role[responder_role] = None

        need_by_role = {
            role: (role in active_request_roles) if active_request_roles else raw_need_by_role[role]
            for role in responder_roles
        }

        offers_are_mutually_clear = True

        if use_offer_confirmation and request_state == C.send_request:
            for responder_role in responder_roles:
                if need_by_role[responder_role]:
                    accept_state_by_role[responder_role] = C.offer
                    response_messages.append(
                        emit_v2x_message_event(
                            time,
                            responder_role,
                            requester_role,
                            requesting_priority,
                            "maneuver_offer",
                            config_by_role,
                            {"stage": responder_role},
                        )
                    )
        elif not use_offer_confirmation and request_state == C.send_request:
            for responder_role in responder_roles:
                if need_by_role[responder_role]:
                    accept_state_by_role[responder_role] = C.request_accepted
                    speed_state_by_role[responder_role] = (
                        speed_state_for_priority(requesting_priority, increase=False)
                        if responder_role == "target_lane_rear"
                        else speed_state_for_priority(requesting_priority, increase=True)
                    )

        next_request_state = request_state_for_multi_acceptor(
            motion_state,
            need_by_role,
            accept_state_by_role,
            offer_states_for_request if use_offer_confirmation and offers_are_mutually_clear else None,
        )
        if next_request_state == C.execute_request and request_state == C.no_request_sent:
            next_request_state = C.send_request
        request_state = next_request_state
        if request_state in (C.execute_request, C.no_need_for_request):
            motion_state = C.lane_change

        request_candidates = [
            build_request_message_candidates(
                requester_role,
                responder_role,
                request_state,
                requesting_priority,
                responder_role,
            )
            for responder_role in responder_roles
            if need_by_role[responder_role]
        ]
        active_receiver_roles = get_active_receiver_roles(
            [
                build_requester_stage_spec(
                    requester_role,
                    motion_state,
                    responder_role,
                    responder_role,
                    secondary=False,
                )
                for responder_role in responder_roles
                if need_by_role[responder_role]
            ],
            {
                requester_role: {
                    "request_state": request_state,
                }
            },
        )
        step_messages = emit_request_messages(time, request_candidates, config_by_role)
        step_messages.extend(response_messages)
        step_messages.extend(confirm_messages)
        message_log.extend(step_messages)
        append_message_events(event_log, step_messages, echo=verbose_events)
        vehicle_inboxes = {
            config.vehicle_id: get_vehicle_inbox(step_messages, config.vehicle_id)
            for config in vehicle_configs
        }

        log_requester_state_changes(
            vehicle_configs,
            {
                requester_role: {
                    "motion_state": motion_state,
                    "request_state": request_state,
                    "accept_state": (
                        C.request_accepted
                        if any(state == C.request_accepted for state in accept_state_by_role.values())
                        else next(
                            (state for state in accept_state_by_role.values() if state == C.offer),
                            C.no_request_received,
                        )
                    ),
                }
            },
            last_coordination_snapshot,
            event_log,
            time,
            active_receiver_roles,
            echo=verbose_events,
        )
        log_receiver_state_changes(
            vehicle_configs,
            accept_state_by_role,
            last_coordination_snapshot,
            event_log,
            time,
            active_receiver_roles,
            echo=verbose_events,
        )

        for idx, (state, history, config) in enumerate(zip(states, histories, vehicle_configs)):
            if history.goal_flag:
                steers[idx] = 0.0
                state.v = 0.0
                continue
            default_path = paths_by_role[config.role]
            px, py, pyaw, pk, sp = role_specific_paths.get(
                config.role,
                (default_path.x, default_path.y, default_path.yaw, default_path.curvature, default_path.speed_profile),
            )
            steer, history.target_ind = rear_wheel_feedback_control(state, px, py, pyaw, pk, history.target_ind)
            steers[idx] = steer
            selected_acceleration = acceleration_by_role.get(config.role)
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

        updated_requester_state = get_role_item(states, role_to_index, requester_role)
        target_lane_center_y = lane_center_y(config_by_role[requester_role].target_lane_id)
        requester_in_target_lane = (
            infer_lane_id_from_y(updated_requester_state.y, vehicle_configs)
            == config_by_role[requester_role].target_lane_id
        )
        requester_centered_in_target_lane = abs(updated_requester_state.y - target_lane_center_y) <= 0.35
        if (
            request_state in (send_request, C.confirm_offers, C.execute_request, C.no_need_for_request)
            and requester_in_target_lane
            and requester_centered_in_target_lane
        ):
            request_state = C.no_request_sent
            motion_state = C.follow_second_lane
            coordination_completed = True
            active_request_roles.clear()
            for responder_role in responder_roles:
                accept_state_by_role[responder_role] = C.no_request_received
                speed_state_by_role[responder_role] = C.follow_planned
            requested_plot_cache.pop(requester_role, None)

        if any(history.goal_flag for history in histories):
            break

        if states_by_role[requester_role].x >= priority_road_end_x - 0.25:
            get_role_item(histories, role_to_index, requester_role).goal_flag = True
            break

        collisions = detect_vehicle_collisions(states, vehicle_configs, min_distance=2.0)
        if collisions:
            collision_text = ", ".join(f"{first} vs {second}" for first, second in collisions)
            raise RuntimeError(f"Vehicle collision detected: {collision_text}")

        if get_role_item(histories, role_to_index, requester_role).goal_flag:
            break

        plot_requests = [
            {
                "xref": get_requested_plot_xref(
                    xreq,
                    req_traj,
                    request_state,
                    motion_state,
                    requested_plot_cache,
                    requester_role,
                ),
                "state": ego_state,
            }
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
            requester_role=requester_role,
            coordination_by_requester=None,
            active_receiver_roles=active_receiver_roles,
            receiver_accept_states=accept_state_by_role,
            recent_events=event_log,
        )
        frame_log.append(frame_data)

        if show_animation:
            connect_escape_key()
            render_simulation_frame(
                plt.gca(),
                vehicle_configs,
                frame_data,
                road_end_x=100.0,
                lane_lines=(0.0, 4.0, 8.0),
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

def run_three_vehicle_coordination_scenario(
    vehicle_configs: Sequence[VehicleConfig],
    ox: Sequence[float],
    oy: Sequence[float],
    obstacle_list,
    show_animation: bool = True,
    verbose_events: bool = True,
):
    """Compatibility wrapper for the simple three-vehicle coordination flow."""
    return run_multi_acceptor_three_vehicle_coordination_scenario(
        vehicle_configs,
        ox,
        oy,
        obstacle_list,
        use_offer_confirmation=False,
        show_animation=show_animation,
        verbose_events=verbose_events,
    )

def run_three_vehicle_coordination_4_messages_scenario(
    vehicle_configs: Sequence[VehicleConfig],
    ox: Sequence[float],
    oy: Sequence[float],
    obstacle_list,
    show_animation: bool = True,
    verbose_events: bool = True,
):
    """Compatibility wrapper for the four-message offer-confirmation flow."""
    return run_multi_acceptor_three_vehicle_coordination_scenario(
        vehicle_configs,
        ox,
        oy,
        obstacle_list,
        use_offer_confirmation=True,
        show_animation=show_animation,
        verbose_events=verbose_events,
    )
