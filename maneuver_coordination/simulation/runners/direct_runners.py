"""Closed-loop runners for direct two-vehicle and rejection/fallback scenarios."""

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination.constants import (
    find_lane_change,
    follow_planned,
    no_request_received,
    no_request_sent,
    send_request,
)
from maneuver_coordination.coordination.planner import BehaviouralLocalPlanner
from maneuver_coordination.simulation.motion.controllers import pid_control, rear_wheel_feedback_control
from maneuver_coordination.simulation.motion.vehicle_dynamics import update
from maneuver_coordination.simulation.core.history import append_history, init_vehicle_history
from maneuver_coordination.simulation.core.math_helpers import reached_goal
from maneuver_coordination.simulation.motion.speed_profiles import calc_speed_profile_accepting_vehicle
from maneuver_coordination.simulation.core.types import State, V2XMessage, VehicleConfig
from maneuver_coordination.vehicle.plotting import connect_escape_key

from maneuver_coordination.simulation.motion.adaptation import (
    build_candidate_refs_for_accelerations,
    cooperative_acceleration_bounds,
    latched_candidate_value,
    select_first_feasible_reduction_candidate,
)
from maneuver_coordination.simulation.coordination.coordination_flow import (
    build_requester_conflict_sources,
    build_requester_local_paths,
    build_requester_path_inputs,
    build_requester_requested_trajectories,
    build_requester_speed_profiles,
    build_requester_stage_spec,
    build_role_reference_trajectories,
    choose_cooperating_role,
    find_conflicting_vehicle_roles,
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
    build_received_request_context,
    build_request_message_candidates,
    emit_request_messages,
    get_config_by_role,
    get_vehicle_inbox,
)
from maneuver_coordination.simulation.motion.paths import build_paths_for_two_vehicle_coordination_configs
from maneuver_coordination.simulation.coordination.roles import (
    build_acc_speed_profile,
    build_delayed_speed_profile,
    build_target_lane_reference_candidates,
    detect_vehicle_collisions,
    find_vehicle_ahead_in_lane,
    get_requester_candidate_roles,
    get_role_item,
    get_role_to_index,
    require_roles,
    should_trigger_maneuver_search,
)
from maneuver_coordination.simulation.visualization import (
    capture_simulation_frame,
    get_requested_plot_xref,
    render_simulation_frame,
    update_requested_plot_cache,
)


def run_two_vehicle_coordination_scenario(
    vehicle_configs: Sequence[VehicleConfig],
    ox: Sequence[float],
    oy: Sequence[float],
    obstacle_list,
    show_animation: bool = True,
    verbose_events: bool = True,
):
    """Run the direct ego-to-acceptor coordination scenario."""
    dl = 0.1
    sim_t = 500.0
    goal_dis = 1.0
    stop_speed = 0.05
    priority_road_end_x = next(config.goal[0] for config in vehicle_configs if config.role == "ego")

    paths_by_role = build_paths_for_two_vehicle_coordination_configs(vehicle_configs, ox, oy)
    role_to_index = get_role_to_index(vehicle_configs)
    config_by_role = get_config_by_role(vehicle_configs)
    require_roles(role_to_index, ["ego", "lead", "acceptor"])
    requester_roles = get_requester_candidate_roles(vehicle_configs)
    primary_requester_role = requester_roles[0]

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

    accepting_vehicle_state = no_request_received
    speed_profile_state = follow_planned
    cooperating_speed_delta: float | None = None
    cooperating_acceleration: float | None = None
    requesting_vehicle_state = no_request_sent
    time = 0.0
    message_log: List[V2XMessage] = []
    event_log: List[str] = []
    vehicle_inboxes: Dict[int, List[V2XMessage]] = {config.vehicle_id: [] for config in vehicle_configs}
    last_coordination_snapshot: Dict[str, Dict[str, int]] = {}
    braking_trigger_logged = False
    active_request_roles: set[str] = set()
    requested_plot_cache: Dict[str, Dict[str, object]] = {}
    frame_log: List[Dict[str, object]] = []
    while sim_t >= time:
        states_by_role = {config.role: state for config, state in zip(vehicle_configs, states)}
        ego_state = states_by_role["ego"]
        lead_config = config_by_role["lead"]
        lead_braking_start_time = float(lead_config.metadata.get("braking_start_time", "-1.0"))
        lead_braking_target_speed = float(lead_config.metadata.get("braking_target_speed", str(lead_config.target_speed)))
        lead_role = find_vehicle_ahead_in_lane("ego", states_by_role, vehicle_configs)
        lead_state = states_by_role[lead_role] if lead_role is not None else states_by_role["ego"]
        lead_history = get_role_item(histories, role_to_index, lead_role) if lead_role is not None else None
        acceptor_state = get_role_item(states, role_to_index, "acceptor")

        current_state = planner.transition_state(ego_state, lead_state, requesting_vehicle_state)
        if should_trigger_maneuver_search(
            ego_state,
            lead_state,
            lead_history,
            existing_path_length=planner.generated_path_length(),
        ):
            current_state = find_lane_change
            if not braking_trigger_logged and lead_role is not None:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time)}: coordination need detected, ID {config_by_role[primary_requester_role].vehicle_id} sees vehicle ahead ID {config_by_role[lead_role].vehicle_id} decelerating",
                    echo=verbose_events,
                )
                braking_trigger_logged = True

        ego_path = paths_by_role["ego"]
        lead_path = paths_by_role["lead"]
        acceptor_path = paths_by_role["acceptor"]
        requester_states = {
            primary_requester_role: ego_state,
        }
        requester_stage_specs = [
            build_requester_stage_spec(
                primary_requester_role,
                current_state,
                "acceptor",
                "primary",
                secondary=False,
            )
        ]
        requester_path_inputs = build_requester_path_inputs(
            [primary_requester_role],
            config_by_role,
            paths_by_role,
        )
        requester_local_paths = build_requester_local_paths(
            planner,
            requester_stage_specs,
            requester_path_inputs,
            requester_states,
            obstacle_list,
            {
                primary_requester_role: {
                    "accept_state": accepting_vehicle_state,
                }
            },
        )
        local_path_x, local_path_y, local_path_yaw, local_path_k = requester_local_paths[primary_requester_role]
        histories_by_role = {
            primary_requester_role: get_role_item(histories, role_to_index, "ego"),
        }
        requester_target_speeds = {
            primary_requester_role: get_role_item(vehicle_configs, role_to_index, "ego").target_speed,
        }
        requester_speed_profiles = build_requester_speed_profiles(
            requester_stage_specs,
            requester_local_paths,
            requester_states,
            requester_target_speeds,
            histories_by_role,
            {
                primary_requester_role: {
                    "speed_state": follow_planned,
                }
            },
            lead_state,
        )
        speed_profile_ego = requester_speed_profiles[primary_requester_role]
        speed_profile_acceptor = calc_speed_profile_accepting_vehicle(
            acceptor_path.x,
            acceptor_path.y,
            acceptor_path.yaw,
            get_role_item(vehicle_configs, role_to_index, "acceptor").target_speed,
            acceptor_state,
            current_state,
            speed_profile_state,
            get_role_item(histories, role_to_index, "acceptor").target_ind,
            cooperating_speed_delta,
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
        update_requested_plot_cache(
            requested_plot_cache,
            primary_requester_role,
            xreq,
            req_traj,
            requesting_vehicle_state,
            current_state,
        )

        responder_histories = {
            "lead": get_role_item(histories, role_to_index, "lead"),
            "acceptor": get_role_item(histories, role_to_index, "acceptor"),
        }
        responder_reference_inputs = {
            "lead": {
                "state": get_role_item(states, role_to_index, "lead"),
                "path_x": lead_path.x,
                "path_y": lead_path.y,
                "path_yaw": lead_path.yaw,
                "path_k": lead_path.curvature,
                "speed_profile": lead_path.speed_profile,
            },
            "acceptor": {
                "state": acceptor_state,
                "path_x": acceptor_path.x,
                "path_y": acceptor_path.y,
                "path_yaw": acceptor_path.yaw,
                "path_k": acceptor_path.curvature,
                "speed_profile": speed_profile_acceptor,
            },
        }
        role_reference_trajectories = build_role_reference_trajectories(
            responder_reference_inputs,
            responder_histories,
            dl,
        )

        target_lane_reference_candidates = build_target_lane_reference_candidates(
            primary_requester_role,
            role_reference_trajectories,
            states_by_role,
            vehicle_configs,
        )
        conflicting_roles = find_conflicting_vehicle_roles(
            primary_requester_role,
            xreq,
            target_lane_reference_candidates,
            states_by_role,
            vehicle_configs,
        )
        cooperating_role = choose_cooperating_role(
            primary_requester_role,
            conflicting_roles,
            states_by_role,
            vehicle_configs,
            "acceptor",
        )
        cooperating_state = get_role_item(states, role_to_index, cooperating_role)
        cooperating_reference = target_lane_reference_candidates[cooperating_role]

        requesting_priority = planner.calc_priority(ego_state, lead_state)
        candidate_refs = build_candidate_refs_for_accelerations(
            cooperating_state,
            paths_by_role[cooperating_role],
            dl,
            requesting_priority,
            increase=False,
        )
        conflict_free_req = planner.check_req_tr_conflict(
            ego_state,
            cooperating_state,
            xreq,
            cooperating_reference,
            req_traj,
        )
        selected_candidate = select_first_feasible_reduction_candidate(
            ego_state,
            xreq,
            req_traj,
            candidate_refs,
        )
        conflict_free_new = selected_candidate is not None

        requesting_vehicle_state = planner.requesting_vehicle_states(
            current_state,
            accepting_vehicle_state,
            conflict_free_req,
        )

        message_stage_specs = [
            build_requester_stage_spec(
                primary_requester_role,
                current_state,
                cooperating_role,
                "primary",
                secondary=False,
            )
        ]
        active_receiver_roles = get_active_receiver_roles(
            message_stage_specs,
            {
                primary_requester_role: {
                    "request_state": requesting_vehicle_state,
                }
            },
        )
        request_candidates = [
            build_request_message_candidates(
                stage_spec["requester_role"],
                stage_spec["receiver_role"],
                requesting_vehicle_state,
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
        receiver_id = config_by_role[cooperating_role].vehicle_id
        request_context = build_received_request_context(
            step_messages,
            receiver_id,
            build_requester_conflict_sources(
                primary_requester_role,
                conflict_free_req,
                conflict_free_req,
                conflict_free_new,
            ),
            primary_requester_role,
            requesting_priority,
        )

        if requesting_vehicle_state == send_request and not request_context["delivered"]:
            accepting_vehicle_state, speed_profile_state = no_request_received, follow_planned
            cooperating_speed_delta = None
            cooperating_acceleration = None
        else:
            accepting_vehicle_state, speed_profile_state = planner.accepting_vehicle_states_direct(
                requesting_vehicle_state,
                request_context["priority"],
                conflict_free_req,
                conflict_free_new,
            )
            cooperating_speed_delta = latched_candidate_value(
                cooperating_speed_delta,
                accepting_vehicle_state,
                selected_candidate,
                "speed_delta",
            )
            cooperating_acceleration = latched_candidate_value(
                cooperating_acceleration,
                accepting_vehicle_state,
                selected_candidate,
                "acceleration",
            )

        log_requester_state_changes(
            vehicle_configs,
            {
                primary_requester_role: {
                    "motion_state": current_state,
                    "request_state": requesting_vehicle_state,
                    "accept_state": accepting_vehicle_state,
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
            {cooperating_role: accepting_vehicle_state},
            last_coordination_snapshot,
            event_log,
            time,
            active_receiver_roles,
            echo=verbose_events,
        )

        role_specific_paths = {
            "ego": (local_path_x, local_path_y, local_path_yaw, local_path_k, speed_profile_ego),
            "lead": (lead_path.x, lead_path.y, lead_path.yaw, lead_path.curvature, lead_path.speed_profile),
            "acceptor": (
                acceptor_path.x,
                acceptor_path.y,
                acceptor_path.yaw,
                acceptor_path.curvature,
                speed_profile_acceptor,
            ),
        }

        for idx, (state, history, config) in enumerate(zip(states, histories, vehicle_configs)):
            default_path = paths_by_role[config.role]
            px, py, pyaw, pk, sp = role_specific_paths.get(
                config.role,
                (default_path.x, default_path.y, default_path.yaw, default_path.curvature, default_path.speed_profile),
            )
            steer, history.target_ind = rear_wheel_feedback_control(state, px, py, pyaw, pk, history.target_ind)
            steers[idx] = steer
            selected_acceleration = cooperating_acceleration if config.role == cooperating_role else None
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

        ego_state = get_role_item(states, role_to_index, "ego")
        if ego_state.x >= priority_road_end_x - 0.25:
            get_role_item(histories, role_to_index, "ego").goal_flag = True
            break

        collisions = detect_vehicle_collisions(states, vehicle_configs, min_distance=2.0)
        if collisions:
            collision_text = ", ".join(f"{first} vs {second}" for first, second in collisions)
            raise RuntimeError(f"Vehicle collision detected: {collision_text}")

        ego_reached_goal = get_role_item(histories, role_to_index, "ego").goal_flag
        if ego_reached_goal:
            break

        plot_requests = [
            {
                "xref": get_requested_plot_xref(
                    xreq,
                    req_traj,
                    requesting_vehicle_state,
                    current_state,
                    requested_plot_cache,
                    primary_requester_role,
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
            requester_role=primary_requester_role,
            coordination_by_requester=None,
            active_receiver_roles=active_receiver_roles,
            receiver_accept_states={cooperating_role: accepting_vehicle_state},
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

def run_rejected_request_then_free_lane_scenario(
    vehicle_configs: Sequence[VehicleConfig],
    ox: Sequence[float],
    oy: Sequence[float],
    show_animation: bool = True,
    verbose_events: bool = True,
):
    """Run the scenario where rejection triggers ACC following before a later lane change."""
    dl = 0.1
    sim_t = 500.0
    goal_dis = 1.0
    stop_speed = 0.05
    priority_road_end_x = next(config.goal[0] for config in vehicle_configs if config.role == "ego")

    paths_by_role = build_paths_for_two_vehicle_coordination_configs(vehicle_configs, ox, oy)
    role_to_index = get_role_to_index(vehicle_configs)
    config_by_role = get_config_by_role(vehicle_configs)
    require_roles(role_to_index, ["ego", "lead", "acceptor"])

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

    accepting_vehicle_state = no_request_received
    speed_profile_state = follow_planned
    cooperating_speed_delta: float | None = None
    cooperating_acceleration: float | None = None
    requesting_vehicle_state = no_request_sent
    time = 0.0
    message_log: List[V2XMessage] = []
    event_log: List[str] = []
    vehicle_inboxes: Dict[int, List[V2XMessage]] = {config.vehicle_id: [] for config in vehicle_configs}
    last_coordination_snapshot: Dict[str, Dict[str, int]] = {}
    braking_trigger_logged = False
    waiting_for_free_lane_after_rejection = False
    lane_change_committed = False
    lane_change_completed = False
    requested_plot_cache: Dict[str, Dict[str, object]] = {}
    frame_log: List[Dict[str, object]] = []
    requester_role = "ego"
    cooperating_role = "acceptor"

    while sim_t >= time:
        states_by_role = {config.role: state for config, state in zip(vehicle_configs, states)}
        ego_state = states_by_role["ego"]
        lead_state = states_by_role["lead"]
        acceptor_state = states_by_role["acceptor"]
        lead_history = get_role_item(histories, role_to_index, "lead")

        lead_config = config_by_role["lead"]
        lead_braking_start_time = float(lead_config.metadata.get("braking_start_time", "-1.0"))
        lead_braking_target_speed = float(lead_config.metadata.get("braking_target_speed", str(lead_config.target_speed)))

        current_state = planner.transition_state(ego_state, lead_state, requesting_vehicle_state)
        if lane_change_completed:
            current_state = C.follow_second_lane
            requesting_vehicle_state = C.no_request_sent
            accepting_vehicle_state = C.no_request_received
            cooperating_speed_delta = None
        elif lane_change_committed:
            current_state = C.lane_change
        elif waiting_for_free_lane_after_rejection:
            current_state = C.follow_lane
        elif should_trigger_maneuver_search(
            ego_state,
            lead_state,
            lead_history,
            existing_path_length=planner.generated_path_length(),
        ):
            current_state = find_lane_change
            if not braking_trigger_logged:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time)}: coordination need detected, ID {config_by_role[requester_role].vehicle_id} sees vehicle ahead ID {config_by_role['lead'].vehicle_id} decelerating",
                    echo=verbose_events,
                )
                braking_trigger_logged = True
        elif (
            time >= lead_braking_start_time
            and lead_state.x > ego_state.x
            and (lead_state.x - ego_state.x - 4.0) < max(C.PLANNER_PARAMS.safe_time_gap * ego_state.v + 2.0, 40.0)
        ):
            current_state = find_lane_change
            if not braking_trigger_logged:
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time)}: coordination need detected, ID {config_by_role[requester_role].vehicle_id} sees vehicle ahead ID {config_by_role['lead'].vehicle_id} decelerating",
                    echo=verbose_events,
                )
                braking_trigger_logged = True

        ego_path = paths_by_role["ego"]
        ego_alt_path = paths_by_role[config_by_role["ego"].fallback_path_key]
        lead_path = paths_by_role["lead"]
        acceptor_path = paths_by_role["acceptor"]

        planning_motion_state = C.find_lane_change if waiting_for_free_lane_after_rejection else current_state
        planning_local_paths = {
            requester_role: (
                ego_alt_path.x,
                ego_alt_path.y,
                ego_alt_path.yaw,
                ego_alt_path.curvature,
            )
        }

        lead_speed_profile = build_delayed_speed_profile(
            lead_path,
            lead_config.target_speed,
            lead_braking_target_speed,
            time,
            lead_braking_start_time,
        )
        speed_profile_acceptor = [config_by_role["acceptor"].target_speed + (cooperating_speed_delta or 0.0) for _ in acceptor_path.x]
        responder_histories = {
            "lead": get_role_item(histories, role_to_index, "lead"),
            "acceptor": get_role_item(histories, role_to_index, "acceptor"),
        }
        role_reference_trajectories = build_role_reference_trajectories(
            {
                "lead": {
                    "state": lead_state,
                    "path_x": lead_path.x,
                    "path_y": lead_path.y,
                    "path_yaw": lead_path.yaw,
                    "path_k": lead_path.curvature,
                    "speed_profile": lead_speed_profile,
                },
                "acceptor": {
                    "state": acceptor_state,
                    "path_x": acceptor_path.x,
                    "path_y": acceptor_path.y,
                    "path_yaw": acceptor_path.yaw,
                    "path_k": acceptor_path.curvature,
                    "speed_profile": speed_profile_acceptor,
                },
            },
            responder_histories,
            dl,
        )

        requester_requested_trajectories = build_requester_requested_trajectories(
            planner,
            [build_requester_stage_spec(requester_role, planning_motion_state, "acceptor", "primary", secondary=False)],
            {requester_role: ego_state},
            planning_local_paths,
            {requester_role: [config_by_role["ego"].target_speed for _ in ego_alt_path.x]},
            dl,
        )
        xreq = requester_requested_trajectories[requester_role]["xref"]
        req_traj = requester_requested_trajectories[requester_role]["enabled"]
        update_requested_plot_cache(
            requested_plot_cache,
            requester_role,
            xreq,
            req_traj,
            requesting_vehicle_state,
            current_state,
        )

        target_lane_reference_candidates = build_target_lane_reference_candidates(
            requester_role,
            {"acceptor": role_reference_trajectories["acceptor"]},
            states_by_role,
            vehicle_configs,
        )
        conflicting_roles = find_conflicting_vehicle_roles(
            requester_role,
            xreq,
            target_lane_reference_candidates,
            states_by_role,
            vehicle_configs,
        )
        acceptor_gap = acceptor_state.x - ego_state.x
        target_lane_is_free = not conflicting_roles and acceptor_gap > 12.0
        cooperating_reference = target_lane_reference_candidates.get(cooperating_role, role_reference_trajectories["acceptor"])

        requesting_priority = planner.calc_priority(ego_state, lead_state)
        candidate_refs = build_candidate_refs_for_accelerations(
            acceptor_state,
            acceptor_path,
            dl,
            requesting_priority,
            increase=False,
        )
        conflict_free_req = planner.check_req_tr_conflict(
            ego_state,
            acceptor_state,
            xreq,
            cooperating_reference,
            req_traj,
        )
        selected_candidate = select_first_feasible_reduction_candidate(
            ego_state,
            xreq,
            req_traj,
            candidate_refs,
        )
        conflict_free_new = selected_candidate is not None

        if lane_change_completed:
            requesting_vehicle_state = C.no_request_sent
            accepting_vehicle_state = C.no_request_received
            cooperating_speed_delta = None
            cooperating_acceleration = None
        elif waiting_for_free_lane_after_rejection:
            if target_lane_is_free:
                waiting_for_free_lane_after_rejection = False
                lane_change_committed = True
                accepting_vehicle_state = no_request_received
                cooperating_speed_delta = None
                cooperating_acceleration = None
                requesting_vehicle_state = C.no_need_for_request
                current_state = C.lane_change
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time)}: adjacent lane free, ID {config_by_role[requester_role].vehicle_id} starts lane change",
                    echo=verbose_events,
                )
            else:
                accepting_vehicle_state = C.request_rejected
                cooperating_speed_delta = None
                cooperating_acceleration = None
                requesting_vehicle_state = C.no_request_sent
        elif not lane_change_committed:
            if current_state == find_lane_change and not target_lane_is_free:
                requesting_vehicle_state = send_request
            else:
                requesting_vehicle_state = planner.requesting_vehicle_states(
                    current_state,
                    accepting_vehicle_state,
                    conflict_free_req,
                )

            message_stage_specs = [
                build_requester_stage_spec(
                    requester_role,
                    current_state,
                    cooperating_role,
                    "primary",
                    secondary=False,
                )
            ]
            active_receiver_roles = get_active_receiver_roles(
                message_stage_specs,
                {requester_role: {"request_state": requesting_vehicle_state}},
            )
            request_candidates = [
                build_request_message_candidates(
                    requester_role,
                    cooperating_role,
                    requesting_vehicle_state,
                    requesting_priority,
                    "primary",
                )
            ]
            step_messages = emit_request_messages(time, request_candidates, config_by_role)
            message_log.extend(step_messages)
            append_message_events(event_log, step_messages, echo=verbose_events)
            vehicle_inboxes = {
                config.vehicle_id: get_vehicle_inbox(step_messages, config.vehicle_id)
                for config in vehicle_configs
            }
            request_context = build_received_request_context(
                step_messages,
                config_by_role[cooperating_role].vehicle_id,
                build_requester_conflict_sources(requester_role, conflict_free_req, conflict_free_req, conflict_free_new),
                requester_role,
                requesting_priority,
            )

            if requesting_vehicle_state == send_request:
                if request_context["delivered"]:
                    accepting_vehicle_state, speed_profile_state = C.request_rejected, follow_planned
                    cooperating_speed_delta = None
                    cooperating_acceleration = None
                else:
                    accepting_vehicle_state, speed_profile_state = no_request_received, follow_planned
                    cooperating_speed_delta = None
                    cooperating_acceleration = None
            else:
                accepting_vehicle_state, speed_profile_state = planner.accepting_vehicle_states_direct(
                    requesting_vehicle_state,
                    request_context["priority"],
                    conflict_free_req,
                    conflict_free_new,
                )
                cooperating_speed_delta = latched_candidate_value(
                    cooperating_speed_delta,
                    accepting_vehicle_state,
                    selected_candidate,
                    "speed_delta",
                )
                cooperating_acceleration = latched_candidate_value(
                    cooperating_acceleration,
                    accepting_vehicle_state,
                    selected_candidate,
                    "acceleration",
                )

            if accepting_vehicle_state == C.request_rejected:
                waiting_for_free_lane_after_rejection = True
                append_simulation_event(
                    event_log,
                    f"{format_event_time(time)}: ID {config_by_role[requester_role].vehicle_id} request rejected, fallback to ACC",
                    echo=verbose_events,
                )
        else:
            requesting_vehicle_state = C.no_need_for_request
            accepting_vehicle_state = C.no_request_received
            cooperating_speed_delta = None
            cooperating_acceleration = None

        if lane_change_completed:
            speed_profile_ego = [config_by_role["ego"].target_speed for _ in ego_alt_path.x]
            ego_drive_path = ego_alt_path
        elif waiting_for_free_lane_after_rejection:
            speed_profile_ego = build_acc_speed_profile(
                ego_path,
                ego_state,
                config_by_role["ego"].target_speed,
                lead_state,
            )
            ego_drive_path = ego_path
        elif current_state == C.lane_change or lane_change_committed:
            speed_profile_ego = [config_by_role["ego"].target_speed for _ in ego_alt_path.x]
            ego_drive_path = ego_alt_path
        else:
            speed_profile_ego = build_acc_speed_profile(
                ego_path,
                ego_state,
                config_by_role["ego"].target_speed,
                lead_state,
            )
            ego_drive_path = ego_path

        active_receiver_roles = [cooperating_role] if requesting_vehicle_state in (send_request, C.execute_request) else []

        log_requester_state_changes(
            vehicle_configs,
            {
                requester_role: {
                    "motion_state": current_state,
                    "request_state": requesting_vehicle_state,
                    "accept_state": accepting_vehicle_state,
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
            {cooperating_role: accepting_vehicle_state},
            last_coordination_snapshot,
            event_log,
            time,
            active_receiver_roles,
            echo=verbose_events,
        )

        role_specific_paths = {
            "ego": (ego_drive_path.x, ego_drive_path.y, ego_drive_path.yaw, ego_drive_path.curvature, speed_profile_ego),
            "lead": (lead_path.x, lead_path.y, lead_path.yaw, lead_path.curvature, lead_speed_profile),
            "acceptor": (acceptor_path.x, acceptor_path.y, acceptor_path.yaw, acceptor_path.curvature, speed_profile_acceptor),
        }

        for idx, (state, history, config) in enumerate(zip(states, histories, vehicle_configs)):
            default_path = paths_by_role[config.role]
            px, py, pyaw, pk, sp = role_specific_paths.get(
                config.role,
                (default_path.x, default_path.y, default_path.yaw, default_path.curvature, default_path.speed_profile),
            )
            steer, history.target_ind = rear_wheel_feedback_control(state, px, py, pyaw, pk, history.target_ind)
            steers[idx] = steer
            selected_acceleration = cooperating_acceleration if config.role == cooperating_role else None
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

        if states_by_role[requester_role].y >= C.PLANNER_PARAMS.second_lane_y:
            lane_change_completed = True
            lane_change_committed = False

        if any(history.goal_flag for history in histories):
            break

        collisions = detect_vehicle_collisions(states, vehicle_configs, min_distance=2.0)
        if collisions:
            collision_text = ", ".join(f"{first} vs {second}" for first, second in collisions)
            raise RuntimeError(f"Vehicle collision detected: {collision_text}")

        if get_role_item(states, role_to_index, "ego").x >= priority_road_end_x - 0.25:
            get_role_item(histories, role_to_index, "ego").goal_flag = True
            break

        frame_data = capture_simulation_frame(
            time=time,
            states=states,
            steers=steers,
            vehicle_configs=vehicle_configs,
            paths_by_role=paths_by_role,
            role_specific_paths=role_specific_paths,
            plot_requests=[],
            priority=requesting_priority,
            requester_role=requester_role,
            coordination_by_requester=None,
            active_receiver_roles=active_receiver_roles,
            receiver_accept_states={cooperating_role: accepting_vehicle_state},
            recent_events=event_log,
        )
        frame_log.append(frame_data)

        if show_animation:
            connect_escape_key()
            render_simulation_frame(
                plt.gca(),
                vehicle_configs,
                frame_data,
                road_end_x=125.0,
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
