"""Pure coordination decision helpers used by the behavioral planner."""

import math

import maneuver_coordination.coordination.constants as C


def transition_state(
    lane_change_maneuver,
    lane_change_path_found,
    lane_change_path_length,
    state_y,
    requesting_vehicle_state,
):
    """Advance requester motion state for the direct lane-change flow."""
    lane_change_complete = (
        lane_change_path_length > 2
        and state_y >= C.PLANNER_PARAMS.second_lane_y
    )

    if lane_change_complete:
        return C.MotionState.FOLLOW_SECOND_LANE
    if requesting_vehicle_state == C.RequestState.SEND_REQUEST:
        return C.MotionState.FIND_LANE_CHANGE
    if requesting_vehicle_state in (
        C.RequestState.EXECUTE_REQUEST,
        C.RequestState.NO_NEED_FOR_REQUEST,
    ):
        return C.MotionState.LANE_CHANGE
    if lane_change_maneuver and not lane_change_path_found:
        return C.MotionState.FIND_LANE_CHANGE
    return C.MotionState.FOLLOW_LANE


def transition_state_cascading(
    lane_change_path_length,
    state_y,
    requesting_vehicle_state,
    secondary_requesting_vehicle_state,
):
    """Advance requester motion state for the cascading coordination flow."""
    lane_change_complete = (
        lane_change_path_length > 2
        and state_y >= C.PLANNER_PARAMS.cascading_second_lane_y
    )
    lane_change_in_progress = (
        secondary_requesting_vehicle_state == C.RequestState.EXECUTE_REQUEST
    )

    if lane_change_complete:
        return C.MotionState.FOLLOW_SECOND_LANE
    if requesting_vehicle_state == C.RequestState.SEND_REQUEST:
        return C.MotionState.FIND_LANE_CHANGE
    if lane_change_in_progress:
        return C.MotionState.LANE_CHANGE
    return C.MotionState.FOLLOW_LANE


def calc_priority(state, state2):
    """Estimate request priority from the gap to the leading vehicle."""
    params = C.PLANNER_PARAMS
    requesting_priority = C.Priority.LOW_PRIORITY
    dx1 = state.x - state2.x
    dy1 = state.y - state2.y
    lead_car_distance = math.hypot(dx1, dy1) - 4

    time_gap = 1.0
    distance_gap_safe = params.safe_time_gap * state.v
    distance_gap = time_gap * state.v
    distance_gap_low = params.low_priority_time_gap * state.v

    if state2.x > state.x and distance_gap_safe < lead_car_distance < distance_gap_low:
        requesting_priority = C.Priority.LOW_PRIORITY
    if state2.x > state.x and distance_gap < lead_car_distance < distance_gap_safe:
        requesting_priority = C.Priority.MEDIUM_PRIORITY
    if state2.x > state.x and lead_car_distance < distance_gap:
        requesting_priority = C.Priority.HIGH_PRIORITY
    if state.y > params.no_priority_y or state.x > state2.x:
        requesting_priority = C.Priority.NO_PRIORITY_REQUEST

    return requesting_priority


def request_state_from_context(
    current_state,
    accepting_vehicle_state,
    conflict_free_req,
    can_send_request,
):
    """Map maneuver context into the next requester message state."""
    if current_state == C.MotionState.LANE_CHANGE:
        return C.RequestState.EXECUTE_REQUEST
    if current_state == C.MotionState.FOLLOW_SECOND_LANE:
        return C.RequestState.NO_REQUEST_SENT
    if current_state != C.MotionState.FIND_LANE_CHANGE:
        return C.RequestState.NO_REQUEST_SENT
    if accepting_vehicle_state == C.AcceptState.REQUEST_ACCEPTED:
        return C.RequestState.EXECUTE_REQUEST
    if conflict_free_req is True:
        return C.RequestState.NO_NEED_FOR_REQUEST
    if can_send_request:
        return C.RequestState.SEND_REQUEST
    return C.RequestState.NO_REQUEST_SENT


def accepting_vehicle_states(
    requesting_vehicle_state,
    requesting_priority,
    conflict_free,
    conflict_free_new,
    downstream_accepting_vehicle_state,
):
    """Decide accept/reject and speed adaptation for cascading cooperation."""
    accepting_vehicle_state = C.AcceptState.NO_REQUEST_RECEIVED
    speed_profile_state = C.SpeedState.FOLLOW_PLANNED

    if requesting_vehicle_state == C.RequestState.SEND_REQUEST:
        if requesting_priority == C.Priority.LOW_PRIORITY:
            if conflict_free:
                accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
            elif conflict_free_new:
                accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
                speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_LOW
            else:
                accepting_vehicle_state = C.AcceptState.REQUEST_REJECTED
        elif requesting_priority in (C.Priority.MEDIUM_PRIORITY, C.Priority.HIGH_PRIORITY):
            if not conflict_free and not conflict_free_new:
                if downstream_accepting_vehicle_state == C.AcceptState.REQUEST_ACCEPTED:
                    accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
                elif downstream_accepting_vehicle_state == C.AcceptState.REQUEST_REJECTED:
                    accepting_vehicle_state = C.AcceptState.REQUEST_REJECTED

    return accepting_vehicle_state, speed_profile_state


def accepting_vehicle_states_direct(
    requesting_vehicle_state,
    requesting_priority,
    conflict_free,
    conflict_free_new,
):
    """Decide accept/reject and speed adaptation for direct cooperation."""
    accepting_vehicle_state = C.AcceptState.NO_REQUEST_RECEIVED
    speed_profile_state = C.SpeedState.FOLLOW_PLANNED

    if requesting_vehicle_state == C.RequestState.EXECUTE_REQUEST:
        accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
        if requesting_priority == C.Priority.LOW_PRIORITY:
            speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_LOW
        elif requesting_priority == C.Priority.HIGH_PRIORITY:
            speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_HIGH
        else:
            speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_MEDIUM
    elif requesting_vehicle_state == C.RequestState.SEND_REQUEST:
        if requesting_priority == C.Priority.LOW_PRIORITY:
            if conflict_free:
                accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
            elif conflict_free_new:
                accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
                speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_LOW
            else:
                accepting_vehicle_state = C.AcceptState.REQUEST_REJECTED
        elif requesting_priority == C.Priority.MEDIUM_PRIORITY:
            if conflict_free_new:
                accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
                speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_MEDIUM
            else:
                accepting_vehicle_state = C.AcceptState.REQUEST_REJECTED
        elif requesting_priority == C.Priority.HIGH_PRIORITY:
            if conflict_free_new:
                accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
                speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_HIGH
            else:
                accepting_vehicle_state = C.AcceptState.REQUEST_REJECTED

    return accepting_vehicle_state, speed_profile_state


def accepting_vehicle_states_downstream(
    secondary_requesting_vehicle_state,
    requesting_priority,
    conflict_free,
    conflict_free_after_adaptation,
):
    """Decide accept/reject for the downstream vehicle in a cascade."""
    accepting_vehicle_state = C.AcceptState.NO_REQUEST_RECEIVED
    speed_profile_state = C.SpeedState.FOLLOW_PLANNED

    if secondary_requesting_vehicle_state == C.RequestState.EXECUTE_REQUEST:
        accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
        speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_MEDIUM
    elif secondary_requesting_vehicle_state == C.RequestState.SEND_REQUEST:
        if requesting_priority == C.Priority.LOW_PRIORITY:
            if conflict_free:
                accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
            elif conflict_free_after_adaptation:
                accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
                speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_LOW
            else:
                accepting_vehicle_state = C.AcceptState.REQUEST_REJECTED
        elif requesting_priority == C.Priority.MEDIUM_PRIORITY:
            if conflict_free_after_adaptation:
                accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
                speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_MEDIUM
            else:
                accepting_vehicle_state = C.AcceptState.REQUEST_REJECTED
        elif requesting_priority == C.Priority.HIGH_PRIORITY:
            if conflict_free_after_adaptation:
                accepting_vehicle_state = C.AcceptState.REQUEST_ACCEPTED
                speed_profile_state = C.SpeedState.REDUCE_SPEED_FOR_REQ_HIGH
            else:
                accepting_vehicle_state = C.AcceptState.REQUEST_REJECTED

    return accepting_vehicle_state, speed_profile_state


def check_for_lane_change(state, state2, existing_rrt_length):
    """Detect when a slower lead vehicle should trigger lane-change planning."""
    params = C.PLANNER_PARAMS
    dx1 = state.x - state2.x
    dy1 = state.y - state2.y
    lead_car_distance = math.hypot(dx1, dy1) - 4
    distance_gap = params.safe_time_gap * state.v

    if lead_car_distance < distance_gap and state2.v < params.lane_change_trigger_speed and existing_rrt_length < 2:
        return True
    return False


def check_for_double_lane_change(state, state2, existing_rrt_length):
    """Detect when the legacy double-lane-change return maneuver should start."""
    params = C.PLANNER_PARAMS
    dx2 = state.x - state2.x
    dy2 = state.y - state2.y
    lead_car_distance = math.hypot(dx2, dy2) - 4
    distance_gap = params.double_lane_change_time_gap * state.v
    return existing_rrt_length > params.lane_change_path_points and lead_car_distance > distance_gap
