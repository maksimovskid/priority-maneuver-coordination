"""Requested and reference trajectory helpers for the behavioral planner."""

import numpy as np

import maneuver_coordination.coordination.constants as C


def _advance_prediction_travel(current_speed: float, acceleration: float, dt: float) -> tuple[float, float]:
    step_speed = max(0.0, current_speed)
    step_travel = step_speed * dt + 0.5 * acceleration * (dt ** 2)
    next_speed = max(0.0, current_speed + acceleration * dt)
    return max(0.0, step_travel), next_speed


def empty_requested_trajectory():
    """Return an empty requested-trajectory placeholder."""
    params = C.PLANNER_PARAMS
    return (
        np.zeros((params.nx, params.horizon_steps + params.lane_change_path_padding)),
        0,
        np.zeros((1, params.horizon_steps + params.lane_change_path_padding)),
        False,
    )


def select_requested_path(has_path_fn, path_tuple_fn, generated_path_key, committed_path_key, current_state):
    """Choose the generated or committed path to advertise as requested."""
    if has_path_fn(generated_path_key) and current_state == C.MotionState.FIND_LANE_CHANGE:
        return path_tuple_fn(generated_path_key), True

    valid_states = (
        C.MotionState.FIND_LANE_CHANGE,
        C.MotionState.LANE_CHANGE,
        C.MotionState.FOLLOW_SECOND_LANE,
    )
    if has_path_fn(committed_path_key) and current_state in valid_states:
        return path_tuple_fn(committed_path_key), True

    return None, False


def build_requested_trajectory(state, path_segment, sp, dl, nearest_index_fn):
    """Sample the requester's future cooperative trajectory on a local path."""
    params = C.PLANNER_PARAMS
    cx, cy, cyaw, _ = path_segment
    xreq = np.zeros((params.nx, params.horizon_steps + params.lane_change_path_padding))
    dreq = np.zeros((1, params.horizon_steps + params.lane_change_path_padding))

    if len(cx) == 0:
        return xreq, 0, dreq, False

    indq, _ = nearest_index_fn(state, cx, cy, cyaw)
    ncourse = len(cx)

    xreq[0, 0] = state.x
    xreq[1, 0] = state.y
    xreq[2, 0] = state.v
    xreq[3, 0] = state.yaw
    dreq[0, 0] = 0.0

    travel = 0.0
    predicted_speed = state.v
    predicted_acceleration = getattr(state, "a", 0.0)
    for i in range(1, params.horizon_steps):
        step_travel, predicted_speed = _advance_prediction_travel(
            predicted_speed,
            predicted_acceleration,
            params.dt,
        )
        travel += step_travel
        dind = int(round(travel / dl))
        sample_index = min(indq + dind, ncourse - 1)

        xreq[0, i] = cx[sample_index]
        xreq[1, i] = cy[sample_index]
        xreq[2, i] = sp[sample_index]
        xreq[3, i] = cyaw[sample_index]
        dreq[0, i] = 0.0

    return xreq, indq, dreq, True


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, nearest_index_fn):
    """Build a short-horizon planned trajectory for conflict checks."""
    _ = ck
    params = C.PLANNER_PARAMS
    xref = np.zeros((params.nx, params.horizon_steps + params.lane_change_path_padding))
    dref = np.zeros((1, params.horizon_steps + params.lane_change_path_padding))
    ncourse = len(cx)

    ind, _ = nearest_index_fn(state, cx, cy, cyaw)

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0

    travel = 0.0
    predicted_speed = state.v
    predicted_acceleration = getattr(state, "a", 0.0)
    for i in range(params.horizon_steps):
        step_travel, predicted_speed = _advance_prediction_travel(
            predicted_speed,
            predicted_acceleration,
            params.dt,
        )
        travel += step_travel
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref
