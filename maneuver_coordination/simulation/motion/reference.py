"""Reference trajectory and geometric path generation helpers."""

from typing import Sequence, Tuple

import numpy as np

from maneuver_coordination.motion_planning.cubic_spline_planner import Spline2D
from maneuver_coordination.motion_planning.hybrid_a_star import hybrid_a_star_planning

from maneuver_coordination.coordination.constants import DT, NX, T
from maneuver_coordination.simulation.motion.vehicle_dynamics import calc_nearest_index
from maneuver_coordination.simulation.core.types import State


XY_GRID_RESOLUTION = 1.0
YAW_GRID_RESOLUTION = np.deg2rad(15.0)


def _advance_prediction_travel(current_speed: float, acceleration: float, dt: float) -> tuple[float, float]:
    step_speed = max(0.0, current_speed)
    step_travel = step_speed * dt + 0.5 * acceleration * (dt ** 2)
    next_speed = max(0.0, current_speed + acceleration * dt)
    return max(0.0, step_travel), next_speed


def calc_ref_trajectory(
    state: State,
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
    ck: Sequence[float],
    sp: Sequence[float],
    dl: float,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Build the short-horizon reference trajectory tracked by the controller."""
    _ = ck
    xref = np.zeros((NX, T + 100))
    dref = np.zeros((1, T + 100))
    ncourse = len(cx)

    if ncourse == 0:
        xref[0, :] = state.x
        xref[1, :] = state.y
        xref[2, :] = state.v
        xref[3, :] = state.yaw
        return xref, 0, dref

    ind, _ = calc_nearest_index(state, cx, cy, cyaw)

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0

    travel = 0.0
    predicted_speed = state.v
    predicted_acceleration = getattr(state, "a", 0.0)
    for i in range(T):
        step_travel, predicted_speed = _advance_prediction_travel(
            predicted_speed,
            predicted_acceleration,
            DT,
        )
        travel += step_travel
        dind = int(round(travel / dl))
        idx = min(ind + dind, ncourse - 1)

        xref[0, i] = cx[idx]
        xref[1, i] = cy[idx]
        xref[2, i] = sp[idx]
        xref[3, i] = cyaw[idx]
        dref[0, i] = 0.0

    return xref, ind, dref


def build_reference_for_speed(
    state: State,
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
    speed_profile: Sequence[float],
    dl: float,
    speed_offset: float,
    use_abs_diff: bool = True,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Build a reference using a speed-offset prediction model."""
    xref = np.zeros((NX, T + 100))
    dref = np.zeros((1, T + 100))
    ncourse = len(cx)

    if ncourse == 0:
        xref[0, :] = state.x
        xref[1, :] = state.y
        xref[2, :] = state.v
        xref[3, :] = state.yaw
        return xref, 0, dref

    ind, _ = calc_nearest_index(state, cx, cy, cyaw)

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = speed_profile[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0

    travel = 0.0
    predicted_speed = state.v
    predicted_acceleration = getattr(state, "a", 0.0)
    for i in range(T):
        _, predicted_speed = _advance_prediction_travel(
            predicted_speed,
            predicted_acceleration,
            DT,
        )
        step_speed = abs(predicted_speed - speed_offset) if use_abs_diff else (abs(predicted_speed) - speed_offset)
        travel += max(0.0, step_speed) * DT
        dind = int(round(travel / dl))
        idx = min(ind + dind, ncourse - 1)

        xref[0, i] = cx[idx]
        xref[1, i] = cy[idx]
        xref[2, i] = speed_profile[idx]
        xref[3, i] = cyaw[idx]
        dref[0, i] = 0.0

    return xref, ind, dref


def build_reference_for_target_speed(
    state: State,
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
    dl: float,
    target_speed: float,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Build a reference assuming the vehicle tracks a constant target speed."""
    xref = np.zeros((NX, T + 100))
    dref = np.zeros((1, T + 100))
    ncourse = len(cx)

    if ncourse == 0:
        xref[0, :] = state.x
        xref[1, :] = state.y
        xref[2, :] = target_speed
        xref[3, :] = state.yaw
        return xref, 0, dref

    ind, _ = calc_nearest_index(state, cx, cy, cyaw)

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = target_speed
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0

    travel = 0.0
    step_speed = max(0.0, target_speed)
    for i in range(T):
        travel += step_speed * DT
        dind = int(round(travel / dl))
        idx = min(ind + dind, ncourse - 1)

        xref[0, i] = cx[idx]
        xref[1, i] = cy[idx]
        xref[2, i] = target_speed
        xref[3, i] = cyaw[idx]
        dref[0, i] = 0.0

    return xref, ind, dref


def build_reference_for_acceleration(
    state: State,
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
    dl: float,
    acceleration: float,
) -> Tuple[np.ndarray, int, np.ndarray, float]:
    """Build a reference under constant acceleration and return final speed."""
    xref = np.zeros((NX, T + 100))
    dref = np.zeros((1, T + 100))
    ncourse = len(cx)

    if ncourse == 0:
        xref[0, :] = state.x
        xref[1, :] = state.y
        xref[2, :] = state.v
        xref[3, :] = state.yaw
        return xref, 0, dref, state.v

    ind, _ = calc_nearest_index(state, cx, cy, cyaw)

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = state.v
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0

    travel = 0.0
    predicted_speed = state.v
    for i in range(T):
        step_travel, predicted_speed = _advance_prediction_travel(
            predicted_speed,
            acceleration,
            DT,
        )
        travel += step_travel
        dind = int(round(travel / dl))
        idx = min(ind + dind, ncourse - 1)

        xref[0, i] = cx[idx]
        xref[1, i] = cy[idx]
        xref[2, i] = predicted_speed
        xref[3, i] = cyaw[idx]
        dref[0, i] = 0.0

    return xref, ind, dref, predicted_speed


def build_three_candidate_refs(
    state: State,
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
    dl: float,
    reductions: Sequence[float],
    use_abs_diff: bool = True,
):
    """Build the legacy low/medium/high speed-reduction references."""
    refs = []
    ind = 0
    dref = np.zeros((1, T + 100))

    for reduction in reductions:
        new_target_speed = state.v - reduction
        speed_profile = [new_target_speed] * len(cx)
        xref, ind, dref = build_reference_for_speed(
            state,
            cx,
            cy,
            cyaw,
            speed_profile,
            dl,
            reduction,
            use_abs_diff,
        )
        refs.append(xref)

    return refs[0], refs[1], refs[2], ind, dref


def plan_path(start: Sequence[float], goal: Sequence[float], ox: Sequence[float], oy: Sequence[float]):
    """Plan a geometric path between start and goal using hybrid A*."""
    return hybrid_a_star_planning(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)


def build_curvature(path_x: Sequence[float], path_y: Sequence[float], ds: float = 0.1):
    """Fit a spline through path points and sample curvature along it."""
    spl = Spline2D(path_x, path_y)
    s = np.arange(0, spl.s[-1], ds)
    ck = [spl.calc_curvature(i_s) for i_s in s]
    return spl, s, ck
