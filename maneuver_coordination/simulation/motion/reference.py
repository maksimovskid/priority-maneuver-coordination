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
    xref = np.zeros((NX, T if len(cx) else 0))
    dref = np.zeros((1, xref.shape[1]))
    ncourse = len(cx)

    if ncourse == 0:
        return xref, 0, dref

    ind, _ = calc_nearest_index(state, cx, cy, cyaw)

    xref[:, 0] = [state.x, state.y, state.v, state.yaw]
    dref[0, 0] = 0.0

    travel = 0.0
    predicted_speed = state.v
    predicted_acceleration = getattr(state, "a", 0.0)
    for i in range(1, T):
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
    xref = np.zeros((NX, T if len(cx) else 0))
    dref = np.zeros((1, xref.shape[1]))
    ncourse = len(cx)

    if ncourse == 0:
        return xref, 0, dref

    ind, _ = calc_nearest_index(state, cx, cy, cyaw)

    xref[:, 0] = [state.x, state.y, state.v, state.yaw]
    dref[0, 0] = 0.0

    travel = 0.0
    predicted_speed = state.v
    predicted_acceleration = getattr(state, "a", 0.0)
    for i in range(1, T):
        next_speed = _advance_prediction_travel(
            predicted_speed,
            predicted_acceleration,
            DT,
        )[1]
        start_speed = predicted_speed - speed_offset
        end_speed = next_speed - speed_offset
        if use_abs_diff:
            start_speed, end_speed = abs(start_speed), abs(end_speed)
        else:
            start_speed, end_speed = max(0.0, start_speed), max(0.0, end_speed)
        travel += 0.5 * (start_speed + end_speed) * DT
        predicted_speed = next_speed
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
    xref = np.zeros((NX, T if len(cx) else 0))
    dref = np.zeros((1, xref.shape[1]))
    ncourse = len(cx)

    if ncourse == 0:
        return xref, 0, dref

    ind, _ = calc_nearest_index(state, cx, cy, cyaw)

    xref[:, 0] = [state.x, state.y, state.v, state.yaw]
    dref[0, 0] = 0.0

    travel = 0.0
    step_speed = max(0.0, target_speed)
    for i in range(1, T):
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
    xref = np.zeros((NX, T if len(cx) else 0))
    dref = np.zeros((1, xref.shape[1]))
    ncourse = len(cx)

    if ncourse == 0:
        return xref, 0, dref, state.v

    ind, _ = calc_nearest_index(state, cx, cy, cyaw)

    xref[:, 0] = [state.x, state.y, state.v, state.yaw]
    dref[0, 0] = 0.0

    travel = 0.0
    predicted_speed = state.v
    for i in range(1, T):
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
    dref = np.zeros((1, T if len(cx) else 0))

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
    """Return curvature at each original path knot, retaining repeated points.

    ds is retained for call compatibility; input geometry is not resampled.
    """
    if len(path_x) != len(path_y):
        raise ValueError("Path x and y must have equal lengths.")
    if len(path_x) == 0:
        return None, np.empty(0), []
    points = np.column_stack((path_x, path_y))
    if not np.isfinite(points).all():
        raise ValueError("Path coordinates must be finite.")
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0.0, axis=1)]
    unique_points = points[keep]
    if len(unique_points) < 2:
        return None, np.zeros(len(points)), [0.0] * len(points)

    spl = Spline2D(unique_points[:, 0], unique_points[:, 1])
    curvature = []
    for sample in spl.s:
        # The legacy derivative API cannot evaluate its terminal knot.
        parameter = min(sample, np.nextafter(spl.s[-1], -np.inf))
        dx, dy = spl.sx.calcd(parameter), spl.sy.calcd(parameter)
        ddx, ddy = spl.sx.calcdd(parameter), spl.sy.calcdd(parameter)
        speed_squared = dx * dx + dy * dy
        if speed_squared == 0.0:
            raise ValueError("Curvature is undefined at a stationary spline point.")
        curvature.append((dx * ddy - dy * ddx) / speed_squared ** 1.5)
    indices = np.cumsum(keep) - 1
    s = np.asarray(spl.s)[indices]
    return spl, s, [curvature[index] for index in indices]
