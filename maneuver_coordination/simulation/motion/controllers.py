"""Longitudinal and lateral feedback controllers used by scenario runners."""

import math
from typing import Sequence, Tuple

from maneuver_coordination.simulation.core.types import State
from maneuver_coordination.simulation.motion.vehicle_dynamics import calc_nearest_index, pi_2_pi
from maneuver_coordination.vehicle.model import WB


Kp = 10.0
Kd = 0.1
KTH = 0.7
KE = 0.3


def pid_control(target: float, current: float, min_acceleration: float = -4.0, max_acceleration: float = 4.0) -> float:
    """Compute bounded longitudinal acceleration toward a target speed."""
    acceleration = Kp * (target - current)
    return max(min_acceleration, min(max_acceleration, acceleration))


def rear_wheel_feedback_control(
    state: State,
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
    ck: Sequence[float],
    preind: int,
) -> Tuple[float, int]:
    """Compute steering with rear-wheel feedback path tracking."""
    _ = preind
    ind, e = calc_nearest_index(state, cx, cy, cyaw)

    if not ck:
        return 0.0, ind

    curvature_index = min(ind, len(ck) - 1)
    k = ck[curvature_index]
    v = state.v
    th_e = pi_2_pi(state.yaw - cyaw[ind])
    sin_term = 1.0 if abs(th_e) < 1e-6 else math.sin(th_e) / th_e

    omega = (
        v * k * math.cos(th_e) / (1.0 - k * e)
        - KTH * abs(v) * th_e
        - KE * v * sin_term * e
    )

    if omega == 0.0 or v == 0.0:
        return 0.0, ind

    delta = math.atan2(WB * omega / v, 1.0)
    return min(delta, 0.6), ind
