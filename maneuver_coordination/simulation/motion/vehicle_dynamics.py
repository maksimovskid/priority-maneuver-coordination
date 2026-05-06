"""Simple bicycle-model vehicle dynamics helpers."""

import math
from typing import Sequence, Tuple

from maneuver_coordination.coordination.constants import DT
from maneuver_coordination.simulation.core.types import State


L = 2.7


def update(state: State, a: float, delta: float) -> State:
    """Propagate vehicle state by one fixed simulation step."""
    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / L * math.tan(delta) * DT
    state.v = state.v + a * DT
    state.a = a
    return state


def pi_2_pi(angle: float) -> float:
    """Normalize an angle to the [-pi, pi] range."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def calc_nearest_index(
    state: State,
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
) -> Tuple[int, float]:
    """Find the nearest path point and signed lateral error."""
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = [idx ** 2 + idy ** 2 for idx, idy in zip(dx, dy)]

    mind = min(d)
    ind = d.index(mind)
    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind
