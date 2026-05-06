"""Small kinematic helper formulas shared by simulation modules."""

import math
from typing import Sequence

from maneuver_coordination.simulation.core.types import State


def calc_distance(v_i: float, v_f: float, a: float) -> float:
    """Return distance needed to change speed under constant acceleration."""
    return (v_f * v_f - v_i * v_i) / (2 * a)


def calc_time(v_i: float, v_f: float, a: float) -> float:
    """Return time needed to change speed under constant acceleration."""
    return (v_f - v_i) / a


def calc_final_speed(v_i: float, a: float, d: float) -> float:
    """Return final speed after travelling distance `d` with acceleration `a`."""
    temp = v_i * v_i + 2 * d * a
    return 0.000001 if temp < 0 else math.sqrt(temp)


def reached_goal(state: State, goal: Sequence[float], goal_dis: float = 1.0) -> bool:
    """Check whether a vehicle is close enough to its configured goal point."""
    return math.hypot(state.x - goal[0], state.y - goal[1]) <= goal_dis

