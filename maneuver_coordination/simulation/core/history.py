"""Vehicle history helpers used for plotting and closed-loop bookkeeping."""

from typing import Sequence

from maneuver_coordination.simulation.motion.vehicle_dynamics import calc_nearest_index
from maneuver_coordination.simulation.core.types import State, VehicleHistory


def init_vehicle_history(
    state: State,
    cx: Sequence[float],
    cy: Sequence[float],
    cyaw: Sequence[float],
) -> VehicleHistory:
    """Initialize a history object and nearest-path index for one vehicle."""
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw)
    return VehicleHistory(
        x=[state.x],
        y=[state.y],
        yaw=[state.yaw],
        v=[state.v],
        d=[0.0],
        t=[0.0],
        goal_flag=False,
        target_ind=target_ind,
    )


def append_history(hist: VehicleHistory, state: State, steer: float, time: float) -> None:
    """Append one simulation sample to a vehicle history."""
    hist.x.append(state.x)
    hist.y.append(state.y)
    hist.yaw.append(state.yaw)
    hist.v.append(state.v)
    hist.d.append(steer)
    hist.t.append(time)
