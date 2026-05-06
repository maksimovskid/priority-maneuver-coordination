"""Shared helpers for trajectory conflict checks."""

import math


def conflict_time_indices(speed, horizon_length, lane_change_distance=12.0, dt=0.1):
    """Choose representative start/middle/end samples for conflict checking."""
    safe_speed = max(abs(speed), 1e-6)
    lane_change_time = lane_change_distance / safe_speed

    t1 = 0
    t2 = int(round((lane_change_time / 2.0) / dt))
    t3 = int(round(lane_change_time / dt) - 1)

    max_index = max(0, horizon_length - 1)
    return min(t1, max_index), min(t2, max_index), min(t3, max_index)


def sampled_clearances(xreq, xref, speed, first_offset=2.0, later_offset=4.0, lane_change_distance=12.0, dt=0.1):
    """Compute clearance margins between a requested and planned trajectory."""
    t1, t2, t3 = conflict_time_indices(speed, xreq.shape[1], lane_change_distance=lane_change_distance, dt=dt)

    dx1 = xreq[0, t1] - xref[0, t1]
    dy1 = xreq[1, t1] - xref[1, t1]
    dx2 = xreq[0, t2] - xref[0, t2]
    dy2 = xreq[1, t2] - xref[1, t2]
    dx3 = xreq[0, t3] - xref[0, t3]
    dy3 = xreq[1, t3] - xref[1, t3]

    return (
        math.hypot(dx1, dy1) - first_offset,
        math.hypot(dx2, dy2) - later_offset,
        math.hypot(dx3, dy3) - later_offset,
    )


def is_sampled_trajectory_conflict_free(clearances, distance_gap, initial_gap=1.0):
    """Return whether all sampled clearances satisfy the required gaps."""
    tr_1_distance, tr_2_distance, tr_3_distance = clearances
    return (
        tr_1_distance > initial_gap
        and tr_2_distance > distance_gap
        and tr_3_distance > distance_gap
    )
