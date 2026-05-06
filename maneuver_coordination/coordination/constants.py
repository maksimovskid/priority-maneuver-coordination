"""Shared state constants used by the coordination logic."""

from dataclasses import dataclass, field
from enum import IntEnum


class MotionState(IntEnum):
    """Discrete lateral/maneuver execution states for a requester vehicle."""
    FOLLOW_LANE = 0
    FIND_LANE_CHANGE = 1
    LANE_CHANGE = 2
    EXTEND_PATH_AFTER_LANE_CHANGE = 3
    FOLLOW_SECOND_LANE = 4
    FIND_DOUBLE_LANE_CHANGE = 5
    DOUBLE_LANE_CHANGE = 6
    EXTEND_PATH_AFTER_DOUBLE_LANE_CHANGE = 7
    FOLLOW_LANE_AGAIN = 8


class RequestState(IntEnum):
    """Message/request state from the requester perspective."""
    NO_REQUEST_SENT = 10
    SEND_REQUEST = 11
    CONFIRM_OFFERS = 16
    EXECUTE_REQUEST = 17
    NO_NEED_FOR_REQUEST = 18
    CASCADING_WAIT = 19


class AcceptState(IntEnum):
    """Message/request state from the accepting vehicle perspective."""
    NO_REQUEST_RECEIVED = 21
    REQUEST_RECEIVED = 22
    REQUEST_REJECTED = 23
    REQUEST_ACCEPTED = 25
    OFFER = 26


class SpeedState(IntEnum):
    """Longitudinal adaptation mode selected for cooperative motion."""
    FOLLOW_PLANNED = 31
    REDUCE_SPEED_FOR_REQ_LOW = 32
    REDUCE_SPEED_FOR_REQ_MEDIUM = 33
    REDUCE_SPEED_FOR_REQ_HIGH = 34
    INCREASE_SPEED_FOR_REQ_LOW = 35
    INCREASE_SPEED_FOR_REQ_MEDIUM = 36
    INCREASE_SPEED_FOR_REQ_HIGH = 37


class Priority(IntEnum):
    """Priority level attached to a maneuver request."""
    NO_PRIORITY_REQUEST = 50
    LOW_PRIORITY = 51
    MEDIUM_PRIORITY = 52
    HIGH_PRIORITY = 53
    EMERGENCY_MANEUVER = 54


@dataclass(frozen=True)
class PlannerParams:
    """Central tuning parameters used by behavior and trajectory planning."""
    nx: int = 4
    nu: int = 2
    horizon_steps: int = 20
    dt: float = 0.1
    lane_change_path_padding: int = 100
    lane_change_path_points: int = 120
    lane_change_goal_offset: int = 110
    rrt_search_area: list[float] = field(default_factory=lambda: [0.0, 70.0])
    second_lane_y: float = 5.8
    cascading_second_lane_y: float = 9.8
    no_priority_y: float = 4.8
    safe_time_gap: float = 1.5
    low_priority_time_gap: float = 1.8
    double_lane_change_time_gap: float = 1.0
    lane_change_trigger_speed: float = 28.0 / 3.6


follow_lane = MotionState.FOLLOW_LANE
find_lane_change = MotionState.FIND_LANE_CHANGE
lane_change = MotionState.LANE_CHANGE
extend_path_after_lane_change = MotionState.EXTEND_PATH_AFTER_LANE_CHANGE
follow_second_lane = MotionState.FOLLOW_SECOND_LANE
find_double_lane_change = MotionState.FIND_DOUBLE_LANE_CHANGE
double_lane_change = MotionState.DOUBLE_LANE_CHANGE
extend_path_after_double_lane_change = MotionState.EXTEND_PATH_AFTER_DOUBLE_LANE_CHANGE
follow_lane_again = MotionState.FOLLOW_LANE_AGAIN

no_request_sent = RequestState.NO_REQUEST_SENT
send_request = RequestState.SEND_REQUEST
confirm_offers = RequestState.CONFIRM_OFFERS
execute_request = RequestState.EXECUTE_REQUEST
no_need_for_request = RequestState.NO_NEED_FOR_REQUEST
cascading_wait = RequestState.CASCADING_WAIT

no_request_received = AcceptState.NO_REQUEST_RECEIVED
request_received = AcceptState.REQUEST_RECEIVED
request_rejected = AcceptState.REQUEST_REJECTED
request_accepted = AcceptState.REQUEST_ACCEPTED
offer = AcceptState.OFFER

follow_planned = SpeedState.FOLLOW_PLANNED
reduce_speed_for_req_low = SpeedState.REDUCE_SPEED_FOR_REQ_LOW
reduce_speed_for_req_medium = SpeedState.REDUCE_SPEED_FOR_REQ_MEDIUM
reduce_speed_for_req_high = SpeedState.REDUCE_SPEED_FOR_REQ_HIGH
increase_speed_for_req_low = SpeedState.INCREASE_SPEED_FOR_REQ_LOW
increase_speed_for_req_medium = SpeedState.INCREASE_SPEED_FOR_REQ_MEDIUM
increase_speed_for_req_high = SpeedState.INCREASE_SPEED_FOR_REQ_HIGH

no_priority_request = Priority.NO_PRIORITY_REQUEST
low_priority = Priority.LOW_PRIORITY
medium_priority = Priority.MEDIUM_PRIORITY
high_priority = Priority.HIGH_PRIORITY
emergency_maneuver = Priority.EMERGENCY_MANEUVER

PLANNER_PARAMS = PlannerParams()

NX = PLANNER_PARAMS.nx
NU = PLANNER_PARAMS.nu
T = PLANNER_PARAMS.horizon_steps
DT = PLANNER_PARAMS.dt

LANE_CHANGE_PATH_PADDING = PLANNER_PARAMS.lane_change_path_padding
LANE_CHANGE_PATH_POINTS = PLANNER_PARAMS.lane_change_path_points
LANE_CHANGE_GOAL_OFFSET = PLANNER_PARAMS.lane_change_goal_offset
RRT_SEARCH_AREA = PLANNER_PARAMS.rrt_search_area
SECOND_LANE_Y = PLANNER_PARAMS.second_lane_y
CASCADING_SECOND_LANE_Y = PLANNER_PARAMS.cascading_second_lane_y
NO_PRIORITY_Y = PLANNER_PARAMS.no_priority_y
SAFE_TIME_GAP = PLANNER_PARAMS.safe_time_gap
LOW_PRIORITY_TIME_GAP = PLANNER_PARAMS.low_priority_time_gap
DOUBLE_LANE_CHANGE_TIME_GAP = PLANNER_PARAMS.double_lane_change_time_gap
LANE_CHANGE_TRIGGER_SPEED = PLANNER_PARAMS.lane_change_trigger_speed
