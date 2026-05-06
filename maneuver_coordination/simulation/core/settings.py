from maneuver_coordination.coordination import constants as C

ROAD_END_X = 79.0
EGO_STOP_MARGIN = 0.25
VEHICLE_COLLISION_DISTANCE = 3.5
BRAKING_SPEED_DELTA = 0.3
ACC_TIME_GAP = 1.0
ACC_MIN_GAP = 6.0
ACC_SPEED_MARGIN = 0.5
ADAPTATION_ACCEL_STEP = 0.5
ADAPTATION_SPEED_STEP_RATIO = 0.05
ADAPTATION_ACCEL_LIMIT_BY_PRIORITY = {
    C.low_priority: 2.0,
    C.medium_priority: 4.0,
    C.high_priority: 9.5,
}
ADAPTATION_SPEED_LIMIT_RATIO_BY_PRIORITY = {
    C.low_priority: 0.20,
    C.medium_priority: 0.40,
    C.high_priority: 0.75,
}
