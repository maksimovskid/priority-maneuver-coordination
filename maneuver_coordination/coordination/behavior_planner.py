"""Behavior planner facade used by the scenario runners.

The class keeps the legacy method names from the original prototype, but routes
most work to smaller decision/trajectory helpers. It owns generated and
committed lane-change paths for requester, secondary requester, and return
paths.

author: Daniel Maksimovski
"""

import math
from typing import NamedTuple

import numpy as np

import maneuver_coordination.coordination.constants as C
from maneuver_coordination.coordination.conflicts import (
    is_sampled_trajectory_conflict_free,
    sampled_clearances,
)
from maneuver_coordination.coordination.decision_logic import (
    accepting_vehicle_states as compute_accepting_vehicle_states,
    accepting_vehicle_states_downstream as compute_accepting_vehicle_states_downstream,
    accepting_vehicle_states_direct as compute_accepting_vehicle_states_direct,
    calc_priority as compute_priority,
    check_for_double_lane_change as compute_double_lane_change_maneuver,
    check_for_lane_change as compute_lane_change_maneuver,
    request_state_from_context as compute_request_state,
    transition_state as compute_transition_state,
    transition_state_cascading as compute_transition_state_cascading,
)
from maneuver_coordination.coordination.trajectory_logic import (
    build_requested_trajectory as compute_requested_trajectory,
    calc_ref_trajectory as compute_ref_trajectory,
    empty_requested_trajectory as compute_empty_requested_trajectory,
    select_requested_path as compute_selected_requested_path,
)
from maneuver_coordination.motion_planning.cubic_spline_planner import Spline2D
try:
    from maneuver_coordination.motion_planning.rrt_reeds_shepp import RRTReedsShepp
except ImportError:
    raise

DEBUG = False

SHOW_ANIMATION = True


def _debug(*args):
    if DEBUG:
        print(*args)


PATH_AXES = ("x", "y", "yaw", "k")


class ReferencePath(NamedTuple):
    """Named reference path tuple used to avoid ambiguous cx/cx_2 arguments."""
    x: list
    y: list
    yaw: list
    curvature: list


def _legacy_axis_attrs(path_key, prefix, suffix=""):
    return {
        f"{prefix}_{axis}{suffix}": (path_key, index)
        for index, axis in enumerate(PATH_AXES)
    }


class BehaviouralLocalPlanner:
    """Stateful planner wrapper for maneuver search, path buffers, and requests."""
    AXES = PATH_AXES
    REQUESTER_ROLE = "requester"
    SECONDARY_ROLE = "secondary"
    RETURN_ROLE = "return"
    REQUESTER_GENERATED_PATH_KEY = "requester_generated"
    REQUESTER_COMMITTED_PATH_KEY = "requester_committed"
    RETURN_PATH_KEY = "return_path"
    SECONDARY_GENERATED_PATH_KEY = "secondary_generated"
    SECONDARY_COMMITTED_PATH_KEY = "secondary_committed"

    # Compatibility aliases for older code/tests that used RRT-centric names.
    PRIMARY_RRT_KEY = REQUESTER_GENERATED_PATH_KEY
    PRIMARY_COMMITTED_KEY = REQUESTER_COMMITTED_PATH_KEY
    DOUBLE_RRT_KEY = RETURN_PATH_KEY
    CASCADING_RRT_KEY = SECONDARY_GENERATED_PATH_KEY
    CASCADING_COMMITTED_KEY = SECONDARY_COMMITTED_PATH_KEY
    ROLE_TO_PATH_KEYS = {
        REQUESTER_ROLE: {
            "generated": REQUESTER_GENERATED_PATH_KEY,
            "committed": REQUESTER_COMMITTED_PATH_KEY,
        },
        SECONDARY_ROLE: {
            "generated": SECONDARY_GENERATED_PATH_KEY,
            "committed": SECONDARY_COMMITTED_PATH_KEY,
        },
        RETURN_ROLE: {
            "generated": RETURN_PATH_KEY,
            "committed": RETURN_PATH_KEY,
        },
    }
    ROLE_ALIASES = {
        "ego": REQUESTER_ROLE,
        "acceptor_1": SECONDARY_ROLE,
        "return": RETURN_ROLE,
    }
    LEGACY_PATH_ATTRS = {
        **_legacy_axis_attrs(REQUESTER_GENERATED_PATH_KEY, "rrt_lane_change"),
        **_legacy_axis_attrs(REQUESTER_COMMITTED_PATH_KEY, "lane_change_state"),
        **_legacy_axis_attrs(RETURN_PATH_KEY, "rrt_lane_change", "_2"),
        **_legacy_axis_attrs(SECONDARY_GENERATED_PATH_KEY, "rrt_lane_change", "3"),
        **_legacy_axis_attrs(SECONDARY_COMMITTED_PATH_KEY, "lane_change_state", "3"),
    }

    def __init__(self, cx, cy, cyaw, ck):
        _ = (cx, cy, cyaw, ck)
        self.follow_lead_vehicle = False
        self.lane_change_maneuver = False
        self.double_lane_change_maneuver = False
        self.lane_change_path_found = False
        self.double_lane_change_path_found = False
        self.lane_change_path_found_3 = False
        self.double_lane_change_path_found_3 = False
        self.path_buffers = {
            self.PRIMARY_RRT_KEY: [[] for _ in self.AXES],
            self.PRIMARY_COMMITTED_KEY: [[] for _ in self.AXES],
            self.DOUBLE_RRT_KEY: [[] for _ in self.AXES],
            self.CASCADING_RRT_KEY: [[] for _ in self.AXES],
            self.CASCADING_COMMITTED_KEY: [[] for _ in self.AXES],
        }

    def __getattr__(self, name):
        if name in self.LEGACY_PATH_ATTRS and "path_buffers" in self.__dict__:
            path_key, index = self.LEGACY_PATH_ATTRS[name]
            return self.path_buffers[path_key][index]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name, value):
        legacy_attrs = getattr(type(self), "LEGACY_PATH_ATTRS", {})
        if name in legacy_attrs and "path_buffers" in self.__dict__:
            path_key, index = legacy_attrs[name]
            self.path_buffers[path_key][index] = list(value)
            return
        super().__setattr__(name, value)

    def _reset_path_group(self, path_key):
        self.path_buffers[path_key] = [[] for _ in self.AXES]

    def _normalize_role(self, role):
        return self.ROLE_ALIASES.get(role, role)

    def _ensure_path_key(self, path_key):
        if path_key not in self.path_buffers:
            self.path_buffers[path_key] = [[] for _ in self.AXES]
        return path_key

    def _generated_path_key(self, role):
        normalized_role = self._normalize_role(role)
        if normalized_role in self.ROLE_TO_PATH_KEYS:
            return self._ensure_path_key(self.ROLE_TO_PATH_KEYS[normalized_role]["generated"])
        return self._ensure_path_key(f"{normalized_role}_generated")

    def _committed_path_key(self, role):
        normalized_role = self._normalize_role(role)
        if normalized_role in self.ROLE_TO_PATH_KEYS:
            return self._ensure_path_key(self.ROLE_TO_PATH_KEYS[normalized_role]["committed"])
        return self._ensure_path_key(f"{normalized_role}_committed")

    def _path_found_attr(self, role):
        normalized_role = self._normalize_role(role)
        if normalized_role == self.SECONDARY_ROLE:
            return "lane_change_path_found_3"
        return "lane_change_path_found"

    def _get_path_found(self, role):
        return getattr(self, self._path_found_attr(role))

    def _set_path_found(self, role, value):
        setattr(self, self._path_found_attr(role), value)

    def _append_path_segment(self, path_key, path_segment):
        for values, buffer_values in zip(path_segment, self.path_buffers[path_key]):
            buffer_values.extend(values)

    @staticmethod
    def _reference_path(cx, cy, cyaw, ck):
        return ReferencePath(cx, cy, cyaw, ck)

    def _append_reference_suffix(self, path_key, reference_path, start_index):
        if len(reference_path.x) <= 1:
            return

        start_index = min(start_index, len(reference_path.x) - 1)
        suffix_segment = (
            reference_path.x[start_index:-1],
            reference_path.y[start_index:-1],
            reference_path.yaw[start_index:-1],
            reference_path.curvature[start_index:-1],
        )
        self._append_path_segment(path_key, suffix_segment)

    def _path_tuple(self, path_key):
        return tuple(self.path_buffers[path_key])

    def _path_length(self, path_key):
        return len(self.path_buffers[path_key][0])

    def generated_path_length(self, role=REQUESTER_ROLE):
        return self._path_length(self._generated_path_key(role))

    def committed_path_length(self, role=REQUESTER_ROLE):
        return self._path_length(self._committed_path_key(role))

    def _has_path(self, path_key):
        return len(self.path_buffers[path_key][0]) > 0

    @staticmethod
    def _base_path(reference_path):
        return (
            reference_path.x,
            reference_path.y,
            reference_path.yaw,
            reference_path.curvature,
        )

    @staticmethod
    def _empty_requested_trajectory():
        return compute_empty_requested_trajectory()

    def _select_requested_path(self, generated_path_key, committed_path_key, current_state, find_only=False):
        _ = find_only
        return compute_selected_requested_path(
            self._has_path,
            self._path_tuple,
            generated_path_key,
            committed_path_key,
            current_state,
        )

    def _build_requested_trajectory(self, state, path_segment, sp, dl):
        return compute_requested_trajectory(
            state,
            path_segment,
            sp,
            dl,
            calc_nearest_index,
        )


    def _lane_change_local_path_for_role(
        self,
        role,
        lane_change_builder,
        current_lane_path,
        target_lane_path,
        current_state,
        state,
        obstacleList,
        accepting_vehicle_state,
    ):
        generated_key = self._generated_path_key(role)
        committed_key = self._committed_path_key(role)
        path_found = self._get_path_found(role)
        if current_state == C.MotionState.FOLLOW_LANE:
            _debug('lane change path found:', path_found)
            return *self._base_path(current_lane_path), path_found

        elif current_state == C.MotionState.FIND_LANE_CHANGE:
            cx, cy, cyaw, ck = current_lane_path
            target_cx, target_cy, target_cyaw, target_ck = target_lane_path
            self._reset_path_group(generated_key)
            lane_change_segment = lane_change_builder(
                cx,
                cy,
                cyaw,
                ck,
                target_cx,
                target_cy,
                target_cyaw,
                target_ck,
                state,
                obstacleList,
            )

            target_index, _ = calc_nearest_index(state, target_cx, target_cy, target_cyaw)
            target_suffix_start = target_index + C.LANE_CHANGE_PATH_PADDING

            self._append_path_segment(generated_key, lane_change_segment)
            self._set_path_found(role, True)
            self._append_reference_suffix(
                generated_key,
                target_lane_path,
                target_suffix_start,
            )

            if accepting_vehicle_state == C.AcceptState.REQUEST_ACCEPTED:
                self._reset_path_group(committed_key)
                self._append_path_segment(committed_key, lane_change_segment)
                self._append_reference_suffix(
                    committed_key,
                    target_lane_path,
                    target_suffix_start,
                )

            path_found = self._get_path_found(role)
            _debug('lane change path found:', path_found)
            _debug('generated lane-change length:', self.generated_path_length(role))
            _debug('committed lane-change length:', self.committed_path_length(role))
            return *self._base_path(current_lane_path), path_found
        elif current_state == C.MotionState.LANE_CHANGE:
            self._set_path_found(role, True)
            path_found = self._get_path_found(role)
            if self._has_path(committed_key):
                return *self._path_tuple(committed_key), path_found
            if self._has_path(generated_key):
                return *self._path_tuple(generated_key), path_found
            return *self._base_path(current_lane_path), path_found


        elif current_state == C.MotionState.FOLLOW_SECOND_LANE:
            path_found = self._get_path_found(role)
            if self._has_path(committed_key):
                return *self._path_tuple(committed_key), path_found
            if self._has_path(generated_key):
                return *self._path_tuple(generated_key), path_found
            return *self._base_path(current_lane_path), path_found

        raise ValueError('cannot find a state')

    def _lane_change_builder_for_role(self, role):
        if self._normalize_role(role) == self.SECONDARY_ROLE:
            return self.rrt_lane_change_path_secondary
        return self.rrt_lane_change_path

    def local_path_for_reference_paths(
        self,
        role,
        current_lane_path,
        target_lane_path,
        current_state,
        state,
        obstacleList,
        accepting_vehicle_state,
    ):
        """Select or build the local path for one role using named lane paths."""
        path_found = self._get_path_found(role)

        if role != self.REQUESTER_ROLE:
            return self._lane_change_local_path_for_role(
                role,
                self._lane_change_builder_for_role(role),
                current_lane_path,
                target_lane_path,
                current_state,
                state,
                obstacleList,
                accepting_vehicle_state,
            )

        if current_state in (
            C.MotionState.FOLLOW_LANE,
            C.MotionState.FIND_LANE_CHANGE,
            C.MotionState.LANE_CHANGE,
            C.MotionState.FOLLOW_SECOND_LANE,
        ):
            return self._lane_change_local_path_for_role(
                role,
                self._lane_change_builder_for_role(role),
                current_lane_path,
                target_lane_path,
                current_state,
                state,
                obstacleList,
                accepting_vehicle_state,
            )

        elif current_state == C.MotionState.FIND_DOUBLE_LANE_CHANGE:
            cx, cy, cyaw, ck = current_lane_path
            target_cx, target_cy, target_cyaw, target_ck = target_lane_path
            double_lane_segment = self.rrt_double_lane_change_path(
                cx,
                cy,
                cyaw,
                ck,
                target_cx,
                target_cy,
                target_cyaw,
                target_ck,
                state,
                obstacleList,
            )

            current_lane_index, _ = calc_nearest_index(state, cx, cy, cyaw)
            current_lane_suffix_start = current_lane_index + C.LANE_CHANGE_PATH_PADDING

            return_key = self._generated_path_key(self.RETURN_ROLE)
            committed_key = self._committed_path_key(self.REQUESTER_ROLE)
            self._reset_path_group(return_key)
            self._append_path_segment(return_key, double_lane_segment)
            self.double_lane_change_path_found = True
            self._append_reference_suffix(
                return_key,
                current_lane_path,
                current_lane_suffix_start,
            )

            return *self._path_tuple(committed_key), path_found

        elif current_state == C.MotionState.DOUBLE_LANE_CHANGE:
            return_key = self._generated_path_key(self.RETURN_ROLE)
            if self._has_path(return_key):
                return *self._path_tuple(return_key), path_found
            return *self._base_path(current_lane_path), path_found


        elif current_state == C.MotionState.FOLLOW_LANE_AGAIN:
            return_key = self._generated_path_key(self.RETURN_ROLE)
            if self._has_path(return_key):
                return *self._path_tuple(return_key), path_found
            return *self._base_path(current_lane_path), path_found
        else:
            raise ValueError('cannot find a state')

    def local_path_for_role(self, role, cx, cy, cyaw, ck, cx_2, cy_2, cyaw_2, ck_2, current_state, state, obstacleList, accepting_vehicle_state):
        return self.local_path_for_reference_paths(
            role,
            self._reference_path(cx, cy, cyaw, ck),
            self._reference_path(cx_2, cy_2, cyaw_2, ck_2),
            current_state,
            state,
            obstacleList,
            accepting_vehicle_state,
        )

    def local_path(self, cx, cy, cyaw, ck, cx_2, cy_2, cyaw_2, ck_2, current_state, state, obstacleList, accepting_vehicle_state):
        return self.local_path_for_role(
            self.REQUESTER_ROLE,
            cx,
            cy,
            cyaw,
            ck,
            cx_2,
            cy_2,
            cyaw_2,
            ck_2,
            current_state,
            state,
            obstacleList,
            accepting_vehicle_state,
        )

    def local_path_secondary(
        self,
        current_lane_x,
        current_lane_y,
        current_lane_yaw,
        current_lane_curvature,
        target_lane_x,
        target_lane_y,
        target_lane_yaw,
        target_lane_curvature,
        secondary_motion_state,
        secondary_state,
        obstacleList,
        downstream_accepting_vehicle_state,
    ):
        return self.local_path_for_reference_paths(
            self.SECONDARY_ROLE,
            self._reference_path(
                current_lane_x,
                current_lane_y,
                current_lane_yaw,
                current_lane_curvature,
            ),
            self._reference_path(
                target_lane_x,
                target_lane_y,
                target_lane_yaw,
                target_lane_curvature,
            ),
            secondary_motion_state,
            secondary_state,
            obstacleList,
            downstream_accepting_vehicle_state,
        )

    def local_path_3_cascading(
        self,
        current_lane_x,
        current_lane_y,
        current_lane_yaw,
        current_lane_curvature,
        target_lane_x,
        target_lane_y,
        target_lane_yaw,
        target_lane_curvature,
        secondary_motion_state,
        secondary_state,
        obstacleList,
        downstream_accepting_vehicle_state,
    ):
        return self.local_path_secondary(
            current_lane_x,
            current_lane_y,
            current_lane_yaw,
            current_lane_curvature,
            target_lane_x,
            target_lane_y,
            target_lane_yaw,
            target_lane_curvature,
            secondary_motion_state,
            secondary_state,
            obstacleList,
            downstream_accepting_vehicle_state,
        )

    def path_found(self):
        if not self.lane_change_path_found:
            return False
        return True

    def transition_state(self, state, state2, requesting_vehicle_state):
        """Update the primary requester's motion state from traffic and request state."""
        self.lane_change_maneuver = self.check_for_lane_change(state, state2)
        current_state = compute_transition_state(
            lane_change_maneuver=self.lane_change_maneuver,
            lane_change_path_found=self.lane_change_path_found,
            lane_change_path_length=self.committed_path_length(self.REQUESTER_ROLE),
            state_y=state.y,
            requesting_vehicle_state=requesting_vehicle_state,
        )
        _debug('current state:', current_state)
        return current_state

    def transition_state_secondary(self, secondary_state, requesting_vehicle_state, secondary_requesting_vehicle_state):
        secondary_motion_state = compute_transition_state_cascading(
            lane_change_path_length=self.committed_path_length(self.SECONDARY_ROLE),
            state_y=secondary_state.y,
            requesting_vehicle_state=requesting_vehicle_state,
            secondary_requesting_vehicle_state=secondary_requesting_vehicle_state,
        )
        _debug('secondary motion state:', secondary_motion_state)
        return secondary_motion_state

    def transition_state_cascading_3(self, secondary_state, requesting_vehicle_state, secondary_requesting_vehicle_state):
        return self.transition_state_secondary(
            secondary_state,
            requesting_vehicle_state,
            secondary_requesting_vehicle_state,
        )

    def check_ref_tr_collision (self, state, xref, xref2):                  # for vehicle sin same lane  for the lead vehicle
        clearances = sampled_clearances(
            xref,
            xref2,
            state.v,
            first_offset=4.0,
            later_offset=4.0,
        )
        distance_gap = 1.0 * state.v
        return is_sampled_trajectory_conflict_free(
            clearances,
            distance_gap,
            initial_gap=distance_gap,
        )

    def calc_ref_req_trajectory(self, state, cx, cy, cyaw, sp, dl, current_state):  ###added  to calclulate requested trajectory to send
        return self.calc_ref_req_trajectory_for_role(
            self.REQUESTER_ROLE,
            state,
            cx,
            cy,
            cyaw,
            sp,
            dl,
            current_state,
        )

    def calc_ref_req_trajectory_for_role(self, role, state, cx, cy, cyaw, sp, dl, current_state):  ###added  to calclulate requested trajectory to send
        _ = (cx, cy, cyaw)
        generated_key = self._generated_path_key(role)
        committed_key = self._committed_path_key(role)
        selected_path, req_traj = self._select_requested_path(
            generated_key,
            committed_key,
            current_state,
        )
        if not req_traj:
            return self._empty_requested_trajectory()
        return self._build_requested_trajectory(state, selected_path, sp, dl)

    def calc_ref_req_trajectory_secondary(self, secondary_state, cx, cy, cyaw, sp, dl, secondary_motion_state):  ###added  to calclulate requested trajectory to send
        return self.calc_ref_req_trajectory_for_role(
            self.SECONDARY_ROLE,
            secondary_state,
            cx,
            cy,
            cyaw,
            sp,
            dl,
            secondary_motion_state,
        )

    def calc_ref_req_trajectory_cascading(self, secondary_state, cx, cy, cyaw, sp, dl, secondary_motion_state):  ###added  to calclulate requested trajectory to send
        return self.calc_ref_req_trajectory_secondary(
            secondary_state,
            cx,
            cy,
            cyaw,
            sp,
            dl,
            secondary_motion_state,
        )

    @staticmethod                                        ## to calculate new reference Pt
    def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl):  ###added
        return compute_ref_trajectory(
            state,
            cx,
            cy,
            cyaw,
            ck,
            sp,
            dl,
            calc_nearest_index,
        )

    def check_req_tr_conflict (self, state, other_state, xreq, other_xref, req_traj):      #check requested trajectory if conflict free  vehicles in different lanes

        if req_traj is not True:
            return False

        clearances = sampled_clearances(xreq, other_xref, state.v)
        distance_gap = 1.0 * other_state.v
        return is_sampled_trajectory_conflict_free(clearances, distance_gap)

    def check_req_tr_conflict_4 (self, state, xreq, xref4, req_traj):      #check requested trajectory if conflict free  vehicles in different lanes

        if req_traj is not True:
            return False

        clearances = sampled_clearances(xreq, xref4, state.v)
        distance_gap = 1.0 * state.v
        return is_sampled_trajectory_conflict_free(clearances, distance_gap)

    def check_req_tr_conflict_new (self, state, other_state, xreq, other_xref_low, other_xref_medium, other_xref_high, req_traj, requesting_priority):      #check requested trajectory if conflict free  vehicles in different lanes

        if req_traj is not True:
            return False

        candidate_refs = {
            C.Priority.LOW_PRIORITY: (other_xref_low, other_state.v - 1.3),
            C.Priority.MEDIUM_PRIORITY: (other_xref_medium, other_state.v - 2.6),
            C.Priority.HIGH_PRIORITY: (other_xref_high, other_state.v - 5.0),
        }
        candidate_ref, candidate_speed = candidate_refs.get(requesting_priority, (None, None))
        if candidate_ref is None:
            return False

        clearances = sampled_clearances(xreq, candidate_ref, state.v)
        _debug("distances 1, 2, 3:", *clearances)
        return is_sampled_trajectory_conflict_free(clearances, 1.0 * candidate_speed)

    def check_req_tr_conflict_new_4 (self, state, xreq, new_xref4_low, new_xref4_medium, new_xref4_high, req_traj, requesting_priority):      #check requested trajectory if conflict free  vehicles in different lanes

        if req_traj is not True:
            return False

        candidate_refs = {
            C.Priority.LOW_PRIORITY: new_xref4_low,
            C.Priority.MEDIUM_PRIORITY: new_xref4_medium,
            C.Priority.HIGH_PRIORITY: new_xref4_high,
        }
        candidate_ref = candidate_refs.get(requesting_priority)
        if candidate_ref is None:
            return False

        clearances = sampled_clearances(xreq, candidate_ref, state.v)
        return is_sampled_trajectory_conflict_free(clearances, 1.0 * state.v)

    def calc_priority(self, state, state2):
        requesting_priority = compute_priority(state, state2)
        _debug("requesting priority:", requesting_priority)
        return requesting_priority

    def _request_state_from_context(
        self,
        current_state,
        accepting_vehicle_state,
        conflict_free_req,
        can_send_request,
    ):
        return compute_request_state(
            current_state=current_state,
            accepting_vehicle_state=accepting_vehicle_state,
            conflict_free_req=conflict_free_req,
            can_send_request=can_send_request,
        )

    def requesting_vehicle_states(self, current_state, accepting_vehicle_state, conflict_free_req):
        """Convert motion/acceptance/conflict context into a request state."""
        requesting_vehicle_state = self._request_state_from_context(
            current_state,
            accepting_vehicle_state,
            conflict_free_req,
            can_send_request=True,
        )

        _debug('requesting vehicle state:', requesting_vehicle_state)
        return requesting_vehicle_state

    def requesting_vehicle_states_secondary(self, secondary_motion_state, downstream_accepting_vehicle_state, conflict_free_req, requesting_priority):
        secondary_requesting_vehicle_state = self._request_state_from_context(
            secondary_motion_state,
            downstream_accepting_vehicle_state,
            conflict_free_req,
            can_send_request=requesting_priority == C.Priority.MEDIUM_PRIORITY,
        )

        _debug('secondary requesting vehicle state:', secondary_requesting_vehicle_state)
        return secondary_requesting_vehicle_state

    def requesting_vehicle_states_3_cascading(self, secondary_motion_state, downstream_accepting_vehicle_state, conflict_free_req, requesting_priority):
        return self.requesting_vehicle_states_secondary(
            secondary_motion_state,
            downstream_accepting_vehicle_state,
            conflict_free_req,
            requesting_priority,
        )

    def accepting_vehicle_states(self, requesting_vehicle_state, requesting_priority, conflict_free, conflict_free_new, downstream_accepting_vehicle_state):
        accepting_vehicle_state, speed_profile_state = compute_accepting_vehicle_states(
            requesting_vehicle_state=requesting_vehicle_state,
            requesting_priority=requesting_priority,
            conflict_free=conflict_free,
            conflict_free_new=conflict_free_new,
            downstream_accepting_vehicle_state=downstream_accepting_vehicle_state,
        )
        _debug('accepting vehicle state:', accepting_vehicle_state)
        _debug('accepting vehicle speed profile state:', speed_profile_state)
        return accepting_vehicle_state, speed_profile_state

    def accepting_vehicle_states_direct(self, requesting_vehicle_state, requesting_priority, conflict_free, conflict_free_new):
        accepting_vehicle_state, speed_profile_state = compute_accepting_vehicle_states_direct(
            requesting_vehicle_state=requesting_vehicle_state,
            requesting_priority=requesting_priority,
            conflict_free=conflict_free,
            conflict_free_new=conflict_free_new,
        )
        _debug('accepting vehicle state (direct):', accepting_vehicle_state)
        _debug('accepting vehicle speed profile state (direct):', speed_profile_state)
        return accepting_vehicle_state, speed_profile_state

    def accepting_vehicle_states_downstream(
        self,
        secondary_requesting_vehicle_state,
        requesting_priority,
        conflict_free,
        conflict_free_after_adaptation,
    ):
        accepting_vehicle_state, speed_profile_state = compute_accepting_vehicle_states_downstream(
            secondary_requesting_vehicle_state=secondary_requesting_vehicle_state,
            requesting_priority=requesting_priority,
            conflict_free=conflict_free,
            conflict_free_after_adaptation=conflict_free_after_adaptation,
        )
        _debug('downstream accepting vehicle state:', accepting_vehicle_state)
        _debug('downstream accepting vehicle speed profile state:', speed_profile_state)
        return accepting_vehicle_state, speed_profile_state

    def check_for_lane_change(self, state, state2):
        self.lane_change_maneuver = compute_lane_change_maneuver(
            state,
            state2,
            existing_rrt_length=self.generated_path_length(self.REQUESTER_ROLE),
        )
        return self.lane_change_maneuver

    def check_for_double_lane_change(self, state, state2):
        self.double_lane_change_maneuver = compute_double_lane_change_maneuver(
            state,
            state2,
            existing_rrt_length=self.generated_path_length(self.REQUESTER_ROLE),
        )
        return self.double_lane_change_maneuver

    @staticmethod
    def _curvature_from_xy(path_x, path_y):
        if len(path_x) < 2 or len(path_y) < 2:
            return []

        spline = Spline2D(path_x, path_y)
        if not spline.s:
            return []

        samples = np.arange(0, spline.s[-1], 0.1)
        if len(samples) == 0:
            return [spline.calc_curvature(0.0)]

        return [spline.calc_curvature(i_s) for i_s in samples]

    @staticmethod
    def _build_rrt_path(target_x, target_y, target_yaw, state, obstacle_list, goal_offset):
        params = C.PLANNER_PARAMS
        if not target_x or not target_y or not target_yaw:
            raise ValueError("Target path for RRT lane change is empty.")

        start_rrt = [state.x, state.y, state.yaw]
        target_index, _ = calc_nearest_index(state, target_x, target_y, target_yaw)
        goal_index = min(target_index + goal_offset, len(target_x) - 1)
        goal_rrt = [target_x[goal_index], target_y[goal_index], target_yaw[goal_index]]

        rrt_reeds_shepp = RRTReedsShepp(start_rrt, goal_rrt, obstacle_list, params.rrt_search_area)
        rrt_path = rrt_reeds_shepp.planning(animation=SHOW_ANIMATION)
        if not rrt_path:
            raise ValueError("RRT lane change planning failed to produce a path.")

        path_x = np.array([node[0] for node in rrt_path])[::-1]
        path_y = np.array([node[1] for node in rrt_path])[::-1]
        path_yaw = np.array([node[2] for node in rrt_path])[::-1]
        path_k = BehaviouralLocalPlanner._curvature_from_xy(path_x, path_y)
        return path_x, path_y, path_yaw, path_k

    @staticmethod
    def rrt_lane_change_path(cx, cy, cyaw, ck, cx_2, cy_2, cyaw_2, ck_2, state, obstacleList):
        _ = (cx, cy, cyaw, ck, ck_2)
        return BehaviouralLocalPlanner._build_rrt_path(
            cx_2,
            cy_2,
            cyaw_2,
            state,
            obstacleList,
            C.PLANNER_PARAMS.lane_change_goal_offset,
        )

    @staticmethod
    def rrt_lane_change_path_secondary(
        current_lane_x,
        current_lane_y,
        current_lane_yaw,
        current_lane_curvature,
        target_lane_x,
        target_lane_y,
        target_lane_yaw,
        target_lane_curvature,
        secondary_state,
        obstacleList,
    ):
        _ = (
            current_lane_x,
            current_lane_y,
            current_lane_yaw,
            current_lane_curvature,
            target_lane_curvature,
        )
        return BehaviouralLocalPlanner._build_rrt_path(
            target_lane_x,
            target_lane_y,
            target_lane_yaw,
            secondary_state,
            obstacleList,
            C.PLANNER_PARAMS.lane_change_goal_offset,
        )

    @staticmethod
    def rrt_lane_change_path_3(
        current_lane_x,
        current_lane_y,
        current_lane_yaw,
        current_lane_curvature,
        target_lane_x,
        target_lane_y,
        target_lane_yaw,
        target_lane_curvature,
        secondary_state,
        obstacleList,
    ):
        return BehaviouralLocalPlanner.rrt_lane_change_path_secondary(
            current_lane_x,
            current_lane_y,
            current_lane_yaw,
            current_lane_curvature,
            target_lane_x,
            target_lane_y,
            target_lane_yaw,
            target_lane_curvature,
            secondary_state,
            obstacleList,
        )

    @staticmethod
    def rrt_double_lane_change_path( cx, cy, cyaw, ck, cx_2, cy_2, cyaw_2, ck_2, state, obstacleList):
        _ = (ck, cx_2, cy_2, cyaw_2, ck_2)
        return BehaviouralLocalPlanner._build_rrt_path(
            cx,
            cy,
            cyaw,
            state,
            obstacleList,
            C.PLANNER_PARAMS.lane_change_path_padding,
        )


def pi_2_pi(angle):                       ### keep heading withing [-pi, pi] so optimizer behaves well
    """Normalize an angle to the [-pi, pi] range."""
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def calc_nearest_index(state, cx, cy, cyaw):    # calculating nearest index on the planned path next to the ego car (ind)
    """Find nearest path point and signed lateral error for legacy planner code."""
    dx = [state.x - icx for icx in cx]          # calculating also the distance between ego and closest planned point on path (e or mind)
    dy = [state.y - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind                 # calculating the target index and the e(distance between the path and the tracked path)
