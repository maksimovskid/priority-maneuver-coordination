import unittest

from maneuver_coordination.coordination.behavior_planner import (
    BehaviouralLocalPlanner,
    ReferencePath,
)
from maneuver_coordination.coordination.constants import (
    SECOND_LANE_Y,
    execute_request,
    find_lane_change,
    follow_lane,
    follow_planned,
    follow_second_lane,
    high_priority,
    lane_change,
    no_request_sent,
    medium_priority,
    no_need_for_request,
    request_accepted,
    request_rejected,
    reduce_speed_for_req_high,
    reduce_speed_for_req_medium,
    send_request,
)
from maneuver_coordination.simulation.core.types import State


def make_state(x=0.0, y=0.0, yaw=0.0, v=10.0):
    return State(x=x, y=y, yaw=yaw, v=v)


class BehaviorPlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = BehaviouralLocalPlanner([], [], [], [])

    def test_transition_state_stays_in_follow_second_lane_when_lane_change_complete(self):
        self.planner.path_buffers[
            self.planner.REQUESTER_COMMITTED_PATH_KEY
        ][0] = [0.0, 1.0, 2.0]

        current_state = self.planner.transition_state(
            make_state(y=SECOND_LANE_Y),
            make_state(x=20.0),
            no_need_for_request,
        )

        self.assertEqual(current_state, follow_second_lane)

    def test_transition_state_enters_lane_change_on_execute_request(self):
        current_state = self.planner.transition_state(
            make_state(y=2.0),
            make_state(x=20.0),
            execute_request,
        )

        self.assertEqual(current_state, lane_change)

    def test_transition_state_stays_in_find_lane_change_while_request_pending(self):
        current_state = self.planner.transition_state(
            make_state(y=2.0),
            make_state(x=20.0),
            send_request,
        )

        self.assertEqual(current_state, find_lane_change)

    def test_transition_state_enters_find_lane_change_when_maneuver_needed(self):
        lead_state = make_state(x=5.0, v=5.0)
        ego_state = make_state(x=0.0, y=2.0, v=10.0)

        current_state = self.planner.transition_state(
            ego_state,
            lead_state,
            follow_lane,
        )

        self.assertEqual(current_state, find_lane_change)

    def test_local_path_returns_base_path_while_following_lane(self):
        cx = [0.0, 1.0, 2.0]
        cy = [2.0, 2.0, 2.0]
        cyaw = [0.0, 0.0, 0.0]
        ck = [0.0, 0.0, 0.0]

        path_x, path_y, path_yaw, path_k, path_found = self.planner.local_path(
            cx,
            cy,
            cyaw,
            ck,
            cx,
            cy,
            cyaw,
            ck,
            follow_lane,
            make_state(),
            [],
            request_accepted,
        )

        self.assertEqual(path_x, cx)
        self.assertEqual(path_y, cy)
        self.assertEqual(path_yaw, cyaw)
        self.assertEqual(path_k, ck)
        self.assertFalse(path_found)

    def test_local_path_for_reference_paths_uses_named_current_and_target_paths(self):
        current_lane_path = ReferencePath(
            x=[0.0, 1.0, 2.0],
            y=[2.0, 2.0, 2.0],
            yaw=[0.0, 0.0, 0.0],
            curvature=[0.0, 0.0, 0.0],
        )
        target_lane_path = ReferencePath(
            x=[0.0, 1.0, 2.0],
            y=[6.0, 6.0, 6.0],
            yaw=[0.0, 0.0, 0.0],
            curvature=[0.0, 0.0, 0.0],
        )

        path_x, path_y, path_yaw, path_k, path_found = (
            self.planner.local_path_for_reference_paths(
                self.planner.REQUESTER_ROLE,
                current_lane_path,
                target_lane_path,
                follow_lane,
                make_state(),
                [],
                request_accepted,
            )
        )

        self.assertEqual(path_x, current_lane_path.x)
        self.assertEqual(path_y, current_lane_path.y)
        self.assertEqual(path_yaw, current_lane_path.yaw)
        self.assertEqual(path_k, current_lane_path.curvature)
        self.assertFalse(path_found)

    def test_local_path_builds_committed_lane_change_when_request_accepted(self):
        planner = BehaviouralLocalPlanner([], [], [], [])
        planner.rrt_lane_change_path = lambda *args: (
            [1.0, 2.0],
            [3.0, 4.0],
            [0.1, 0.2],
            [0.01, 0.02],
        )

        cx = [0.0, 1.0, 2.0]
        cy = [2.0, 2.0, 2.0]
        cyaw = [0.0, 0.0, 0.0]
        ck = [0.0, 0.0, 0.0]
        cx_2 = list(range(150))
        cy_2 = [6.0] * 150
        cyaw_2 = [0.0] * 150
        ck_2 = [0.0] * 150

        _, _, _, _, path_found = planner.local_path(
            cx,
            cy,
            cyaw,
            ck,
            cx_2,
            cy_2,
            cyaw_2,
            ck_2,
            find_lane_change,
            make_state(x=0.0, y=2.0, v=10.0),
            [],
            request_accepted,
        )

        self.assertTrue(path_found)
        self.assertEqual(
            planner.path_buffers[planner.REQUESTER_COMMITTED_PATH_KEY][0][:2],
            [1.0, 2.0],
        )
        self.assertGreater(planner.generated_path_length(), 2)

    def test_legacy_path_attributes_still_map_to_path_buffers(self):
        self.planner.lane_change_state_x = [0.0, 1.0]
        self.planner.rrt_lane_change_x3 = [2.0, 3.0]

        self.assertEqual(
            self.planner.path_buffers[self.planner.REQUESTER_COMMITTED_PATH_KEY][0],
            [0.0, 1.0],
        )
        self.assertEqual(
            self.planner.path_buffers[self.planner.SECONDARY_GENERATED_PATH_KEY][0],
            [2.0, 3.0],
        )

    def test_accepting_vehicle_states_direct_accepts_medium_priority_with_speed_reduction(self):
        accepting_vehicle_state, speed_profile_state = self.planner.accepting_vehicle_states_direct(
            send_request,
            medium_priority,
            False,
            True,
        )

        self.assertEqual(accepting_vehicle_state, request_accepted)
        self.assertEqual(speed_profile_state, reduce_speed_for_req_medium)

    def test_accepting_vehicle_states_direct_rejects_high_priority_without_safe_adjustment(self):
        accepting_vehicle_state, speed_profile_state = self.planner.accepting_vehicle_states_direct(
            send_request,
            high_priority,
            False,
            False,
        )

        self.assertEqual(accepting_vehicle_state, request_rejected)
        self.assertEqual(speed_profile_state, follow_planned)

    def test_accepting_vehicle_states_direct_keeps_acceptance_during_execution(self):
        accepting_vehicle_state, speed_profile_state = self.planner.accepting_vehicle_states_direct(
            execute_request,
            high_priority,
            False,
            True,
        )

        self.assertEqual(accepting_vehicle_state, request_accepted)
        self.assertEqual(speed_profile_state, reduce_speed_for_req_high)

    def test_requesting_vehicle_states_3_cascading_sends_only_for_medium_priority(self):
        requesting_vehicle_state = self.planner.requesting_vehicle_states_3_cascading(
            find_lane_change,
            request_rejected,
            False,
            medium_priority,
        )

        self.assertEqual(requesting_vehicle_state, send_request)

    def test_requesting_vehicle_states_3_cascading_does_not_send_for_high_priority(self):
        requesting_vehicle_state = self.planner.requesting_vehicle_states_3_cascading(
            find_lane_change,
            request_rejected,
            False,
            high_priority,
        )

        self.assertEqual(requesting_vehicle_state, no_request_sent)


if __name__ == "__main__":
    unittest.main()
