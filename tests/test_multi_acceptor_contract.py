import unittest
import math
from unittest.mock import patch

import numpy as np

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.scenarios.three_vehicles_coordination import build_obstacles
from maneuver_coordination.simulation.coordination.multi_acceptor import offers_are_clear, response_to_message
from maneuver_coordination.simulation.runners import multi_acceptor_runner as runner
from maneuver_coordination.simulation.runners.scenario_configs import (
    build_default_three_vehicle_coordination_configs,
    build_default_three_vehicle_coordination_4_messages_configs,
)
from maneuver_coordination.vehicle.model import LF, LB, W


class MultiAcceptorContractTests(unittest.TestCase):
    def run_scenario(self, four=False):
        configs = (build_default_three_vehicle_coordination_4_messages_configs()
                   if four else build_default_three_vehicle_coordination_configs())
        ox, oy, _, _, obstacles = build_obstacles()
        result = runner.run_multi_acceptor_three_vehicle_coordination_scenario(
            configs, ox, oy, obstacles, use_offer_confirmation=four,
            show_animation=False, verbose_events=False,
        )
        self.assert_safe_motion(result)
        return result

    def assert_safe_motion(self, result):
        histories = result["histories"]
        for history in histories:
            self.assertTrue(all(b >= a - 1e-9 for a, b in zip(history.x, history.x[1:])))
        for index in range(len(histories[0].t)):
            polygons, axes_by_vehicle = [], []
            for history in histories:
                c, s = math.cos(history.yaw[index]), math.sin(history.yaw[index])
                polygon = [(history.x[index] + dx*c - dy*s,
                            history.y[index] + dx*s + dy*c)
                           for dx in (-LB, LF) for dy in (-W/2, W/2)]
                self.assertGreaterEqual(min(y for _, y in polygon), -1e-9)
                self.assertLessEqual(max(y for _, y in polygon), 8.0 + 1e-9)
                polygons.append(polygon)
                axes_by_vehicle.append(((c, s), (-s, c)))
            for first in range(len(polygons)):
                for second in range(first + 1, len(polygons)):
                    separated = False
                    for ax, ay in axes_by_vehicle[first] + axes_by_vehicle[second]:
                        p = [x*ax + y*ay for x, y in polygons[first]]
                        q = [x*ax + y*ay for x, y in polygons[second]]
                        if max(p) <= min(q) or max(q) <= min(p):
                            separated = True
                            break
                    self.assertTrue(separated, (index, first, second))

    def executed(self, result):
        return any("ID 1 motion -> lane_change" in event for event in result["events"])

    def test_infeasible_required_cv_cannot_be_overwritten(self):
        for four in (False, True):
            with self.subTest(four=four), patch.object(
                runner.BehaviouralLocalPlanner, "check_req_tr_conflict", return_value=False
            ), patch.object(runner, "select_first_feasible_reduction_candidate", return_value=None):
                result = self.run_scenario(four)
                self.assertFalse(self.executed(result))
                self.assertFalse(any(message.sender_role == "target_lane_rear" and
                                     message.message_type in ("maneuver_offer", "maneuver_accept")
                                     for message in result["messages"]))
                self.assertTrue(any(message.sender_role == "target_lane_rear" and
                                    message.message_type == "maneuver_reject"
                                    for message in result["messages"]))
                if four:
                    self.assertTrue(any(message.sender_role == "target_lane_front" and
                                        message.message_type == "maneuver_offer"
                                        for message in result["messages"]))
                    self.assertFalse(any(message.message_type == "confirm_offer"
                                         for message in result["messages"]))

    def test_all_requests_dropped_prevents_execution(self):
        for four in (False, True):
            with self.subTest(four=four), patch.object(runner, "emit_request_messages", return_value=[]):
                result = self.run_scenario(four)
                self.assertFalse(self.executed(result))
                self.assertGreater(result["histories"][0].t[-1], 5.0)
                if not four:
                    self.assertLess(min(result["histories"][0].v), 8.0)
                self.assertLess(max(abs(y - 2.0) for y in result["histories"][0].y), 0.15)
                self.assertFalse(any(message.message_type == "maneuver_offer"
                                     for message in result["messages"]))

    def test_missing_offer_confirm_or_accept_blocks_execution(self):
        inbox = runner.get_vehicle_inbox
        for kind in ("maneuver_offer", "confirm_offer", "maneuver_accept"):
            def drop_kind(messages, receiver_id):
                return [message for message in inbox(messages, receiver_id)
                        if message.message_type != kind]

            with self.subTest(kind=kind), patch.object(runner, "get_vehicle_inbox", side_effect=drop_kind):
                self.assertFalse(self.executed(self.run_scenario(True)))

    def test_unavailable_maneuver_is_not_replaced_by_current_lane_path(self):
        with patch.object(runner.BehaviouralLocalPlanner, "_build_rrt_path", return_value=([], [], [], [])):
            result = self.run_scenario()
        self.assertFalse(self.executed(result))
        self.assertLess(max(abs(y - 2.0) for y in result["histories"][0].y), 0.15)

    def test_one_required_request_dropped_prevents_execution(self):
        emit = runner.emit_request_messages

        def drop_rear(*args, **kwargs):
            return [message for message in emit(*args, **kwargs)
                    if message.receiver_role != "target_lane_rear"]

        for four in (False, True):
            with self.subTest(four=four), patch.object(runner, "emit_request_messages", side_effect=drop_rear):
                self.assertFalse(self.executed(self.run_scenario(four)))

    def test_normal_delivery_executes_after_required_accepts(self):
        for four in (False, True):
            with self.subTest(four=four):
                result = self.run_scenario(four)
                self.assertTrue(self.executed(result))
                accepts = [message for message in result["messages"] if message.message_type == "maneuver_accept"]
                required = {message.receiver_role for message in result["messages"]
                            if message.message_type == "maneuver_request"}
                self.assertTrue(required)
                self.assertEqual({message.sender_role for message in accepts}, required)
                for accepted in accepts:
                    requests = [message for message in result["messages"]
                                if message.message_type == "maneuver_request"
                                and message.receiver_role == accepted.sender_role]
                    self.assertTrue(any(message.time_s < accepted.time_s for message in requests))
                if four:
                    for accepted in accepts:
                        self.assertTrue(any(message.message_type == "confirm_offer"
                                            and message.receiver_role == accepted.sender_role
                                            and message.time_s < accepted.time_s
                                            for message in result["messages"]))

    def test_conflicting_combined_offers_are_not_confirmable(self):
        reference = np.ones((4, 20))
        proposal = {"xref": reference, "final_speed": 8.0}
        self.assertFalse(offers_are_clear({"rear": proposal, "front": proposal},
                                         {"rear", "front"}, reference, 8.0))
        self.assertFalse(offers_are_clear({"rear": None}, {"rear"}, reference, 8.0))

    def test_missing_message_or_proposal_cannot_approve(self):
        self.assertEqual(response_to_message(C.request_rejected, None, "maneuver_request",
                                            use_offer_confirmation=True), C.request_rejected)
        self.assertEqual(response_to_message(C.no_request_received, {"xref": None}, None,
                                            use_offer_confirmation=False), C.no_request_received)


if __name__ == "__main__":
    unittest.main()
