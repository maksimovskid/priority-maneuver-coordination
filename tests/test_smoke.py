import unittest
from unittest.mock import patch

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination.planner import BehaviouralLocalPlanner
from maneuver_coordination.scenarios.cascading_coordination import build_obstacles as build_cascading_obstacles
from maneuver_coordination.scenarios.rejected_request_then_free_lane import build_obstacles as build_rejected_request_obstacles
from maneuver_coordination.scenarios.three_vehicles_coordination import build_obstacles as build_three_vehicle_obstacles
from maneuver_coordination.coordination.trajectory_logic import build_requested_trajectory
from maneuver_coordination.simulation.motion.reference import calc_ref_trajectory as calc_simulation_ref_trajectory
from maneuver_coordination.simulation.motion.controllers import rear_wheel_feedback_control
from maneuver_coordination.simulation.runner import (
    ROAD_END_X,
    build_acceptance_stage_context,
    build_acc_speed_profile,
    build_fallback_paths,
    build_requester_path_inputs,
    build_role_reference_trajectories,
    build_target_lane_reference_candidates,
    build_default_cascading_vehicle_configs,
    build_default_rejected_request_then_free_lane_configs,
    build_default_three_vehicle_coordination_4_messages_configs,
    build_default_three_vehicle_coordination_configs,
    build_requester_local_paths,
    build_requester_speed_profiles,
    build_request_message_candidates,
    build_requester_requested_trajectories,
    build_requester_stage_spec,
    build_requester_conflict_bundle,
    build_requester_conflict_sources,
    build_default_two_vehicle_coordination_configs,
    build_received_request_context,
    emit_v2x_request_message,
    emit_request_messages,
    find_conflicting_vehicle_roles,
    infer_lane_id_from_y,
    find_vehicle_ahead_in_lane,
    find_vehicle_behind_in_lane,
    find_target_lane_neighbors,
    get_latest_v2x_request,
    get_message_priority,
    has_v2x_request_message,
    get_vehicle_inbox,
    get_requester_candidate_roles,
    is_vehicle_braking,
    select_secondary_requester_role,
    select_request_message_for_receiver,
    select_lowest_cost_role,
    should_trigger_maneuver_search,
    run_cascading_scenario,
    run_rejected_request_then_free_lane_scenario,
    run_three_vehicle_coordination_4_messages_scenario,
    run_three_vehicle_coordination_scenario,
)
from maneuver_coordination.simulation.core.types import PlannedPath, State
from maneuver_coordination.vehicle.model import LENGTH, MAX_STEER, WB
import numpy as np


class StructureSmokeTests(unittest.TestCase):
    def test_default_vehicle_configs_are_built(self):
        configs = build_default_cascading_vehicle_configs()

        self.assertEqual(len(configs), 5)
        self.assertEqual(configs[0].role, "ego")
        self.assertEqual([config.vehicle_id for config in configs], [1, 2, 3, 4, 5])
        self.assertTrue(all(config.goal[0] == ROAD_END_X for config in configs))

    def test_two_vehicle_coordination_configs_are_built(self):
        configs = build_default_two_vehicle_coordination_configs()

        self.assertEqual(len(configs), 3)
        self.assertEqual([config.role for config in configs], ["ego", "lead", "acceptor"])
        self.assertEqual([config.vehicle_id for config in configs], [1, 2, 3])
        self.assertEqual(configs[0].lane_id, 0)
        self.assertEqual(configs[0].target_lane_id, 1)

    def test_three_vehicle_coordination_configs_are_built(self):
        configs = build_default_three_vehicle_coordination_configs()

        self.assertEqual(len(configs), 4)
        self.assertEqual([config.vehicle_id for config in configs], [1, 2, 3, 4])
        self.assertEqual([config.role for config in configs], ["ego", "lead", "target_lane_rear", "target_lane_front"])

    def test_rejected_request_then_free_lane_configs_are_built(self):
        configs = build_default_rejected_request_then_free_lane_configs()

        self.assertEqual(len(configs), 3)
        self.assertEqual([config.vehicle_id for config in configs], [1, 2, 3])
        self.assertEqual(configs[1].metadata["braking_start_time"], "2.0")

    def test_three_vehicle_coordination_4_messages_configs_are_built(self):
        configs = build_default_three_vehicle_coordination_4_messages_configs()

        self.assertEqual(len(configs), 4)
        self.assertEqual(configs[1].start[0], 44.0)
        self.assertEqual(configs[2].initial_speed, 23.0 / 3.6)

    def test_rear_wheel_feedback_control_steers_toward_parallel_offset_path(self):
        state = State(x=4.5, y=2.0, yaw=0.0, v=8.0)
        cx = [4.5, 8.5, 12.5]
        cy = [6.0, 6.0, 6.0]
        cyaw = [0.0, 0.0, 0.0]
        ck = [0.0, 0.0, 0.0]

        delta, _ = rear_wheel_feedback_control(state, cx, cy, cyaw, ck, 0)

        self.assertNotEqual(delta, 0.0)

    def test_v2x_message_uses_vehicle_ids_and_roles(self):
        configs = build_default_two_vehicle_coordination_configs()
        message = emit_v2x_request_message(
            1.2,
            configs[0],
            configs[2],
            C.medium_priority,
            "maneuver_request",
            {"stage": "primary"},
        )

        self.assertEqual(message.sender_id, 1)
        self.assertEqual(message.receiver_id, 3)
        self.assertEqual(message.sender_role, "ego")
        self.assertEqual(message.receiver_role, "acceptor")
        self.assertEqual(message.priority_label, "medium_priority")

    def test_vehicle_inbox_filters_messages_by_receiver_id(self):
        configs = build_default_cascading_vehicle_configs()
        messages = [
            emit_v2x_request_message(0.0, configs[0], configs[2], C.low_priority, "maneuver_request"),
            emit_v2x_request_message(0.1, configs[2], configs[3], C.medium_priority, "maneuver_request"),
        ]

        inbox = get_vehicle_inbox(messages, 4)
        self.assertEqual(len(inbox), 1)
        self.assertEqual(inbox[0].sender_id, 3)

    def test_has_v2x_request_message_detects_sender_receiver_pair(self):
        configs = build_default_two_vehicle_coordination_configs()
        messages = [
            emit_v2x_request_message(0.0, configs[0], configs[2], C.low_priority, "maneuver_request"),
        ]

        self.assertTrue(has_v2x_request_message(messages, 1, 3))
        self.assertFalse(has_v2x_request_message(messages, 1, 2))

    def test_get_latest_v2x_request_returns_matching_message(self):
        configs = build_default_cascading_vehicle_configs()
        messages = [
            emit_v2x_request_message(0.0, configs[0], configs[2], C.low_priority, "maneuver_request"),
            emit_v2x_request_message(0.1, configs[0], configs[2], C.high_priority, "maneuver_request"),
        ]

        latest = get_latest_v2x_request(messages, receiver_id=3, sender_id=1)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.priority_label, "high_priority")

    def test_get_message_priority_reads_priority_from_payload(self):
        configs = build_default_two_vehicle_coordination_configs()
        message = emit_v2x_request_message(0.0, configs[0], configs[2], C.medium_priority, "maneuver_request")

        self.assertEqual(get_message_priority(message, C.low_priority), C.medium_priority)

    def test_select_request_message_for_receiver_prefers_higher_priority(self):
        configs = build_default_cascading_vehicle_configs()
        messages = [
            emit_v2x_request_message(0.0, configs[0], configs[2], C.low_priority, "maneuver_request"),
            emit_v2x_request_message(0.1, configs[1], configs[2], C.high_priority, "maneuver_request"),
        ]

        selected = select_request_message_for_receiver(messages, receiver_id=3)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.sender_id, 2)
        self.assertEqual(selected.priority_label, "high_priority")

    def test_build_received_request_context_uses_selected_message_role_and_priority(self):
        configs = build_default_cascading_vehicle_configs()
        messages = [
            emit_v2x_request_message(0.0, configs[0], configs[2], C.low_priority, "maneuver_request"),
        ]
        requester_conflict_sources = {}
        requester_conflict_sources.update(build_requester_conflict_sources("ego", False, False, True))
        requester_conflict_sources.update(build_requester_conflict_sources("acceptor_1", True, True, True))

        context = build_received_request_context(
            messages,
            receiver_id=3,
            requester_conflict_sources=requester_conflict_sources,
            fallback_requester_role="acceptor_1",
            fallback_priority=C.no_priority_request,
        )

        self.assertTrue(context["delivered"])
        self.assertEqual(context["requester_role"], "ego")
        self.assertEqual(context["priority"], C.low_priority)
        self.assertEqual(
            context["requester_conflicts"],
            build_requester_conflict_bundle("ego", requester_conflict_sources),
        )

    def test_build_requester_conflict_sources_creates_role_keyed_source(self):
        sources = build_requester_conflict_sources("ego", False, True, False)
        self.assertEqual(sources, {"ego": (False, True, False)})

    def test_emit_request_messages_creates_message_for_send_request_candidate(self):
        configs = build_default_two_vehicle_coordination_configs()
        config_by_role = {config.role: config for config in configs}
        candidates = [
            build_request_message_candidates("ego", "acceptor", C.send_request, C.low_priority, "primary"),
            build_request_message_candidates("ego", "lead", C.no_request_sent, C.low_priority, "primary"),
        ]

        messages = emit_request_messages(0.2, candidates, config_by_role)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].sender_id, 1)
        self.assertEqual(messages[0].receiver_id, 3)

    def test_build_requester_stage_spec_creates_role_based_stage_metadata(self):
        spec = build_requester_stage_spec("ego", C.follow_lane, "acceptor", "primary", secondary=False)
        self.assertEqual(spec["requester_role"], "ego")
        self.assertEqual(spec["receiver_role"], "acceptor")
        self.assertEqual(spec["stage_label"], "primary")
        self.assertFalse(spec["secondary"])

    def test_build_acceptance_stage_context_creates_stage_metadata(self):
        context = build_acceptance_stage_context(
            "ego",
            {"request_clear": False, "accept_clear": False, "accept_clear_new": True},
            C.low_priority,
            True,
            secondary=False,
            peer_accept_state=C.no_request_received,
        )
        self.assertEqual(context["requester_role"], "ego")
        self.assertTrue(context["delivered"])
        self.assertEqual(context["priority"], C.low_priority)
        self.assertEqual(context["peer_accept_state"], C.no_request_received)

    def test_build_requester_requested_trajectories_uses_stage_specs(self):
        class FakePlanner:
            def calc_ref_req_trajectory(self, state, local_path_x, local_path_y, local_path_yaw, speed_profile, dl, motion_state):
                _ = (state, local_path_x, local_path_y, local_path_yaw, speed_profile, dl, motion_state)
                return np.array([[1.0], [2.0], [0.0], [0.0]]), None, None, True

            def calc_ref_req_trajectory_secondary(self, state, local_path_x, local_path_y, local_path_yaw, speed_profile, dl, motion_state):
                _ = (state, local_path_x, local_path_y, local_path_yaw, speed_profile, dl, motion_state)
                return np.array([[3.0], [4.0], [0.0], [0.0]]), None, None, True

        trajectories = build_requester_requested_trajectories(
            FakePlanner(),
            [
                build_requester_stage_spec("ego", C.follow_lane, "", "primary", secondary=False),
                build_requester_stage_spec("acceptor_1", C.find_lane_change, "", "secondary", secondary=True),
            ],
            {"ego": State(), "acceptor_1": State()},
            {
                "ego": ([0.0], [0.0], [0.0], [0.0]),
                "acceptor_1": ([0.0], [0.0], [0.0], [0.0]),
            },
            {"ego": [1.0], "acceptor_1": [1.0]},
            0.1,
        )

        self.assertTrue(trajectories["ego"]["enabled"])
        self.assertEqual(float(trajectories["ego"]["xref"][0, 0]), 1.0)
        self.assertEqual(float(trajectories["acceptor_1"]["xref"][0, 0]), 3.0)

    def test_build_requester_local_paths_uses_stage_specs(self):
        class FakePlanner:
            def local_path(self, *args):
                _ = args
                return [1.0], [2.0], [0.1], [0.01], True

            def local_path_secondary(self, *args):
                _ = args
                return [3.0], [4.0], [0.2], [0.02], True

        path = PlannedPath([0.0], [0.0], [0.0], [0.0], [0.0, 0.0, 0.0], [1.0])
        local_paths = build_requester_local_paths(
            FakePlanner(),
            [
                build_requester_stage_spec("ego", C.follow_lane, "", "primary", secondary=False),
                build_requester_stage_spec("acceptor_1", C.find_lane_change, "", "secondary", secondary=True),
            ],
            {
                "ego": {"base": path, "alt": path},
                "acceptor_1": {"base": path, "alt": path},
            },
            {"ego": State(), "acceptor_1": State()},
            [],
            {
                "ego": {"accept_state": C.no_request_received},
                "acceptor_1": {"accept_state": C.no_request_received},
            },
        )

        self.assertEqual(local_paths["ego"][0], [1.0])
        self.assertEqual(local_paths["acceptor_1"][0], [3.0])

    def test_build_requester_path_inputs_uses_fallback_path_key(self):
        configs = build_default_cascading_vehicle_configs()
        config_by_role = {config.role: config for config in configs}
        path = PlannedPath([0.0], [0.0], [0.0], [0.0], [0.0, 0.0, 0.0], [1.0])
        alt_path = PlannedPath([1.0], [1.0], [0.0], [0.0], [1.0, 1.0, 0.0], [1.0])
        paths_by_role = {
            "ego": path,
            "acceptor_1": path,
            "ego_alt": alt_path,
            "acceptor_1_alt": alt_path,
        }

        requester_inputs = build_requester_path_inputs(
            ["ego", "acceptor_1"],
            config_by_role,
            paths_by_role,
        )

        self.assertIs(requester_inputs["ego"]["alt"], alt_path)
        self.assertIs(requester_inputs["acceptor_1"]["alt"], alt_path)

    def test_build_fallback_paths_uses_vehicle_metadata(self):
        configs = build_default_two_vehicle_coordination_configs()
        fake_path = PlannedPath([0.0], [0.0], [0.0], [0.0], [0.0, 0.0, 0.0], [1.0])
        with patch("maneuver_coordination.simulation.runner._build_planned_path", return_value=fake_path):
            fallback_paths = build_fallback_paths(configs, [0.0, 1.0], [0.0, 1.0])
        self.assertIn("ego_alt", fallback_paths)

    def test_build_requester_speed_profiles_uses_stage_specs(self):
        requester_stage_specs = [
            build_requester_stage_spec("ego", C.follow_lane, "", "primary", secondary=False),
            build_requester_stage_spec("acceptor_1", C.find_lane_change, "", "secondary", secondary=True),
        ]
        requester_local_paths = {
            "ego": ([0.0], [0.0], [0.0], [0.0]),
            "acceptor_1": ([0.0], [0.0], [0.0], [0.0]),
        }
        requester_states = {"ego": State(), "acceptor_1": State()}
        requester_target_speeds = {"ego": 1.0, "acceptor_1": 2.0}

        class History:
            def __init__(self, target_ind):
                self.target_ind = target_ind

        histories_by_role = {"ego": History(0), "acceptor_1": History(0)}
        coordination_by_requester = {
            "ego": {"speed_state": C.follow_planned},
            "acceptor_1": {"speed_state": C.follow_planned},
        }

        profiles = build_requester_speed_profiles(
            requester_stage_specs,
            requester_local_paths,
            requester_states,
            requester_target_speeds,
            histories_by_role,
            coordination_by_requester,
            State(),
        )

        self.assertEqual(profiles["ego"], [1.0])
        self.assertEqual(profiles["acceptor_1"], [2.0])

    def test_build_role_reference_trajectories_updates_histories_by_role(self):
        class History:
            def __init__(self):
                self.target_ind = 0

        histories_by_role = {"lead": History()}
        references = build_role_reference_trajectories(
            {
                "lead": {
                    "state": State(),
                    "path_x": [0.0, 1.0],
                    "path_y": [0.0, 0.0],
                    "path_yaw": [0.0, 0.0],
                    "path_k": [0.0, 0.0],
                    "speed_profile": [1.0, 1.0],
                }
            },
            histories_by_role,
            0.1,
        )

        self.assertIn("lead", references)
        self.assertGreaterEqual(histories_by_role["lead"].target_ind, 0)

    def test_find_conflicting_vehicle_roles_filters_to_target_lane(self):
        configs = build_default_two_vehicle_coordination_configs()
        xreq = np.zeros((4, 20))
        xreq[0, :3] = [0.0, 1.0, 2.0]
        xreq[1, :3] = [6.0, 6.0, 6.0]

        conflicting_ref = np.zeros((4, 20))
        conflicting_ref[0, :3] = [0.5, 1.5, 2.5]
        conflicting_ref[1, :3] = [6.0, 6.0, 6.0]

        non_target_lane_ref = np.zeros((4, 20))
        non_target_lane_ref[0, :3] = [0.5, 1.5, 2.5]
        non_target_lane_ref[1, :3] = [2.0, 2.0, 2.0]

        conflicts = find_conflicting_vehicle_roles(
            "ego",
            xreq,
            {"acceptor": conflicting_ref, "lead": non_target_lane_ref},
            {
                "acceptor": State(x=0.0, y=6.0, v=5.0),
                "lead": State(x=0.0, y=2.0, v=5.0),
            },
            configs,
        )

        self.assertEqual(conflicts, ["acceptor"])

    def test_find_vehicle_ahead_in_lane_returns_closest_vehicle_ahead(self):
        configs = build_default_cascading_vehicle_configs()
        states_by_role = {
            "ego": State(x=10.0, y=2.0, v=5.0),
            "lead": State(x=20.0, y=2.0, v=5.0),
            "acceptor_1": State(x=12.0, y=6.0, v=5.0),
            "acceptor_2": State(x=15.0, y=10.0, v=5.0),
            "non_cooperating_vehicle": State(x=18.0, y=6.0, v=5.0),
        }

        lead_role = find_vehicle_ahead_in_lane("ego", states_by_role, configs)
        self.assertEqual(lead_role, "lead")

    def test_infer_lane_id_from_y_uses_runtime_position(self):
        configs = build_default_cascading_vehicle_configs()
        self.assertEqual(infer_lane_id_from_y(2.1, configs), 0)
        self.assertEqual(infer_lane_id_from_y(6.2, configs), 1)
        self.assertEqual(infer_lane_id_from_y(9.7, configs), 2)

    def test_find_vehicle_ahead_in_lane_uses_runtime_lane_after_lane_change(self):
        configs = build_default_cascading_vehicle_configs()
        states_by_role = {
            "non_cooperating_vehicle": State(x=5.0, y=6.0, v=5.0),
            "acceptor_1": State(x=8.0, y=10.0, v=5.0),
            "acceptor_2": State(x=14.0, y=10.0, v=5.0),
            "ego": State(x=1.0, y=2.0, v=5.0),
            "lead": State(x=20.0, y=2.0, v=5.0),
        }

        lead_role = find_vehicle_ahead_in_lane("non_cooperating_vehicle", states_by_role, configs)
        self.assertIsNone(lead_role)

    def test_find_vehicle_behind_in_lane_returns_closest_vehicle_behind(self):
        configs = build_default_cascading_vehicle_configs()
        states_by_role = {
            "ego": State(x=10.0, y=2.0, v=5.0),
            "lead": State(x=20.0, y=2.0, v=5.0),
            "acceptor_1": State(x=12.0, y=6.0, v=5.0),
            "acceptor_2": State(x=15.0, y=10.0, v=5.0),
            "non_cooperating_vehicle": State(x=8.0, y=6.0, v=5.0),
        }

        rear_role = find_vehicle_behind_in_lane("ego", states_by_role, configs, lane_id=1)
        self.assertEqual(rear_role, "non_cooperating_vehicle")

    def test_find_target_lane_neighbors_returns_front_and_rear_roles(self):
        configs = build_default_cascading_vehicle_configs()
        states_by_role = {
            "ego": State(x=10.0, y=2.0, v=5.0),
            "lead": State(x=20.0, y=2.0, v=5.0),
            "acceptor_1": State(x=12.0, y=6.0, v=5.0),
            "acceptor_2": State(x=15.0, y=10.0, v=5.0),
            "non_cooperating_vehicle": State(x=8.0, y=6.0, v=5.0),
        }

        neighbors = find_target_lane_neighbors("ego", states_by_role, configs)
        self.assertEqual(neighbors["front"], "acceptor_1")
        self.assertEqual(neighbors["rear"], "non_cooperating_vehicle")

    def test_select_secondary_requester_role_prefers_runtime_target_lane_neighbor(self):
        configs = build_default_cascading_vehicle_configs()
        states_by_role = {
            "ego": State(x=10.0, y=2.0, v=5.0),
            "lead": State(x=20.0, y=2.0, v=5.0),
            "acceptor_1": State(x=12.0, y=6.0, v=5.0),
            "acceptor_2": State(x=15.0, y=10.0, v=5.0),
            "non_cooperating_vehicle": State(x=8.0, y=6.0, v=5.0),
        }

        selected_role = select_secondary_requester_role(
            "ego",
            ["ego", "acceptor_1"],
            states_by_role,
            configs,
            "acceptor_1",
        )

        self.assertEqual(selected_role, "acceptor_1")

    def test_build_target_lane_reference_candidates_orders_rear_then_front(self):
        configs = build_default_cascading_vehicle_configs()
        states_by_role = {
            "ego": State(x=10.0, y=2.0, v=5.0),
            "lead": State(x=20.0, y=2.0, v=5.0),
            "acceptor_1": State(x=12.0, y=6.0, v=5.0),
            "acceptor_2": State(x=15.0, y=10.0, v=5.0),
            "non_cooperating_vehicle": State(x=8.0, y=6.0, v=5.0),
        }
        candidates = build_target_lane_reference_candidates(
            "ego",
            {"acceptor_1": np.zeros((4, 2)), "non_cooperating_vehicle": np.ones((4, 2))},
            states_by_role,
            configs,
        )

        self.assertEqual(list(candidates.keys()), ["non_cooperating_vehicle", "acceptor_1"])

    def test_is_vehicle_braking_detects_speed_drop(self):
        class History:
            def __init__(self, speeds):
                self.v = speeds

        self.assertTrue(is_vehicle_braking(History([10.0, 9.5])))
        self.assertFalse(is_vehicle_braking(History([10.0, 9.9])))

    def test_should_trigger_maneuver_search_for_close_braking_lead(self):
        class History:
            def __init__(self, speeds):
                self.v = speeds

        self.assertTrue(
            should_trigger_maneuver_search(
                State(x=10.0, y=2.0, v=10.0),
                State(x=20.0, y=2.0, v=4.0),
                History([6.0, 4.0]),
                existing_path_length=0,
            )
        )

    def test_select_lowest_cost_role_prefers_lower_cooperation_cost(self):
        configs = build_default_cascading_vehicle_configs()
        selected_role = select_lowest_cost_role(["non_cooperating_vehicle", "acceptor_1"], configs, "non_cooperating_vehicle")
        self.assertEqual(selected_role, "acceptor_1")

    def test_build_acc_speed_profile_caps_speed_for_close_lead(self):
        path = PlannedPath(
            x=[0.0, 1.0, 2.0],
            y=[0.0, 0.0, 0.0],
            yaw=[0.0, 0.0, 0.0],
            curvature=[0.0, 0.0, 0.0],
            goal=[2.0, 0.0, 0.0],
            speed_profile=[5.0, 5.0, 5.0],
        )
        state = State(x=0.0, y=0.0, v=6.0)
        lead_state = State(x=6.0, y=0.0, v=4.0)

        speed_profile = build_acc_speed_profile(path, state, 7.0, lead_state)

        self.assertTrue(all(speed <= 3.5 for speed in speed_profile))

    def test_build_acc_speed_profile_blends_smoothly_for_intermediate_gap(self):
        path = PlannedPath(
            x=[0.0, 1.0, 2.0],
            y=[0.0, 0.0, 0.0],
            yaw=[0.0, 0.0, 0.0],
            curvature=[0.0, 0.0, 0.0],
            goal=[2.0, 0.0, 0.0],
            speed_profile=[5.0, 5.0, 5.0],
        )
        state = State(x=0.0, y=0.0, v=6.0)
        lead_state = State(x=14.0, y=0.0, v=4.0)

        speed_profile = build_acc_speed_profile(path, state, 7.0, lead_state)

        self.assertTrue(all(3.5 < speed < 7.0 for speed in speed_profile))

    def test_requested_trajectory_with_zero_acceleration_matches_constant_velocity_progression(self):
        state = State(x=0.0, y=0.0, yaw=0.0, v=10.0, a=0.0)
        cx = list(range(40))
        cy = [0.0] * 40
        cyaw = [0.0] * 40
        speed_profile = [10.0] * 40

        xreq, indq, _, enabled = build_requested_trajectory(
            state,
            (cx, cy, cyaw, []),
            speed_profile,
            1.0,
            lambda state, cx, cy, cyaw: (0, 0.0),
        )

        self.assertTrue(enabled)
        self.assertEqual(indq, 0)
        self.assertEqual(xreq[0, 0], 0.0)
        self.assertEqual(xreq[1, 0], 0.0)
        self.assertEqual(xreq[0, 1], 1.0)

    def test_requested_trajectory_progresses_when_vehicle_is_already_far_along_path(self):
        state = State(x=30.0, y=0.0, yaw=0.0, v=10.0, a=0.0)
        cx = list(range(120))
        cy = [0.0] * 120
        cyaw = [0.0] * 120
        speed_profile = [10.0] * 120

        xreq, indq, _, enabled = build_requested_trajectory(
            state,
            (cx, cy, cyaw, []),
            speed_profile,
            1.0,
            lambda state, cx, cy, cyaw: (30, 0.0),
        )

        self.assertTrue(enabled)
        self.assertEqual(indq, 30)
        self.assertGreater(xreq[0, 2], xreq[0, 1])

    def test_reference_trajectory_with_positive_acceleration_advances_farther(self):
        cx = list(range(60))
        cy = [0.0] * 60
        cyaw = [0.0] * 60
        ck = [0.0] * 60
        speed_profile = [10.0] * 60

        xref_constant, _, _ = calc_simulation_ref_trajectory(
            State(x=0.0, y=0.0, yaw=0.0, v=10.0, a=0.0),
            cx,
            cy,
            cyaw,
            ck,
            speed_profile,
            1.0,
        )
        xref_accelerating, _, _ = calc_simulation_ref_trajectory(
            State(x=0.0, y=0.0, yaw=0.0, v=10.0, a=2.0),
            cx,
            cy,
            cyaw,
            ck,
            speed_profile,
            1.0,
        )

        self.assertGreater(xref_accelerating[0, 15], xref_constant[0, 15])

    def test_get_requester_candidate_roles_uses_lane_targets(self):
        configs = build_default_cascading_vehicle_configs()
        self.assertEqual(get_requester_candidate_roles(configs), ["ego", "acceptor_1"])

    def test_behavior_planner_can_be_constructed(self):
        planner = BehaviouralLocalPlanner([], [], [], [])
        self.assertFalse(planner.path_found())

    def test_vehicle_constants_are_available(self):
        self.assertGreater(WB, 0.0)
        self.assertGreater(LENGTH, 0.0)
        self.assertGreater(MAX_STEER, 0.0)

    def test_grouped_state_enums_are_available(self):
        self.assertEqual(C.MotionState.FOLLOW_LANE, C.follow_lane)
        self.assertEqual(C.RequestState.SEND_REQUEST, C.send_request)
        self.assertEqual(C.AcceptState.REQUEST_ACCEPTED, C.request_accepted)
        self.assertEqual(C.SpeedState.FOLLOW_PLANNED, C.follow_planned)
        self.assertEqual(C.Priority.HIGH_PRIORITY, C.high_priority)

    def test_planner_params_aliases_are_available(self):
        self.assertEqual(C.PLANNER_PARAMS.horizon_steps, C.T)
        self.assertEqual(C.PLANNER_PARAMS.dt, C.DT)
        self.assertEqual(C.PLANNER_PARAMS.lane_change_path_padding, C.LANE_CHANGE_PATH_PADDING)
        self.assertEqual(C.PLANNER_PARAMS.second_lane_y, C.SECOND_LANE_Y)

    def test_cascading_scenario_regression_id1_requests_id3_then_id3_requests_id4(self):
        configs = build_default_cascading_vehicle_configs()
        ox, oy, ox2, oy2, obstacle_list = build_cascading_obstacles()

        result = run_cascading_scenario(
            vehicle_configs=configs,
            ox=ox,
            oy=oy,
            obstacle_list=obstacle_list,
            show_animation=False,
            verbose_events=False,
        )

        sender_receiver_pairs = [(message.sender_id, message.receiver_id) for message in result["messages"]]
        self.assertIn((1, 3), sender_receiver_pairs)
        self.assertIn((3, 4), sender_receiver_pairs)

        events = result["events"]
        self.assertTrue(any("ID 1 motion -> lane_change" in event for event in events))
        self.assertTrue(any("ID 3 motion -> lane_change" in event for event in events))

    def test_three_vehicle_coordination_regression_ego_requests_target_lane_vehicle(self):
        configs = build_default_three_vehicle_coordination_configs()
        ox, oy, ox2, oy2, obstacle_list = build_three_vehicle_obstacles()

        result = run_three_vehicle_coordination_scenario(
            vehicle_configs=configs,
            ox=ox,
            oy=oy,
            obstacle_list=obstacle_list,
            show_animation=False,
            verbose_events=False,
        )

        sender_receiver_pairs = [(message.sender_id, message.receiver_id) for message in result["messages"]]
        self.assertTrue((1, 3) in sender_receiver_pairs or (1, 4) in sender_receiver_pairs)
        self.assertTrue(any("ID 1 motion -> lane_change" in event for event in result["events"]))
        self.assertGreater(max(result["histories"][0].y), 4.0)
        self.assertLessEqual(result["histories"][3].x[-1], configs[3].goal[0] + 0.5)

    def test_three_vehicle_coordination_4_messages_emits_offer_or_confirm(self):
        configs = build_default_three_vehicle_coordination_4_messages_configs()
        ox, oy, ox2, oy2, obstacle_list = build_three_vehicle_obstacles()

        result = run_three_vehicle_coordination_4_messages_scenario(
            vehicle_configs=configs,
            ox=ox,
            oy=oy,
            obstacle_list=obstacle_list,
            show_animation=False,
            verbose_events=False,
        )

        message_types = [message.message_type for message in result["messages"]]
        self.assertIn("maneuver_request", message_types)
        self.assertTrue("maneuver_offer" in message_types or "confirm_offer" in message_types)

    def test_rejected_request_then_free_lane_regression(self):
        configs = build_default_rejected_request_then_free_lane_configs()
        ox, oy, ox2, oy2, obstacle_list = build_rejected_request_obstacles()

        result = run_rejected_request_then_free_lane_scenario(
            vehicle_configs=configs,
            ox=ox,
            oy=oy,
            show_animation=False,
            verbose_events=False,
        )

        events_text = "\n".join(result["events"])
        self.assertIn("request rejected", events_text)
        self.assertIn("adjacent lane free", events_text)
        self.assertGreater(max(result["histories"][0].y), 4.0)


if __name__ == "__main__":
    unittest.main()
