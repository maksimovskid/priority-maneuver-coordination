import math
import unittest
from unittest.mock import patch

import numpy as np

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination.behavior_planner import BehaviouralLocalPlanner, ReferencePath
from maneuver_coordination.scenarios.cascading_coordination import build_obstacles as cascading_obstacles
from maneuver_coordination.scenarios.coordination_2_vehicles import build_obstacles as direct_obstacles
from maneuver_coordination.scenarios.rejected_request_then_free_lane import build_obstacles as rejected_obstacles
from maneuver_coordination.scenarios.three_vehicles_coordination import build_obstacles as multi_obstacles
from maneuver_coordination.scenarios.three_vehicles_coordination_4_messages import build_obstacles as four_obstacles
from maneuver_coordination.simulation.core.types import State
from maneuver_coordination.simulation.motion.paths import (
    build_paths_for_configs,
    build_paths_for_two_vehicle_coordination_configs,
)
from maneuver_coordination.simulation.motion.reference import build_curvature
from maneuver_coordination.simulation.runners.scenario_configs import (
    build_default_cascading_vehicle_configs,
    build_default_rejected_request_then_free_lane_configs,
    build_default_three_vehicle_coordination_4_messages_configs,
    build_default_three_vehicle_coordination_configs,
    build_default_two_vehicle_coordination_configs,
)


class PointwiseCurvatureTests(unittest.TestCase):
    def test_nonuniform_points_use_original_knots_not_resampled_positions(self):
        x = [0.0, 0.15, 0.9, 2.2, 3.0]
        y = [0.0, 0.02, 0.6, 1.7, 1.8]
        spline, samples, curvature = build_curvature(x, y)
        np.testing.assert_allclose(samples, np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        self.assertEqual(len(curvature), len(x))
        expected = []
        for s in samples:
            s = min(s, np.nextafter(samples[-1], -np.inf))
            dx, dy = spline.sx.calcd(s), spline.sy.calcd(s)
            ddx, ddy = spline.sx.calcdd(s), spline.sy.calcdd(s)
            expected.append((dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5)
        np.testing.assert_allclose(curvature, expected)
        np.testing.assert_allclose(BehaviouralLocalPlanner._curvature_from_xy(x, y), expected)
        np.testing.assert_allclose(build_curvature(x, y, ds=10.0)[2], expected)

    def test_duplicate_terminal_and_degenerate_points_keep_one_curvature_per_point(self):
        for x, y in [([], []), ([1.0], [2.0]), ([1.0, 1.0], [2.0, 2.0]),
                     ([0.0, 1.0, 1.0, 2.0, 2.0], [0.0, 0.5, 0.5, 0.0, 0.0])]:
            with self.subTest(x=x):
                _, s, curvature = build_curvature(x, y)
                self.assertEqual(len(s), len(x))
                self.assertEqual(len(curvature), len(x))
                self.assertTrue(np.isfinite(curvature).all())
                if len(x) == 5:
                    self.assertEqual(curvature[1], curvature[2])
                    self.assertEqual(curvature[3], curvature[4])

    def test_invalid_geometry_is_not_normalized(self):
        for x, y in [([0.0], []), ([0.0, float("nan")], [0.0, 1.0])]:
            with self.subTest(x=x), self.assertRaises(ValueError):
                BehaviouralLocalPlanner._curvature_from_xy(x, y)


class PathJoinTests(unittest.TestCase):
    def test_suffix_uses_endpoint_preserves_resolution_and_includes_goal(self):
        for yaw in (0.0, math.pi / 2.0, math.pi):
            for endpoint in (11.0, 11.04, 11.06):
                with self.subTest(yaw=yaw, endpoint=endpoint):
                    planner = BehaviouralLocalPlanner([], [], [], [])
                    key = planner.PRIMARY_RRT_KEY
                    distances = np.arange(0.0, 15.01, 0.1)
                    reference = ReferencePath(
                        distances * math.cos(yaw), distances * math.sin(yaw),
                        np.full(len(distances), yaw), np.zeros(len(distances)),
                    )
                    planner._append_path_segment(key, ([endpoint * math.cos(yaw)],
                                                      [endpoint * math.sin(yaw)], [yaw], [0.0]))
                    planner._append_reference_suffix(key, reference)
                    x, y, headings, curvature = planner._path_tuple(key)
                    progress = np.asarray(x) * math.cos(yaw) + np.asarray(y) * math.sin(yaw)
                    self.assertTrue(np.all(np.diff(progress) > 0.0))
                    self.assertLessEqual(progress[1] - progress[0], 0.1 + 1e-9)
                    self.assertAlmostEqual(progress[-1], 15.0)
                    self.assertEqual(len({len(x), len(y), len(headings), len(curvature)}), 1)
                    expected = distances[distances > endpoint + 1e-9]
                    np.testing.assert_allclose(progress[1:], expected)

    def test_empty_generated_segment_does_not_acquire_reference_suffix(self):
        planner = BehaviouralLocalPlanner([], [], [], [])
        planner._append_reference_suffix(planner.PRIMARY_RRT_KEY, ReferencePath([1.0], [2.0], [0.0], [0.0]))
        self.assertEqual(planner._path_tuple(planner.PRIMARY_RRT_KEY), ([], [], [], []))

    def test_misaligned_segment_is_rejected_before_mutating_buffers(self):
        planner = BehaviouralLocalPlanner([], [], [], [])
        with self.assertRaises(ValueError):
            planner._append_path_segment(planner.PRIMARY_RRT_KEY, ([1.0], [2.0], [0.0], []))
        self.assertEqual(planner.generated_path_length(), 0)


class UnavailableRrtTests(unittest.TestCase):
    def test_planning_failure_returns_explicit_unavailable_axes(self):
        with patch("maneuver_coordination.coordination.behavior_planner.RRTReedsShepp") as rrt:
            for unavailable in (None, []):
                rrt.return_value.planning.return_value = unavailable
                self.assertEqual(BehaviouralLocalPlanner._build_rrt_path(
                    [0.0, 10.0], [6.0, 6.0], [0.0, 0.0], State(), [], 1,
                ), ([], [], [], []))

    def test_failed_search_keeps_base_and_committed_path_without_suffix(self):
        base = ReferencePath([0.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0])
        target = ReferencePath([0.0, 1.0, 2.0], [6.0] * 3, [0.0] * 3, [0.0] * 3)
        for role in (BehaviouralLocalPlanner.REQUESTER_ROLE, BehaviouralLocalPlanner.SECONDARY_ROLE):
            with self.subTest(role=role):
                planner = BehaviouralLocalPlanner([], [], [], [])
                committed = planner._committed_path_key(role)
                planner._append_path_segment(committed, base)
                with patch.object(planner, "_lane_change_builder_for_role", return_value=lambda *args: ([], [], [], [])):
                    result = planner.local_path_for_reference_paths(
                        role, base, target, C.find_lane_change, State(), [], C.request_accepted,
                    )
                self.assertEqual(result, (*base, False))
                self.assertFalse(planner._get_path_found(role))
                self.assertEqual(planner.generated_path_length(role), 0)
                self.assertEqual(planner._path_tuple(committed), tuple(base))

    def test_failed_return_search_keeps_valid_base_path(self):
        planner = BehaviouralLocalPlanner([], [], [], [])
        base = ReferencePath([0.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0])
        with patch.object(planner, "rrt_double_lane_change_path", return_value=([], [], [], [])):
            result = planner.local_path_for_reference_paths(
                planner.REQUESTER_ROLE, base, base, C.find_double_lane_change, State(), [], C.no_request_received,
            )
        self.assertEqual(result[:4], tuple(base))
        self.assertFalse(planner.double_lane_change_path_found)
        self.assertEqual(planner.generated_path_length(planner.RETURN_ROLE), 0)


class DefaultPathContractTests(unittest.TestCase):
    def test_all_default_nominal_fallback_and_actual_rrt_paths_are_aligned(self):
        cases = (
            (build_default_cascading_vehicle_configs, cascading_obstacles, build_paths_for_configs),
            (build_default_two_vehicle_coordination_configs, direct_obstacles, build_paths_for_two_vehicle_coordination_configs),
            (build_default_rejected_request_then_free_lane_configs, rejected_obstacles, build_paths_for_two_vehicle_coordination_configs),
            (build_default_three_vehicle_coordination_configs, multi_obstacles, build_paths_for_two_vehicle_coordination_configs),
            (build_default_three_vehicle_coordination_4_messages_configs, four_obstacles, build_paths_for_two_vehicle_coordination_configs),
        )
        for configs_builder, obstacles_builder, paths_builder in cases:
            configs = configs_builder()
            ox, oy, _, _, obstacle_list = obstacles_builder()
            paths = paths_builder(configs, ox, oy)
            for name, path in paths.items():
                with self.subTest(scenario=configs_builder.__name__, path=name):
                    axes = (path.x, path.y, path.yaw, path.curvature, path.speed_profile)
                    self.assertGreater(len(path.x), 1)
                    self.assertEqual(len({len(axis) for axis in axes}), 1)
                    self.assertTrue(all(np.isfinite(axis).all() for axis in axes))
                    self.assertAlmostEqual(path.x[-1], path.goal[0])
                    self.assertAlmostEqual(path.y[-1], path.goal[1])
            for config in configs:
                if "transition_length" not in config.metadata:
                    continue
                with self.subTest(scenario=configs_builder.__name__, analytic_fallback=config.role):
                    path = paths[config.fallback_path_key]
                    length = float(config.metadata["transition_length"])
                    delta = float(config.metadata["alt_goal_y"]) - config.start[1]
                    progress = np.clip((np.asarray(path.x) - config.start[0]) / length, 0.0, 1.0)
                    interior = (progress > 0.0) & (progress < 1.0)
                    dy = np.where(interior, delta * 6.0 * progress * (1.0 - progress) / length, 0.0)
                    ddy = np.where(interior, delta * (6.0 - 12.0 * progress) / length ** 2, 0.0)
                    np.testing.assert_allclose(path.y, config.start[1] + delta * progress ** 2 * (3.0 - 2.0 * progress))
                    np.testing.assert_allclose(path.yaw, np.arctan2(dy, 1.0))
                    np.testing.assert_allclose(path.curvature, ddy / (1.0 + dy ** 2) ** 1.5)
            for config in configs:
                if not config.fallback_path_key:
                    continue
                with self.subTest(scenario=configs_builder.__name__, rrt=config.role):
                    target = paths[config.fallback_path_key]
                    state = State(x=config.start[0], y=config.start[1], yaw=config.start[2], v=config.initial_speed)
                    segment = BehaviouralLocalPlanner._build_rrt_path(
                        target.x, target.y, target.yaw, state, obstacle_list, C.PLANNER_PARAMS.lane_change_goal_offset,
                    )
                    self.assertGreater(len(segment[0]), 1)
                    self.assertEqual(len({len(axis) for axis in segment}), 1)
                    self.assertTrue(all(np.isfinite(axis).all() for axis in segment))
                    np.testing.assert_allclose(segment[3], build_curvature(segment[0], segment[1])[2])
                    distances = np.hypot(np.asarray(target.x) - segment[0][-1], np.asarray(target.y) - segment[1][-1])
                    endpoint_index = int(np.argmin(distances))
                    self.assertLess(distances[endpoint_index], 1e-9)
                    self.assertAlmostEqual(segment[2][-1], target.yaw[endpoint_index])
                    planner = BehaviouralLocalPlanner([], [], [], [])
                    key = planner.PRIMARY_RRT_KEY
                    planner._append_path_segment(key, segment)
                    planner._append_reference_suffix(key, ReferencePath(target.x, target.y, target.yaw, target.curvature))
                    joined = planner._path_tuple(key)
                    self.assertEqual(len({len(axis) for axis in joined}), 1)
                    join = len(segment[0])
                    self.assertGreater(len(joined[0]), join)
                    dx, dy = joined[0][join] - joined[0][join - 1], joined[1][join] - joined[1][join - 1]
                    self.assertLessEqual(math.hypot(dx, dy), 0.11)
                    self.assertGreaterEqual(dx * math.cos(segment[2][-1]) + dy * math.sin(segment[2][-1]), -1e-9)
                    self.assertAlmostEqual(joined[0][-1], target.x[-1])
                    self.assertAlmostEqual(joined[1][-1], target.y[-1])
                    for axis, reference_axis in zip(joined, (target.x, target.y, target.yaw, target.curvature)):
                        np.testing.assert_allclose(axis[join:], reference_axis[endpoint_index + 1:])


if __name__ == "__main__":
    unittest.main()
