import unittest
from fractions import Fraction

import numpy as np

from maneuver_coordination.coordination import constants as C
from maneuver_coordination.coordination import trajectory_logic
from maneuver_coordination.coordination.conflicts import (
    conflict_time_indices,
    is_sampled_trajectory_conflict_free,
    sampled_clearances,
)
from maneuver_coordination.simulation.core.types import State
from maneuver_coordination.simulation.motion import reference
from maneuver_coordination.simulation.motion.vehicle_dynamics import calc_nearest_index


class TrajectoryContractTests(unittest.TestCase):
    def build_all(self, state, cx, cy, cyaw, sp, dl=1.0):
        ck = [0.0] * len(cx)
        return {
            "requested": trajectory_logic.build_requested_trajectory(
                state, (cx, cy, cyaw, ck), sp, dl, calc_nearest_index,
            ),
            "coordination": trajectory_logic.calc_ref_trajectory(
                state, cx, cy, cyaw, ck, sp, dl, calc_nearest_index,
            ),
            "simulation": reference.calc_ref_trajectory(
                state, cx, cy, cyaw, ck, sp, dl,
            ),
            "speed_offset": reference.build_reference_for_speed(
                state, cx, cy, cyaw, sp, dl, 0.0,
            ),
            "target_speed": reference.build_reference_for_target_speed(
                state, cx, cy, cyaw, dl, state.v,
            ),
            "acceleration": reference.build_reference_for_acceleration(
                state, cx, cy, cyaw, dl, state.a,
            ),
        }

    def assert_horizon(self, result):
        trajectory, _, steering = result[:3]
        self.assertEqual(trajectory.shape, (C.NX, 20))
        self.assertEqual(steering.shape, (1, 20))
        self.assertTrue(np.isfinite(trajectory).all())
        np.testing.assert_array_equal(steering, 0.0)

    def test_all_builders_preserve_actual_current_state_not_nearest_path_sample(self):
        state = State(x=30.2, y=5.5, v=10.0, yaw=0.25)
        results = self.build_all(state, list(range(100)), [6.0] * 100,
                                 [0.1] * 100, [8.0] * 100)
        for name, result in results.items():
            with self.subTest(builder=name):
                self.assert_horizon(result)
                self.assertEqual(result[1], 30)
                np.testing.assert_array_equal(
                    result[0][:, 0], [state.x, state.y, state.v, state.yaw],
                )
                np.testing.assert_allclose(result[0][0, 1:], np.arange(31, 50))
                np.testing.assert_allclose(result[0][1, 1:], 6.0)
                np.testing.assert_allclose(result[0][3, 1:], 0.1)

    def test_requested_and_reference_samples_represent_the_same_times(self):
        state = State(x=10.0, y=6.0, v=10.0)
        results = self.build_all(state, list(range(100)), [6.0] * 100,
                                 [0.0] * 100, [10.0] * 100)
        expected = np.vstack((np.arange(10, 30), np.full(20, 6.0),
                              np.full(20, 10.0), np.zeros(20)))
        for name, result in results.items():
            with self.subTest(builder=name):
                self.assert_horizon(result)
                np.testing.assert_allclose(result[0], expected)

    def test_accelerated_position_and_speed_are_aligned_to_sample_time(self):
        state = State(x=10.0, y=6.0, v=2.0, a=2.0)
        dl = 0.01
        cx = np.arange(0.0, 40.0, dl).tolist()
        results = self.build_all(state, cx, [6.0] * len(cx),
                                 [0.0] * len(cx), [2.0] * len(cx), dl)
        times = np.arange(20) * C.DT
        expected_x = state.x + state.v * times + 0.5 * state.a * times ** 2
        for name in ("requested", "coordination", "simulation", "speed_offset", "acceleration"):
            with self.subTest(builder=name):
                np.testing.assert_allclose(results[name][0][0], expected_x)
        accelerated = results["acceleration"]
        np.testing.assert_allclose(accelerated[0][2], state.v + state.a * times)
        self.assertAlmostEqual(accelerated[3], state.v + state.a * times[-1])

    def test_short_paths_repeat_endpoint_instead_of_zero_padding(self):
        state = State(x=10.0, y=6.0, v=10.0)
        results = self.build_all(state, [10.0, 11.0], [6.0, 6.0],
                                 [0.1, 0.2], [10.0, 10.0])
        for name, result in results.items():
            with self.subTest(builder=name):
                self.assert_horizon(result)
                np.testing.assert_allclose(result[0][0, 1:], 11.0)
                np.testing.assert_allclose(result[0][1, 1:], 6.0)
                np.testing.assert_allclose(result[0][3, 1:], 0.2)

    def test_empty_paths_and_disabled_requests_are_unavailable(self):
        state = State(x=10.0, y=6.0, v=2.0, yaw=0.25)
        results = self.build_all(state, [], [], [], [])
        for name, result in results.items():
            with self.subTest(builder=name):
                self.assertEqual(result[0].shape, (C.NX, 0))
                self.assertEqual(result[2].shape, (1, 0))
                self.assertEqual(result[1], 0)
                if name == "requested":
                    self.assertFalse(result[3])
                if name == "acceleration":
                    self.assertEqual(result[3], state.v)
        disabled = trajectory_logic.empty_requested_trajectory()
        self.assertEqual(disabled[0].shape, (C.NX, 0))
        self.assertEqual(disabled[2].shape, (1, 0))
        self.assertFalse(disabled[3])
        candidates = reference.build_three_candidate_refs(
            state, [], [], [], 1.0, [1.0, 2.0, 3.0],
        )
        for candidate in candidates[:3]:
            self.assertEqual(candidate.shape, (C.NX, 0))
        self.assertEqual(candidates[4].shape, (1, 0))

    def test_target_and_offset_candidates_only_change_future_samples(self):
        state = State(x=10.0, y=6.0, v=10.0)
        cx, cy, cyaw = list(range(100)), [6.0] * 100, [0.0] * 100
        target, _, _ = reference.build_reference_for_target_speed(
            state, cx, cy, cyaw, 1.0, 20.0,
        )
        self.assertEqual(target[2, 0], state.v)
        np.testing.assert_allclose(target[0], 10.0 + 2.0 * np.arange(20))
        np.testing.assert_allclose(target[2, 1:], 20.0)
        low, medium, high, _, steering = reference.build_three_candidate_refs(
            state, cx, cy, cyaw, 1.0, [1.0, 2.0, 3.0],
        )
        for reduction, candidate in zip([1.0, 2.0, 3.0], [low, medium, high]):
            with self.subTest(reduction=reduction):
                self.assert_horizon((candidate, 0, steering))
                np.testing.assert_array_equal(candidate[:, 0], [10.0, 6.0, 10.0, 0.0])
                # Exact decimal times keep half-grid ties out of float artifacts.
                step_distance = Fraction(str(state.v - reduction)) * Fraction(str(C.DT))
                expected_x = [state.x + round(step_distance * i) for i in range(20)]
                np.testing.assert_allclose(candidate[0], expected_x)
                np.testing.assert_allclose(candidate[2, 1:], state.v - reduction)

    def test_conflict_indices_use_current_state_time_origin(self):
        self.assertEqual(conflict_time_indices(10.0, 20), (0, 6, 12))
        self.assertEqual(conflict_time_indices(12.0, 20, dt=0.2), (0, 2, 5))

    def test_low_speed_conflicts_use_last_populated_sample_not_padding(self):
        for speed in (0.0, 0.1, 1.0):
            with self.subTest(speed=speed):
                state = State(x=10.0, y=6.0, v=speed)
                results = self.build_all(state, list(range(100)), [6.0] * 100,
                                         [0.0] * 100, [speed] * 100)
                requested = results["requested"][0]
                planned = results["simulation"][0].copy()
                planned[0] += 30.0
                self.assertEqual(conflict_time_indices(speed, requested.shape[1]), (0, 19, 19))
                clearances = sampled_clearances(requested, planned, speed)
                np.testing.assert_allclose(clearances, (28.0, 26.0, 26.0))
                self.assertTrue(is_sampled_trajectory_conflict_free(clearances, speed))
                planned[0, -1] = requested[0, -1]
                self.assertFalse(is_sampled_trajectory_conflict_free(
                    sampled_clearances(requested, planned, speed), speed,
                ))

    def test_clearances_use_shared_horizon_in_either_argument_order(self):
        longer = np.zeros((4, 20))
        shorter = np.zeros((4, 3))
        shorter[0] = [10.0, 20.0, 30.0]
        for requested, planned in ((longer, shorter), (shorter, longer)):
            with self.subTest(requested_length=requested.shape[1]):
                np.testing.assert_allclose(
                    sampled_clearances(requested, planned, 0.1), (8.0, 26.0, 26.0),
                )

    def test_empty_horizon_has_no_valid_conflict_sample(self):
        with self.assertRaises(ValueError):
            conflict_time_indices(1.0, 0)
        for requested, planned in ((np.zeros((4, 0)), np.zeros((4, 20))),
                                   (np.zeros((4, 20)), np.zeros((4, 0)))):
            clearances = sampled_clearances(requested, planned, 1.0)
            self.assertEqual(clearances, (-np.inf, -np.inf, -np.inf))
            self.assertFalse(is_sampled_trajectory_conflict_free(clearances, 1.0))

    def test_padded_external_inputs_require_explicit_populated_prefix(self):
        requested = np.zeros((4, 120))
        planned = np.zeros((4, 120))
        planned[0, :3] = [10.0, 20.0, 30.0]
        np.testing.assert_allclose(
            sampled_clearances(requested, planned, 0.1, valid_count=3),
            (8.0, 26.0, 26.0),
        )
        np.testing.assert_allclose(
            sampled_clearances(requested[:, :2], planned, 0.1, valid_count=3),
            (8.0, 16.0, 16.0),
        )
        clearances = sampled_clearances(requested, planned, 1.0, valid_count=0)
        self.assertFalse(is_sampled_trajectory_conflict_free(clearances, 1.0))
        with self.assertRaises(ValueError):
            sampled_clearances(requested, planned, 1.0, valid_count=-1)

    def test_existing_conflict_callers_reject_unavailable_references(self):
        from maneuver_coordination.coordination.planner import BehaviouralLocalPlanner

        planner = BehaviouralLocalPlanner([], [], [], [])
        state = State(v=1.0)
        available = np.ones((4, 20))
        unavailable = np.empty((4, 0))
        for requested, planned in ((available, unavailable), (unavailable, available),
                                   (unavailable, unavailable)):
            self.assertFalse(planner.check_ref_tr_collision(state, requested, planned))
            self.assertFalse(planner.check_req_tr_conflict(
                state, state, requested, planned, True,
            ))
            self.assertFalse(planner.check_req_tr_conflict_4(
                state, requested, planned, True,
            ))

    def test_curvature_preserves_original_knots_and_repeated_endpoint(self):
        cx = [0.0, 0.5, 0.5, 2.0, 3.0, 3.0]
        cy = [0.0, 0.2, 0.2, 1.0, 1.2, 1.2]
        spline, distances, curvature = reference.build_curvature(cx, cy, ds=10.0)
        self.assertEqual(len(curvature), len(cx))
        np.testing.assert_allclose(distances, np.r_[0.0, np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
        self.assertEqual(curvature[1], curvature[2])
        self.assertEqual(curvature[-1], curvature[-2])
        for index, distance in enumerate(distances):
            parameter = min(distance, np.nextafter(spline.s[-1], -np.inf))
            dx, dy = spline.sx.calcd(parameter), spline.sy.calcd(parameter)
            ddx, ddy = spline.sx.calcdd(parameter), spline.sy.calcdd(parameter)
            expected = (dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
            self.assertAlmostEqual(curvature[index], expected)


if __name__ == "__main__":
    unittest.main()
