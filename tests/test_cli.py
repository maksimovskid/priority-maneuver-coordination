import sys
import unittest
from unittest.mock import patch

from maneuver_coordination import cli


class CliTests(unittest.TestCase):
    def test_default_scenario_is_cascading_coordination(self):
        parser = cli.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.scenario, cli.CASCADING_COORDINATION)

    def test_two_vehicles_coordination_dispatches(self):
        with patch.object(cli, "run_two_vehicle_priority") as run_priority:
            with patch.object(cli, "run_cascading_lane_change") as run_cascading:
                with patch.object(sys, "argv", ["run.py", cli.TWO_VEHICLES_COORDINATION]):
                    cli.main()

        run_priority.assert_called_once()
        run_cascading.assert_not_called()

    def test_legacy_coordination_2_vehicles_alias_dispatches(self):
        with patch.object(cli, "run_two_vehicle_priority") as run_priority:
            with patch.object(cli, "run_cascading_lane_change") as run_cascading:
                with patch.object(sys, "argv", ["run.py", "coordination_2_vehicles"]):
                    cli.main()

        run_priority.assert_called_once()
        run_cascading.assert_not_called()

    def test_legacy_priority_alias_dispatches(self):
        with patch.object(cli, "run_two_vehicle_priority") as run_priority:
            with patch.object(cli, "run_cascading_lane_change") as run_cascading:
                with patch.object(sys, "argv", ["run.py", "priority"]):
                    cli.main()

        run_priority.assert_called_once()
        run_cascading.assert_not_called()

    def test_three_vehicles_coordination_dispatches(self):
        with patch.object(cli, "run_three_vehicle_coordination") as run_three_vehicle:
            with patch.object(cli, "run_two_vehicle_priority") as run_priority:
                with patch.object(cli, "run_rejected_request_then_free_lane") as run_rejected:
                    with patch.object(cli, "run_cascading_lane_change") as run_cascading:
                        with patch.object(sys, "argv", ["run.py", cli.THREE_VEHICLES_COORDINATION]):
                            cli.main()

        run_three_vehicle.assert_called_once()
        run_priority.assert_not_called()
        run_rejected.assert_not_called()
        run_cascading.assert_not_called()

    def test_three_vehicles_coordination_4_messages_dispatches(self):
        with patch.object(cli, "run_three_vehicle_coordination_4_messages") as run_three_vehicle:
            with patch.object(cli, "run_two_vehicle_priority") as run_priority:
                with patch.object(cli, "run_rejected_request_then_free_lane") as run_rejected:
                    with patch.object(cli, "run_cascading_lane_change") as run_cascading:
                        with patch.object(sys, "argv", ["run.py", cli.THREE_VEHICLES_COORDINATION_4_MESSAGES]):
                            cli.main()

        run_three_vehicle.assert_called_once()
        run_priority.assert_not_called()
        run_rejected.assert_not_called()
        run_cascading.assert_not_called()

    def test_rejected_request_then_free_lane_dispatches(self):
        with patch.object(cli, "run_rejected_request_then_free_lane") as run_rejected:
            with patch.object(cli, "run_two_vehicle_priority") as run_priority:
                with patch.object(cli, "run_cascading_lane_change") as run_cascading:
                    with patch.object(sys, "argv", ["run.py", cli.REJECTED_REQUEST_THEN_FREE_LANE]):
                        cli.main()

        run_rejected.assert_called_once()
        run_priority.assert_not_called()
        run_cascading.assert_not_called()

    def test_save_animation_flag_is_forwarded(self):
        with patch.object(cli, "run_cascading_lane_change") as run_cascading:
            with patch.object(sys, "argv", ["run.py", cli.CASCADING_COORDINATION, "--no-animation", "--save-output-dir", "output", "--save-animation"]):
                cli.main()

        run_cascading.assert_called_once_with(
            show_animation=False,
            save_output_dir="output",
            save_animation=True,
        )


if __name__ == "__main__":
    unittest.main()
