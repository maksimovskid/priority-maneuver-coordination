"""Default vehicle configurations for the bundled scenarios."""

from typing import List

from maneuver_coordination.simulation.core.settings import ROAD_END_X
from maneuver_coordination.simulation.core.types import VehicleConfig


def build_default_cascading_vehicle_configs() -> List[VehicleConfig]:
    """Configure the five-vehicle cascading lane-change scenario."""
    return [
        VehicleConfig(
            vehicle_id=1,
            name="ego",
            role="ego",
            start=[9.0, 2.0, 0.0],
            goal=[ROAD_END_X, 2.0, 0.0],
            initial_speed=32.0 / 3.6,
            target_speed=32.0 / 3.6,
            color="b",
            lane_id=0,
            target_lane_id=1,
            cooperation_cost=1.0,
            fallback_path_key="ego_alt",
            metadata={"alt_start_x": "9.0", "alt_start_y": "6.0", "alt_goal_x": f"{ROAD_END_X}", "alt_goal_y": "6.0"},
        ),
        VehicleConfig(
            vehicle_id=2,
            name="lead",
            role="lead",
            start=[44.0, 2.0, 0.0],
            goal=[ROAD_END_X, 2.0, 0.0],
            initial_speed=32.0 / 3.6,
            target_speed=32.0 / 3.6,
            color="r",
            lane_id=0,
            target_lane_id=0,
            cooperation_cost=1.2,
        ),
        VehicleConfig(
            vehicle_id=3,
            name="acceptor_1",
            role="acceptor_1",
            start=[9.0, 6.0, 0.0],
            goal=[ROAD_END_X, 6.0, 0.0],
            initial_speed=28.0 / 3.6,
            target_speed=28.0 / 3.6,
            color="g",
            lane_id=1,
            target_lane_id=2,
            cooperation_cost=0.8,
            fallback_path_key="acceptor_1_alt",
            metadata={"alt_start_x": "9.0", "alt_start_y": "10.0", "alt_goal_x": f"{ROAD_END_X}", "alt_goal_y": "10.0"},
        ),
        VehicleConfig(
            vehicle_id=4,
            name="acceptor_2",
            role="acceptor_2",
            start=[3.5, 10.0, 0.0],
            goal=[ROAD_END_X, 10.0, 0.0],
            initial_speed=24.0 / 3.6,
            target_speed=24.0 / 3.6,
            color="m",
            lane_id=2,
            target_lane_id=2,
            cooperation_cost=0.8,
        ),
        VehicleConfig(
            vehicle_id=5,
            name="non_cooperating_vehicle",
            role="non_cooperating_vehicle",
            start=[1.5, 6.0, 0.0],
            goal=[ROAD_END_X, 6.0, 0.0],
            initial_speed=10.0 / 3.6,
            target_speed=23.5 / 3.6,
            color="c",
            lane_id=1,
            target_lane_id=1,
            cooperation_cost=1.5,
        ),
    ]


def build_default_two_vehicle_coordination_configs() -> List[VehicleConfig]:
    """Configure the direct two-vehicle lane-change request scenario."""
    return [
        VehicleConfig(
            vehicle_id=1,
            name="ego",
            role="ego",
            start=[4.5, 2.0, 0.0],
            goal=[75.0, 2.0, 0.0],
            initial_speed=32.0 / 3.6,
            target_speed=32.0 / 3.6,
            color="b",
            lane_id=0,
            target_lane_id=1,
            cooperation_cost=1.0,
            fallback_path_key="ego_alt",
            metadata={"alt_start_x": "4.5", "alt_start_y": "6.0", "alt_goal_x": "75.0", "alt_goal_y": "6.0", "transition_length": "36.0"},
        ),
        VehicleConfig(
            vehicle_id=2,
            name="lead",
            role="lead",
            start=[24.0, 2.0, 0.0],
            goal=[75.0, 2.0, 0.0],
            initial_speed=20.0 / 3.6,
            target_speed=20.0 / 3.6,
            color="r",
            lane_id=0,
            target_lane_id=0,
            cooperation_cost=1.2,
        ),
        VehicleConfig(
            vehicle_id=3,
            name="acceptor",
            role="acceptor",
            start=[1.5, 6.0, 0.0],
            goal=[75.0, 6.0, 0.0],
            initial_speed=25.0 / 3.6,
            target_speed=25.0 / 3.6,
            color="g",
            lane_id=1,
            target_lane_id=1,
            cooperation_cost=0.8,
        ),
    ]


def build_default_rejected_request_then_free_lane_configs() -> List[VehicleConfig]:
    """Configure the reject-then-ACC-follow-then-free-lane scenario."""
    scenario_road_end_x = 120.0
    requester_goal_x = 113.0
    return [
        VehicleConfig(
            vehicle_id=1,
            name="ego",
            role="ego",
            start=[4.5, 2.0, 0.0],
            goal=[requester_goal_x, 2.0, 0.0],
            initial_speed=32.0 / 3.6,
            target_speed=32.0 / 3.6,
            color="b",
            lane_id=0,
            target_lane_id=1,
            cooperation_cost=1.0,
            fallback_path_key="ego_alt",
            metadata={"alt_start_x": "4.5", "alt_start_y": "6.0", "alt_goal_x": f"{requester_goal_x}", "alt_goal_y": "6.0", "transition_length": "36.0"},
        ),
        VehicleConfig(
            vehicle_id=2,
            name="lead",
            role="lead",
            start=[36.0, 2.0, 0.0],
            goal=[scenario_road_end_x, 2.0, 0.0],
            initial_speed=32.0 / 3.6,
            target_speed=32.0 / 3.6,
            color="r",
            lane_id=0,
            target_lane_id=0,
            cooperation_cost=1.2,
            metadata={"braking_start_time": "2.0", "braking_target_speed": f"{20.0 / 3.6}"},
        ),
        VehicleConfig(
            vehicle_id=3,
            name="acceptor",
            role="acceptor",
            start=[12.0, 6.0, 0.0],
            goal=[scenario_road_end_x, 6.0, 0.0],
            initial_speed=34.0 / 3.6,
            target_speed=34.0 / 3.6,
            color="g",
            lane_id=1,
            target_lane_id=1,
            cooperation_cost=0.8,
        ),
    ]


def build_default_three_vehicle_coordination_configs() -> List[VehicleConfig]:
    """Configure the multi-acceptor scenario with rear/front gap creation."""
    scenario_road_end_x = 95.0
    requester_goal_x = 88.0
    return [
        VehicleConfig(
            vehicle_id=1,
            name="ego",
            role="ego",
            start=[4.5, 2.0, 0.0],
            goal=[requester_goal_x, 2.0, 0.0],
            initial_speed=32.0 / 3.6,
            target_speed=32.0 / 3.6,
            color="b",
            lane_id=0,
            target_lane_id=1,
            cooperation_cost=1.0,
            fallback_path_key="ego_alt",
            metadata={"alt_start_x": "4.5", "alt_start_y": "6.0", "alt_goal_x": f"{requester_goal_x}", "alt_goal_y": "6.0", "transition_length": "36.0"},
        ),
        VehicleConfig(
            vehicle_id=2,
            name="lead",
            role="lead",
            start=[36.0, 2.0, 0.0],
            goal=[scenario_road_end_x, 2.0, 0.0],
            initial_speed=32.0 / 3.6,
            target_speed=32.0 / 3.6,
            color="r",
            lane_id=0,
            target_lane_id=0,
            cooperation_cost=1.3,
            metadata={"braking_start_time": "2.0", "braking_target_speed": f"{20.0 / 3.6}"},
        ),
        VehicleConfig(
            vehicle_id=3,
            name="target_lane_rear",
            role="target_lane_rear",
            start=[0.0, 6.0, 0.0],
            goal=[scenario_road_end_x, 6.0, 0.0],
            initial_speed=23.0 / 3.6,
            target_speed=23.0 / 3.6,
            color="g",
            lane_id=1,
            target_lane_id=1,
            cooperation_cost=0.8,
        ),
        VehicleConfig(
            vehicle_id=4,
            name="target_lane_front",
            role="target_lane_front",
            start=[24.0, 6.0, 0.0],
            goal=[scenario_road_end_x, 6.0, 0.0],
            initial_speed=29.0 / 3.6,
            target_speed=29.0 / 3.6,
            color="m",
            lane_id=1,
            target_lane_id=1,
            cooperation_cost=0.9,
        ),
    ]


def build_default_three_vehicle_coordination_4_messages_configs() -> List[VehicleConfig]:
    """Configure the three-vehicle scenario using offer/confirm/accept messages."""
    scenario_road_end_x = 95.0
    requester_goal_x = 88.0
    return [
        VehicleConfig(
            vehicle_id=1,
            name="ego",
            role="ego",
            start=[4.5, 2.0, 0.0],
            goal=[requester_goal_x, 2.0, 0.0],
            initial_speed=32.0 / 3.6,
            target_speed=32.0 / 3.6,
            color="b",
            lane_id=0,
            target_lane_id=1,
            cooperation_cost=1.0,
            fallback_path_key="ego_alt",
            metadata={"alt_start_x": "4.5", "alt_start_y": "6.0", "alt_goal_x": f"{requester_goal_x}", "alt_goal_y": "6.0", "transition_length": "36.0"},
        ),
        VehicleConfig(
            vehicle_id=2,
            name="lead",
            role="lead",
            start=[44.0, 2.0, 0.0],
            goal=[scenario_road_end_x, 2.0, 0.0],
            initial_speed=32.0 / 3.6,
            target_speed=32.0 / 3.6,
            color="r",
            lane_id=0,
            target_lane_id=0,
            cooperation_cost=1.3,
            metadata={"braking_start_time": "2.0", "braking_target_speed": f"{20.0 / 3.6}"},
        ),
        VehicleConfig(
            vehicle_id=3,
            name="target_lane_rear",
            role="target_lane_rear",
            start=[0.0, 6.0, 0.0],
            goal=[scenario_road_end_x, 6.0, 0.0],
            initial_speed=23.0 / 3.6,
            target_speed=23.0 / 3.6,
            color="g",
            lane_id=1,
            target_lane_id=1,
            cooperation_cost=0.8,
        ),
        VehicleConfig(
            vehicle_id=4,
            name="target_lane_front",
            role="target_lane_front",
            start=[24.0, 6.0, 0.0],
            goal=[scenario_road_end_x, 6.0, 0.0],
            initial_speed=29.0 / 3.6,
            target_speed=29.0 / 3.6,
            color="m",
            lane_id=1,
            target_lane_id=1,
            cooperation_cost=0.9,
        ),
    ]
