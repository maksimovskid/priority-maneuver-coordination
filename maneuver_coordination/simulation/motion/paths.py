"""Path construction helpers for scenario vehicle configurations."""

import math
from typing import Dict, Sequence

import numpy as np

from maneuver_coordination.simulation.motion.reference import build_curvature, plan_path
from maneuver_coordination.simulation.core.settings import ROAD_END_X
from maneuver_coordination.simulation.motion.speed_profiles import calc_speed_profile, calc_speed_profile5
from maneuver_coordination.simulation.core.types import PlannedPath, VehicleConfig


def build_non_cooperating_vehicle_path_specs() -> Dict[str, Sequence[float]]:
    """Return legacy alternate-path specs used by cascading scenarios."""
    return {
        "ego_alt_start": [9.0, 6.0, 0.0],
        "ego_alt_goal": [ROAD_END_X, 6.0, 0.0],
        "acceptor_1_alt_start": [9.0, 10.0, 0.0],
        "acceptor_1_alt_goal": [ROAD_END_X, 10.0, 0.0],
    }


def build_support_path_specs() -> Dict[str, Sequence[float]]:
    """Compatibility alias for the old support-vehicle alternate path specs."""
    return build_non_cooperating_vehicle_path_specs()


def build_two_vehicle_coordination_path_specs() -> Dict[str, Sequence[float]]:
    """Return alternate-path specs for the two-vehicle coordination scenario."""
    return {
        "ego_alt_start": [4.5, 6.0, 0.0],
        "ego_alt_goal": [75.0, 6.0, 0.0],
    }


def build_fallback_paths(
    vehicle_configs: Sequence[VehicleConfig],
    ox: Sequence[float],
    oy: Sequence[float],
) -> Dict[str, PlannedPath]:
    """Create configured alternate paths used as lane-change targets."""
    fallback_paths: Dict[str, PlannedPath] = {}
    for config in vehicle_configs:
        if not config.fallback_path_key:
            continue

        metadata = config.metadata
        required_keys = ("alt_start_x", "alt_start_y", "alt_goal_x", "alt_goal_y")
        if not all(key in metadata for key in required_keys):
            continue

        alt_start = [float(metadata["alt_start_x"]), float(metadata["alt_start_y"]), 0.0]
        alt_goal = [float(metadata["alt_goal_x"]), float(metadata["alt_goal_y"]), 0.0]
        transition_length_value = metadata.get("transition_length")
        if transition_length_value is None:
            fallback_paths[config.fallback_path_key] = _build_planned_path(
                alt_start,
                alt_goal,
                ox,
                oy,
                calc_speed_profile,
                config.target_speed,
            )
            continue

        transition_length = float(transition_length_value)
        start_x = float(config.start[0])
        start_y = float(config.start[1])
        target_y = alt_goal[1]
        target_x = alt_goal[0]

        if abs(target_y - start_y) < 1e-6 or target_x <= start_x + 5.0:
            fallback_paths[config.fallback_path_key] = _build_planned_path(
                alt_start,
                alt_goal,
                ox,
                oy,
                calc_speed_profile,
                config.target_speed,
            )
            continue

        step = 0.1
        cx = list(np.arange(start_x, target_x + step, step))
        cy = []
        cyaw = []
        lane_delta = target_y - start_y

        for x in cx:
            progress = min(1.0, max(0.0, (x - start_x) / transition_length))
            smooth = progress * progress * (3.0 - 2.0 * progress)
            y = start_y + lane_delta * smooth
            cy.append(y)

            if 0.0 < progress < 1.0:
                dy_dx = lane_delta * (6.0 * progress * (1.0 - progress)) / transition_length
            else:
                dy_dx = 0.0
            cyaw.append(math.atan2(dy_dx, 1.0))

        _, _, ck = build_curvature(cx, cy)
        speed_profile = calc_speed_profile(cx, cy, cyaw, config.target_speed)
        fallback_paths[config.fallback_path_key] = PlannedPath(cx, cy, cyaw, ck, alt_goal, speed_profile)

    return fallback_paths


def _build_planned_path(
    start: Sequence[float],
    goal: Sequence[float],
    ox: Sequence[float],
    oy: Sequence[float],
    speed_profile_builder,
    target_speed: float,
) -> PlannedPath:
    path = plan_path(start, goal, ox, oy)
    cx, cy, cyaw = path.xlist, path.ylist, path.yawlist
    _, _, ck = build_curvature(cx, cy)
    speed_profile = speed_profile_builder(cx, cy, cyaw, target_speed)
    return PlannedPath(cx, cy, cyaw, ck, goal, speed_profile)


def build_paths_for_configs(
    vehicle_configs: Sequence[VehicleConfig],
    ox: Sequence[float],
    oy: Sequence[float],
) -> Dict[str, PlannedPath]:
    """Build nominal and fallback paths for a generic scenario config list."""
    paths: Dict[str, PlannedPath] = {}

    for config in vehicle_configs:
        speed_builder = calc_speed_profile5 if config.role == "non_cooperating_vehicle" else calc_speed_profile
        paths[config.role] = _build_planned_path(
            config.start,
            config.goal,
            ox,
            oy,
            speed_builder,
            config.target_speed,
        )

    paths.update(build_fallback_paths(vehicle_configs, ox, oy))

    return paths


def build_paths_for_two_vehicle_coordination_configs(
    vehicle_configs: Sequence[VehicleConfig],
    ox: Sequence[float],
    oy: Sequence[float],
) -> Dict[str, PlannedPath]:
    """Build paths for the two-vehicle runner using the default speed profile."""
    paths: Dict[str, PlannedPath] = {}

    for config in vehicle_configs:
        paths[config.role] = _build_planned_path(
            config.start,
            config.goal,
            ox,
            oy,
            calc_speed_profile,
            config.target_speed,
        )

    paths.update(build_fallback_paths(vehicle_configs, ox, oy))

    return paths
