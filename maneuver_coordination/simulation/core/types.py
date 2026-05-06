"""Shared data structures passed between simulation, planning, and plotting."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class State:
    """Vehicle state used by dynamics, controllers, and reference generation."""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    a: float = 0.0


@dataclass
class VehicleHistory:
    """Time-series values recorded for one simulated vehicle."""
    x: List[float]
    y: List[float]
    yaw: List[float]
    v: List[float]
    d: List[float]
    t: List[float]
    goal_flag: bool
    target_ind: int


@dataclass
class PlannedPath:
    """Geometric path plus the nominal speed profile attached to it."""
    x: Sequence[float]
    y: Sequence[float]
    yaw: Sequence[float]
    curvature: Sequence[float]
    goal: Sequence[float]
    speed_profile: Sequence[float]


@dataclass
class VehicleConfig:
    """Static scenario setup for one vehicle."""
    vehicle_id: int
    name: str
    role: str
    start: Sequence[float]
    goal: Sequence[float]
    initial_speed: float
    target_speed: float
    color: str
    lane_id: Optional[int] = None
    target_lane_id: Optional[int] = None
    cooperation_cost: float = 1.0
    fallback_path_key: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class V2XMessage:
    """Simplified maneuver-coordination message exchanged during simulation."""
    time_s: float
    sender_id: int
    receiver_id: int
    sender_role: str
    receiver_role: str
    message_type: str
    priority_label: str
    payload: Dict[str, str] = field(default_factory=dict)
