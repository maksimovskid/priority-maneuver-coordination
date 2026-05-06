# Creating New Scenarios

## Purpose

This guide explains how to add new scenarios to the repository without returning to the older pattern of copying full behavioral-planner and coordination scripts.

The main idea is:

- reuse the shared planner and runner logic
- express most new scenarios as configuration, geometry, and timing differences
- only add new coordination logic when the behavior truly cannot be represented by the existing scenario flows

## Relation To The PriMa Paper

The current repository is based on the same general PriMa coordination idea described in the ITSC 2021 paper:

- a vehicle detects that normal lane following is no longer sufficient
- it requests a coordinated maneuver
- relevant vehicles in the target lane cooperate by adapting their motion
- a safe gap is created for the requesting vehicle

Using the paper terminology, the important concepts are:

- `PT` = planned trajectory, continuously broadcast and updated
- `RT` = requested trajectory, sent when a maneuver is negotiated
- offer-style response trajectories in multi-vehicle coordination
- priority levels `low`, `medium`, and `high`

The paper assumes:

- decentralized V2V coordination
- a message/update rate of `10 Hz`
- a short prediction horizon of about `2 s`
- conflict checks against surrounding vehicles before maneuver execution

Those assumptions match the style of the current simulation closely:

- the simulation step is `0.1 s`
- vehicles negotiate based on requested future trajectories
- acceptance depends on conflict-free adaptation
- priority is computed from the traffic situation ahead

The current implementation already covers the main paper coordination structures as implementation examples:

- direct coordination in `two_vehicles_coordination`
- multi-vehicle gap creation in `three_vehicles_coordination`
- request / offer / confirm / accept flow in `three_vehicles_coordination_4_messages`
- cascading coordination in `cascading_coordination`

The current implementation also reflects the paper's requester / acceptor / cascading-role idea:

- requester vehicle: wants to execute the maneuver
- accepting vehicle(s): adapt by decelerating, accelerating, or rejecting
- cascading vehicle: first accepts, then becomes a requester downstream

At the same time, the codebase also contains practical engineering extensions that go beyond a minimal paper reproduction, for example:

- explicit runtime vehicle IDs
- dynamic requester / acceptor display states
- simple V2X-style inbox/message handling
- Docker support
- headless saved-output mode
- rejection-and-fallback behavior in `rejected_request_then_free_lane`

And there are still paper-level details that are only partially implemented or not yet represented end to end, for example:

- explicit request IDs across every message step
- `cancel`
- `abort`
- `emergency`
- unreliable communication behavior beyond the current simplified message delivery model

For conceptual understanding and easier further development, it is strongly recommended to read the paper together with this code. The paper should be used as the conceptual reference and as an example of what can be implemented with PriMa-style coordination, not as the claim that every scenario in this repository is an exact paper reproduction. In practice, the best approach is:

1. read the paper flow first
2. identify which current scenario is closest
3. implement the new case by reusing the existing runner structure

## Design Principle

When developing a new scenario, first ask:

- is this mostly a different traffic setup?
- or does it really require new coordination logic?

In many cases, a new scenario only needs changes to:

- vehicle starts and goals
- target lane assignments
- vehicle speeds
- braking start time
- braking target speed
- road length
- obstacle boundaries
- fallback lane-change geometry
- whether the message exchange is direct, offer/confirm, cascading, or rejection-based

Those differences should usually be implemented as scenario data, not as a new copied planner.

## Which Existing Scenario To Start From

Use the closest current scenario as your base.

### `two_vehicles_coordination`

Use this when you need:

- one requester
- one relevant cooperating vehicle in the target lane
- a simpler direct coordination story

### `three_vehicles_coordination`

Use this when you need:

- one requester
- one rear target-lane vehicle creating space by slowing down
- one front target-lane vehicle creating space by speeding up

This is the best base for "safe gap created by two cooperating vehicles".

### `three_vehicles_coordination_4_messages`

Use this when the explicit message chain matters:

- request
- offer
- confirm
- execute

### `cascading_coordination`

Use this when:

- one vehicle first cooperates
- then that vehicle becomes a requester itself
- downstream coordination is part of the scenario logic

### `rejected_request_then_free_lane`

Use this when:

- a request is rejected
- the requester falls back to ACC-style following
- the lane change only happens later when the adjacent lane becomes free

## Files You Usually Need To Touch

For a typical new scenario, the expected files are:

- `maneuver_coordination/scenarios/<new_scenario>.py`
- `maneuver_coordination/simulation/runners/scenario_configs.py`
- one existing runner in `maneuver_coordination/simulation/runners/`
- `maneuver_coordination/cli.py`
- `tests/test_smoke.py`
- `README.md` if the scenario should be publicly documented

Usually you should not edit `maneuver_coordination/simulation/runner.py` directly. It is kept as the public facade so existing scenario scripts and tests can import from one stable place.

## Step-By-Step Workflow

### 1. Define The Road Geometry

In the new scenario file, add a `build_obstacles()` function.

That function should define:

- outer road boundaries
- lane divider lines
- obstacle list for planning

Keep this pattern close to the existing scenario files so the geometry is easy to read.

### 2. Add Vehicle Configurations

Add or reuse a config builder in `maneuver_coordination/simulation/runners/scenario_configs.py`, usually named like:

- `build_default_<scenario_name>_configs()`

Each vehicle should define:

- `vehicle_id`
- `name`
- `role`
- `start`
- `goal`
- `initial_speed`
- `target_speed`
- `color`
- `lane_id`
- `target_lane_id`
- `cooperation_cost`

Optional metadata can hold scenario-specific tuning such as:

- `braking_start_time`
- `braking_target_speed`
- `fallback_path_key`
- `alt_start_x`
- `alt_start_y`
- `alt_goal_x`
- `alt_goal_y`
- `transition_length`

These metadata fields are a good place to represent paper-style scenario assumptions without changing the planner itself, for example:

- when a leading vehicle starts decelerating
- how strongly a cooperating vehicle adapts
- how long the negotiated lane-change path should be

### 3. Reuse The Closest Runner Flow

Prefer reusing an existing runner flow rather than adding a new monolithic one.

Current useful runner bases are:

- `run_two_vehicle_coordination_scenario(...)`
- `run_multi_acceptor_three_vehicle_coordination_scenario(...)`
- `run_cascading_scenario(...)`
- `run_rejected_request_then_free_lane_scenario(...)`

Their implementations live in:

- `maneuver_coordination/simulation/runners/direct_runners.py`
- `maneuver_coordination/simulation/runners/multi_acceptor_runner.py`
- `maneuver_coordination/simulation/runners/cascading_runner.py`

If your scenario differs only in:

- config
- road geometry
- timing
- message flavor

then you probably do not need a new planner implementation.

This is especially important for PriMa-style work. Most new use cases should first be treated as:

- different blocking-vehicle IDs
- different requester / acceptor arrangement
- different priority timing
- different front/rear target-lane behavior

and only later as new code, if the existing runners truly cannot express the coordination story.

### 4. Add A Scenario Entry Point

Create a new file in `maneuver_coordination/scenarios/` using the same structure as the current scenario files:

- `build_obstacles()`
- `main(show_animation=True, save_output_dir=None)`

That keeps local runs, Docker runs, and saved-output runs consistent.

Import shared runner functions and config builders from `maneuver_coordination.simulation.runner` unless you are actively changing runner internals. This keeps scenario files independent from the internal folder layout.

### 5. Register It In The CLI

Update `maneuver_coordination/cli.py`:

- add the scenario name constant
- add it to parser choices
- route it to the new scenario `main()`

### 6. Add A Regression Test

Add at least one scenario-level test in `tests/test_smoke.py`.

A good scenario regression test should check the main intended story, for example:

- which request messages happen
- which vehicle accepts or rejects
- whether the ego reaches the target lane
- whether fallback occurs
- whether cascading actually happens
- whether the intended rear/front target-lane adaptations match the scenario design

That protects the scenario when shared coordination code evolves.

## What To Avoid

Try to avoid:

- copying the old full behavioral planner into a new file
- copying the old full coordination script into a new file
- adding vehicle-number-specific names like `vehicle_3_logic` or `states_4`
- hardcoding scenario behavior into the planner when it only belongs in scenario configuration

The current repo structure is specifically intended to reduce that kind of duplication.

## When New Logic Is Actually Justified

A new helper or runner path is justified when the scenario introduces a genuinely new behavior pattern, for example:

- a new acceptance policy
- a new message phase
- a new fallback mode
- a new multi-stage coordination structure
- a new paper-level message type that is not currently modeled, such as cancel, abort, or emergency

Even then, prefer adding:

- one reusable helper
- one runner extension

instead of creating another full copied scenario stack.

## Practical Checklist

Before considering a scenario complete, check:

- the ego behaves as intended
- cooperating vehicles adapt in the intended way
- the vehicle stays in the correct lane after the maneuver
- the request / offer / confirm / accept story matches the intended coordination style
- the request trajectory is plotted correctly if that scenario uses it
- terminal events tell the correct story
- the scenario runs headlessly
- a regression test exists
- README is updated if the scenario is part of the public repo interface

## Minimal Scenario Template

Most new scenario entry-point files should stay small. A good target shape is:

```python
"""Short explanation of what the scenario demonstrates."""

import matplotlib.pyplot as plt

from maneuver_coordination.scenarios.output import save_scenario_animation, save_summary_plots
from maneuver_coordination.simulation.runner import (
    build_default_<scenario>_configs,
    run_<scenario_family>_scenario,
)


def build_obstacles():
    """Build road boundaries, lane dividers, and planner obstacles."""
    road_end_x = 100
    ox, oy = [], []
    # Add road boundaries and lane divider points here.
    obstacle_list = []
    return ox, oy, obstacle_list


def main(*, show_animation=True, save_output_dir=None, save_animation=False):
    """Set up the scenario, run the shared runner, and save/plot results."""
    vehicle_configs = build_default_<scenario>_configs()
    ox, oy, obstacle_list = build_obstacles()

    result = run_<scenario_family>_scenario(
        vehicle_configs=vehicle_configs,
        ox=ox,
        oy=oy,
        obstacle_list=obstacle_list,
        show_animation=show_animation,
    )

    if save_output_dir:
        save_summary_plots(...)
        if save_animation:
            save_scenario_animation(...)
```

The important point is that scenario files should describe the traffic scene.
They should not contain a copied behavior planner, copied message logic, or
vehicle-ID-specific control code.

## Where To Put Scenario-Specific Values

Prefer this split:

- Road length, lane lines, and obstacle boundaries belong in `build_obstacles()`.
- Vehicle start/goal/speed/lane/cost values belong in `scenario_configs.py`.
- Trigger timing such as `braking_start_time` belongs in `VehicleConfig.metadata`.
- Lane-change geometry such as `alt_start_y`, `alt_goal_y`, and `transition_length` belongs in `VehicleConfig.metadata`.
- Shared message and acceptance behavior belongs in `simulation/coordination/`.
- Shared reference, speed, and path generation belongs in `simulation/motion/`.

If a new scenario needs a magic number inside a runner, pause and ask whether it
could instead become a metadata field on the relevant vehicle.

## Recommended Development Pattern

The cleanest pattern for future work is:

1. choose the closest existing scenario
2. create new obstacle geometry
3. create new vehicle configs
4. reuse the existing runner path
5. add only the smallest missing helper if needed
6. add a regression test

That is the main path that keeps this repository maintainable while new scenarios are added.
