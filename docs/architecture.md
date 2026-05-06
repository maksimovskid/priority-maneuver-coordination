# Architecture Overview

## Layers

### `maneuver_coordination.simulation`

Contains the closed-loop simulation mechanics:

- `runner.py`: public facade used by scenarios and older imports
- `visualization.py`: shared live plotting and saved-frame rendering
- `core/`: vehicle state dataclasses, settings, history, and small math helpers
- `motion/`: dynamics, controllers, path/reference generation, speed profiles, and trajectory adaptation
- `coordination/`: runtime role state, events, V2X-style messages, and coordination-flow helpers
- `runners/`: scenario-family runner implementations and default vehicle configurations

The important dependency direction is:

- scenario entry points call the public `simulation.runner` facade
- `simulation.runner` re-exports configs and scenario runner functions from `simulation.runners`
- runner implementations use `core`, `motion`, `coordination`, and `visualization`

This keeps the user-facing API stable while still allowing the implementation to be grouped by responsibility.

### `maneuver_coordination.coordination`

Contains cooperative decision-making logic:

- shared constants
- planner adapter
- conflict-check helpers

### `maneuver_coordination.motion_planning`

Contains the geometric path-planning algorithms used by the maneuver planners:

- `hybrid_a_star.py`: Hybrid A* path search used for lane-change path generation
- `a_star.py`: dynamic-programming/A* helper used internally by Hybrid A*
- `reeds_shepp_path_planning.py`: Reeds-Shepp curve generation
- `rrt.py`: basic RRT implementation
- `rrt_reeds_shepp.py`: RRT with Reeds-Shepp steering, used by the behavior planner
- `cubic_spline_planner.py`: spline interpolation for road/reference paths
- `rrt_1.py` and `rrt_reeds_shepp_demo.py`: small demo/import wrappers kept for experimenting

### `maneuver_coordination.scenarios`

Contains runnable scenario definitions. These files define road geometry, choose vehicle configurations, and call one of the shared simulation runner flows.
