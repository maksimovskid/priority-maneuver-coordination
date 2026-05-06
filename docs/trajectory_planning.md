# Trajectory Planning And Plotting

This document explains the difference between paths, planned trajectories, and
requested cooperative trajectories. These names are easy to mix up, but they
represent different layers of the simulation.

## Path

A path is the geometric road/lane curve a vehicle can follow.

Examples:

- lane-centered path in the current lane
- alternate/fallback path into a target lane
- generated local lane-change path

Paths are mostly spatial:

- `x`
- `y`
- `yaw`
- `curvature`

Important files:

- `simulation.motion.paths`
- `simulation.motion.reference`
- `motion_planning.hybrid_a_star`
- `coordination.behavior_planner`

## Speed Profile

A speed profile attaches desired speed values to a path.

Examples:

- normal target speed
- ACC-style slower following speed
- cooperative deceleration
- cooperative acceleration

Important files:

- `simulation.motion.speed_profiles`
- `simulation.motion.adaptation`
- `simulation.coordination.roles.build_acc_speed_profile`

## Planned Trajectory

A planned trajectory is the short future horizon sampled from the current path
and speed profile. It answers:

"Where does this vehicle expect to be over the next few seconds if it continues
with its current plan?"

In the current simulation, this follows the paper style of a short horizon:

- simulation step: `0.1 s`
- horizon steps: `20`
- horizon duration: about `2 s`

Planned trajectories are used for:

- conflict checks
- future trajectory plotting
- acceptor evaluation

In the live plot, planned future trajectories are shown as vehicle-colored
points. They should show only the short future horizon, not a long line to the
goal.

## Requested Cooperative Trajectory

A requested cooperative trajectory is the future path the requester wants to
execute as part of a negotiated maneuver.

It is different from the normal planned trajectory because it represents:

- the proposed lane change
- the trajectory sent in the maneuver request
- the trajectory acceptors evaluate for conflicts

In the live plot, requested cooperative trajectories are shown as yellow points.

Important rule:

The yellow requested trajectory should be visible only while the cooperative
request/execution is active. After the agreed cooperative maneuver is completed,
it should disappear and the vehicles should continue in normal operation with
only their normal planned future trajectories.

## Why Both Planned And Requested Trajectories Are Plotted

During negotiation, plotting both is useful:

- vehicle-colored points show what each vehicle currently plans to do
- yellow points show what the requester is asking to do cooperatively

This helps debug whether:

- the requester is asking for the correct lane-change trajectory
- the accepting vehicles are part of the actual conflict
- the future planned trajectories are only a 2-second horizon
- the yellow requested trajectory disappears after execution

## Lane-Change Length

Lane-change smoothness is mostly controlled by the fallback/generated local path
geometry, especially metadata such as `transition_length` in `VehicleConfig`.

Longer transition length means:

- smoother lateral motion
- slower visual lane transition
- less sharp RRT/local-path behavior

Shorter transition length means:

- faster lane change
- tighter lateral motion
- more risk of looking abrupt, even if the controller can track it

For scenario consistency, prefer configuring lane-change geometry through
vehicle metadata before changing controller gains.

## Conflict Checking

Conflict checks compare the requester's requested trajectory with another
vehicle's planned trajectory at representative points across the maneuver.

The current helper computes clearances around:

- beginning of the maneuver
- middle of the maneuver
- end of the maneuver

Then it checks whether those clearances satisfy the required gap. This keeps the
check simple and fast while still catching the important target-lane conflicts in
the current scenarios.

Important files:

- `coordination.conflicts`
- `simulation.coordination.coordination_flow.find_conflicting_vehicle_roles`
- `simulation.motion.adaptation`

## Common Plotting Mistakes

If a plot looks wrong, check:

- Is a full path being plotted instead of the 2-second planned horizon?
- Is a line being drawn where points should be drawn?
- Is the requested trajectory plotted after execution has finished?
- Is the requested trajectory built from the current vehicle position forward,
  or accidentally from a stale/behind path segment?
- Is the vehicle changing lane twice because the post-maneuver target path still
  points back to the original lane?

