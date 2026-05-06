# Coordination Flow

This document explains the runtime maneuver-coordination flow used by the
current scenarios. It is intentionally higher level than the code docstrings so
future scenarios can be designed without first reading every runner.

## Main Idea

Vehicles normally follow their planned lane-centered trajectories. A vehicle
only becomes a requester at runtime when the traffic situation makes normal lane
following insufficient, for example when a vehicle ahead starts slowing down and
the gap becomes too small.

Runtime labels are behavioral states, not permanent identities:

- `normal_operation`: vehicle follows its planned path.
- `requester`: vehicle is searching for or executing a coordinated maneuver.
- `acceptor`: vehicle is evaluating or supporting another vehicle's request.

The scenario configuration still uses stable internal roles such as `ego`,
`lead`, or `target_lane_rear`, but the visible behavior should come from runtime
state changes.

## Direct Request / Accept Flow

The simplest coordination flow is used by `two_vehicles_coordination`.

1. The requester detects a slowing vehicle ahead.
2. The requester creates a lane-change path and requested trajectory.
3. The requester sends a maneuver request to the relevant target-lane vehicle.
4. The receiver checks whether the requested trajectory conflicts with its own
   planned trajectory.
5. If needed, the receiver evaluates cooperative speed/acceleration candidates.
6. The receiver accepts if a conflict-free adaptation exists within the allowed
   priority limit.
7. The requester executes the lane change.
8. Both vehicles return to normal operation after the cooperative maneuver is
   complete.

Important helpers:

- `simulation.coordination.roles.should_trigger_maneuver_search`
- `coordination.decision_logic.calc_priority`
- `coordination.conflicts.sampled_clearances`
- `simulation.motion.adaptation.build_candidate_refs_for_accelerations`
- `simulation.coordination.messages.emit_request_messages`

## Multi-Acceptor Flow

The `three_vehicles_coordination` scenario uses two target-lane vehicles to
create a safe gap:

- a rear target-lane vehicle can decelerate
- a front target-lane vehicle can accelerate

The requester should only coordinate with vehicles that are actually part of the
conflict. The runner determines relevant front/rear target-lane neighbors at
runtime, checks the requested trajectory against their planned trajectories, and
then selects adaptation candidates.

This scenario is useful for testing whether a lane-change gap can be shaped by a
pair of cooperating vehicles rather than only one acceptor.

## Four-Message Flow

The `three_vehicles_coordination_4_messages` scenario adds a more explicit
message sequence:

1. `request`: requester asks target-lane vehicles for cooperation.
2. `offer`: each feasible acceptor offers an adapted trajectory.
3. `confirm`: requester confirms the combined offer set.
4. `accept`: acceptors commit and the requester executes.

The purpose of the confirm step is to avoid committing individual vehicles to
adaptations before the requester has checked the combined response.

Current implementation note: this is still a simplified message model. Messages
carry sender/receiver IDs, roles, priority labels, and small payload metadata,
but complete paper-level request IDs, cancel, abort, emergency, and unreliable
communication behavior are not fully modeled yet.

## Cascading Flow

The `cascading_coordination` scenario demonstrates a chain:

```text
ID 1 -> ID 3
ID 3 -> ID 4
```

ID 1 first requests cooperation from ID 3. If ID 3 cannot simply adapt within
its current lane without creating another conflict, ID 3 becomes a secondary
requester and asks ID 4 for cooperation. Once the downstream request is accepted,
ID 3 can support ID 1, and ID 1 can execute the original lane change.

This is the most important scenario for understanding why "requester" and
"acceptor" should be runtime states. A vehicle can be an acceptor in one message
stage and a requester in the next.

## Rejection And Fallback Flow

The `rejected_request_then_free_lane` scenario demonstrates a practical fallback:

1. ID 1 detects a need for coordination.
2. ID 1 sends a request.
3. The request is rejected because the target lane is not safely available.
4. ID 1 switches to ACC-style following behind the slowing lead vehicle.
5. When the adjacent lane becomes free, ID 1 changes lane without cooperation.

This scenario is useful for validating safety behavior when coordination fails.
The requester should not force the maneuver through a conflict.

## Priority And Adaptation

Priority is computed from the gap to the lead vehicle:

- low priority: comfortable but relevant need
- medium priority: shrinking gap
- high priority: urgent/safety-critical gap

Cooperative vehicles compare candidate adaptations instead of using a single
hardcoded speed change. Current candidates are evaluated in 5 percent speed
steps, with priority-dependent limits:

- low priority: up to about 20 percent speed adaptation
- medium priority: up to about 40 percent speed adaptation
- high priority: stronger adaptation if needed

Acceleration/deceleration bounds are also priority dependent:

- low priority: about 2.0 m/s^2 maximum adaptation
- medium priority: about 4.0 m/s^2 maximum adaptation
- high priority: stronger adaptation if needed

## What To Watch When Adding Scenarios

When a new scenario behaves strangely, check these points first:

- Did the maneuver trigger too early because the lead vehicle starts braking at
  `t = 0.0 s`?
- Is the target-lane vehicle actually in conflict, or is it accepting even
  though it is irrelevant?
- Does the requester remain in the target lane after the lane change?
- Does the requested yellow trajectory disappear after execution?
- Does the terminal event log tell the same story as the plot?
- Are lane IDs and runtime y-position lane inference consistent?

