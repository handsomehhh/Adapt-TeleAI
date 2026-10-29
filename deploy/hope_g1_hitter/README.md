# HOPE G1 hitter deployment

The 104-D actor consumes a world-frame racket position, velocity and strike
deadline. Real execution uses the external ROS planner and the existing
`deploy.io.mocap` ROS-to-UDP bridge. Run commands from the repository root.

## Forehand/backhand selection

`configs/rally.yaml` enables `hitter.side_selection: auto`. At the beginning of
a swing the controller checks both sides' position boxes, station bounds and
timing window, then chooses the reachable side requiring the least lateral
travel from the **measured pelvis Y**. Ties keep the previous side. The selected
side remains locked through follow-through; target updates still use that side.
The actor uses the chosen side's velocity bounds and swing timing. Its input
layout and checkpoint are unchanged; there is no extra side input to the actor.

Override with `--side-selection auto|forehand|backhand`. Auto starts from the
backhand idle posture. Returning to AMP and re-entering the policy resets this
posture while preserving the first-UP return anchor.

## Return to the first-UP anchor

The first accepted UP press captures the measured base position in `hope_table`,
before AMP upper-body prepositioning. The anchor stays fixed for the controller's
lifetime, including DOWN → UP cycles; restart deployment to capture a new one.
Entry without AMP captures at policy entry. Direct offline controller callers
fall back to the first policy observation.

Once follow-through finishes, the controller stays `ARMED` and sets:

- Base target XY to the captured anchor XY, continuously until the next shot.
- Waiting racket target XY to anchor XY plus the selected side's position-box
  centre offset. Racket Z stays relative to the calibrated floor/table; captured
  base Z is diagnostic only because the actor has no independent base Z command.
- Waiting racket velocity to that side's velocity-box centre, and TTS to its
  fixed windup time, as before. Action history is retained.

The visible phase is `RETURN` outside `hitter.return_tolerance_m` (default 0.05 m)
and `IDLE` inside it. The target remains the anchor even after arrival; it is
never replaced by the latest measured position. There is no return timer or
blocking recovery phase. New commands are evaluated every policy tick, can
select either side and immediately interrupt the return. New-shot lateral
reach is checked from measured Y, not from the not-yet-reached anchor.

Planner positions/deadlines, the calibrated strike station X and the global
strike workspace are unchanged. The return anchor does not translate incoming
world-frame ball targets or move the external planner's hit plane.

Logs include `return_anchor_captured`, `return_to_anchor_started`,
`return_interrupted_by_command` and `return_anchor_reached` events; policy rows
contain `return_anchor_world`, `return_anchor_error_m`, `returning_to_anchor`,
`goal_base_xy` and `goal_position`. `completed_swings` counts ended swings;
`recovered_swings` now counts completed returns within the XY tolerance, rather
than simply duplicating the swing count. Interrupted returns are not counted.

Both sides need a physically consistent planner target. With station X=-0.65,
the current contract gives these world X ranges:

| Side | World X range (m) |
| --- | --- |
| Forehand | [-0.222358, -0.142358] |
| Backhand | [-0.295784, -0.215784] |
| Intersection | [-0.222358, -0.215784] |

The supplied ROS config uses **X=-0.219 m**, inside this intersection. The
controller preserves the incoming position and deadline; it does not relabel a
prediction at one plane as a prediction at another. X outside a side's box is
rejected as `x_outside_training_box`. In particular, the former X=-0.256 plane
only permits backhand. Recalculate the intersection if station X or contract X
ranges change.

Each side keeps its own contract Z range. The current forehand and backhand
ranges are both **[-0.10, 0.50] m above the tabletop** (`[0.66, 1.26]` in the
contract), aligned at the operator's request. These expanded acceptance bounds
do not change the trained actor. Planner publication bounds are a separate gate
and do not override either side's controller bounds.

## Start with the existing ROS interface workspace

Keep VRPN, `vrpn_to_pose` and the ROS-to-UDP bridge running as described in
`../hope_g1/REAL_DEPLOYMENT_QUICKSTART.md`. Stop the previous planner, then use
the existing installed planner package:

```zsh
ros2 run hope_planner_optimized_2 hope_planner_optimized_2_node \
  --ros-args -r __node:=hope_planner_optimized_2 \
  --params-file /home/lxz/Adapt-Deploy/deploy/hope_g1_hitter/configs/hope_planner_ros2.yaml
```

The startup output must show `x=[-0.219,-0.219]`. Remove previous command-line
overrides setting `x_hit` or publication X bounds to `-0.256`. The deploy-owned
package can alternatively run the same YAML with
`ros2 run hope_g1_deploy_planner hope_g1_deploy_planner_node --ros-args --params-file ...`.

```zsh
.venv/bin/python -m deploy.hope_g1_hitter.deploy_agent --env check
.venv/bin/python -m deploy.hope_g1_hitter.deploy_agent \
  --env real --network-interface eno1 --side-selection auto \
  --duration 600 --enable-motors
```

Remote sequence: START → POSE_HOLD → A → STABLE → UP. Logs record the selected
`side`, both candidate decisions (including `selected` and `lateral_travel_m`),
and the resolved selection mode in `rally_config.yaml` / the summary.

## Offline checks

```zsh
.venv/bin/python -m pytest deploy/validation/hitter_planner_checks.py -q
```

These cover command decoding, both sides' real actor inference, measured-position
selection, side locking, fixed-side overrides, X-plane validation, anchor capture,
return observations and command preemption during return. They do
not establish real-robot balance or hitting accuracy at the new common plane.
