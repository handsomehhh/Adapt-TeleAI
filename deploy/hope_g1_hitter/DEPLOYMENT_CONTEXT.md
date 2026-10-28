# Hitter 9500 Deployment Context for lxz@192.168.152.200

**Date**: 2024-10-28  
**Model**: `deploy/hope_g1_hitter/artifacts/hitter_20261027_9500/model_9500.pt`  
**Task**: Deploy 104-D continuous hitter policy on Unitree G1, backhand only

---

## 1. What is this checkpoint

This is a **new ping-pong hitting policy** trained in mjlab with:
- **Actor input**: 104 dimensions (no reference motion tracking)
- **Actor output**: 29 joint targets
- **Training task**: `unitree_g1_hitter_pingpong_mjlab` 
- **Control mode**: Direct target-driven, not motion-library-based
- **Sides**: Trained on both forehand/backhand; deployment uses **backhand only** (side=1)

### Key differences from existing hope_g1 (model_18000.pt, 175-D):

| Aspect | Original HOPE (175-D) | New Hitter (104-D) |
|--------|----------------------|-------------------|
| Reference motion | 22 NPZ motions, frame-by-frame tracking | No reference tracking; motions only for metadata |
| Observation | 175-D with reference joints/anchor/velocity | 104-D: state + target + time-to-strike |
| Motion selection | Choose from 22 motions per strike | Fixed backhand geometry box |
| Swing clock | Frame index progression | Deadline countdown (windup → strike → follow-through) |
| Recovery | Separate RECOVER state with home motion | Direct return to idle target after swing |
| Planner mode | Supports both embedded and ROS external | **ROS external only** (no embedded planner) |

---

## 2. Deployment architecture

### Control flow:
```
Motion capture (/poses)
    ↓
External ROS planner (hope_planner_optimized_2_A30808)
    ↓
/racket/command (RacketCommand message)
    ↓
sim-bridge or deploy mocap → UDP transport
    ↓
HitterController.update_external_planner_command()
    ↓
HitterController.select() → validate target in training box
    ↓
HitterController.step() → HitterActor (104-D observation)
    ↓
29 joint targets → shared PD control → DDS → G1 motors
```

### Shared components (same as hope_g1):
- `deploy/backends/unitree.py` — main real-robot control loop
- AMP stable policy (ONNX) for START/idle
- Remote controller state machine (START → AMP → policy)
- DDS SDK2 low-level communication
- Mocap integration via ROS2 bridge
- Safety: tilt limits, timeout, damping

### New components:
- `deploy/hope_g1_hitter/controller.py` — HitterController
- `deploy/hope_g1_hitter/deploy_agent.py` — entry point
- `deploy/hope_g1_hitter/artifacts/hitter_20261027_9500/` — model + contract
- `deploy/hope_g1_hitter/configs/` — rally.yaml, real.yaml, hope_planner_ros2.yaml

---

## 3. Coordinate system: hope_table

**All deployment uses `hope_table` world frame:**
- Origin: table center (net position)
- X axis: perpendicular to net; **negative toward robot**
- Y axis: parallel to net, along table width
- Z axis: vertical; **Z=0 is tabletop surface**

**Key positions:**
- Robot station (pelvis): `[-0.65, -0.7625]` m
- Backhand hit plane: `X = -0.256` m
- Forehand hit plane: `X = -0.182` m (not used in current deployment)
- Table near edge: `X ≈ 0.0`
- Table far edge: `X ≈ 2.74`

**Training and deployment use identical coordinates.** No coordinate transformation needed between planner output and controller input.

---

## 4. Recent fixes (applied in test/HOPE_G1_new branch)

Four critical bugs were fixed on 2024-10-28:

### Fix 1: PlannerCommand does not exist
**Problem**: `controller.py:257` imported non-existent `PlannerCommand` from `hope_planner_adapter`.  
**Fix**: Use `SimpleNamespace(position, velocity, deadline)` instead.  
**Impact**: Command reception would crash on first valid planner message.

### Fix 2: Strike deadline repeatedly delayed
**Problem**: `select()` recalculated `deadline = now + time_to_strike` every frame.  
**Fix**: Compute `external_planner_deadline` once in `update_external_planner_command()`.  
**Impact**: Robot would never swing; deadline kept moving into the future.

### Fix 3: Backhand config reset to forehand
**Problem**: `reset_state()` set `self.side = 0`, overriding `__init__` backhand setting.  
**Fix**: Remove `self.side` assignment from `reset_state()`.  
**Impact**: Deployment configured for backhand would execute forehand geometry.

### Fix 4: Planner hit plane mismatch
**Problem**: `hope_planner_ros2.yaml` had `x_hit: 0.0` (table edge), not `-0.256` (backhand plane).  
**Fix**: Set `x_hit: -0.256` to match training contract.  
**Impact**: Planner would publish targets 25 cm forward of trained range.

**All fixes verified with `--env check` on local machine.**

---

## 5. Required planner configuration on lxz200

### File to modify:
`/home/lxz/HOPE/hope_ws/src/hope_planner_optimized_2_A30808/config/hope_planner.yaml`

### Change:
```yaml
x_hit: -0.256  # was 0.0; backhand hit plane for hitter
```

### Other critical parameters (verify, do not change unless needed):
```yaml
publish_intercept_z_min: -0.2  # hitter strikes near tabletop
publish_intercept_z_max: 0.2
publish_intercept_y_min: -1.54
publish_intercept_y_max: 0.0
```

**After editing, rebuild and restart the planner node:**
```bash
cd ~/HOPE/hope_ws
colcon build --packages-select hope_planner_optimized_2
source install/setup.bash
# Then restart your planner launch (e.g., via sim_planner.zsh or deployment launch script)
```

---

## 6. Deployment checklist on lxz200

### Pre-flight (no motors):
1. Pull latest `test/HOPE_G1_new` branch from Adapt-TeleAI
2. Verify model artifacts exist: `deploy/hope_g1_hitter/artifacts/hitter_20261027_9500/`
3. Install dependencies: `uv sync` or check `.venv` has mjlab, tensordict, SDK2
4. Run preflight check:
   ```bash
   cd ~/Adapt-TeleAI  # or /home/lxz/deploy_g1_pingpong
   uv run python -m deploy.hope_g1_hitter.deploy_agent --env check
   ```
   Expected output: contract validation, actor load, geometry summary

5. Update planner config (see section 5)
6. Start planner node with updated `x_hit = -0.256`
7. Start mocap bridge (if not already running)
8. Run read-only real mode (no motor enable):
   ```bash
   uv run python -m deploy.hope_g1_hitter.deploy_agent --env real --duration 30
   ```
   **Do NOT use `--enable-motors` yet.** This subscribes to DDS/mocap but sends zero torque.

9. Verify in logs:
   - SDK2 imports successfully
   - DDS LowState received (joint feedback)
   - Mocap `/poses` → planner → `/racket/command` → UDP → controller chain works
   - No `ImportError`, `KeyError`, or `RuntimeError`

### Motor-enabled testing (with extreme caution):
10. Follow existing G1 safety procedures (e-stop ready, clear space, etc.)
11. Use remote controller to: START → AMP → UP (enter hitter policy)
12. Feed slow, predictable balls within training range
13. Monitor joint targets, actual motion, tilt, and fallback triggers
14. Record logs with `--output` for offline analysis

### Known limitations before first motor test:
- **No sim2sim validation yet** (MuJoCo loop not tested with external planner)
- **No trajectory verification** (position/velocity/deadline fields not logged in actual receipt)
- **Forehand disabled** (only backhand active; forehand plane exists but unused)
- **No force/torque clipping** beyond existing G1 limits (no hitter-specific soft limits added)

---

## 7. Observation contract (104-D input to actor)

The actor expects exactly 104 floats in this order:

| Index | Content | Size |
|-------|---------|------|
| 0-2 | Body angular velocity (gyro) | 3 |
| 3-31 | Current joint positions (q - default_q) | 29 |
| 32-60 | Current joint velocities (dq) | 29 |
| 61-89 | Previous actor action | 29 |
| 90-92 | Gravity projection (IMU) | 3 |
| 93-94 | Body forward direction XY (world frame) | 2 |
| 95-96 | Base target XY offset (station - current base) | 2 |
| 97-99 | Racket target position (relative to station) | 3 |
| 100-102 | Racket target velocity | 3 |
| 103 | Time to strike (seconds, can be negative post-strike) | 1 |

**No normalization.** All values are raw physical units (rad, rad/s, m, m/s, s).

See `controller.py:observation()` for implementation.

---

## 8. Important file paths

### On lxz200 after pull:
```
~/Adapt-TeleAI/  (or /home/lxz/deploy_g1_pingpong/)
├── deploy/hope_g1_hitter/
│   ├── deploy_agent.py          # Entry point
│   ├── controller.py            # HitterController
│   ├── artifacts/hitter_20261027_9500/
│   │   ├── model_9500.pt        # 50 MB checkpoint
│   │   ├── contract.json        # Training metadata
│   │   ├── manifest.json        # SHA256 checksums
│   │   └── motions/*.npz        # Reference motions (metadata only)
│   └── configs/
│       ├── real.yaml            # Robot/SDK config
│       ├── rally.yaml           # Ball/world/planner transport config
│       └── hope_planner_ros2.yaml  # Planner parameters (for reference)
└── deploy/tools/
    └── live_viewer.py           # Real-time MuJoCo visualization (optional)
```

### Planner on lxz200:
```
~/HOPE/hope_ws/src/hope_planner_optimized_2_A30808/
├── config/hope_planner.yaml     # ← EDIT x_hit HERE
└── hope_planner_optimized_2/
    └── (Python source)
```

---

## 9. Debugging tips

### If robot doesn't swing:
- Check planner publishes `/racket/command` with `valid: true`
- Check `external_planner_deadline` is set (add log in `update_external_planner_command`)
- Check `geometry.evaluate()` doesn't reject target (see diagnostic in logs)
- Verify `time_to_strike` in planner output is reasonable (0.3-1.0 s)

### If swing happens at wrong location:
- Verify planner `x_hit = -0.256` (not 0.0)
- Check target position in logs (should be near `[-0.256, Y, Z]`)
- Verify coordinate frame in `/racket/command` header is `hope_table` or `world`

### If robot falls or oscillates:
- Check tilt monitor (logged as `tilt_degrees`)
- Verify `default_q` matches contract (look for large `q - default_q` in obs)
- Check PD gains match training (should auto-validate, but worth verifying)

### If "no valid command" persists:
- Check planner node is running and publishing
- Check mocap bridge forwards `/racket/command` → UDP → controller
- Add print in `update_external_planner_command()` to see reception

---

## 10. Rollback plan

If hitter deployment fails and you need to revert to stable HOPE policy:

1. Switch entry point:
   ```bash
   python -m deploy.hope_g1.deploy_agent --env real  # original 175-D
   ```
2. Restore original planner config:
   ```bash
   cd ~/HOPE/hope_ws/src/hope_planner_optimized_2_A30808/config
   git checkout hope_planner.yaml  # restore x_hit=0.0
   colcon build --packages-select hope_planner_optimized_2
   ```
3. Restart planner node with original config

**Original deployment is unaffected** — it uses separate entry point and controller.

---

## 11. Success criteria for initial test

**Minimal success** (motors enabled, first 10 strikes):
- Robot enters hitter policy without crash
- Accepts at least one valid planner command
- Executes swing motion toward target
- Returns to idle after follow-through
- No tilt/torque/timeout faults

**Good success** (30+ minutes):
- Consistent swings on backhand balls within training range
- Low hit rate acceptable (target is motion execution, not ball contact)
- Smooth transition between idle and swing
- Logs show reasonable observation/action values

**Full success** (metrics comparable to sim2sim):
- Similar swing frequency to MuJoCo (limited by ball feeding)
- Joint trajectories roughly match MuJoCo recorded logs
- No drift in standing position over extended rally

---

## 12. Contact and next steps

**This deployment**:
- Authored by Claude (Opus 5.5) assisting yiranwang
- Tested locally: `--env check` passed; `--env mujoco` partial (no external planner)
- **Not tested**: `--env real`, motor-enabled execution, external planner integration

**After initial motor test**:
- Record full deployment log with `--output`
- Share log for offline analysis (view with `deploy/tools/view_real_log.py`)
- If forehand needed, add second planner instance or modify geometry selection
- If embedded planner preferred, restore `HopePlannerAdapter` in controller

**Known missing**:
- Torque soft limits (training used ±5 Nm actor, ±25 Nm simulation; real has only SDK hard limits)
- Multi-strike endurance testing
- Forehand deployment path
- Workflow integration with existing HOPE runbook

---

**End of context document. Good luck with deployment!**
