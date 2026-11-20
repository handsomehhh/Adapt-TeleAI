"""Zero-command AMP standing, adapted from 225 mjlab-prior/deploy_lhy.

The ONNX, fixed depth input, SDK joint order, gains and observation contract
are preserved. No camera or walking command is needed for this staging policy.
"""

import json
from pathlib import Path

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.core import rotation
from deploy.motion_20261119.reference.g1_pingpong.pd_gains import deployment_gains


class StableAMP:
  def __init__(self, joint_names, sdk_names, asset_dir):
    import onnxruntime as ort

    asset_dir = Path(asset_dir)
    metadata = json.loads((asset_dir / "metadata.json").read_text())
    self.metadata = metadata
    self.to_sdk = np.array([joint_names.index(n) for n in sdk_names])
    self.to_policy = np.array([sdk_names.index(n) for n in joint_names])
    self.default = np.asarray(metadata["default_dof_pos"], dtype=np.float32)
    self.used = np.asarray(metadata["used_joint_idx"])
    self.default_q = self.default[self.to_policy].copy()
    self.kp, self.kd = deployment_gains(joint_names, profile="stable")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(
      str(asset_dir / "amp_slope_walk_stand.onnx"),
      options,
      providers=["CPUExecutionProvider"],
    )
    self.inputs = [v.name for v in self.session.get_inputs()]
    if len(self.inputs) != 3:
      raise ValueError("AMP requires current observation, history and fixed depth")
    depth = np.load(asset_dir / "depth_imgs_clean.npy")[0]
    self.depth = np.stack([depth, depth])[None].astype(np.float32)
    self.reset()
    # Warm inference before creating a robot command channel; reset history.
    self.step(self.default_q, np.zeros(29), np.zeros(3), np.array([1.0, 0, 0, 0]))
    self.reset()

  def reset(self):
    self.action = np.zeros(29, dtype=np.float32)
    self.history = np.zeros((1, 10, 57), dtype=np.float32)
    self.reset_id = getattr(self, "reset_id", 0) + 1
    self.step_index = 0
    self.debug = {}

  def step(self, q, dq, gyro, imu_quat):
    q, dq = np.asarray(q)[self.to_sdk], np.asarray(dq)[self.to_sdk]
    obs = np.zeros(57, dtype=np.float32)
    # command [vx, vy, wz] stays exactly zero, irrespective of joystick axes.
    obs[3:6] = np.asarray(gyro) * 0.25
    obs[6:9] = rotation(imu_quat).T @ [0.0, 0.0, -1.0]
    obs[9:25] = (q - self.default)[self.used]
    obs[25:41] = dq[self.used] * 0.05
    obs[41:57] = self.action[self.used]
    if not np.isfinite(obs).all():
      raise ValueError("Nonfinite AMP observation")
    self.history = np.concatenate([self.history[:, 1:], obs[None, None]], axis=1)
    action = np.asarray(
      self.session.run(
        None,
        dict(
          zip(
            self.inputs,
            [obs[None], self.history.reshape(1, 570), self.depth],
            strict=True,
          )
        ),
      )[0]
    ).reshape(-1)
    if action.shape != (16,) or not np.isfinite(action).all():
      raise ValueError("Invalid AMP output")
    self.action[self.used] = action
    target = (self.default + 0.25 * self.action)[self.to_policy].copy()
    self.debug = dict(
      observation=obs.copy(),
      action=action.copy(),
      target_q=target.copy(),
      reset_id=self.reset_id,
      step_index=self.step_index,
    )
    self.step_index += 1
    return target


class HomePosture:
  """Prepare the trained upper-body pose while AMP keeps leg balance."""

  def __init__(self, names, start, home, now, seconds, hold_seconds, pose_error):
    legs = (
      "left_hip",
      "right_hip",
      "left_knee",
      "right_knee",
      "left_ankle",
      "right_ankle",
    )
    self.upper = np.array([i for i, n in enumerate(names) if not n.startswith(legs)])
    self.start, self.home = np.asarray(start).copy(), np.asarray(home).copy()
    self.started, self.seconds = now, seconds
    self.hold_seconds, self.pose_error = hold_seconds, pose_error
    if min(seconds, hold_seconds, pose_error) <= 0 or not len(self.upper):
      raise ValueError("Invalid home-posture preparation")
    self.stable_since = None
    self.debug = {}

  def command(self, amp_target, now):
    alpha = np.clip((now - self.started) / self.seconds, 0, 1)
    weight = alpha * alpha * (3 - 2 * alpha)
    result = np.asarray(amp_target, dtype=float).copy()
    result[self.upper] = ((1 - weight) * self.start + weight * self.home)[self.upper]
    return result

  def ready(self, q, dq, gyro, now, max_velocity, max_gyro, *, measured=True):
    error = float(np.max(np.abs(np.asarray(q)[self.upper] - self.home[self.upper])))
    stable = (
      now - self.started >= self.seconds
      and error <= self.pose_error
      and np.max(np.abs(dq)) <= max_velocity
      and np.max(np.abs(gyro)) <= max_gyro
    )
    self.stable_since = (
      (now if self.stable_since is None else self.stable_since) if stable else None
    )
    self.debug = dict(
      upper_pose_error=error,
      stable=bool(stable),
      stable_seconds=now - self.stable_since if self.stable_since is not None else 0.0,
      readiness_mode="measured" if measured else "timed",
      elapsed_seconds=now - self.started,
    )
    if not measured:
      return now - self.started >= self.seconds + self.hold_seconds
    return stable and now - self.stable_since >= self.hold_seconds


class SmoothHandoff:
  """Blend PD effort via gain-weighted targets, then apply the handoff bound."""

  def __init__(self, command, now, seconds, dt, max_velocity, timeout=5.0):
    if min(seconds, dt, max_velocity) <= 0 or (timeout is not None and timeout <= 0):
      raise ValueError("Handoff durations and velocity must be positive")
    self.source = tuple(np.asarray(x).copy() for x in command)
    self.previous = self.source[0].copy()
    self.started, self.seconds = now, seconds
    self.step_limit, self.timeout = dt * max_velocity, timeout
    self.alpha = 0.0

  def apply(self, command, now, *, source=None):
    elapsed = now - self.started
    self.alpha = float(np.clip(elapsed / self.seconds, 0, 1))
    blend = self.alpha**2 * (3 - 2 * self.alpha)
    source_command = self.source if source is None else source
    mixed = [
      a + blend * (np.asarray(b) - a)
      for a, b in zip(source_command, command, strict=True)
    ]
    if not all(np.isfinite(x).all() for x in mixed):
      raise ValueError("Invalid handoff command")
    # Separate position/gain interpolation adds an unwanted cross term:
    # beta*(1-beta)*(K_source-K_dest)*(q_dest-q_source).
    # Weight each target by its gain so PD torque is the convex combination
    # of the two controllers (before the explicit target slew bound).
    weighted = (1 - blend) * np.asarray(source_command[1]) * np.asarray(
      source_command[0]
    ) + blend * np.asarray(command[1]) * np.asarray(command[0])
    np.divide(weighted, mixed[1], out=mixed[0], where=mixed[1] > 0)
    mixed[0] = np.clip(
      mixed[0], self.previous - self.step_limit, self.previous + self.step_limit
    )
    self.previous = mixed[0].copy()
    done = self.alpha >= 1 and np.allclose(mixed[0], command[0], atol=1e-7, rtol=0)
    if not done and self.timeout is not None and elapsed > self.seconds + self.timeout:
      raise TimeoutError("Policy handoff target did not settle")
    return tuple(mixed), done
