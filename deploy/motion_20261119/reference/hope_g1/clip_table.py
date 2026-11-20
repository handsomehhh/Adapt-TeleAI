"""Per-clip strike timing and Isaac's start-frame window over the whole library."""

import numpy as np

from deploy.motion_20261119.reference.g1_pingpong.motions import heading


class ClipTable:
  """Strike frame, fps and clip-frame strike point of every library clip.

  The strike points are RebasedMotion.strike_position applied to all clips at
  once, so scoring thousands of clips is one vectorized step per control tick.
  ``startable`` is the only implementation of the training entry window.
  """

  def __init__(self, motions, strike_frames, ball, *, max_start_frame, min_lead_frames):
    self.strike_frames = np.asarray(strike_frames)
    if len(self.strike_frames) != len(motions):
      raise ValueError("One strike frame per motion is required")
    self.fps = np.array([clip.fps for clip in motions])
    anchors = [clip.sample(0) for clip in motions]
    self.anchor = np.array([a[1] for a in anchors])
    self.yaw = np.array([heading(a[2]) for a in anchors])
    self.strike = np.array([clip.strike_position(ball) for clip in motions])
    self.max_start_frame = max_start_frame
    # None: the task has no minimum lead, as with strike_phase libraries.
    self.min_lead_frames = min_lead_frames or 0

  def rebased_strikes(self, indices, position, quaternion):
    """RebasedMotion(clip, position, quaternion).strike_position per clip."""
    yaw = heading(quaternion) - self.yaw[indices]
    offset = self.strike[indices] - self.anchor[indices]
    c, s = np.cos(yaw), np.sin(yaw)
    xy = np.stack(
      [c * offset[:, 0] - s * offset[:, 1], s * offset[:, 0] + c * offset[:, 1]],
      axis=1,
    )
    return position + np.column_stack([xy, offset[:, 2]])

  def aligned_strikes(self, indices, home):
    """Strike points of AlignedLibrary clips, which share the home transform."""
    return self.strike[indices] @ home.rotation.T + home.translation

  def start_frames(self, indices, remaining):
    """Frame to enter each clip so that it strikes ``remaining`` seconds later."""
    strikes, fps = self.strike_frames[indices], self.fps[indices]
    return np.floor((strikes / fps - remaining) * fps + 0.5).astype(int)

  def startable(self, indices, starts):
    """Isaac enters a clip at [0, min(random_start_frame_max, strike - lead)].

    A negative start means the clip is shorter than the time to strike and
    would swing early, so it is rejected rather than clamped to frame zero.
    """
    return (
      (starts >= 0)
      & (starts <= self.max_start_frame)
      & (starts <= self.strike_frames[indices] - self.min_lead_frames)
    )
