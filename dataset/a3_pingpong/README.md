# A3 ping-pong reference motions

The raw `.npz` archives and generated `motions/` directory are local training
artifacts and are intentionally excluded from Git. Keep the archives under
`raw/` when reproducing the conversion; `manifest.json` records their expected
checksums.

The 61 supplied archives are copied without changes into `raw/` from
`/home/lxz/Agibot_A3/motion/robot/A3/pingpong_kjp_0814_hard_0814_f_0804_2_0623-npz--clip-racket-peak`
on the user-provided reference host. Source SHA-256 checksums are recorded in
`manifest.json`. No credentials are stored.

`motions/` contains the training inputs: initial pelvis XY is `(0, 0)` and its
initial heading is +X. Every world position, orientation and velocity is
transformed consistently. Root height, joint positions/velocities, body/joint
names, frame rate and clip timing are retained. Configure `MotionCommand` with
`align_heading_to_frame="none"` because this transform is already applied.

Regenerate and validate against the imported A3 kinematics:

```bash
uv run --no-sync python scripts/tools/prepare_a3_motion.py \
  --source dataset/a3_pingpong/raw \
  --output dataset/a3_pingpong/motions \
  --xml src/mjlab/asset_zoo/robots/agibot_a3/xmls/a3_pingpong.xml
```

All clips have 94 frames at 50 Hz (1.86 seconds between first and last frame),
31 named joints and 32 named bodies. Quaternions use `(w, x, y, z)` and body
linear/angular velocities are in world coordinates. The supplied joint
velocities agree with central position differences; quaternion norms differ
from one by less than 2.4e-7. Raw initial yaw is approximately -95 to -82 degrees;
root roll is -1.60 to 0.89 degrees and pitch -0.50 to 9.93 degrees. The lowest
initial ankle-link origin is 0.050 to 0.067 m above ground; this describes link
origins, not sole penetration.

`strike_frame=43` and `strike_time_s=0.86` preserve HOPE's nominal strike phase.
The separately stored `racket_speed_peak_frame` is the global maximum of the
mounted racket speed: it is frame 43 in 44 of the 61 clips, so the two fields
must not be treated as interchangeable impact annotations. The source does not
contain measured ball trajectories or confirmed ball-contact annotations.

Additional fields `racket_pos_w`, `racket_lin_vel_w` and `racket_normal_w` use
the right wrist mount offset `(0.2102113992, 0.0320784995, 0.0320358706)` m and
the local +Y face normal. Racket velocity includes `omega × offset`.

The model validation compares named poses at the first, strike and last frames
of every clip (183 frames): maximum non-ankle position error 4.52e-6 m and
quaternion L2 error 9.08e-7. The approximately 1.5 mm ankle discrepancy is the
known source difference: the URDF has local ankle Y offsets omitted by the
archived motion FK. Original motion FK is preserved.
