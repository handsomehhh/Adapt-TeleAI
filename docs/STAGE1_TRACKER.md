# Generic G1 stage-1 tracker

`Mjlab-Stage1-Tracker-Flat-Unitree-G1` trains a reusable 29-DoF Unitree G1
motion tracker. It uses the adaptive reference-time stage-1 setup already used
by the project, including interpolated motion references and randomized
per-step reference advancement. The tracker reward contains continuous root,
body pose, body velocity, action-rate, and joint-limit terms; it has no
ping-pong strike or racket-specific terms.

The motion command reads every `.npz` below its `motion_directory` recursively.
The supplied cluster dataset is the default path:

```text
/data_zcy/zcy/datasets/motion_data_used_g1
```

The loader accepts the dataset's legacy seven-array format (`fps`, joint and
body trajectories) as well as self-describing clips with `joint_names` and
`body_names`. Legacy clips are interpreted in the native 29-DoF G1 order.
To keep each distributed worker's GPU memory bounded, the default task selects
up to 1024 clips evenly across the recursive archive. Set
`ADAPT_STAGE1_MAX_MOTIONS=0` to load every clip, or choose a different cap.
Clips are also limited to their first 4096 frames by default so a single very
long recording cannot make the padded batch exceed GPU memory; override this
with `ADAPT_STAGE1_MAX_FRAMES=0` when the archive is known to fit.

On the cluster, after installing the repository environment, start an 8-GPU
run with:

```bash
export ADAPT_STAGE1_NUM_ENVS=1024
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/train Mjlab-Stage1-Tracker-Flat-Unitree-G1 \
  --gpu-ids all
```

For a smoke run or a different dataset, override the environment variables:

```bash
ADAPT_STAGE1_MOTION_DIR=/path/to/npz/root \
ADAPT_STAGE1_NUM_ENVS=64 \
  .venv/bin/train Mjlab-Stage1-Tracker-Flat-Unitree-G1 --gpu-ids 0
```

The regular nested Tyro options remain available, for example
`--env.scene.num-envs 64` and
`--env.commands.motion.motion-directory /path/to/npz/root`.
