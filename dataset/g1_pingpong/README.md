# G1 ping-pong reference motions

The raw `.npz` archives and generated `motions/` directory are local training
artifacts and are intentionally excluded from Git. Keep the archives under
`raw/` when reproducing the conversion; the manifests record their expected
checksums.

This dataset contains the 43 G1 clips used by the user-provided HOPE reference
run `2026-10-02_01-15-34_g1_bvh_v2_30k_save2000_seed0`.

`raw/` is an unchanged copy of the reference run's canonical 29-DoF G1 motion
archives. `source_manifest.json` records their original BVH provenance,
retarget method, checksums, hit labels, and independent acceptance. No access
credentials are stored in this repository.

`motions/` contains the inputs used by this task. Each world pose and velocity
is rotated so the initial pelvis yaw faces +X, and the initial pelvis XY is
translated to `(0, 0)`. Root height, joint motion, names, timing, and racket
kinematics are retained. All 43 clips have 94 frames at 50 Hz, so the interval
between the first and last frame is 1.86 seconds. The annotated strike is frame
43, or 0.86 seconds.

Regenerate the processed clips and repeat named MuJoCo forward-kinematics
validation from the repository root:

```bash
uv run --no-sync python scripts/tools/prepare_g1_pingpong_motion.py
```

The generated `manifest.json` contains source and output SHA256 hashes plus the
measured FK error. The motion arrays use WXYZ quaternions and world-frame body
linear/angular velocities. The task sets `align_heading_to_frame="none"`
because the coordinate transform has already been applied consistently.

The reference asset is the separate 29-DoF model under
`src/mjlab/asset_zoo/robots/unitree_g1_pingpong`. These clips must not be loaded
against the repository's older 27-DoF G1 tennis model: the waist joint set,
racket mount, body schema, and action contract differ.
