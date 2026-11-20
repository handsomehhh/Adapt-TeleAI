# Unitree G1 29 DoF ping pong training asset

The active source is `source/g1_29dof_pingpong_fast_extended_27mm.urdf`, copied
on 2026-11-02 from
`/home/lhy/table-tennis/HOPE-TeleAI/G1/urdf/g1_29dof_pingpong_fast_extended_27mm.urdf`.
Its SHA-256 is
`ff69e37651da1f2f1ae302000e694433fca4f7ad056473dbf8f60dfabab56c7b`.
The training environment loads the converted `g1.xml`. All 40 referenced mesh
files are local copies, with source paths and checksums in `asset_manifest.json`.

The MJCF merges fixed-link inertias into 30 articulated bodies and preserves
all 29 joint transforms, ranges, and the native servo ordering. It has 69 geoms,
including the 31 collision shapes from the supplied fast URDF. Each foot now
uses one box. Total compiled mass is 33.69842032240642 kg. The racket centre in
the right wrist yaw frame is `[0.242, 0.003, 0]` m, normal `[0, 1, 0]`; the centre
moves forward 27 mm compared with the previous model. The longer wood mesh,
grip collision and updated racket inertia are included.

Stage 1, motion correction, MVAE FK, rally, standing and runtime share this asset.
`G1_PINGPONG_MOUNT_OFFSET` is read from the manifest, so ball contact uses the
updated racket centre. The existing training choice to disable self collision
is preserved. The separate original tennis asset `unitree_g1` is unchanged.

Regenerate from the repository root with:

```bash
uv run python -m mjlab.asset_zoo.robots.unitree_g1_pingpong.convert_urdf --urdf /absolute/path/to/G1/urdf/g1_29dof_pingpong_fast_extended_27mm.urdf
uv run python -m mjlab.asset_zoo.robots.unitree_g1_pingpong.validate_urdf --samples 32 --write-report
```

The converter preserves the training joint tree and servo settings, and rejects
joint-transform/range changes requiring motion retargeting. It analytically
merges full inertias and writes explicit principal inertias. Validation uses
MuJoCo's independent native URDF import for FK and geometry, and the original
URDF full inertia tensors for physical inertia comparisons. The report includes
32 sampled configurations, fixed-link mass/COM/inertia, every visual/collision
transform, joint ranges, racket frame and the assembled 29-servo entity.
Native URDF/MJCF mesh principalization differs by approximately 2.14e-6 in the
rotation matrix; a 5e-6 tolerance is recorded. Other reported errors are below
1e-9. Reports and hashes are in `asset_validation.json` and `asset_manifest.json`.

The position servo gains, armatures and effort limits retain the previous G1
configuration; each action scale is `0.25 * effort_limit / stiffness`.
The neutral imported root height is 0.82 m; the bent-knee reset height is 0.76 m.

Motion clips do not belong to this asset package. Fresh tracker training and
new motion correction should use this model; earlier validation runs describe
the previous geometry. No additional licensing claim is made for source meshes.
