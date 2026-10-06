# Unitree G1 29-DoF ping-pong asset

This package contains the 29-DoF Unitree G1 with the supplied right-hand
table-tennis mount and racket. It is deliberately separate from
`asset_zoo/robots/unitree_g1`, whose racket MJCFs describe a 27-DoF model.

The files were imported on 2026-10-03 from the supplied frozen reference
bundle at `/tmp/adapt_g1_reference/g1`. The bundle's `asset_manifest.json`
records the original URDF as
`/home/lxz/HOPE/G1/urdf/g1_29dof_pingpong.urdf` and records SHA-256
`127f42c7fbb4a6df60e1ac3b76b64af0c05e3575f3bdda2a5210c9252342fe1e`.
That URDF is retained under `source/` for provenance. The MJCF and all 41
mesh files are byte-for-byte copies from the frozen bundle; their paths and
checksums are recorded in `asset_manifest.json`.

The MJCF conversion analytically merges fixed-link inertias into the 30
articulated bodies while preserving the URDF joint transforms, limits, mesh
geometry, racket parts, and full inertial properties. It contains 29 hinge
joints, 82 geoms, 44 contact geoms, and a compiled mass of
33.68916754973783 kg. The model's primitive robot contacts and supplied
racket mesh colliders disable self collision. `asset_validation.json`
preserves the conversion acceptance results, including 32 sampled FK,
center-of-mass, inertia, and limit comparisons against the source URDF.

`g1_constants.py` adapts the frozen `g1_robot.py` reference module to this
repository's package layout and two-space Python style. It preserves the
native MJCF joint order for the 29 position actuators. Gains and armatures
follow the pinned G1 actuator conventions, effort and velocity limits come
from this URDF, and each action scale is `0.25 * effort_limit / stiffness`.
The `robot_module_sha256` in the validation snapshot identifies that frozen
reference module rather than this repository-formatted adaptation.
The neutral imported pose uses a root height of 0.82 m to keep the extended
legs above the floor before reset; the configured bent-knee initial pose
retains the reference height of 0.76 m.

No motion clips or task configuration are part of this asset package. The
reference bundle did not include a separate license notice for these G1
source assets, so this import makes no new licensing claim for the URDF or
meshes.
