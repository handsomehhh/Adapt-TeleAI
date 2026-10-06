# Agibot A3 table-tennis robot

Imported from the user-provided HOPE checkout at commit
`67a1a720e985b01ec157821c9809e63bcbb13d9a` on 2026-09-29.
Upstream: https://github.com/hitchopen/HOPE.

The MJCF is adapted from
`hope_training/mjlab_pingpong/hope_mjlab/assets/a3_primitive.xml`.
Its original Agibot meshes are copied without simplification. It preserves the
31 actuated joints, 32 moving bodies, 58.27723163 kg mass, joint transforms,
limits and full inertia tensors. Fixed hand, shell and paddle inertias are
already merged into their parent bodies. The primitive contacts originate in
HOPE's August 6 URDF and are an explicit simulation-speed choice; they are not
the August 5 baseline's mesh contacts. Self collisions are disabled.

`baseline_robot.json` preserves the original HOPE training pose, PD gains,
torque limits, armatures and action scales. `source_manifest.json` preserves
the source paths, URDF checksums and ordering from HOPE. These source paths are
provenance only and are not required at runtime. The asset is self-contained.

Local changes to the MJCF:

- Initial pelvis height is 1.0684 m so the pre-reset model is above the ground.
- Added the massless `racket_center` site at the URDF paddle center.
- Renamed the paddle contact proxy to `pingpong_racket_collision`.

The paddle center is offset `(0.210211399202899, 0.0320784994676765,
0.0320358706296689)` m in `right_wrist_yaw_Link`. The paddle normal is **local
Y**, since the paddle lies in the wrist's XZ plane. The site uses the wrist's
orientation, so its local Y axis is also the normal.

The original URDF ankle-pitch origins have +/-1.5 mm local Y offsets absent
from the supplied motion's FK. The URDF is preserved. Named motion conversion
should regenerate FK against this model instead of copying source body arrays.

HOPE's Apache-2.0 license is retained as `LICENSE-HOPE` (copyright 2025-2026
Intelligent Racing Inc., dba Hitch Interactive). The original Agibot package
metadata declares BSD and is retained as `source_package.xml`; the supplied
asset bundle did not contain a separate BSD license text or author notice.
No new licensing claim is made for third-party meshes.
