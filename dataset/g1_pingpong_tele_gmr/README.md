# G1 Tele-GMR/Football ping-pong motions

The raw `.npz` archives and generated `motions/` directory are local training
artifacts and are intentionally excluded from Git. Keep the archives under
`raw/` when reproducing the conversion; the manifests record their expected
checksums.

This is the 43-clip motion set requested for the second full Stage-1 training
run. The source was copied byte-for-byte from:

```text
lxz@192.168.152.200:/home/lxz/HOPE/hope_training/mjlab_pingpong/hope_mjlab/assets/g1_motion_sets/pingpong_kjp_0804_2_0623-tele_gmr-football-20261003-npz--clip-racket-peak
```

No access credentials are stored in the repository. `raw/` contains the 43
unchanged NPZ files. `source_manifest.json` and `hit_type.json` are unchanged
copies from that directory. Their SHA-256 values are respectively
`87f402472ec33317c93a5a921b386f27f487d1bd6259bdaf7065e53be1b8f690`
and `85283503511fa4e1401d7fb65df3f557b503348e96e572f735348e6edb3cc5f7`.

The source pipeline is Xsens BVH at 240 Hz, Tele-GMR G1 PKL at 30 Hz, then
Football/IsaacLab G1 NPZ at 50 Hz. Each accepted clip has 94 frames and uses
strike frame 43. The source manifest binds the files to the local 29-DoF G1
MJCF and asset manifest by SHA-256 and records a separate 43/43 acceptance run.

`motions/` contains the files consumed by mjlab. The preparation step verifies
all raw hashes, schemas, finite values, MuJoCo forward kinematics, source COM
velocities, and root link velocities. It then converts COM linear velocities
to link-origin velocities, translates the initial pelvis XY to zero, and
rotates the initial pelvis yaw to zero. The complete output hashes and measured
errors are in `manifest.json`.

Regenerate the processed set from the repository root:

```bash
uv run --no-sync python scripts/tools/prepare_g1_pingpong_motion.py \
  --source dataset/g1_pingpong_tele_gmr/raw \
  --output dataset/g1_pingpong_tele_gmr/motions \
  --source-manifest dataset/g1_pingpong_tele_gmr/source_manifest.json \
  --xml src/mjlab/asset_zoo/robots/unitree_g1_pingpong/g1.xml
```

The dedicated task ID is
`Mjlab-PingPong-Tracking-G1-TeleGMR-Stage1-RandomDt`. It uses the same Stage-1
environment and runner configuration as
`Mjlab-PingPong-Tracking-G1-Stage1-RandomDt`; only the verified 43-file motion
tuple differs.
