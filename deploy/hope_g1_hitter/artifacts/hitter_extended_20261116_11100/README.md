# G1 extended 11100 deployment bundle

Default deployment release exported from
`2026-11-16_20-29-38/model_11100.pt`. The checkpoint is unchanged and tracked
with Git LFS. `model_11200.pt` is not part of this release.

- Actor: 104 observations → 29 actions, 50 Hz, no observation normalization.
- Both reference strike points and command X use the common 0.44 m plane.
- Forehand command Y: [-0.55, -0.15] m; backhand: [-0.15, 0.25] m,
  relative to the target station. Position curriculum is fully expanded.
- Station Y remains [-1.25, -0.35] m with a maximum 0.35 m change per command.
- The checkpoint's 47 explicit self-contact pairs are in `contract.json`;
  the deployment loader adds them to the bundled base robot XML.

Contents: checkpoint, `contract.json`, reference NPZs, original training
`params/env.yaml` and `params/agent.yaml`, checkpoint `training_recipe.json`,
robot XML/meshes/source URDFs and provenance. `manifest.json` gives byte counts
and SHA256 for every other file. Absolute training paths inside the archived
recipe/configs are provenance only; runtime resources resolve inside this bundle.
The training Python package is not required to load this bundle.

## Reproduce the export

From Adapt-TeleAI, using the training repository's Python environment:

```bash
/path/to/PingPong/hope_training/mjlab_training/.venv/bin/python \
  deploy/hope_g1_hitter/export_contract.py /path/to/new-bundle \
  --checkpoint /path/to/2026-11-16_20-29-38/model_11100.pt
```

The output directory must not exist. The exporter copies the checkpoint,
training configs, hash-verified reference motions and robot resources before
generating the contract and manifest. After export, update the default task
path and copy the exact contract bytes to the ROS planner's
`src/hope_planner_g1/config/policy_contract.json`.

## Synchronize to lxz after the commits have been pushed

```bash
cd /home/lxz/Adapt-Deploy
git pull --ff-only
git lfs install
git lfs pull

cd /home/lxz/g1_planner_ros2
git pull --ff-only
source /opt/ros/humble/setup.zsh
colcon build --symlink-install
source install/setup.zsh

cmp /home/lxz/Adapt-Deploy/deploy/hope_g1_hitter/artifacts/hitter_extended_20261116_11100/contract.json \
  /home/lxz/g1_planner_ros2/src/hope_planner_g1/config/policy_contract.json
cd /home/lxz/Adapt-Deploy
.venv/bin/python -m deploy.hope_g1_hitter.deploy_agent --env check
```

The existing planner launch and real deployment commands then use this release
by default. The real deployment terminal must start in `/home/lxz/Adapt-Deploy`.
This bundle changes the policy contract and resources; it does not recalibrate
planner ball/racket physics or contain the PingPong multiplayer ball-material XML.

## Validation

See `validation.json`: source checkpoint/config byte equality, reference hashes
and common plane, matching ROS contract, relocated bundle load, controller
initialization, 47 compiled self-contact pairs, and 100 actor input comparisons
(maximum absolute difference 0). Planner standalone suite: 45 passed.
The default `--env check` entry also passed. No hardware run was performed.
