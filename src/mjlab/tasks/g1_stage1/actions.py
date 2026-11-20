"""Joint targets use the same semantic order as G1 motion references."""

from dataclasses import dataclass

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class OrderedJointPositionAction(JointPositionAction):
  def _find_targets(self, cfg):
    return self._entity.find_joints(cfg.actuator_names, preserve_order=True)


@dataclass(kw_only=True)
class OrderedJointPositionActionCfg(JointPositionActionCfg):
  def build(self, env):
    return OrderedJointPositionAction(self, env)
