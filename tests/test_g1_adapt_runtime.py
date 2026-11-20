"""Optional real-simulator parity test with an exported trained/smoke bundle."""

import copy
import os
from pathlib import Path

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.g1_adapt.actions import LatentTrackerAction
from mjlab.tasks.g1_adapt.ball import RallyBall
from mjlab.tasks.g1_adapt.env_cfg import g1_adapt_env_cfg
from mjlab.tasks.g1_adapt.reference import GeneratedMotionCommand
from mjlab.tasks.g1_adapt.runtime import (
  HierarchicalController,
  RobotState,
  observation_prefix,
)


def test_exported_runtime_matches_simulator_observations_and_joint_targets():
  directory = os.environ.get("ADAPT_G1_TEST_BUNDLE")
  if not directory:
    pytest.skip("Set ADAPT_G1_TEST_BUNDLE for GPU simulator/runtime parity")
  root = Path(directory)
  cfg = g1_adapt_env_cfg(play=True)
  cfg.scene.num_envs = 1
  cfg.commands["motion"].mvae_file = str(root / "mvae.pt")
  cfg.actions["joint_pos"].tracker_file = str(root / "tracker.pt")
  controller = HierarchicalController(root, device="cuda:0")
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    observations, _ = env.reset()
    with torch.inference_mode():
      for _ in range(12):
        observations, _, _, _, _ = env.step(torch.zeros(1, 33, device=env.device))
      ball = env.command_manager.get_term("ball")
      assert isinstance(ball, RallyBall)
      assert ball.observed_valid.all()
      motion = env.command_manager.get_term("motion")
      assert isinstance(motion, GeneratedMotionCommand)
      assert motion.generator is not None
      action = env.action_manager.get_term("joint_pos")
      assert isinstance(action, LatentTrackerAction)
      robot = env.scene["robot"].data
      ids = action._target_ids
      # Include a nonzero encoder bias to verify observation/actuation signs.
      robot.encoder_bias[:, ids] = 0.005
      observations = {
        name: env.observation_manager.compute_group(name)
        for name in ("actor", "tracker")
      }
      state = RobotState(
        robot.root_link_pos_w,
        robot.root_link_quat_w,
        robot.joint_pos_biased[:, ids],
        robot.joint_vel[:, ids],
        robot.root_link_ang_vel_b,
        robot.encoder_bias[:, ids],
      )
      controller.generator = copy.deepcopy(motion.generator)
      controller.estimator = copy.deepcopy(ball.estimator)
      controller.high_action.copy_(env.action_manager.action)
      controller.joint_action.copy_(action.joint_action)
      controller.initialized = True
      manual = observation_prefix(
        controller.generator,
        state,
        controller.default_joints,
        action.joint_action,
        motion.last_dt,
      )
      torch.testing.assert_close(manual, observations["tracker"], atol=2e-5, rtol=2e-5)
      targets, info = controller.step(state, None, ball.elapsed, target=ball.target)
      torch.testing.assert_close(
        info["planner_observation"], observations["actor"], atol=2e-5, rtol=2e-5
      )
      high_action = controller.rally(observations["actor"])
      action.process_actions(high_action)
      assert state.encoder_bias is not None
      expected = action._processed_actions - state.encoder_bias
      torch.testing.assert_close(targets, expected, atol=2e-5, rtol=2e-5)
      # A stale/missing position stream switches to the learned standing policy.
      targets, info = controller.step(state, None, ball.elapsed + 0.2)
      assert info["standing"].all() and targets.isfinite().all()
  finally:
    env.close()
