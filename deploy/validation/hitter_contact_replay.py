"""Isolate contact-law differences using recorded incoming/racket states.

This is a counterfactual replay after contact, not a new closed-loop policy run.
Air/table physics, position and the measured racket state stay fixed; spin is
zero before contact and generated spin is not propagated by the legacy SimBall.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from deploy.backends.mujoco import SimBall
from deploy.hope_g1.world import mujoco_ball_config, simulation_translation
from hope_planner_g1.physics.ball_contact import predict_paddle_contact
from hope_planner_g1.physics.constants import BallPhysics, PlannerConfig


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("audits", nargs="+", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(1)
  options = yaml.safe_load(
    Path("deploy/hope_g1_hitter/configs/rally_mujoco.yaml").read_text()
  )
  translation = simulation_translation(options["world"])
  cfg = mujoco_ball_config(options["ball"], translation)
  physics, contact_cfg = BallPhysics(), PlannerConfig()
  rows = []

  def rollout(position, velocity):
    ball = SimBall.from_launch(cfg, position, velocity)
    ball.hit = True
    state = SimpleNamespace(
      racket_pos=np.asarray(position), racket_normal=np.array([1.0, 0, 0])
    )
    for _ in range(1500):
      ball.substep(state, 0.001)
      if ball.first_return_landing is not None or ball.return_net_hit:
        break
    result = ball.return_report()
    return dict(
      outcome=result["outcome"],
      legal=result["legal_return"],
      landing=(
        None
        if result["first_landing_plane_crossing"] is None
        else (
          np.asarray(result["first_landing_plane_crossing"]["position"]) + translation
        ).tolist()
      ),
    )

  for path in args.audits:
    audit = json.loads(path.read_text())
    for b in audit["balls"]:
      if not b["hit"]:
        continue
      c = b["contact"]
      vin, vr, n = (
        np.asarray(c[k])
        for k in ("incoming_velocity", "racket_velocity", "racket_normal")
      )
      vnew, _ = predict_paddle_contact(vin, vr, n, np.zeros(3), physics, contact_cfg)
      old = rollout(c["position"], c["outgoing_velocity"])
      new = rollout(c["position"], vnew)
      last = b["commands"][-1]
      wanted = np.asarray(last["velocity"])
      normal = wanted / np.linalg.norm(wanted)
      ideal, _ = predict_paddle_contact(
        vin, wanted, normal, np.zeros(3), physics, contact_cfg
      )
      rows.append(
        dict(
          seed=audit["seed"],
          ball=b["feed_id"],
          original_outcome=b["outcome"],
          old_replay=old,
          matched_contact_replay=new,
          ideal_racket_at_actual_contact_replay=rollout(c["position"], ideal),
          outgoing_speed_error=float(np.linalg.norm(vnew - c["outgoing_velocity"])),
          racket_velocity_error=float(np.linalg.norm(vr - wanted)),
          normal_error_degrees=float(
            np.degrees(np.arccos(np.clip(abs(n @ normal), 0, 1)))
          ),
          effective_restitution=float(
            contact_cfg.e_exp_g1 * np.exp(contact_cfg.e_exp_g2 * abs((vin - vr) @ n))
          ),
          old_velocity=c["outgoing_velocity"],
          matched_velocity=vnew.tolist(),
          requested_velocity=last["velocity"],
        )
      )
  result = dict(
    method=__doc__,
    total_contacts=len(rows),
    original_outcomes=dict(Counter(r["original_outcome"] for r in rows)),
    old_replay_legal=sum(r["old_replay"]["legal"] for r in rows),
    matched_contact_replay_legal=sum(
      r["matched_contact_replay"]["legal"] for r in rows
    ),
    ideal_racket_at_actual_contact_replay_legal=sum(
      r["ideal_racket_at_actual_contact_replay"]["legal"] for r in rows
    ),
    mean_outgoing_velocity_difference=float(
      np.mean([r["outgoing_speed_error"] for r in rows])
    ),
    mean_racket_velocity_error=float(
      np.mean([r["racket_velocity_error"] for r in rows])
    ),
    mean_normal_error_degrees=float(np.mean([r["normal_error_degrees"] for r in rows])),
    rows=rows,
  )
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(
    json.dumps(
      {k: v for k, v in result.items() if k not in ("rows", "method")}, indent=2
    )
  )


if __name__ == "__main__":
  main()
