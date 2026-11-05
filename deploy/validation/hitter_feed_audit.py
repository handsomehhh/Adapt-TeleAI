"""Exercise deploy_agent startup and the actual F/'one ball' callback offline.

No live IO, viewer, policy, planner or feeder changes; scripted button presses
replace the browser for repeatability. Per-ball decisions are captured alongside
physical returns. Run as a module from the deployment repository.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--balls", type=int, default=12)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--legacy-contact",
    action="store_true",
    help="Reproduce the old fixed restitution=0.88",
  )
  parser.add_argument(
    "--keep-track",
    action="store_true",
    help="Reproduce the old missing reset at simulated ball spawn",
  )
  args = parser.parse_args()
  from deploy.g1_pingpong.simulation_runtime import MujocoRuntime
  from deploy.hope_g1_hitter.controller import HitterController
  from deploy.hope_g1_hitter.deploy_agent import main as deploy_main
  from hope_planner_g1.planner import G1Planner

  sessions, records = [], {}
  original_init, original_buttons = MujocoRuntime.__init__, MujocoRuntime.buttons
  original_plan, original_select = G1Planner.plan, HitterController.select
  if args.keep_track:
    HitterController.reset_ball_observations = lambda self: None

  def init(session, *a, **kw):
    original_init(session, *a, **kw)
    session.continuous_feed = False
    session.audit_next_ball = None
    sessions.append(session)

  def buttons(session):
    now = session.monotonic()
    if session.engine.lifecycle.state in ("ARMED", "EXECUTE"):
      if session.audit_next_ball is None:
        session.audit_next_ball = now
      if session.served < args.balls and now >= session.audit_next_ball:
        session.key(ord("F"))
        session.audit_next_ball += 4.0
      if session.served >= args.balls and now >= session.audit_next_ball:
        return 1 << 3
    return original_buttons(session)

  def record():
    if not sessions or not sessions[-1].served:
      return None
    return records.setdefault(
      sessions[-1].served,
      dict(plan_calls=0, accepted=0, rejected_attempts=Counter(), commands=[]),
    )

  def plan(planner, *a, **kw):
    command = original_plan(planner, *a, **kw)
    r = record()
    if r is not None:
      r["plan_calls"] += 1
      for attempt in planner.diagnostic.get("attempts", []):
        if attempt["reason"] != "accepted":
          r["rejected_attempts"][attempt["reason"]] += 1
      if command is not None:
        r["last_solve"] = dict(
          side=command.side,
          position=command.position.tolist(),
          velocity=command.velocity.tolist(),
          normal=command.normal.tolist(),
          station=command.station.tolist(),
          strike_time=command.strike_time,
          residual=command.residual,
          outgoing=command.outgoing_velocity.tolist(),
          requested=command.outgoing_velocity_requested.tolist(),
        )
    return command

  def select(controller, now):
    shot, diagnostic = original_select(controller, now)
    r = record()
    if r is not None and shot is not None:
      r["accepted"] += 1
      r["commands"].append(
        dict(
          time=now,
          deadline=shot.strike_at,
          side=shot.side,
          position=shot.position.tolist(),
          velocity=shot.velocity.tolist(),
          station=shot.station.tolist(),
        )
      )
    return shot, diagnostic

  MujocoRuntime.__init__, MujocoRuntime.buttons = init, buttons
  G1Planner.plan, HitterController.select = plan, select
  sys.argv = [
    "deploy_agent",
    "--env",
    "mujoco",
    "--sim-controls",
    "auto",
    "--sim-startup",
    "suspended",
    "--duration",
    str(20 + 4 * args.balls),
    "--episodes",
    "999",
    "--seed",
    str(args.seed),
    "--output",
    str(args.output),
  ]
  if args.legacy_contact:
    # Deliberate plant/planner mismatch for this diagnostic only. Apply after
    # automatic profile selection; production entry always keeps them aligned.
    from deploy.hope_g1_hitter import deploy_agent

    resolve = deploy_agent.resolve_physics

    def legacy_physics(options, mode):
      result = resolve(options, mode)
      result["ball"].pop("racket_contact_model", None)
      result["ball"]["racket_restitution"] = 0.88
      return result

    deploy_agent.resolve_physics = legacy_physics
  try:
    deploy_main()
  except SystemExit as exc:
    if exc.code not in (0, 1):
      raise
  summary = json.loads((args.output / "summary.json").read_text())
  balls = []
  for b in summary["ball_returns"]:
    r = records.get(b["feed_id"], {})
    balls.append(
      dict(
        feed_id=b["feed_id"],
        feed=b["feed"],
        outcome=b["outcome"],
        hit=b["hit"],
        legal_return=b["legal_return"],
        contact=b["contact"],
        landing=b["first_landing_plane_crossing"],
        net_crossing=b["net_crossing"],
        **r,
      )
    )
  result = dict(
    seed=args.seed,
    balls=balls,
    totals=dict(
      balls=len(balls),
      planned=sum(b.get("accepted", 0) > 0 for b in balls),
      hit=sum(b["hit"] for b in balls),
      legal=sum(b["legal_return"] for b in balls),
    ),
    physics_validation=summary["physics_validation"],
    button_schedule="Actual F callback every 4 seconds after ARMED; automatic START/lower/A/UP, same controller, no rendering",
  )
  (args.output / "feed_audit.json").write_text(json.dumps(result, indent=2) + "\n")
  print("AUDIT", json.dumps(result["totals"]))


if __name__ == "__main__":
  main()
