"""Physical MuJoCo-deployment ball trajectories resembling recorded real serves.

Uses the production SimBall plant (analytical contacts, as in deployment), an
actual MuJoCo scene/clock, and unchanged estimator/A3 planner. No actor, motion
library, racket collision or reverse construction from a racket command.
"""

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
import yaml

from deploy.backends.mujoco import SimBall
from deploy.backends.playback import HELP as PLAYBACK_HELP
from deploy.backends.playback import PlaybackControl
from deploy.g1_pingpong.perception import BallEstimate, BallEstimator
from deploy.hope_g1.a3_planner import A3Planner


def make_suite(diagnosis):
  cases = []
  for run in diagnosis["runs"]:
    for index, swing in enumerate(run["swings"]):
      samples = np.asarray(swing["raw_samples"])
      # Fit only observed incoming positions BEFORE the recorded trigger/bounce.
      # Never infer a feed from a motion, command or desired hit location.
      pre = samples[samples[:, 0] <= -0.04]
      start = float(pre[0, 0])
      t = pre[:, 0] - start
      design = np.column_stack([np.ones(len(t)), t, t**2])
      coefficients = np.linalg.lstsq(design, pre[:, 1:], rcond=None)[0]
      cases.append(dict(
        id=f"{Path(run['run']).name}_{index + 1}",
        initial_position=coefficients[0].tolist(),
        initial_velocity=coefficients[1].tolist(),
        recorded_trigger_time=-start,
        provenance="Quadratic fit to pre-trigger real mocap positions; not a motion-derived serve",
        prefit_rms_m=float(np.sqrt(np.mean((design @ coefficients - pre[:, 1:])**2))),
      ))
  return cases


def scene(config):
  b = config
  x = (b["table_near_x"] + b["table_far_x"]) / 2
  y, z = b["table_center_y"], b["table_height"]
  length = (b["table_far_x"] - b["table_near_x"]) / 2
  xml = f'''<mujoco><option timestep="0.001"/><visual><global offwidth="960" offheight="640"/></visual>
    <worldbody><light pos="1 -1 4"/><geom type="plane" pos="0 0 -0.76" size="6 6 .1" rgba=".2 .2 .2 1"/>
    <geom type="box" pos="{x} {y} {z-.015}" size="{length} {b['table_width']/2} .015" rgba=".05 .25 .42 1"/>
    <geom type="box" pos="{x} {y} {z+b['net_height']/2}" size=".003 {b['table_width']/2+b.get('net_overhang',0)} {b['net_height']/2}" rgba=".8 .8 .8 .8"/>
    <geom type="box" pos="{x} {y} {z+.0006}" size="{length} .0015 .0005" rgba="1 1 1 1"/>
    <body mocap="true"><geom type="sphere" size="{b['ball_radius']}" rgba="1 .65 .05 1" contype="0" conaffinity="0"/></body>
    </worldbody></mujoco>'''
  model = mujoco.MjModel.from_xml_string(xml)
  return model, mujoco.MjData(model)


def simulate(case, config):
  """Production ball dynamics, free of any policy/racket/command influence."""
  shift = np.array([0., config["table_center_y"], 0.])
  physical = copy.deepcopy(config)
  physical["target_xy"] = (np.asarray(config["target_xy"]) - shift[:2]).tolist()
  plant = SimBall.from_launch(physical, np.asarray(case["initial_position"]) - shift,
                              case["initial_velocity"], lifetime=2.)
  remote_racket = SimpleNamespace(racket_pos=np.array([-100., 0., 0.]),
                                  racket_normal=np.array([1., 0., 0.]))
  model, data = scene(config)
  states, bounce_times = [], []
  crossing = None
  for step in range(1801):
    now = step * .001
    p, v = plant.position + shift, plant.velocity.copy()
    states.append([now, *p, *v])
    data.mocap_pos[0] = p
    if crossing is not None and now >= crossing[0] + .08:
      break
    old = plant.position.copy()
    plant.substep(remote_racket, .001)
    mujoco.mj_step(model, data)
    if v[2] < 0 < plant.velocity[2]:
      bounce_times.append(now + .001)
    if crossing is None and old[0] > -.25 >= plant.position[0]:
      f = (old[0] + .25) / (old[0] - plant.position[0])
      crossing = np.r_[now + f*.001, old + f*(plant.position-old) + shift]
  if crossing is None:
    raise ValueError(f"No physical incoming crossing: {case['id']}")
  if plant.hit:
    raise AssertionError("Planner-only plant must never contact a racket")
  return np.asarray(states), crossing, bounce_times


def interpolate(states, times, columns):
  return np.column_stack([np.interp(np.atleast_1d(times), states[:, 0], states[:, col])
                          for col in columns])


def sensor_packets(states, phase, mode, z_bias, noise_mm, seed):
  capture = np.arange(0., states[-1, 0], 1/180)
  position = interpolate(states, capture, range(1, 4))
  position += np.random.default_rng(seed).normal(0, noise_mm*.001, position.shape)
  position[:, 2] += z_bias
  polling = np.ceil((capture-phase)/.01 - 1e-10)*.01 + phase
  rank = np.zeros(len(capture))
  for i in range(1, len(capture)):
    if abs(polling[i]-polling[i-1]) < 1e-8:
      rank[i] = rank[i-1] + 1
  callback = polling + rank*.00003
  stamp = callback if mode == "callback_stamp" else capture
  # Exactly the same delivered positions/batches for the two timestamp modes.
  delivery = callback + .001
  return list(zip(delivery, stamp, position, strict=True))


def evaluate(case, states, crossing, bounces, rally, mocap, mode, phase, bias, noise):
  estimator = BallEstimator(**mocap["estimator"])
  planner = A3Planner(rally["ball"], rally["a3"])
  packets = sensor_packets(states, phase, mode, bias, noise, seed=7)
  index, resets, first = 0, [], None
  evaluations = []
  for now in np.arange(.02, crossing[0]-.005, .02):
    while index < len(packets) and packets[index][0] <= now + 1e-10:
      _delivery, stamp, position = packets[index]
      count = len(estimator.samples)
      # UDPMocap maps source stamp into the local monotonic clock before fitting.
      estimator.update(position, stamp, received=stamp)
      if count >= 2 and len(estimator.samples) < count:
        resets.append(float(now))
      index += 1
    if mode == "oracle":
      state = interpolate(states, now, range(1, 7))[0]
      estimate = BallEstimate(state[:3], state[3:], now, now)
    else:
      estimate = estimator.get(now)
    command = planner.plan(estimate, now, -.25) if estimate else None
    if command is None:
      continue
    tts = command.deadline-now
    row = dict(time=float(now), tts=float(tts), deadline=float(command.deadline),
               position=command.position.tolist(), velocity=command.velocity.tolist(),
               normal=command.normal.tolist(), fit_samples=len(estimator.samples),
               position_error_m=float(np.linalg.norm(command.position-crossing[1:])),
               time_error_s=float(command.deadline-crossing[0]),
               predicted_bounces=command.diagnostic["predicted_bounces"])
    # Only the production TTS/frame window. No selector or reachability claims.
    start_frame = round((.86-tts)*50)
    eligible = bool(0 < tts <= .5 and 0 <= start_frame <= 29)
    row["within_start_time_window"] = eligible
    evaluations.append(row)
    if first is None and eligible:
      first = row
  anchor = min(evaluations, key=lambda r: abs(r["time"]-case["recorded_trigger_time"])) if evaluations else None
  return dict(case=case["id"], sensor=mode, polling_phase_s=phase, sensor_z_bias_m=bias,
              first_time_window_command=first, near_recorded_trigger=anchor,
              estimator_reset_times=resets, actual_bounce_times=bounces,
              actual_crossing=crossing.tolist(), evaluations=evaluations)


def summarize(results):
  groups = {}
  for row in results:
    groups.setdefault((row["profile"], row["sensor"]), []).append(row)
  output = []
  for (profile, sensor), group in groups.items():
    selected = [r["first_time_window_command"] for r in group if r["first_time_window_command"]]
    errors = np.array([r["position_error_m"] for r in selected])
    timing = np.array([abs(r["time_error_s"]) for r in selected])
    output.append(dict(profile=profile, sensor=sensor, total=len(group), commands=len(selected),
      position_median_cm=float(np.median(errors)*100) if len(errors) else None,
      position_max_cm=float(np.max(errors)*100) if len(errors) else None,
      time_median_abs_ms=float(np.median(timing)*1000) if len(timing) else None,
      within_5cm_20ms=int(sum((errors <= .05) & (timing <= .02)))))
  return output


def plots(output, results, trajectories):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  columns = min(3, len(trajectories))
  rows = (len(trajectories)+columns-1)//columns
  fig, axes = plt.subplots(rows, columns, figsize=(14, 4*rows), constrained_layout=True, squeeze=False)
  for ax in axes.flat[len(trajectories):]:
    ax.set_visible(False)
  case_numbers = {name: i+1 for i, name in enumerate(dict.fromkeys(c for _, c in trajectories))}
  for ax, ((profile, case), states) in zip(axes.flat, trajectories.items(), strict=False):
    ax.plot(states[:, 0], states[:, 3], color="black", label="Physical ball Z")
    group = [r for r in results if r["profile"] == profile and r["case"] == case
             and r["polling_phase_s"] == 0]
    for row, color in zip(group, ["#228833", "#4477aa", "#cc3311"], strict=True):
      e = row["first_time_window_command"]
      if e:
        ax.scatter(e["deadline"], e["position"][2], color=color, label=row["sensor"], s=45)
        ax.axvline(e["time"], color=color, alpha=.2)
    cross = group[0]["actual_crossing"]
    ax.scatter(cross[0], cross[3], marker="x", s=70, color="black", label="Actual crossing")
    ax.set(title=f"{profile} / serve {case_numbers[case]}",
           xlabel="Time after launch (s)", ylabel="Height above configured table (m)")
    ax.grid(alpha=.25)
  axes[0, 0].legend(fontsize=8)
  fig.savefig(output / "trajectory_comparison.png", dpi=160)
  plt.close(fig)
  summary = summarize(results)
  fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
  for ax, key, title in zip(axes, ["position_error_m", "time_error_s"],
                           ["First command position error (cm)", "Signed arrival prediction error (ms)"], strict=True):
    groups = [(r["profile"], r["sensor"]) for r in summary]
    data = [[r["first_time_window_command"][key]*(100 if key.startswith('position') else 1000)
             for r in results if (r["profile"],r["sensor"]) == g and r["first_time_window_command"]]
            for g in groups]
    ax.boxplot(data, tick_labels=[f"{a}\n{b}" for a,b in groups], showfliers=True)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.set_title(title)
    ax.grid(axis="y", alpha=.25)
  fig.savefig(output / "planner_errors.png", dpi=160)
  plt.close(fig)


def render(results, trajectories, config, playback_rate):
  from deploy import viewer as deployment_viewer
  model, data = scene(config)
  playback = PlaybackControl(playback_rate)
  print(f"MuJoCo playback: {PLAYBACK_HELP}", flush=True)
  with deployment_viewer.launch_passive(model, data, key_callback=playback.key) as viewer:
    viewer.cam.lookat[:] = [1., -.76, .25]
    viewer.cam.distance = 4.5
    viewer.cam.azimuth, viewer.cam.elevation = 120, -25
    for (profile, case), states in trajectories.items():
      row = next(r for r in results if r["profile"] == profile and r["case"] == case
                 and r["sensor"] == "callback_stamp" and r["polling_phase_s"] == 0)
      command = row["first_time_window_command"]
      playback.reset(0.)
      for state in states[::10]:
        if not viewer.is_running():
          return
        data.mocap_pos[0] = state[1:4]
        mujoco.mj_forward(model, data)
        with viewer.lock():
          viewer.user_scn.ngeom = 0
          if command and state[0] >= command["time"]:
            for position, rgba in [(command["position"], [1,.15,.1,.7]),
                                    (row["actual_crossing"][1:], [.1,1,.2,.7])]:
              geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
              mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3,.025),
                                  np.asarray(position), np.eye(3).ravel(), np.asarray(rgba))
              viewer.user_scn.ngeom += 1
        viewer.set_texts((None, None, "Playback [ / ] / \\", f"{playback.rate:g}x"))
        viewer.sync()
        playback.wait(state[0])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--diagnosis", type=Path, default=Path("deploy/recordings/hope_g1/diagnostics_20261022/latest_two_runs_diagnosis.json"))
  parser.add_argument("--suite", type=Path)
  parser.add_argument("--rally-config", type=Path, default=Path("deploy/hope_g1/configs/rally_mujoco_random.yaml"))
  parser.add_argument("--mocap-config", type=Path, default=Path("deploy/hope_g1/configs/mocap.yaml"))
  parser.add_argument("--phases", type=int, default=6)
  parser.add_argument("--noise-mm", type=float, default=0.)
  parser.add_argument("--render", action="store_true")
  parser.add_argument("--playback-rate", type=float, default=.5)
  parser.add_argument("--output", type=Path, required=True)
  from deploy.viewer import add_viewer_arguments, configure_viewer

  add_viewer_arguments(parser)
  args = parser.parse_args()
  configure_viewer(args)
  if args.phases < 1 or args.noise_mm < 0 or args.playback_rate <= 0:
    parser.error("Invalid phases/noise/playback-rate")
  if (args.output / "results.json").exists():
    parser.error("Use a fresh output directory")
  args.output.mkdir(parents=True, exist_ok=True)
  torch.set_num_threads(1)
  rally, mocap = yaml.safe_load(args.rally_config.read_text()), yaml.safe_load(args.mocap_config.read_text())
  cases = json.loads(args.suite.read_text()) if args.suite else make_suite(json.loads(args.diagnosis.read_text()))
  (args.output / "suite.json").write_text(json.dumps(cases, indent=2))
  profiles = {"matched": {}, "bounce_sensitivity": {"table_restitution": .78, "table_tangential_damping": .74}}
  results, trajectories = [], {}
  for profile, changes in profiles.items():
    for case in cases:
      plant = {**rally["ball"], **changes}
      states, crossing, bounces = simulate(case, plant)
      trajectories[(profile, case["id"])] = states
      for mode in ["oracle", "capture_stamp", "callback_stamp"]:
        for phase in np.arange(1 if mode == "oracle" else args.phases)*.01/args.phases:
          row = evaluate(case, states, crossing, bounces, rally, mocap, mode, float(phase), 0., args.noise_mm)
          row.update(profile=profile, plant_parameters=plant)
          results.append(row)
      print(f"Finished {profile} / {case['id']}", flush=True)
  summary = summarize(results)
  document = dict(scope="Planner only; no actor/motion selector; not a hit-rate test", physical_engine="Existing deployment SimBall ballistic/table/net kernels at1ms in MuJoCo scene; not native MuJoCo sphere/table contacts", physics_note="bounce_sensitivity .78/.74 is an explicit stress condition, not a field calibration", timestamp_note="Identical180Hz observations delivered by100Hz polling; capture vs callback time stamps differ, no measurement noise by default", no_runtime_changes=True, thresholds=dict(position_m=.05, time_s=.02), noise_mm=args.noise_mm, rally=rally, estimator=mocap["estimator"], summary=summary, results=results)
  (args.output / "results.json").write_text(json.dumps(document, indent=2))
  np.savez_compressed(args.output / "physical_trajectories.npz", **{f"{p}_{i}": v for i, ((p,c), v) in enumerate(trajectories.items())})
  plots(args.output, results, trajectories)
  print(json.dumps(summary, indent=2), flush=True)
  if args.render:
    render(results, trajectories, rally["ball"], args.playback_rate)


if __name__ == "__main__":
  main()
