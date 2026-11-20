"""Executor contract: the replaceable unit that turns racket commands into joints.

HopeController owns everything around execution: command sources (embedded or
ROS planner), the lifecycle, selection settling, recovery accounting, standing
checks and events. An executor owns what produces joint targets: a policy and,
when the policy tracks references, its motions and motion selector
(MotionExecutor). A policy that needs no reference motion implements the same
contract on its own, with an empty motion catalogue.

The deployment chooses the executor with ``settings["executor"]``, a dotted
class path or a class; MotionExecutor is the default. It is constructed as

  Executor(deployment, runtime, options, *, command_driven, home_index, allow,
           selection)

``command_driven`` is True when swings start from RacketCommands (the HOPE
planner) rather than from raw ball estimates; executors that only accept
commands raise for False. ``home_index``, ``allow`` and ``selection`` are the
motion CLI choices; an executor without motions accepts only their defaults.

Per tick HopeController calls, in this order:

  prepare(state, now) -> dict    once after (re)entry: freeze the idle frame;
                                 the dict is logged with the home_frozen event
  swing_finished(now) -> bool    polled while EXECUTE; frames or a deadline
  finish(state) -> dict          end the swing using the current measured state;
                                 logged with swing_finished
  release()                      forget the last selection (hysteresis state)
  choose(state, command, now)    opaque choice or None; sets ``decision``
  begin(choice, now) -> Swing    latch the choice and start swinging
  act(state, now) -> target_q    one 29-D joint target, swing or idle

An executor may implement ``update_command(state, command, now) -> Swing | None``
for opt-in raw-command updates during EXECUTE. None preserves the active swing;
a Swing replaces the controller's plan/command and logs racket_command_updated.
It must preserve actor history. Unsupported executors reject enabling this mode.
MotionExecutor additionally supports ``a3.live_motion_reselection`` (default
False): each fresh live command reruns its configured selector before replacing
the active clip/goal, without starting a new lifecycle swing.

plus reset() on policy entry and manual reset. It also exposes ``home_pose``
(29-D idle posture for positioning, lifecycle and standing checks),
``policy_name``, ``debug`` (per-tick policy diagnostics, logged as
command_debug), ``decision``, ``motion_index``, ``motion_name``,
``motion_files``, ``candidate_indices``, ``candidates(state)`` and
``describe()``. ``choose_from_ball(state, estimate, now)`` is optional and only
used by the legacy ball-estimate target source.
"""

from dataclasses import dataclass
from importlib import import_module

DEFAULT_EXECUTOR = "deploy.motion_20261119.reference.hope_g1.motion_executor.MotionExecutor"


@dataclass
class Swing:
  """An accepted command, as reported to the runtime and event log.

  ``plan`` exposes ``diagnostic`` and ``plan == (racket position, deadline)``,
  e.g. a3_planner.ReceivePlan. ``command`` is the latched RacketCommand, or
  None for ball-estimate swings. ``event`` is merged into motion_selected.
  """

  plan: object
  command: object
  event: dict


def executor_class(deployment):
  spec = deployment.settings.get("executor", DEFAULT_EXECUTOR)
  if not isinstance(spec, str):
    return spec
  module, _, name = spec.rpartition(".")
  return getattr(import_module(module), name)
