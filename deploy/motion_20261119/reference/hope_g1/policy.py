"""Actor-only CPU inference for the original HOPE RSL-RL checkpoint format.

``checkpoint`` may be the training ``model_N.pt`` or the ``model_N.onnx`` written
next to it by hope_training's ``scripts/export_deploy_onnx.py``. The ONNX graph
contains normalizer + actor; action clipping still comes from params/agent.yaml.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ONNX_FORMAT = "hope_deploy_actor_v1"


class HopePolicy:
  input_dim = 175
  output_dim = 29
  actor_class = "ActorCritic"
  normalizer_eps = 1e-2

  def __init__(self, checkpoint, agent, *, input_dim=None):
    # Dual-racket HOPE appends a 1-D racket_hand term (176-D actor input).
    if input_dim is not None:
      self.input_dim = int(input_dim)
    cfg = agent["policy"]
    if cfg["class_name"] != self.actor_class or cfg["activation"] != "elu":
      raise ValueError(f"Expected HOPE feed-forward {self.actor_class} with ELU")
    self.clip = agent.get("clip_actions")
    if self.clip is not None and (not np.isfinite(self.clip) or self.clip <= 0):
      raise ValueError("Invalid runner action clipping")
    if Path(checkpoint).suffix == ".onnx":
      self.backend = "onnx"
      self._infer = self.load_onnx(checkpoint, agent)
    else:
      self.backend = "torch"
      self._infer = self.load_torch(checkpoint, agent)
    self(np.zeros(self.input_dim, dtype=np.float32))

  def load_torch(self, checkpoint, agent):
    cfg = agent["policy"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = {
      k.removeprefix("actor."): v
      for k, v in saved["model_state_dict"].items()
      if k.startswith("actor.")
    }
    dims = [self.input_dim, *cfg["actor_hidden_dims"], self.output_dim]
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
      layers.append(nn.Linear(a, b))
      if i < len(dims) - 2:
        layers.append(nn.ELU())
    self.actor = nn.Sequential(*layers).eval()
    self.actor.load_state_dict(weights, strict=True)
    self.mean, self.std, self.eps = self.normalizer(saved, agent)
    if (
      self.mean.shape != (1, self.input_dim)
      or self.std.shape != self.mean.shape
      or not torch.isfinite(self.mean).all()
      or not torch.isfinite(self.std).all()
      or torch.any(self.std < 0)
      or any(not torch.isfinite(w).all() for w in self.actor.parameters())
    ):
      raise ValueError("Invalid actor/normalizer tensors")
    self.iteration = int(saved["iter"])

    def infer(x):
      with torch.inference_mode():
        normalized = (torch.from_numpy(x[None]) - self.mean) / (self.std + self.eps)
        return self.actor(normalized).numpy()[0].copy()

    return infer

  def load_onnx(self, checkpoint, agent):
    import onnxruntime as ort

    options = ort.SessionOptions()
    # One 175x512x256x128x29 MLP per tick: threading only adds wake-up latency.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
      str(checkpoint), sess_options=options, providers=["CPUExecutionProvider"]
    )
    meta = session.get_modelmeta().custom_metadata_map
    cfg = agent["policy"]
    expected = {
      "format": ONNX_FORMAT,
      "policy_class": self.actor_class,
      "input_dim": str(self.input_dim),
      "output_dim": str(self.output_dim),
    }
    for key, value in expected.items():
      if meta.get(key) != value:
        raise ValueError(f"ONNX {checkpoint}: metadata {key}={meta.get(key)!r}, expected {value!r}")
    if json.loads(meta["actor_hidden_dims"]) != list(cfg["actor_hidden_dims"]):
      raise ValueError(f"ONNX {checkpoint}: actor_hidden_dims differ from params/agent.yaml")
    if float(meta["normalizer_eps"]) != self.normalizer_eps:
      raise ValueError(f"ONNX {checkpoint}: normalizer eps differs from this policy contract")
    if meta.get("experiment_name") != str(agent.get("experiment_name")):
      raise ValueError(f"ONNX {checkpoint}: exported from a different experiment")
    (inp,), (out,) = session.get_inputs(), session.get_outputs()
    if inp.shape[-1] != self.input_dim or out.shape[-1] != self.output_dim:
      raise ValueError(f"ONNX {checkpoint}: graph shape {inp.shape}->{out.shape}")
    self.iteration = int(meta["iteration"])
    self.session = session
    input_name, output_names = inp.name, [out.name]
    run = session.run

    def infer(x):
      return run(output_names, {input_name: x[None]})[0][0].copy()

    return infer

  def normalizer(self, saved, agent):
    if not agent.get("empirical_normalization"):
      raise ValueError("This HOPE adapter requires saved empirical normalization")
    norm = saved["obs_norm_state_dict"]
    # rsl_rl.modules.normalizer.EmpiricalNormalization default in old HOPE.
    return norm["_mean"].float(), norm["_std"].float(), self.normalizer_eps

  def __call__(self, observation):
    x = np.asarray(observation, dtype=np.float32)
    if x.shape != (self.input_dim,) or not np.isfinite(x).all():
      raise ValueError(f"HOPE requires a finite {self.input_dim}-dimensional observation")
    action = self._infer(x)
    if self.clip is not None:
      action = np.clip(action, -self.clip, self.clip)
    if action.shape != (29,) or not np.isfinite(action).all():
      raise ValueError("Invalid HOPE actor output")
    return action


class HitterPolicy(HopePolicy):
  """G1 hitter's deterministic actor; exploration std is training-only.

  This run uses RSL-RL's policy-owned normalization switch, not the deprecated
  runner switch. Its saved actor_obs_normalization={} is false and the actor
  has no normalizer tensors: inference is Identity -> actor, NOT /(1+0.01).
  Reject other formats rather than silently omitting normalization.
  """

  input_dim = 104
  actor_class = "BoundedStdActorCritic"
  normalizer_eps = 0.0

  def normalizer(self, saved, agent):
    cfg = agent["policy"]
    if (
      "actor_obs_normalization" not in cfg
      or cfg["actor_obs_normalization"]
      or cfg.get("state_dependent_std", False)
      or cfg.get("entropy_excluded_action_indices")
      or "obs_norm_state_dict" in saved
      or any("normalizer" in k for k in saved["model_state_dict"])
    ):
      raise ValueError("Unsupported hitter normalization/passive-action contract")
    return torch.zeros(1, self.input_dim), torch.ones(1, self.input_dim), 0.0
