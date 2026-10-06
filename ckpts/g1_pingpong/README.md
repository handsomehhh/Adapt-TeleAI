# G1 ping-pong validation checkpoints

The `.pt` files are local training artifacts and are intentionally excluded
from Git. Keep them in this directory for local playback; this README and the
provenance record document the expected checkpoint set.

These files are the selected checkpoints from the two 1000-update G1
validation runs on 2026-10-03. Exact sources and SHA-256 hashes are recorded in
`provenance.json`.

| File | Purpose |
| --- | --- |
| `model_999.pt` | Stage 2 receiving policy: 29 joint residuals and reference speed |
| `tracker.pt` | Frozen Stage 1 JIT tracker required by the Stage 2 policy |
| `tracking_model_999.pt` | Stage 1 PPO checkpoint for standalone motion tracking |

Play the receiving policy from the repository root:

```bash
uv run --no-sync play Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --checkpoint-file ckpts/g1_pingpong/model_999.pt \
  --num-envs 1
```

Keep `model_999.pt` and `tracker.pt` together. The loader validates the frozen
tracker hash stored in the Stage 2 checkpoint. The task also requires the local
motions under `dataset/g1_pingpong/motions`.

See [training and playback instructions](../../docs/G1_PINGPONG.md) and the
[validation report](../../docs/G1_PINGPONG_VALIDATION.md) for measured curves,
fixed-seed rollouts, and adaptive-policy ablations.
