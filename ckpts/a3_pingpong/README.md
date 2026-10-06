# A3 ping-pong trained checkpoints

The `.pt` files are local training artifacts and are intentionally excluded
from Git. Keep them in this directory for local playback; this README and the
provenance record document the expected checkpoint set.

These are the final checkpoints from the two 1000-update validation runs on
2026-09-29. Source paths and SHA256 hashes are in `provenance.json`.

| File | Purpose |
| --- | --- |
| `model_999.pt` | Stage 2 PPO receiving policy: joint residuals and reference speed |
| `tracker.pt` | Frozen Stage 1 JIT tracker required by the receiving policy |
| `tracking_model_999.pt` | Stage 1 PPO checkpoint for standalone motion tracking |

Run from the repository root:

```bash
uv run --no-sync play Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint-file ckpts/a3_pingpong/model_999.pt --num-envs 1
```

Keep `model_999.pt` and `tracker.pt` together. The loader checks the frozen
tracker hash before restoring the receiving checkpoint. The task also uses the
local motions in `dataset/a3_pingpong/motions`.

See [training and playback instructions](../../docs/A3_PINGPONG.md) and the
[validation report](../../docs/A3_PINGPONG_VALIDATION.md) for measured results,
curves, ablations, and scope limitations.
