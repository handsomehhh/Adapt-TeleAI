# A3 乒乓球任务验证记录

本文记录本机实际训练和评估结果。两个阶段各完成 1000 次 PPO 更新：Stage 1
整段动作跟踪 61/61 完成；Stage 2 在 244 次来球中击球 244 次、有效回球 198 次。
本结果限于下面注明的动作库和仿真来球分布，不代表论文全部实验已复现。

## 数据与评估范围

使用用户提供的全部 61 段动作，每段 94 帧、50 Hz，首末帧相隔 1.86 秒。
训练输入统一为初始 pelvis XY=0、yaw=0，保留高度和运动学数据；原始文件、
SHA-256、坐标约定及复现命令见[数据说明](../dataset/a3_pingpong/README.md)与
[数据清单](../dataset/a3_pingpong/manifest.json)。

183 个首帧/击球帧/末帧的 MuJoCo FK 对照通过：非脚踝最大位置误差
4.52e-6 m，四元数 L2 误差 9.08e-7；脚踝约 1.5 mm 差异来自原始运动数据
省略的 URDF 局部偏移。名义击球标记为第 43 帧、0.86 秒，没有实测球轨迹标签。

下面的整段评估复用了这 61 段训练动作，没有留出测试集。结果说明已知动作库上的
训练与运行行为，不构成未见动作、真实机器人或未知来球的泛化证明。

## Stage 1：采用的物理时间训练

验收运行：[2026-09-29_00-53-35_physical_time](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-53-35_physical_time/)。
实际配置为 1024 环境、1000 次 PPO 更新、每次 24 步、seed=42；物理仿真步长
0.005 秒、控制周期 0.02 秒。参考动作仍随机推进 0.01–0.04 秒，并使用失败自适应
初始相位采样；奖励积分和 GAE 折扣按固定物理控制周期计算。
[环境配置](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-53-35_physical_time/params/env.yaml)、
[训练配置](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-53-35_physical_time/params/agent.yaml)
均随日志保存。

TensorBoard 前 100 次更新（0–99）与末 100 次更新（900–999）的均值：

| 指标 | 前 100 次 | 末 100 次 |
| --- | ---: | ---: |
| 平均回报 | 2.3741 | 4.8318 |
| 平均 episode length，控制步 | 43.2720 | 48.1972 |
| 终止时身体相对位置误差，m | 0.14724 | 0.03991 |
| 终止时关节位置误差，31 关节 L2 范数 | 1.36161 | 0.84518 |
| 从采样初始相位到末帧的完成比例 | 0.99280 | 1.00000 |

全部已记录标量为有限值。原始数值见
[training_summary.json](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-53-35_physical_time/validation/training_summary.json)，
曲线见 [PNG](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-53-35_physical_time/validation/training_curves.png)
或 [PDF](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-53-35_physical_time/validation/training_curves.pdf)。

训练长度约 48 步是预期行为：训练经常从动作中段开始，随机参考推进量平均为
0.025 秒，结束后还有短暂末帧保持。它不能直接与 94 个参考帧或 4 秒安全时限比较。
训练中的完成比例也只衡量采样后的剩余片段。`Episode_Termination/*` 是每个控制步
终止环境数的日志平均，不是概率；多个终止原因可以同时发生。

### 固定速度、从首帧开始的整段评估

每段动作分配一个环境，同 seed=42，从第 0 帧开始，参考速度固定为 1；关闭观测、
重置和域随机噪声，仅统计每个环境的第一回合。两个 checkpoint 使用相同动作和协议。
`model_0` 是 iteration 0 保存的初始 checkpoint，不能视为严格未更新过的网络。

| 指标 | model_0 | model_999 |
| --- | ---: | ---: |
| 到达参考动作末帧 | 6/61（9.84%） | 61/61（100%） |
| 包含末帧保持的完整回合无失败 | 2/61（3.28%） | 61/61（100%） |
| 失败终止率 | 96.72% | 0% |
| 平均回合长度 | 79.18 步 / 1.584 秒 | 107 步 / 2.140 秒 |
| 平均回报 | 3.7867 | 11.1912 |
| 身体相对位置误差，m | 0.17155 | 0.04129 |
| 身体全局位置误差，m | 0.19706 | 0.05237 |
| 每关节位置 RMSE，rad | 0.29749 | 0.18182 |

以上跟踪误差为各回合有效采样的时间平均，再对动作取平均；与训练日志中的终止状态
指标口径不同。详细逐动作结果见
[rollouts.json](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-53-35_physical_time/validation/rollouts.json)。
该结果支持 Stage 1 训练、存档、加载及整段跟踪正常，未出现持续极短回合问题。

### 击球部位仍存在的限制

身体平均误差小，不能保证击球瞬间球拍准确。额外诊断在每段动作的物理时间
0.86 秒比较右腕安装点推算的球拍中心、速度和拍面法向；全部 61 条动作当时均存活。

| model_999 的击球时刻指标 | 实测 |
| --- | ---: |
| 球拍中心全局位置误差，平均 / 最大 | 0.1537 / 0.3019 m |
| 对齐 anchor 后的位置误差，平均 | 0.1343 m |
| 实际球拍速度 / 参考速度，平均 | 0.9218 / 3.0804 m/s |
| 拍面法向误差，平均 | 38.34° |
| 击球时球拍中心误差不超过 0.10 m | 12/61 |

这反映出快速手臂和拍面跟踪仍弱于整体平衡跟踪。0.10 m 阈值只是几何诊断，
并非实际碰撞或回球成功判据。即使在击球时刻前后 0.3 秒搜索，最近球拍位置误差
仍平均为 0.1132 m；不能把所有误差归因于单个控制步的时间偏移。
逐动作实际/参考位置、速度与相位诊断见
[strike_diagnostics.json](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-53-35_physical_time/validation/strike_diagnostics.json)。
Stage 2 必须独立检查真实扫掠碰撞、过网与有效落台结果。

### 未采用的参考时间缩放试验

保留诊断运行
[2026-09-29_00-43-48_validation](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-43-48_validation/)。
该运行采用了参考时间奖励缩放及当时的参数化折扣路径；虽然后期回合长度并未塌缩，
前/末 100 次更新的平均回报由 2.3725 降至 1.4568，身体位置误差由 0.1864 增至
0.2200 m。整段评估中，999 的平均回报 3.5939、身体相对误差 0.1896 m，弱于
该运行 250 的 5.3921、0.1638 m。
[曲线统计](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-43-48_validation/validation/training_summary.json)
与[逐动作评估](../logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-43-48_validation/validation/rollouts.json)
均保留，不能用这个退化运行宣称收敛。当前验收采用上面的固定物理时间方案；
随机参考速度与自适应片段采样仍然启用。两个运行不足以构成折扣机制的完整消融研究。

### 复现分析与评估

在仓库根目录运行：

```bash
A3_STAGE1_RUN=logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/2026-09-29_00-53-35_physical_time
uv run --no-sync python scripts/tools/analyze_a3_training.py "$A3_STAGE1_RUN" --window 100
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --checkpoint "$A3_STAGE1_RUN/model_0.pt" "$A3_STAGE1_RUN/model_999.pt" \
  --output "$A3_STAGE1_RUN/validation/rollouts.json"
uv run --no-sync python scripts/tools/diagnose_a3_strike.py \
  --checkpoint "$A3_STAGE1_RUN/model_250.pt" "$A3_STAGE1_RUN/model_750.pt" "$A3_STAGE1_RUN/model_999.pt" \
  --output "$A3_STAGE1_RUN/validation/strike_diagnostics.json"
```

## Stage 2：接球验证

验收运行：[2026-09-29_01-03-16_scheduled_intercept](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/)。
使用上面 Stage 1 的 `modeljit_999.pt` 作为冻结跟踪器，1024 环境、1000 次 PPO 更新、
每次 24 步、seed=42。Stage 2 学习 31 维关节残差和 1 维参考速度动作。
默认来球速度 3.5–4.5 m/s，带位置扰动；名义到球时间由参考击球时刻确定，
本次训练没有额外的到球时间扰动。完整配置保存在该运行的 `params/` 中。

TensorBoard 前 100 次和末 100 次更新的均值：

| 指标 | 前 100 次 | 末 100 次 |
| --- | ---: | ---: |
| 平均回报 | 3.5610 | 7.4966 |
| 平均 episode length，控制步 | 62.8628 | 76.9053 |
| 终止时身体相对位置误差，m | 0.07940 | 0.05629 |
| 终止时关节位置误差，31 关节 L2 范数 | 1.24707 | 1.20174 |

已记录标量均为有限值。见
[training_summary.json](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/validation/training_summary.json)、
[曲线 PNG](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/validation/training_curves.png)
与 [PDF](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/validation/training_curves.pdf)。

### 固定种子的 244 次来球评估

全部 61 段动作各重复 4 次，从动作首帧开始，seed=42；关闭观测、重置和域随机噪声，
保留来球分布与学习的参考速度，仅统计每个环境的第一回合。
成功要求实际球拍扫掠碰撞、击球后过网、落到对方有效台面，且没有机器人失败终止。
所有比率的分母均为 244 次来球，并非仅统计已经击中的球。

| 指标 | model_0 | model_999 |
| --- | ---: | ---: |
| 击球 | 24/244（9.84%） | 244/244（100%） |
| 击球后过网 | 0/244 | 214/244（87.70%） |
| 有效回球 | 0/244 | 198/244（81.15%） |
| 机器人失败终止 | 0/244 | 0/244 |
| 超时或评估上限未结束 | 0/244 | 0/244 |
| 平均回合长度 | 57.02 步 / 1.140 秒 | 76.34 步 / 1.527 秒 |
| 平均回报 | 3.1100 | 8.9300 |
| 参考速度均值 / 标准差 | 0.9665 / 0.0704 | 0.8716 / 0.3511 |

最终模型的 46 次失败均为回球结果失败，没有机器人失稳；其中 30 球没有完成过网，
另 16 球过网后未在对方有效区域落台。中间 `model_500` 的有效回球率为 30.33%，
最终提升至 81.15%。原始逐回合记录见
[rollouts.json](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/validation/rollouts.json)
和 [rollouts_500.json](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/validation/rollouts_500.json)。

最终策略在发生落点判定的 200 球中，平均目标落点误差为 **0.3622 m**；
仅在 198 个有效回球中统计则为 **0.3595 m**。这不是所有来球的无条件精度。
报告中的对应字段是 `landing_error_mean_given_landing_m` 和
`landing_error_mean_given_success_m`，同时保存分母。原始训练的 `landing_error`
包含没有落台的零初始值，不能直接用于落点精度结论。

有效落台后的 `ball_finished` 是正常结束。接球回合通常在球的结果确定后结束，
不要求继续播放到参考动作末帧；因此 Stage 2 应读取 `receiving_task_success_rate`，
不能拿 `completed_episode_without_failure_rate` 当作回球成功率。
约 77 步的训练长度与单次来球的物理时长一致，没有持续极短回合现象。

### 速度适配与残差的作用

以下使用相同的动作、seed、来球配置和 244 回合协议，在推理时关闭相应动作分量。
这属于已训练策略的组件消融，不是分别重新训练各基线的比较。

| 策略 | 击球率 | 过网率 | 有效回球率 |
| --- | ---: | ---: | ---: |
| 冻结跟踪器，速度固定为 1，残差为 0 | 19.26% | 0% | 0% |
| 最终策略，速度固定为 1，保留残差 | 84.02% | 56.56% | 37.30% |
| 最终策略，学习速度，残差置 0 | 23.36% | 0% | 0% |
| 最终策略，学习速度和残差 | 100% | 87.70% | 81.15% |

各组均无机器人失败终止。速度适配对这项接球任务有实际贡献：
保留同一个残差策略、将速度固定后，有效回球率下降约 43.9 个百分点；
同时，仅靠速度也不能弥补冻结跟踪器的拍面和位置误差。
原始报告：[冻结基线](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/validation/frozen_baseline.json)、
[固定速度](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/validation/fixed_speed.json)、
[零残差](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/validation/zero_residual.json)。

额外将来球到达时间独立均匀扰动 ±0.08 秒，使用相同种子的配对来球：

| 策略 | 击球率 | 过网率 | 有效回球率 |
| --- | ---: | ---: | ---: |
| 最终策略，学习速度和残差 | 95.90% | 75.82% | 68.44%（167/244） |
| 最终策略，固定速度，保留残差 | 80.33% | 55.33% | 40.98%（100/244） |

两组仍均无机器人失败。有效回球条件下的目标误差分别为 0.3782 m（167 球）和
0.4659 m（100 球）。见
[时间扰动与自适应策略](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/validation/jitter_008_adaptive.json)、
[时间扰动与固定速度](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/validation/jitter_008_fixed_speed.json)。
训练未使用这项额外时差扰动，但评估仍采用同一动作库和相近来球分布；这只是有限的
时间扰动测试，不代表任意来球、未见挥拍动作或真实机器人泛化。

### 复现与交付

```bash
A3_STAGE2_RUN=logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept
uv run --no-sync python scripts/tools/analyze_a3_training.py "$A3_STAGE2_RUN" --window 100
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint "$A3_STAGE2_RUN/model_0.pt" "$A3_STAGE2_RUN/model_999.pt" \
  --tracker-file "$A3_STAGE2_RUN/tracker.pt" --repeats 4 \
  --output "$A3_STAGE2_RUN/validation/rollouts.json"
```

固定速度消融添加 `--fixed-speed`，零残差添加 `--zero-residual`，冻结基线同时添加二者。
时间扰动测试添加 `--arrival-time-jitter 0.08`，并使用不同的 `--output` 保存报告。

最终模型已复制到 [ckpts/a3_pingpong](../ckpts/a3_pingpong/)，来源与 SHA256 记录于
`provenance.json`。`model_999.pt` 与 `tracker.pt` 需要一起保存。
最终模型的 [8 秒 play 视频](../logs/rsl_rl/a3_pingpong_receive_stage2/2026-09-29_01-03-16_scheduled_intercept/videos/play/rl-video-step-0.mp4)
已经录制并检查，任务使用项目标准 play 入口。

两个阶段均实际验证了从最终 checkpoint 恢复并继续 2 次 PPO 更新。相关单元与回归测试
共 **64 项通过**，覆盖机器人与动作映射、动作数据、跟踪器依赖、球拍/台面/球网事件、
评估终止统计、play 默认数据路径、参数化 GAE timeout 与原有 PPO/storage 行为：

```bash
uv run --no-sync pytest \
  tests/test_a3_assets.py tests/test_a3_motion.py tests/test_a3_tracking.py \
  tests/test_a3_ball.py tests/test_a3_tracker_dependency.py \
  tests/test_a3_motion_evaluation.py tests/test_a3_play.py \
  tests/test_parametric_gae_timeout.py \
  src/rsl_rl_lib/tests/algorithms/test_ppo.py \
  src/rsl_rl_lib/tests/storage/test_rollout_storage.py -q
```

新增模块通过 Ruff 和类型检查。未声称复现论文的 MVAE 运动生成器、旋转球模型和实机实验；
当前可验收的是本地 A3 两阶段训练、随机速度跟踪、失败自适应采样，以及冻结跟踪器上的
速度适配和残差接球策略。
