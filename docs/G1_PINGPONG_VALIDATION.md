# G1 乒乓球任务验证记录

本文记录本机已经完成并保存证据的 G1 两阶段训练与评估结果。Stage 1 和 Stage 2
验收运行各完成 1000 次 PPO 更新：Stage 1 最终 checkpoint 完成全部 43 段动作；
Stage 2 在 172 次来球中击中 166 球、完成 139 次有效回球。两个阶段均完成短程恢复
训练检查，Stage 2 的 5 秒 play 视频已经保存和检查。另有 30000 次更新的 Stage 1
长训练正在后台运行；本文只将它记录为进行中，不把早期状态当作最终结果。

## 数据与评估范围

使用 43 段 G1 动作，其中反手 31 段、正手 12 段。每段 94 帧、50 Hz，首末帧
相隔 1.86 秒；名义击球标记均为第 43 帧、0.86 秒。处理后的动作将初始 pelvis
XY 平移到 `(0, 0)`，将初始 pelvis yaw 旋转到 0，同时保留根高度、关节运动、
身体运动学和球拍运动学。数据来源、处理约定、文件 SHA-256 与复现方法见
[数据说明](../dataset/g1_pingpong/README.md)、
[处理后清单](../dataset/g1_pingpong/manifest.json)和
[源数据清单](../dataset/g1_pingpong/source_manifest.json)。

任务使用独立的 29-DoF G1 模型
[g1.xml](../src/mjlab/asset_zoo/robots/unitree_g1_pingpong/g1.xml)，不能与仓库原有的
27-DoF G1 网球模型混用。43 段源动作的独立验收记录为 43/43 通过；处理脚本又在
首帧、击球帧和末帧共检查 129 个 MuJoCo FK 状态，最大身体位置误差为
`8.73e-8 m`，最大身体四元数 L2 误差为 `8.27e-8`。模型和资产校验信息见
[资产清单](../src/mjlab/asset_zoo/robots/unitree_g1_pingpong/asset_manifest.json)。

下面的训练和逐段评估使用同一组 43 段动作，没有留出测试集。结果只说明已知动作库
上的仿真跟踪行为，不构成对未见动作、未知来球或真实机器人的泛化证明。Stage 1
也不包含来球结果，因此不能从这些结果推导击球率、过网率或有效回球率。

## Stage 1：随机参考速度跟踪

验收运行：
[2026-10-03_22-05-04_g1_stage1_acceptance](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-05-04_g1_stage1_acceptance/)。
实际配置为 1024 个环境、1000 次 PPO 更新、每次 24 步、seed=42，每 250 次更新
保存一次 checkpoint。MuJoCo 仿真步长为 0.005 秒，动作 decimation 为 4，因此
物理控制周期为 0.02 秒。参考动作在每个控制步随机推进 0.01--0.04 秒，并使用失败
自适应初始相位采样；奖励按固定物理控制周期积分，不使用参考时间折扣。
[环境配置](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-05-04_g1_stage1_acceptance/params/env.yaml)与
[训练配置](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-05-04_g1_stage1_acceptance/params/agent.yaml)
随运行保存。

TensorBoard 前 100 次更新（0--99）与末 100 次更新（900--999）的均值如下：

| 指标 | 前 100 次 | 末 100 次 |
| --- | ---: | ---: |
| 平均回报 | 2.6274 | 4.9725 |
| 平均 episode length，控制步 | 39.7704 | 48.3749 |
| 终止时 anchor 位置误差，m | 0.19336 | 0.04219 |
| 终止时身体位置误差，m | 0.12513 | 0.02837 |
| 终止时关节位置误差，29 关节 L2 范数 | 1.20912 | 0.64091 |
| 从采样初始相位到末帧的完成比例 | 0.98905 | 1.00000 |

全部记录标量均为有限值。末 100 次更新中的 anchor-height、anchor-orientation 和
feet-height 失败终止日志均为 0。原始统计见
[training_summary.json](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-05-04_g1_stage1_acceptance/validation/training_summary.json)，
曲线见 [PNG](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-05-04_g1_stage1_acceptance/validation/training_curves.png)
或 [PDF](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-05-04_g1_stage1_acceptance/validation/training_curves.pdf)。

训练并非单调改善。约在 iteration 400--630 出现中段失稳，value loss 的全程峰值
为 `39.9122`，位于 iteration 458；对应的 `model_500` 在后述整段评估中 43 段
全部因 anchor-height 失败。约 iteration 700 后完成率与跟踪误差恢复，最终 100 次
更新保持完整完成率且没有上述失败终止。这个过程说明不能只凭单个训练标量选择
checkpoint，必须结合从首帧开始的逐段回放。

训练中的约 48 步平均回合长度是合理量级：训练会从动作中段采样初始相位，参考时间
平均每步推进 0.025 秒，末帧后还有短暂保持。它不能直接和 94 个参考帧或 4 秒安全
上限比较。`completion_fraction` 也只衡量采样起点之后的剩余片段，不是从首帧开始
完成整段动作的成功率。`Episode_Termination/*` 是每个控制步终止环境数的日志均值，
不是终止概率，并且不同终止项可能重叠。

### 固定速度、从首帧开始的逐段评估

每段动作分配一个环境，seed=42，从第 0 帧开始，参考速度固定为 1；关闭观测噪声、
重置扰动和域随机化，只统计每个环境的第一回合。五个已保存 checkpoint 使用相同
动作和协议。`model_0` 是 iteration 0 保存的 checkpoint，不能视为严格未更新过的
随机网络。

| checkpoint | 到达动作末帧 | 完整回合无失败 | 失败终止率 | 平均回报 | 身体相对位置误差，m | 身体全局位置误差，m | 每关节位置 RMSE，rad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `model_0` | 0/43 | 0/43 | 100% | 3.5434 | 0.13401 | 0.17967 | 0.27590 |
| `model_250` | 43/43 | 43/43 | 0% | 10.7714 | 0.03904 | 0.04846 | 0.17968 |
| `model_500` | 0/43 | 0/43 | 100% | 4.9713 | 0.08792 | 0.21880 | 0.28387 |
| `model_750` | 43/43 | 43/43 | 0% | 11.3568 | 0.03196 | 0.04652 | 0.16005 |
| `model_999` | 43/43 | 43/43 | 0% | **11.6165** | **0.02901** | **0.04318** | **0.14539** |

通过 checkpoint 的平均回合均为 107 步、2.14 秒，并以 `motion_finished` 正常结束；
没有到达评估步数上限后仍未结束的回合。`model_500` 的 43 个回合均触发
anchor-height 失败，印证了训练曲线中的中段退化。`model_999` 在完整完成率相同的
候选中取得最高平均回报和最低身体、关节误差，因此是当前 Stage 1 首选；
`model_750` 可保留为回退候选。逐动作原始记录、终止原因和 checkpoint/JIT SHA-256
见 [rollouts.json](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-05-04_g1_stage1_acceptance/validation/rollouts.json)。

以上误差是在每个回合的有效采样上取时间平均，再对 43 段动作取平均，与训练日志中
回合结束状态的误差不是同一统计口径。

### 击球时刻的球拍跟踪局限

整体身体误差较小并不保证击球瞬间的球拍位置、速度和拍面准确。额外诊断让全部动作
从首帧以固定速度 1 运行，在各动作标注的 0.86 秒比较物理球拍与参考球拍；
`model_999` 的 43 段动作在该时刻均仍存活。

| `model_999` 击球诊断指标 | 实测 |
| --- | ---: |
| 球拍中心全局位置误差，平均 / 最大 | 0.1390 / 0.2631 m |
| 对齐 anchor 后的位置误差，平均 | 0.1191 m |
| 实际球拍速度 / 参考速度，平均 | 0.6390 / 2.9451 m/s |
| 球拍速度误差，平均 | 2.7921 m/s |
| 拍面法向误差，平均 / 最大 | 28.31 / 70.17 deg |
| 标注击球时刻球拍中心误差不超过 0.10 m | 11/43 |
| 击球时刻前后 0.30 秒内曾达到 0.10 m | 23/43 |
| 前后 0.30 秒内最近位置误差，平均 | 0.1071 m |

这些结果表明，最终策略对快速球拍运动的跟踪仍明显弱于整体平衡与身体跟踪。
`0.10 m` 只是几何诊断阈值，并非球拍碰撞或回球成功判据；在前后 0.30 秒搜索后
仍有 20/43 段未进入该范围，也不能把全部误差解释为单个控制步的相位偏移。逐动作的
位置、速度、拍面法向和相位诊断见
[strike_diagnostics.json](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-05-04_g1_stage1_acceptance/validation/strike_diagnostics.json)。
Stage 2 必须另行验收真实扫掠碰撞、击球后过网、有效落台和机器人失败终止。

### 复现分析与评估

在仓库根目录运行：

```bash
G1_STAGE1_RUN=logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-05-04_g1_stage1_acceptance
uv run --no-sync python scripts/tools/analyze_a3_training.py "$G1_STAGE1_RUN" --window 100
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Tracking-G1-Stage1-RandomDt \
  --checkpoint \
    "$G1_STAGE1_RUN/model_0.pt" \
    "$G1_STAGE1_RUN/model_250.pt" \
    "$G1_STAGE1_RUN/model_500.pt" \
    "$G1_STAGE1_RUN/model_750.pt" \
    "$G1_STAGE1_RUN/model_999.pt" \
  --output "$G1_STAGE1_RUN/validation/rollouts.json"
uv run --no-sync python scripts/tools/diagnose_a3_strike.py \
  --task Mjlab-PingPong-Tracking-G1-Stage1-RandomDt \
  --checkpoint "$G1_STAGE1_RUN/model_0.pt" "$G1_STAGE1_RUN/model_999.pt" \
  --output "$G1_STAGE1_RUN/validation/strike_diagnostics.json"
```

## Stage 2：接球验证

验收运行：
[2026-10-03_22-15-38_g1_stage2_acceptance](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/)。
使用 Stage 1 的 `modeljit_999.pt` 作为冻结跟踪器，1024 个环境、1000 次 PPO 更新、
每次 24 步、seed=42。Stage 2 在 29 维跟踪器关节动作上学习 29 维残差和 1 维参考
速度动作；普通及非击球臂关节的残差缩放为 0.25，右侧击球臂关节的缩放为 1.0，
参考速度范围为 0.5--2.0。训练来球的速度范围为
3.5--4.5 m/s、竖直速度为 -3.5 m/s、飞行时间为 0.4--0.5 秒，并带位置扰动；
本次训练的额外到球时间扰动为 0。完整参数见
[环境配置](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/params/env.yaml)与
[训练配置](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/params/agent.yaml)。

冻结 tracker 的 SHA-256 为
`5f982343828188d20f1b588b0183e294f8556e6538cdd6eac591d14c307c4a06`。该值在
[Stage 1 JIT](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-05-04_g1_stage1_acceptance/jit/modeljit_999.pt)、
[Stage 2 tracker](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/tracker.pt)、
评估 JSON 和 Stage 2 checkpoint 保存的 `pingpong_tracker_sha256` 中一致；恢复训练
使用的 tracker 也具有相同哈希。

TensorBoard 前 100 次更新（0--99）与末 100 次更新（900--999）的均值如下：

| 指标 | 前 100 次 | 末 100 次 |
| --- | ---: | ---: |
| 平均回报 | 3.8589 | 7.9927 |
| 平均 episode length，控制步 | 62.4744 | 76.7821 |
| 身体位置误差，m | 0.06359 | 0.04374 |
| 关节位置误差，29 关节 L2 范数 | 1.01122 | 1.08159 |
| 训练击球率 | 51.86% | 94.49% |
| 训练过网率 | 7.30% | 79.15% |
| 训练有效回球率 | 3.57% | 68.07% |

全部记录标量均为有限值。末 100 次仍有少量 anchor-height 终止日志，均值为
`0.01708` 个环境/控制步；feet-height 终止均值为 0。训练日志里的 `landing_error`
包含没有落台时的零默认值，因此不能用它判断落点精度，下面只报告逐回合评估中带
明确分母的条件均值。原始统计见
[training_summary.json](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/validation/training_summary.json)，
曲线见 [PNG](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/validation/training_curves.png)
或 [PDF](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/validation/training_curves.pdf)。

### 固定种子的 172 次来球评估

43 段动作各重复 4 次，共 172 个回合；其中反手 31 段、124 个回合，正手 12 段、
48 个回合。每个回合从动作首帧开始，seed=42，关闭观测噪声、重置扰动和域随机化，
保留配置中的来球分布与学习的参考速度，并且只统计每个环境的第一回合。成功要求
实际球拍扫掠碰撞、击球后过网、球落到对方有效台面，且没有机器人失败终止。下表的
比率均以全部 172 次来球为分母。

| checkpoint | 击球 | 击球后过网 | 有效回球 | 机器人失败 |
| --- | ---: | ---: | ---: | ---: |
| `model_0` | 47/172（27.33%） | 0/172 | 0/172 | 0/172 |
| `model_250` | 138/172（80.23%） | 85/172（49.42%） | 38/172（22.09%） | 0/172 |
| `model_500` | 163/172（94.77%） | 135/172（78.49%） | 97/172（56.40%） | 0/172 |
| `model_750` | 167/172（97.09%） | 147/172（85.47%） | 115/172（66.86%） | 0/172 |
| `model_999` | 166/172（96.51%） | 150/172（87.21%） | **139/172（80.81%）** | 0/172 |

`model_999` 的 172 个回合均正常得到球的最终结果，没有回合停在评估上限。平均回合
长度为 76.73 个控制步、1.535 秒，平均回报为 9.3755；参考速度均值/标准差为
0.8753/0.3476。140 个发生落点判定的球，其目标落点误差均值为 0.2704 m；只在
139 个有效回球中统计时为 0.2689 m。这两个条件均值不能解释为全部 172 球的
无条件落点精度。

正反手结果差异较大：

| `model_999` 分组 | 分母 | 击球 | 击球后过网 | 有效回球 | 机器人失败 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 反手 | 124 | 124/124（100%） | 118/124（95.16%） | 117/124（94.35%） | 0/124 |
| 正手 | 48 | 42/48（87.50%） | 32/48（66.67%） | 22/48（45.83%） | 0/48 |

因此总体 80.81% 主要由反手动作贡献；当前结果不能掩盖正手仅 22/48 的有效回球率。
原始逐回合结果见
[rollouts.json](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/validation/rollouts.json)。

Stage 2 的回合通常在球的结果确定后结束，不要求继续播放到参考动作末帧。
`ball_finished` 包含有效落台，是正常任务结束；报告中的 `failure_termination_rate`
还把 `ball.missed` 计为接球失败，不能把它解释成机器人摔倒率。验收应读取
`receiving_task_success_rate` 和独立的 `robot_failure_rate`，不能用
`full_clip_completion_rate` 或 `completed_episode_without_failure_rate` 代替回球成功率。

### 速度、残差与到球时差消融

以下评估均使用同一个 `model_999`、tracker、43 段动作、seed=42 和 4 次重复。
它们是在推理时关闭动作分量的组件消融，不是分别重新训练的基线。每个比率的分母仍为
全部 172 次来球，所有组的机器人失败和未结束回合均为 0。

| 策略 | 击球 | 击球后过网 | 有效回球 |
| --- | ---: | ---: | ---: |
| 冻结跟踪器，速度固定为 1，残差为 0 | 49/172（28.49%） | 0/172 | 0/172 |
| 最终策略，速度固定为 1，保留残差 | 131/172（76.16%） | 101/172（58.72%） | 86/172（50.00%） |
| 最终策略，学习速度，残差置 0 | 37/172（21.51%） | 0/172 | 0/172 |
| 最终策略，学习速度和残差 | 166/172（96.51%） | 150/172（87.21%） | 139/172（80.81%） |

固定速度但保留残差时，有效回球率比完整策略低 30.81 个百分点；去掉残差后，即使
保留学习速度也没有有效回球。固定速度组的 90 个落点判定误差均值为 0.4096 m，
86 个有效回球的条件均值为 0.3712 m。原始报告见
[冻结基线](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/validation/frozen_baseline.json)、
[固定速度](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/validation/fixed_speed.json)和
[零残差](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/validation/zero_residual.json)。

额外将到球时间独立均匀扰动到名义时间的 +/-0.08 秒，使用相同分母和配对来球：

| 策略 | 击球 | 击球后过网 | 有效回球 |
| --- | ---: | ---: | ---: |
| 学习速度和残差 | 162/172（94.19%） | 140/172（81.40%） | 129/172（75.00%） |
| 固定速度，保留残差 | 126/172（73.26%） | 94/172（54.65%） | 75/172（43.60%） |

完整策略在时间扰动下的反手有效回球仍为 117/124（94.35%），正手降为
12/48（25.00%）。完整策略的 129 个落点与有效回球重合，条件落点误差均值为
0.2858 m；固定速度组在 78 个落点上的均值为 0.3812 m，在 75 个有效回球上的
均值为 0.3493 m。见
[自适应速度时间扰动](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/validation/jitter_008_adaptive.json)和
[固定速度时间扰动](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/validation/jitter_008_fixed_speed.json)。

训练没有使用这项额外时间扰动，评估仍复用训练动作库和配置的仿真来球分布。上述
结果只支持有限的 +/-0.08 秒时差测试，不代表对任意来球、未见动作或真实机器人的
泛化。正手样本只有 48 个回合，且表现显著弱于反手，也应作为后续数据与训练重点。

### 复现 Stage 2 评估

```bash
G1_STAGE2_RUN=logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance
uv run --no-sync python scripts/tools/analyze_a3_training.py "$G1_STAGE2_RUN" --window 100
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --checkpoint \
    "$G1_STAGE2_RUN/model_0.pt" \
    "$G1_STAGE2_RUN/model_250.pt" \
    "$G1_STAGE2_RUN/model_500.pt" \
    "$G1_STAGE2_RUN/model_750.pt" \
    "$G1_STAGE2_RUN/model_999.pt" \
  --tracker-file "$G1_STAGE2_RUN/tracker.pt" --repeats 4 \
  --output "$G1_STAGE2_RUN/validation/rollouts.json"
```

固定速度消融添加 `--fixed-speed`，零残差添加 `--zero-residual`，冻结基线同时添加
两者；时间扰动评估添加 `--arrival-time-jitter 0.08`，并为各组指定不同的
`--output`。

## 恢复训练验证

Stage 1 和 Stage 2 都使用 16 个环境，从各自验收运行的 `model_999.pt` 恢复并继续
2 次 PPO 更新；恢复运行的 `Train/*` 标量步号为 999 和 1000，并成功保存
`model_1000.pt` 与 `jit/modeljit_1000.pt`：

- [Stage 1 resume run](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-33-53_resume_check_16env/)：
  `load_run=2026-10-03_22-05-04_g1_stage1_acceptance`，
  `load_checkpoint=model_999.pt`。
- [Stage 2 resume run](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-34-00_resume_check_16env/)：
  `load_run=2026-10-03_22-15-38_g1_stage2_acceptance`，
  `load_checkpoint=model_999.pt`；同目录 tracker 的 SHA-256 与 checkpoint 内记录均为
  `5f982343828188d20f1b588b0183e294f8556e6538cdd6eac591d14c307c4a06`。

这两次运行验证了 checkpoint、优化器状态、环境状态和 JIT 导出路径可以继续使用。
它们只有 16 个环境和 2 次更新，属于恢复与保存 smoke test，不用于比较策略质量或
替代完整评估。

## Play 视频

Stage 2 验收运行的 [5 秒 play 视频](../logs/rsl_rl/g1_pingpong_receive_stage2/2026-10-03_22-15-38_g1_stage2_acceptance/videos/play/rl-video-step-0.mp4)
已经播放检查。`ffprobe` 报告 H.264、320x240、50 fps、250 帧，视频流和容器时长均为
5.000 秒。该视频用于检查一次可视化运行和文件可播放性；单段视频不能替代上面的
172 回合统计，也不能证明未见来球或真实机器人表现。

## 长训练

30000 次更新的 Stage 1 长训练已经在 tmux 会话 `g1_pingpong_stage1_30k` 中后台启动：
[2026-10-03_22-36-38_g1_bvh_29dof_30k_save2000_seed42](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/2026-10-03_22-36-38_g1_bvh_29dof_30k_save2000_seed42/)。
配置为 4096 个环境、30000 次更新、每 2000 次保存、seed=42，使用全部 43 段动作、
随机参考 `dt` 和失败自适应采样。控制台输出重定向到
[tmux 日志](../logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/g1_bvh_29dof_30k_save2000_seed42.tmux.log)。

启动后的早期审计快照位于 iteration 76：tmux 会话仍存活，GPU 显存约 2880 MiB，
吞吐约 134k steps/s，未见 OOM 或非有限标量；启动时预计总耗时约 6 小时 45 分钟。
这些数字只证明进程在早期正常推进。训练仍在进行中，尚未形成最终曲线、checkpoint
比较或逐段评估，不能宣称 30000 次长训练已经完成或通过验收。完成后仍需按本文
Stage 1 的同一协议分析完整 TensorBoard、比较保存的 checkpoint，并运行 43 段首帧
固定速度评估和击球时刻诊断。
