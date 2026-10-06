# A3 PingPong 训练与推理指南

本文说明如何在本仓库中训练、恢复、播放和评估 A3 乒乓球策略，并解释两阶段策略在每个控制步中如何工作。命令均从仓库根目录 `/home/lenovo/lxz/AdaPT` 执行。

相关资料：

- [任务设计与 AdaPT 对应关系](A3_PINGPONG.md)
- [已完成训练的验证结果](A3_PINGPONG_VALIDATION.md)
- [动作数据说明](../dataset/a3_pingpong/README.md)
- [交付 checkpoint 说明](../ckpts/a3_pingpong/README.md)

## 1. 任务结构

A3 PingPong 采用两阶段训练：

1. **Stage 1：随机速度动作跟踪。**策略根据参考动作和机器人状态，输出 31 个关节位置控制量。训练时参考动作每步以 0.5～2 倍速度推进，使跟踪器能够执行快慢不同的挥拍。
2. **Stage 2：自适应接球。**加载并冻结 Stage 1 跟踪器；新的高层策略观察球和动作进度，输出 31 个关节残差以及 1 个参考速度动作。

整体数据流为：

```text
61 段参考动作
      │
      ▼
Stage 1 跟踪器：参考姿态 + 机器人状态 + 参考时间增量
      │ 输出 31 维基础关节动作
      │ 冻结为 tracker.pt / modeljit_N.pt
      ▼
Stage 2 接球策略：球状态 + 未来球轨迹 + 动作相位 + 上一步高层动作
      │ 输出 31 维残差 + 1 维速度
      ▼
基础动作 + 关节残差 → A3 PD 位置控制
速度动作 → 下一控制步的参考动作推进量
```

这是一项固定动作库上的 AdaPT 迁移。它没有实现论文回球分支的 MVAE 自回归动作生成器，也不包含真实机器人部署链路。

## 2. 环境与数据准备

### 2.1 安装依赖

```bash
cd /home/lenovo/lxz/AdaPT
uv sync
```

Linux 默认安装 CUDA 版本的 PyTorch。训练前可确认 CUDA 可用：

```bash
uv run python -c "import torch; ok=torch.cuda.is_available(); print(ok, torch.cuda.get_device_name(0) if ok else 'CUDA unavailable')"
```

### 2.2 动作数据

训练数据应位于：

```text
dataset/a3_pingpong/motions/
```

当前数据包含 61 段动作，每段 94 帧、50 Hz，首末帧相隔 1.86 秒。每个文件包含 31 个命名关节、32 个命名身体，以及名义击球帧和球拍运动信息；任务从中选取 14 个关键身体计算身体跟踪观测、奖励和误差。

如需从保留的原始数据重新生成：

```bash
uv run --no-sync python scripts/tools/prepare_a3_motion.py \
  --source dataset/a3_pingpong/raw \
  --output dataset/a3_pingpong/motions \
  --xml src/mjlab/asset_zoo/robots/agibot_a3/xmls/a3_pingpong.xml
```

处理会将首帧 pelvis 的 XY 平移到零，并将初始 yaw 对齐到世界坐标 +X；位置、姿态和世界速度会一起变换。训练配置使用 `align_heading_to_frame="none"`，不会再次旋转数据。

## 3. 时间系统

理解两种时间是训练和推理的关键：

| 时间 | 含义 | 默认推进方式 |
| --- | --- | --- |
| 物理时间 | MuJoCo、机器人和球实际经过的时间 | 每个控制步固定 0.02 s |
| 参考时间 | 参考动作播放到的位置 | 每步推进 0.01～0.04 s |

MuJoCo 物理步长为 0.005 s，`decimation=4`，所以一个策略控制步包含 4 个物理子步：

```text
4 × 0.005 s = 0.02 s
```

定义参考倍速为 `speed`，则下一步的参考时间增量为：

```text
reference_dt = 0.02 × speed
```

`speed=0.5/1.0/2.0` 分别对应参考动作每步推进 `0.01/0.02/0.04 s`。无论参考动作如何变速，球和机器人仍按固定物理时间前进。

## 4. Stage 1：随机速度动作跟踪

### 4.1 训练目标

Stage 1 学习一个低层跟踪器：

```text
tracker_observation → 31 维关节动作
```

它需要在不同参考播放速度下保持身体稳定，并跟踪整套挥拍动作。训练时：

- 61 段动作随机选择；
- 每个控制步随机采样 `reference_dt ∈ [0.01, 0.04] s`；
- 动作时间线使用连续插值，关节和位置线性插值，身体四元数使用 SLERP；
- 初始动作相位根据历史失败位置自适应采样；
- 机器人关节、初始 roll/pitch 和编码器偏置带少量扰动。

### 4.2 Actor 观测

Stage 1 actor 的观测共 140 维：

| 观测项 | 维数 | 含义 |
| --- | ---: | --- |
| `command` | 31 | 当前参考关节位置 |
| `motion_anchor_pos_b` | 3 | 参考躯干相对机器人基座的位置 |
| `motion_anchor_ori_b` | 6 | 参考躯干相对朝向的 6D 表示 |
| `projected_gravity` | 3 | 基座坐标系中的重力方向 |
| `base_ang_vel` | 3 | 基座角速度 |
| `joint_pos` | 31 | 当前关节位置相对默认姿态的偏移 |
| `joint_vel` | 31 | 当前关节速度 |
| `actions` | 31 | 上一步关节动作 |
| `motion_dt_sample` | 1 | 产生当前参考姿态的时间增量相对 0.02 s 的差值 |

`motion_dt_sample=-0.01/0/+0.02` 分别对应本次参考以 0.5/1/2 倍速推进。Actor 因而知道当前追踪目标是以多快的节奏生成的。

Critic 还能读取基座线速度和更完整的身体姿态信息；这些额外信息只用于训练 value function，不进入部署用 actor。

### 4.3 动作和控制目标

Actor 输出 31 个关节动作。每个输出经过对应关节的 `action scale`，再叠加默认关节姿态，形成 PD 控制器的位置目标：

```text
joint_target = default_joint_position + action_scale × actor_action
```

关节顺序由 `A3_JOINT_NAMES` 明确指定，并按名称和动作文件匹配，不能依赖数组碰巧具有相同顺序。

### 4.4 奖励与终止

主要奖励包括：

- 躯干 anchor 的全局位置和朝向跟踪；
- 身体相对位置、朝向、线速度和角速度跟踪；
- 右腕三个关节跟踪；
- 击球相位附近的右肘、右腕位置跟踪；
- 动作变化率和关节越界惩罚。

击球窗口中心位于归一化相位约 0.46，对应数据中的名义击球时间 0.86 s。终止条件包括躯干高度误差、躯干朝向误差和脚部高度误差；动作到达末帧并保持 0.25 s 后正常结束。

### 4.5 启动训练

完整训练命令：

```bash
uv run train Mjlab-PingPong-Tracking-A3-Stage1-RandomDt \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.run-name a3_multimotion
```

先做短流程检查时，可将环境数和更新次数调小：

```bash
uv run train Mjlab-PingPong-Tracking-A3-Stage1-RandomDt \
  --env.scene.num-envs 64 \
  --agent.max-iterations 10 \
  --agent.save-interval 5 \
  --agent.run-name smoke_test
```

日志目录格式为：

```text
logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/<时间>_<run-name>/
├── params/env.yaml
├── params/agent.yaml
├── events.out.tfevents.*
├── model_N.pt
└── jit/modeljit_N.pt
```

`model_N.pt` 用于恢复 PPO 训练或通过标准 `play` 入口加载；`jit/modeljit_N.pt` 是冻结推理模型，也是 Stage 2 需要的跟踪器文件。

### 4.6 观察训练过程

```bash
uv run tensorboard \
  --logdir logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt
```

不要直接用平均 episode length 判断是否学会完整动作。训练会从动作中间的自适应相位开始，且参考平均每步推进 0.025 s，因此正常训练回合通常比从首帧以 1 倍速播放的完整回合短。

更有意义的指标包括：

- `Train/mean_reward` 是否总体改善；
- `Metrics/motion/error_body_pos` 和 `error_joint_pos` 是否下降；
- `Metrics/motion/completion_fraction` 是否接近 1；
- `anchor_height`、`anchor_orientation`、`feet_height` 等失败终止是否减少；
- 从首帧、固定速度评估时，61 段动作是否都能完成。

## 5. Stage 2：自适应接球与回球

### 5.1 训练目标

Stage 2 加载 Stage 1 的 JIT 跟踪器，并执行：

```text
base_action = frozen_tracker(tracker_observation)
high_level_action = receiving_policy(receiving_observation)
final_action = base_action + scaled_joint_residual
next_reference_dt = 0.02 × selected_speed
```

冻结意味着 Stage 1 网络参数及其观测归一化统计不参与 Stage 2 优化。Stage 2 只训练接球 actor/critic。

### 5.2 Actor 观测

Stage 2 actor 观测共 196 维，包括 Stage 1 的 140 维观测，再加：

| 观测项 | 维数 | 含义 |
| --- | ---: | --- |
| `ball` | 23 | 球、拦截点、目标落点和未来轨迹 |
| `phase` | 1 | 当前参考动作归一化相位，范围 `[0, 1]` |
| `high_level_action` | 32 | 上一步 Stage 2 输出的 31 维残差和 1 维速度动作 |

23 维球观测按机器人基座坐标系表达，包括：

```text
当前球位置 3
+ 当前球速度 3
+ 计划拦截位置 3
+ 距计划击球的剩余时间 1
+ 目标落点位置 3
+ 未来 0.1/0.2/0.3 s 的预测位置 9
+ 是否已经击球 1
= 23
```

动作相位告诉策略参考挥拍已经执行到哪个阶段；上一高层动作给策略短期控制历史，有助于连续地调整残差和速度。例如，相位为 `0.40`、上一速度为 1.3 倍时，策略可结合当前球状态判断是继续加速接近 0.46 的击球相位，还是减速并修正拍面。

冻结跟踪器使用独立的 `tracker` 观测组，内容和顺序与 Stage 1 actor 一致，不包含 Stage 2 新增的球、相位和高层动作。

### 5.3 32 维高层动作

Stage 2 输出：

- 前 31 维：各关节的控制残差；
- 第 32 维：参考速度动作 `u`。

关节残差先裁剪到 `[-2, 2]`。右肩、右肘、右腕七个击球臂关节使用 1.0 的残差权限，其余关节使用 0.25，最后还要乘各自的 `action scale`：

```text
final_joint_action = tracker_action + residual_authority × clip(residual, -2, 2)
joint_target = default_joint_position + action_scale × final_joint_action
```

速度动作先裁剪到 `[-1, 1]`，再映射为 0.5～2 倍速度：

```text
u < 0: speed = 1 + 0.5 × u
u ≥ 0: speed = 1 + u
```

| `u` | 参考倍速 | 下一步参考推进量 |
| ---: | ---: | ---: |
| -1 | 0.5 | 0.01 s |
| 0 | 1.0 | 0.02 s |
| 1 | 2.0 | 0.04 s |

当前控制步中，冻结跟踪器先根据当前参考及其时间增量生成基础动作；Stage 2 新输出的速度用于**下一次**参考推进，避免参考姿态和速度观测错位。

### 5.4 球模型和成功条件

小球在每个 0.005 s 物理子步中解析更新，并检测：

- 球与移动球拍的扫掠相交；
- 球拍相对速度决定的反弹；
- 球网阻挡或合法过网；
- 球台反弹和对方台面落点。

有效回球要求实际球拍碰撞、击球后过网、落在对方有效台面，并且机器人没有发生失败终止。球靠近球拍只会产生 shaping reward，不会直接判为击球。

当前模型不模拟球旋转或 Magnus 效应，忽略球对机器人的反作用力；来球由空中发球机模型生成，不要求先在己方台面反弹。

### 5.5 奖励

Stage 2 保留 Stage 1 的动作跟踪奖励，但所有以 `motion_` 开头的权重乘 0.3，并加入：

| 奖励 | 权重 | 作用 |
| --- | ---: | --- |
| `ball_approach` | 5 | 鼓励球拍接近球 |
| `ball_intercept` | 3 | 鼓励按真实来球时间到达计划拦截位置 |
| `ball_normal` | 1 | 鼓励拍面朝向能产生目标回球方向 |
| `ball_hit` | 100 | 实际扫掠碰撞发生时的一次性奖励 |
| `ball_net` | 100 | 有效击球后合法过网的一次性奖励 |
| `ball_landing` | 200 | 在对方台面有效落点的一次性奖励，并按目标误差衰减 |
| `residual` | -0.05 | 抑制不必要的关节残差 |
| `speed` | -0.02 | 抑制不必要的速度偏离 |

### 5.6 启动训练

先选择 Stage 1 的 JIT 文件：

```bash
A3_TRACKER=logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/<stage1-run>/jit/modeljit_9999.pt
```

再启动 Stage 2：

```bash
uv run train Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --env.actions.joint-pos.tracker-file "$A3_TRACKER" \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.run-name a3_receive
```

Stage 2 日志写入：

```text
logs/rsl_rl/a3_pingpong_receive_stage2/<时间>_<run-name>/
```

启动时会把冻结跟踪器复制到本次运行目录的 `tracker.pt`。Stage 2 checkpoint 还会保存跟踪器 SHA256；恢复训练或推理时，如加载了不同的跟踪器，会直接报错。

默认训练来球水平速度幅值为 3.5～4.5 m/s，球位置和速度观测带噪声，额外到球时间扰动默认为 0。若希望训练时加入 ±80 ms 到球时差：

```bash
uv run train Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --env.actions.joint-pos.tracker-file "$A3_TRACKER" \
  --env.commands.ball.arrival-time-jitter 0.08 \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10000 \
  --agent.run-name a3_receive_jitter
```

## 6. 恢复训练

### 6.1 恢复 Stage 1

`--agent.load-run` 使用已有日志目录名称，`--agent.load-checkpoint` 指定其中的 checkpoint：

```bash
uv run train Mjlab-PingPong-Tracking-A3-Stage1-RandomDt \
  --agent.resume True \
  --agent.load-run <已有-stage1-日志目录名称> \
  --agent.load-checkpoint model_9999.pt \
  --agent.max-iterations 5000 \
  --agent.run-name continued
```

checkpoint 中会恢复 actor、critic、优化器以及失败自适应采样统计。

### 6.2 恢复 Stage 2

恢复 Stage 2 时仍须明确提供原来的冻结跟踪器：

```bash
uv run train Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --env.actions.joint-pos.tracker-file \
    logs/rsl_rl/a3_pingpong_receive_stage2/<原-run>/tracker.pt \
  --agent.resume True \
  --agent.load-run <已有-stage2-日志目录名称> \
  --agent.load-checkpoint model_9999.pt \
  --agent.max-iterations 5000 \
  --agent.run-name continued
```

如果 tracker 内容与 checkpoint 中保存的 SHA256 不一致，加载会失败。这一检查用于避免“高层策略和底层跟踪器形状相同，但实际行为不同”的隐蔽错误。

## 7. 推理与播放

### 7.1 直接播放交付的 Stage 2 接球策略

```bash
uv run --no-sync play Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint-file ckpts/a3_pingpong/model_999.pt \
  --num-envs 1
```

`play` 会自动查找 checkpoint 同目录下的 `tracker.pt`。因此下面两个文件必须一起保留：

```text
ckpts/a3_pingpong/model_999.pt
ckpts/a3_pingpong/tracker.pt
```

也可以显式指定跟踪器：

```bash
uv run --no-sync play Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint-file <stage2-run>/model_9999.pt \
  --tracker-file <stage2-run>/tracker.pt \
  --num-envs 1
```

调试可视化会显示球、目标点和球台，观察接球时建议保留默认的 `--debug-vis True`。

### 7.2 播放 Stage 1 跟踪器

```bash
uv run --no-sync play Mjlab-PingPong-Tracking-A3-Stage1-RandomDt \
  --checkpoint-file ckpts/a3_pingpong/tracking_model_999.pt \
  --num-envs 1
```

Stage 1 的 play 配置从动作首帧开始，参考速度固定为 1，适合观察完整挥拍。A3 球拍固定安装在右手，不需要 G1 任务的 `--racket-hand` 参数。

### 7.3 推理时每个控制步发生什么

以 Stage 2 为例：

1. 根据当前参考时间插值得到参考关节和身体姿态。
2. 构造 140 维 tracker 观测，冻结跟踪器输出 31 维基础动作。
3. 构造 196 维高层观测，接球策略输出 31 维残差和 1 维速度动作。
4. 合并基础动作与残差，生成 31 个关节位置目标。
5. MuJoCo 执行 4 个 0.005 s 物理子步；每个子步同时更新球并检测碰撞。
6. 使用本步速度动作设置下一步参考时间增量。
7. 更新球事件、奖励、终止状态和下一步观测。

如果球成功落在对方台面，`ball_finished` 是正常任务结束；它不表示机器人失败。Stage 2 也不要求继续播放到参考动作末帧。

### 7.4 录制视频

```bash
uv run --no-sync play Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint-file ckpts/a3_pingpong/model_999.pt \
  --num-envs 1 \
  --video-headless True \
  --video-duration-s 8
```

默认输出目录为 checkpoint 目录下的 `videos/play/`。本项目 CLI 的布尔参数需要显式传入 `True` 或 `False`。

## 8. 批量评估

### 8.1 分析训练曲线

```bash
uv run --no-sync python scripts/tools/analyze_a3_training.py \
  <日志目录> --window 100
```

脚本会在 `<日志目录>/validation/` 生成：

- `training_summary.json`；
- `training_curves.png`；
- `training_curves.pdf`。

### 8.2 Stage 1 全动作评估

```bash
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --checkpoint \
    <stage1-run>/model_0.pt \
    <stage1-run>/model_999.pt \
  --output <stage1-run>/validation/rollouts.json
```

评估器会自动使用各 checkpoint 旁边 `jit/modeljit_N.pt`，让每段动作从第 0 帧开始，以固定速度 1 运行，关闭观测噪声、重置扰动和域随机化，并且只统计每个环境的第一回合。

Stage 1 验收应重点读取：

- `full_clip_completion_rate`；
- `completed_episode_without_failure_rate`；
- `failure_termination_rate`；
- 身体和关节跟踪误差。

### 8.3 Stage 2 接球评估

```bash
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint <stage2-run>/model_999.pt \
  --tracker-file <stage2-run>/tracker.pt \
  --repeats 4 \
  --output <stage2-run>/validation/rollouts.json
```

61 个动作各重复 4 次时共有 244 个回合。Stage 2 重点读取：

- `ball/state_hit`：实际球拍击球率；
- `ball/state_net_cleared`：有效击球后过网率；
- `ball_return_success_rate`：有效落台率；
- `receiving_task_success_rate`：有效回球且机器人未失败；
- `robot_failure_rate`：机器人姿态或脚部终止率；
- `landing_error_mean_given_success_m`：只在成功回球样本中计算的落点误差。

不要使用 `completed_episode_without_failure_rate` 作为 Stage 2 回球成功率，因为接球回合通常在球的结果确定后结束，不要求走到参考动作末帧。

### 8.4 组件消融和时间扰动

固定参考速度、保留残差：

```bash
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint <stage2-run>/model_999.pt \
  --tracker-file <stage2-run>/tracker.pt \
  --repeats 4 --fixed-speed \
  --output <stage2-run>/validation/fixed_speed.json
```

保留学习速度、去除残差：

```bash
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint <stage2-run>/model_999.pt \
  --tracker-file <stage2-run>/tracker.pt \
  --repeats 4 --zero-residual \
  --output <stage2-run>/validation/zero_residual.json
```

冻结跟踪器基线同时使用 `--fixed-speed --zero-residual`。测试 ±80 ms 到球扰动时添加：

```text
--arrival-time-jitter 0.08
```

这些开关是在推理时关闭已训练策略的动作分量，属于组件消融，不等于分别重新训练了不同基线。

## 9. 已保存模型的验证结果

仓库中的交付模型来自两个各 1000 次 PPO 更新的运行。固定 seed 的评估结果为：

| 结果 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| 动作完成 | 61/61 | 不以完成整段动作为目标 |
| 来球击中 | 不适用 | 244/244，100% |
| 击球后过网 | 不适用 | 214/244，87.70% |
| 有效回球 | 不适用 | 198/244，81.15% |
| 机器人失败终止 | 0/61 | 0/244 |

Stage 2 的推理消融结果：

| 推理配置 | 击球率 | 有效回球率 |
| --- | ---: | ---: |
| 固定速度、零残差 | 19.26% | 0% |
| 固定速度、保留残差 | 84.02% | 37.30% |
| 学习速度、零残差 | 23.36% | 0% |
| 学习速度、保留残差 | 100% | 81.15% |

这些结果只覆盖训练使用的 61 段动作和指定来球分布，不代表未见动作、任意来球或真实机器人泛化。

## 10. Checkpoint 与 JIT 文件的区别

| 文件 | 内容和用途 |
| --- | --- |
| `model_N.pt` | 完整 PPO checkpoint，用于恢复训练和标准 `play` |
| `jit/modeljit_N.pt` | 导出的 actor 推理网络，Stage 1 的此文件可作为 Stage 2 冻结跟踪器 |
| `tracker.pt` | Stage 2 运行目录中复制并绑定的 Stage 1 JIT 跟踪器 |

Stage 2 的完整部署单元是：

```text
Stage 2 model_N.pt + 与它匹配的 tracker.pt + 动作数据目录
```

只复制 Stage 2 的 `model_N.pt` 会导致推理缺少底层跟踪器。批量评估脚本还会从 `model_N.pt` 路径自动查找 `jit/modeljit_N.pt`，因此评估训练运行时应使用保留完整 `jit/` 子目录的日志目录。

## 11. 常见问题

### Stage 2 报错找不到 tracker

训练时提供：

```text
--env.actions.joint-pos.tracker-file <stage1-run>/jit/modeljit_N.pt
```

播放时将 `tracker.pt` 放在 Stage 2 checkpoint 同目录，或使用：

```text
--tracker-file <路径>/tracker.pt
```

### Stage 2 报 tracker SHA256 不匹配

当前 tracker 不是训练该 Stage 2 checkpoint 时使用的文件。使用对应运行目录中自动复制的 `tracker.pt`，不要仅根据网络输入输出形状替换模型。

### Stage 1 平均回合只有四五十步

这是自适应初始相位和随机参考推进造成的正常现象。使用从首帧、固定速度的批量评估确认是否能完成 61 段整段动作。

### Stage 2 回合没有走到动作末帧

接球任务在球成功落台或确定失败后结束。应读取 `receiving_task_success_rate`，而不是动作完成率。

### 训练奖励改善，但回球效果不好

分别检查实际击球率、过网率、有效落台率、机器人失败率、球拍残差和参考速度分布。仅有跟踪奖励提升，不保证球拍在击球瞬间的位置、速度和法向足够准确。

### 修改了球台尺寸后结果异常

球台存在两套必须一致的几何参数：`scene.py` 中的可视/物理球台，以及球 command 中用于解析碰撞的边界。修改时必须同步调整。

### 是否应按参考时间缩放奖励和 GAE

当前 A3 默认按固定物理控制时间计算奖励与 GAE，因为机器人和球每步都实际经过 0.02 s。`use_reference_time_discount` 仅用于显式的参考时间折扣实验；不要再同时叠加参考时间奖励缩放。

## 12. 推荐的完整操作顺序

从头训练时按以下顺序执行：

1. `uv sync`，检查 CUDA 和 `dataset/a3_pingpong/motions/`。
2. 用小环境数跑 Stage 1 smoke test，确认环境和日志正常。
3. 用 1024 环境训练 Stage 1，并通过 TensorBoard 和全动作评估选择 checkpoint。
4. 使用选中 checkpoint 对应的 `jit/modeljit_N.pt` 训练 Stage 2。
5. 检查 Stage 2 的击球、过网、落台、机器人失败和速度/残差指标。
6. 使用固定 seed 的 61 动作 × 多次来球评估，并进行固定速度、零残差和时间扰动消融。
7. 交付时一起保存 Stage 2 checkpoint、匹配的 `tracker.pt`、动作数据及配置文件。
