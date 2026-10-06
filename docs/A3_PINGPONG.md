# A3 乒乓球训练

本任务在现有 mjlab / MuJoCo Warp / RSL-RL 训练入口中实现，不依赖远端机器运行。
机器人来自用户提供的 HOPE A3 乒乓球模型，动作来自用户指定的 61 段挥拍数据。
模型参数及来源见 `src/mjlab/asset_zoo/robots/agibot_a3/README.md`，动作坐标变换和
校验结果见 `dataset/a3_pingpong/README.md`、`manifest.json`。

从环境准备到两阶段训练、恢复、推理和批量评估的完整操作说明见
[A3 PingPong 训练与推理指南](A3_PINGPONG_TRAINING_INFERENCE_GUIDE.md)。

## 直接播放已经训练的模型

两个阶段已各训练 1000 次更新，最终模型放在 reload`ckpts/a3_pingpong/`。
在本仓库根目录播放接球策略：

```bash
uv run --no-sync play Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint-file ckpts/a3_pingpong/model_999.pt \
  --num-envs 1
```

`model_999.pt` 和同目录的 `tracker.pt` 是完整接球策略的两个依赖，迁移时需一起保留。
查看 Stage 1 的完整挥拍跟踪：

```bash
uv run --no-sync play Mjlab-PingPong-Tracking-A3-Stage1-RandomDt \
  --checkpoint-file ckpts/a3_pingpong/tracking_model_999.pt \
  --num-envs 1
```

同一动作库上，Stage 1 的 61 段动作全部完成；Stage 2 的 244 次来球中击球率 100%、
有效回球率 81.15%，没有机器人失败终止。配置、曲线、组件消融及适用范围详见
[验证记录](A3_PINGPONG_VALIDATION.md)。

## 与 AdaPT 的对应关系

参考论文：[Towards Professional Tennis Styles for Humanoid Robots with Adaptive Motion Planning and Tracking](https://arxiv.org/html/2608.20087v1)。

| 机制 | 本任务实现 |
| --- | --- |
| 随机执行速度跟踪 | Stage 1 在每个控制步采样参考时间增量 0.01–0.04 秒；物理控制周期始终 0.02 秒 |
| 多动作与失败自适应采样 | 61 段动作统一坐标，按名称映射关节；在片段内按失败统计采样初始相位 |
| 连续参考轨迹 | 关节线性插值和身体四元数 SLERP |
| 跟踪奖励 | 全局 anchor、身体位置/朝向/速度、腕关节、击球窗口的手臂姿态 |
| 冻结跟踪器与残差 | Stage 2 加载 Stage 1 JIT 模型，冻结网络和归一化参数；学习关节控制残差 |
| 可学习速度 | Stage 2 的第 32 个动作决定参考推进速度，范围 0.5–2 倍；球的物理时间保持不变 |
| 来球与回球 | 物理子步更新解析小球；球拍相对运动扫掠碰撞、台面反弹、球网阻挡和对方台面落点 |
| 球预测与扰动 | 当前球状态、未来轨迹采样、目标位置和击球时间观测；来球速度/位置及观测噪声 |
| 训练基础设施 | PPO、KL 自适应学习率、TensorBoard、配置、checkpoint、JIT、继续训练和 play |

这是一项固定动作库上的 AdaPT 迁移。论文回球分支的 MVAE 自回归运动生成器尚未实现，
也没有声称完成真实机器部署或论文全部实验复现。当前球模型未建模旋转/Magnus 效应；
使用解析小球是为了在小球高速飞行时可靠检测接触，球对机器人的反作用力忽略。
来球采用空中发球机模型，不强制来球先在己方台面弹跳；这不是完整的比赛规则引擎。
球拍碰撞采用每个 0.005 秒物理子步的当前法向和中心位移近似扫掠运动。

A3 默认按固定的物理控制时间计算奖励与 GAE 折扣；参考动作随机变速不会改变物理时间。
`use_reference_time_discount` 提供显式的参考时间折扣实验开关。项目原有参数化 GAE 的
timeout bootstrap 重复乘时间积分问题已修复并添加回归测试；不要再同时叠加参考时间奖励缩放。

## 数据准备

本地已经准备好 `dataset/a3_pingpong/motions`。重建方式：

```bash
uv run --no-sync python scripts/tools/prepare_a3_motion.py --help
```

保留了原始数据；训练数据将首帧 pelvis 的 XY 平移到零、yaw 旋转到 +X。
保留 Z、关节姿态和四元数 wxyz 约定，世界线速度/角速度也随坐标系旋转。
每段 94 帧、50 Hz、1.86 秒。击球标记为第 43 帧（0.86 秒），与测量的最大球拍速度帧
分别保存，不把二者混为一谈。

## Stage 1：随机速度跟踪

在仓库根目录运行：

```bash
uv run train Mjlab-PingPong-Tracking-A3-Stage1-RandomDt \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.run-name a3_multimotion
```

日志写入 `logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt/<时间>_<run-name>/`，
包括 `params/env.yaml`、`params/agent.yaml`、TensorBoard events、`model_N.pt` 和
`jit/modeljit_N.pt`，结构与现有 G1 训练一致。

```bash
uv run play Mjlab-PingPong-Tracking-A3-Stage1-RandomDt \
  --checkpoint-file <Stage1日志目录>/model_9999.pt \
  --num-envs 1
```

play 默认固定速度、从动作首帧开始，适合查看完整挥拍。A3 球拍安装在右手，
不使用 G1 专用的 `--racket-hand`。

## Stage 2：接球、速度适配与回球

```bash
uv run train Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --env.actions.joint-pos.tracker-file <Stage1日志目录>/jit/modeljit_9999.pt \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10000 \
  --agent.run-name a3_receive
```

Stage 2 的 actor 输出 31 个残差和 1 个速度动作。跟踪器仍只负责 31 个关节，
其输入顺序、历史关节动作和速度观测与 Stage 1 一致。输入给跟踪器的是产生当前参考姿态的
时间增量；新速度动作影响下一次参考推进。残差先限幅到 ±2，再乘各关节的 action scale：
躯干与非击球关节额外乘 0.25，右侧击球臂乘 1.0，以便调整球拍而保留全身平衡。

日志写入 `logs/rsl_rl/a3_pingpong_receive_stage2/`；冻结跟踪器会复制为该次日志中的
`tracker.pt`，便于保存和迁移整个策略。

```bash
uv run play Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint-file <Stage2日志目录>/model_9999.pt \
  --num-envs 1
```

play 自动寻找同目录 `tracker.pt`，也可以用 `--tracker-file` 显式指定。
球和目标由调试可视化显示，因此观看接球时保留默认 debug visualization。
新 Stage 2 checkpoint 还保存 tracker 的 SHA256，恢复训练/播放时会检查依赖是否一致。

来球时间扰动可通过 `--env.commands.ball.arrival-time-jitter 0.08` 加入 ±80 ms 时差；
默认值为零。改变球台尺寸时，需要同时修改球 command 的几何参数与 `scene.py` 的物理几何。

## 继续训练及日志

```bash
uv run train Mjlab-PingPong-Tracking-A3-Stage1-RandomDt \
  --agent.resume True \
  --agent.load-run <已有日志目录名称> \
  --agent.load-checkpoint model_9999.pt \
  --agent.max-iterations 5000 \
  --agent.run-name continued

uv run tensorboard --logdir logs/rsl_rl/a3_pingpong_tracking_stage1_random_dt
```

Stage 2 继续训练时还需提供同一个 `--env.actions.joint-pos.tracker-file`。

## 验收方法

不能把 Stage 1 的平均 episode length 直接和配置的 4 秒上限比较：训练从动作中段开始，
并且随机速度平均推进量为 0.025 秒。1.86 秒动作的剩余片段通常只需约 40–50 个控制步。
应同时检查动作完成率、摔倒终止原因、跟踪误差，以及固定速度首帧评估。

接球任务额外检查击球率、击球后过网率、有效落台率、落点误差、残差幅度和速度分布。
击球必须来自实际扫掠相交；过网和落台奖励必须发生在有效击球后。
学习曲线改善不等于已达到竞技水平，实际验证数据见验证记录和保存的 JSON 报告。

实际运行结果见 [A3_PINGPONG_VALIDATION.md](A3_PINGPONG_VALIDATION.md)。
可以用下面的脚本复查，评估器会从首帧遍历全部动作并在自动 reset 前保存终止状态：

```bash
uv run --no-sync python scripts/tools/analyze_a3_training.py <日志目录>
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --checkpoint <Stage1日志目录>/model_0.pt <Stage1日志目录>/model_999.pt \
  --output <Stage1日志目录>/validation/rollouts.json

uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Receive-A3-Stage2-Adaptive \
  --checkpoint <Stage2日志目录>/model_999.pt \
  --tracker-file <Stage2日志目录>/tracker.pt \
  --repeats 4 --output <Stage2日志目录>/validation/rollouts.json
```

接球评估可添加 `--fixed-speed`、`--zero-residual` 做消融，
或者添加 `--arrival-time-jitter 0.08` 检查时差扰动下的表现。
上述评估使用训练动作库和指定来球分布，不代表未见动作泛化。

录制 play 视频使用 `--video-headless True --video-duration-s 5`；
本项目 CLI 的布尔参数需要显式传入 `True` 或 `False`。
