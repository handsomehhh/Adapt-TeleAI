# G1 乒乓球训练任务

本任务把 29-DoF Unitree G1 的多动作挥拍跟踪和接球训练接入项目现有的
MuJoCo Warp、RSL-RL、日志、checkpoint、JIT 导出及 `play` 流程。任务分为两个阶段：

1. Stage 1 在随机参考速度下学习完整身体的挥拍动作跟踪。
2. Stage 2 冻结 Stage 1 跟踪器，学习参考动作速度和关节残差，使球拍适应实际来球。

对应的任务 ID 为：

- `Mjlab-PingPong-Tracking-G1-Stage1-RandomDt`
- `Mjlab-PingPong-Tracking-G1-TeleGMR-Stage1-RandomDt`
- `Mjlab-PingPong-Receive-G1-Stage2-Adaptive`
- `Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive`
- `Mjlab-PingPong-Receive-G1-TeleGMR-Hard-Stage2-Adaptive`

所有命令都应在仓库根目录运行，并使用 `uv run --no-sync` 进入项目环境。

## 直接播放验收模型

两个阶段各完成了 1000 次 PPO 更新，选定 checkpoint 位于
`ckpts/g1_pingpong/`。播放最终接球策略：

```bash
uv run --no-sync play Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --checkpoint-file ckpts/g1_pingpong/model_999.pt \
  --num-envs 1
```

同一动作库的固定种子评估中，Stage 1 完成 43/43 段完整动作；Stage 2 在
172 次来球中击球 166 次、有效回球 139 次，且没有机器人失败终止。训练曲线、
正反手拆分、速度/残差消融、到球时差测试和适用范围见
[G1 乒乓球任务验证记录](G1_PINGPONG_VALIDATION.md)。

## 独立的 29-DoF 资产

本任务使用
[`src/mjlab/asset_zoo/robots/unitree_g1_pingpong`](../src/mjlab/asset_zoo/robots/unitree_g1_pingpong)
中的独立机器人包。源 URDF 来自用户提供的 HOPE 参考工程：
`/home/lxz/HOPE/G1/urdf/g1_29dof_pingpong.urdf`。仓库保留了源 URDF、转换后的
MJCF、41 个 mesh、右手球拍安装结构、文件哈希和转换验证记录：

- [`README.md`](../src/mjlab/asset_zoo/robots/unitree_g1_pingpong/README.md)：来源和转换说明；
- [`asset_manifest.json`](../src/mjlab/asset_zoo/robots/unitree_g1_pingpong/asset_manifest.json)：源文件、关节、body、球拍和 actuator 契约；
- [`asset_validation.json`](../src/mjlab/asset_zoo/robots/unitree_g1_pingpong/asset_validation.json)：FK、质量、惯量、关节限制和编译检查结果。

机器人有 29 个受控关节，根 body 为 `pelvis`，动作 anchor 为 `torso_link`。
右手球拍中心使用 MJCF site `right_racket_surface`，安装在
`right_wrist_yaw_link` 上。关节动作严格使用 MJCF 原生顺序；任务构建时也会检查动作数据的
29 个关节和完整 body schema，避免名称相似但拓扑不同的数据被静默加载。

### 与 legacy 27-DoF G1 的隔离

项目原有的 [`unitree_g1`](../src/mjlab/asset_zoo/robots/unitree_g1) 包仍用于论文网球任务，
其左右手球拍模型是另一套 27-DoF articulation。本任务没有修改或替换该包，而是分别隔离：

| 契约 | 29-DoF 乒乓球任务 | legacy 27-DoF G1 |
| --- | --- | --- |
| 资产包 | `unitree_g1_pingpong` | `unitree_g1` |
| 数据目录 | `dataset/g1_pingpong` | 原有网球数据目录 |
| 任务 ID | `Mjlab-PingPong-*-G1-*` | 原有 G1 网球任务 ID |
| 球拍 | 固定右手 HOPE 安装结构 | 原有左右手网球拍配置 |
| 动作维度 | Stage 1 为 29，Stage 2 为 30 | 按原任务配置 |

两套模型的腰部关节集合、body schema、球拍安装和 action contract 不同。不要把
`dataset/g1_pingpong/motions` 加载到 legacy 27-DoF 环境，也不要用 27-DoF checkpoint
初始化本任务。`--racket-hand` 是原 G1 网球任务的切换参数，不适用于本任务。

## 动作数据与来源

本地数据位于 [`dataset/g1_pingpong`](../dataset/g1_pingpong)：

- `raw/` 保存 HOPE 参考运行导出的原始 29-DoF NPZ；
- `motions/` 保存任务实际读取的规范化动作；
- `source_manifest.json` 保存 BVH Mink IK 重定向来源、原始哈希和独立数据审计信息；
- `manifest.json` 保存本地输出哈希、资产绑定、schema 和 FK 校验结果。

动作来自用户提供的 HOPE G1 参考训练
`2026-10-02_01-15-34_g1_bvh_v2_30k_save2000_seed0`。数据共 43 段，每段 94 帧、
50 Hz，首末帧时间跨度为 1.86 秒；标注击球时刻为第 43 帧，即 0.86 秒。处理过程把首帧
pelvis 的 XY 平移到原点并把首帧 yaw 对齐到 +X，同时一致地旋转世界坐标中的姿态和速度。
四元数使用 WXYZ 顺序。原数据中的 body 质心速度会按 MJCF 惯性偏移转换为 link 原点速度，
使训练奖励与运行时 MuJoCo body 定义一致。

重建并重新执行 schema、哈希和 MuJoCo FK 检查：

```bash
uv run --no-sync python scripts/tools/prepare_g1_pingpong_motion.py
```

任务注册时会校验 `manifest.json` 中的 43 个文件、29 个关节、完整 body 列表及每个输出
SHA256。修改资产或数据后需要重新生成 manifest；不要手工绕过这些检查。

另一个独立数据目录
[`dataset/g1_pingpong_tele_gmr`](../dataset/g1_pingpong_tele_gmr) 保存同一批原始
Xsens BVH 经 Tele-GMR 和 Football 转换得到的 43 段动作。对应任务 ID 为
`Mjlab-PingPong-Tracking-G1-TeleGMR-Stage1-RandomDt` 和
`Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive`。它们复用相应阶段的
环境和 runner 配置，替换经过哈希及 MuJoCo FK/速度校验的 motion 列表；
Stage 2 另外加载与该动作库配套的冻结 Stage 1 JIT 跟踪器。

## Stage 1：随机速度多动作跟踪

Stage 1 学习 29 维关节位置动作。每个物理控制步为 0.02 秒，而参考轨迹推进量独立地从
0.01–0.04 秒采样。策略通过 `motion_dt_sample` 观察产生当前参考姿态的时间增量，从而学习
同一挥拍在不同执行速度下的稳定跟踪。关节轨迹使用线性插值，body 四元数使用 SLERP，
不会因为参考时间落在两帧之间而跳帧。

训练还使用失败自适应的动作相位采样。每次终止会更新对应动作和时间 bin 的失败统计，
后续 reset 提高困难区间的采样概率，同时保留均匀采样成分。奖励包含 anchor 位置和朝向、
相对 body 姿态、body 速度、腕关节、击球窗口手臂姿态、动作变化率和关节限制。

先做小规模启动检查：

```bash
uv run --no-sync train Mjlab-PingPong-Tracking-G1-Stage1-RandomDt \
  --env.scene.num-envs 16 \
  --agent.max-iterations 2 \
  --agent.save-interval 1 \
  --agent.run-name g1_stage1_preflight
```

正式训练示例：

```bash
uv run --no-sync train Mjlab-PingPong-Tracking-G1-Stage1-RandomDt \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.run-name g1_stage1_multimotion
```

日志写入
`logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt/<时间>_<run-name>/`，其中包括训练配置、
TensorBoard event、`model_N.pt` 和 `jit/modeljit_N.pt`。Stage 2 需要使用已验证的 Stage 1
JIT 文件，而不是标准 `model_N.pt`：

```bash
mkdir -p ckpts/g1_pingpong
cp <Stage1日志目录>/jit/modeljit_9999.pt ckpts/g1_pingpong/tracker.pt
cp <Stage1日志目录>/model_9999.pt ckpts/g1_pingpong/tracking_model_9999.pt
```

播放完整动作跟踪：

```bash
uv run --no-sync play Mjlab-PingPong-Tracking-G1-Stage1-RandomDt \
  --checkpoint-file <Stage1日志目录>/model_9999.pt \
  --num-envs 1
```

`play` 使用任务内置的 43 段本地数据，从动作首帧开始并固定参考速度为 1。可以用
`--motion-file dataset/g1_pingpong/motions/<文件名>.npz` 查看指定动作。

## Stage 2：接球、速度适配和关节残差

Stage 2 加载并冻结 Stage 1 JIT 跟踪器及其 observation normalization。高层 actor 输出
30 维动作：前 29 维是关节 residual，最后 1 维控制参考速度。速度动作以 1 倍为中心，
映射到 0.5–2.0 倍；它决定下一控制步的参考轨迹推进量。普通关节 residual 额外乘 0.25，
右侧击球臂使用 1.0，使策略能调整球拍轨迹并尽量保留低层全身平衡动作。

小球每个 0.005 秒物理子步更新。环境实现解析弹道、移动球拍扫掠碰撞、台面反弹、球网
阻挡和对方台面落点判定。小球的物理时间始终按真实仿真时间前进，不会随参考动作加速或
减速。actor 观察球的位置、速度、计划击球点、剩余到达时间、目标落点和多个未来预测点；
训练来球包含速度、击球点及观测噪声。奖励只在真实扫掠接触、真实过网和真实落台事件上
发放相应稀疏项，并辅以接近、球拍法向、跟踪、残差和速度正则项。

先检查 tracker 路径和完整环境能否构建：

```bash
uv run --no-sync train Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --env.scene.num-envs 16 \
  --env.actions.joint-pos.tracker-file ckpts/g1_pingpong/tracker.pt \
  --agent.max-iterations 2 \
  --agent.save-interval 1 \
  --agent.run-name g1_stage2_preflight
```

正式训练示例：

```bash
uv run --no-sync train Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --env.scene.num-envs 1024 \
  --env.actions.joint-pos.tracker-file ckpts/g1_pingpong/tracker.pt \
  --agent.max-iterations 10000 \
  --agent.save-interval 500 \
  --agent.run-name g1_stage2_receive
```

日志写入 `logs/rsl_rl/g1_pingpong_receive_stage2/`。runner 会把实际加载的冻结跟踪器复制为
该次运行目录下的 `tracker.pt`，并在 Stage 2 checkpoint 中保存其 SHA256。恢复训练或加载
checkpoint 时，如果 tracker 内容与保存的哈希不一致，运行会立即失败，避免同形状的错误
低层策略被静默替换。

使用新 Tele-GMR/Football motion 的独立 Stage 2 训练命令如下。任务默认读取
`dataset/g1_pingpong_tele_gmr/motions/` 的 43 段动作及
`ckpts/g1_pingpong_tele_gmr/tracker.pt`；后者来自对应 Stage 1 的
`modeljit_29999.pt`，来源和 SHA256 记录在
`ckpts/g1_pingpong_tele_gmr/provenance.json`。

```bash
uv run --no-sync train Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 30000 \
  --agent.save-interval 2000 \
  --agent.run-name g1_tele_gmr_stage2_30k_4096_seed42
```

该实验写入 `logs/rsl_rl/g1_pingpong_receive_stage2_tele_gmr/`。播放其中的 checkpoint：

```bash
uv run --no-sync play Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive \
  --checkpoint-file <TeleGMR-Stage2日志目录>/model_29999.pt \
  --num-envs 1
```

`play` 会自动读取 checkpoint 同目录的 `tracker.pt`，并校验其 SHA256。

## Hard Stage 2：扩大来球分布并加强速度适配

Hard 版本保留同一套 43 段 Tele-GMR motion 和冻结 Stage 1 tracker，单独扩大接球分布：

- 击球点扰动扩大到 X ±0.12 m、Y ±0.25 m、Z ±0.08 m；
- 到达时间加入 ±0.18 s 扰动，飞行时间扩大到 0.30–0.70 s；
- 来球水平速度扩大到 2.5–5.5 m/s，并加入 -1.5–1.5 m/s 的横向速度；
- 对方台面目标落点扰动扩大到 X ±0.30 m、Y ±0.45 m；
- 高层策略的 motion speed 范围扩大到 0.35–2.50 倍。

这使策略必须同时处理更宽的击球空间和更大的参考时间偏差，速度动作会直接决定下一参考步的推进量。
为避免从基线切换时发生分布突变，Hard 随机范围从基线配置开始，在前 5000 个 Hard 训练
iteration 内线性过渡到上述范围。正式实验从基线 30000 轮 checkpoint warm-start，额外训练
30000 轮，使用 4096 个并行环境：

```bash
uv run --no-sync train Mjlab-PingPong-Receive-G1-TeleGMR-Hard-Stage2-Adaptive \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run warmstart_from_baseline \
  --agent.load-checkpoint model_29999.pt \
  --agent.max-iterations 30000 \
  --agent.save-interval 2000 \
  --agent.run-name g1_tele_gmr_hard_stage2_30k_4096_warmstart_seed42
```

日志写入 `logs/rsl_rl/g1_pingpong_receive_stage2_tele_gmr_hard/`，训练会话为
`g1_pingpong_stage2_hard_30k_4096`。完成后播放最终 checkpoint：

```bash
uv run --no-sync play Mjlab-PingPong-Receive-G1-TeleGMR-Hard-Stage2-Adaptive \
  --checkpoint-file <Hard-Stage2日志目录>/model_59998.pt \
  --num-envs 1
```

播放接球策略：

```bash
uv run --no-sync play Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --checkpoint-file <Stage2日志目录>/model_9999.pt \
  --tracker-file <Stage2日志目录>/tracker.pt \
  --num-envs 1
```

如果 `tracker.pt` 与 Stage 2 checkpoint 位于同一目录，可以省略 `--tracker-file`，`play`
会自动使用同目录依赖。录制无窗口视频：

```bash
uv run --no-sync play Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --checkpoint-file <Stage2日志目录>/model_9999.pt \
  --tracker-file <Stage2日志目录>/tracker.pt \
  --num-envs 1 \
  --video-headless True \
  --video-duration-s 5
```

## Adaptive 机制对应关系

本任务中的 “adaptive” 包含三个互相配合的层次：

| 层次 | 实现 |
| --- | --- |
| 训练数据分布 | Stage 1 根据各动作时间 bin 的失败统计提高困难相位的 reset 概率 |
| 速度条件跟踪 | Stage 1 在随机参考 `dt` 下训练，低层策略显式观察上一参考时间增量 |
| 在线动作适配 | Stage 2 针对当前来球逐步选择 0.5–2.0 倍参考速度和 29 维关节 residual |

Stage 2 的 motion command 使用 `start` 模式，每回合从随机选中动作的首帧开始；此阶段的
适应发生在策略输出的速度和 residual，而不是 Stage 1 的失败相位 sampler。新速度影响下一
参考步，跟踪器当前看到的是产生当前参考姿态的时间增量，从而保持训练与执行的因果顺序。

这对应 AdaPT 的随机速度跟踪、冻结低层跟踪器和高层速度/残差适配思路。它是固定挥拍动作
库上的乒乓球任务迁移，不等同于复现论文的所有模型和全部真实机器人实验。

## 训练曲线与批量评估

分析 TensorBoard 标量并生成 PNG、PDF 和 JSON：

```bash
uv run --no-sync python scripts/tools/analyze_a3_training.py <日志目录> --window 100
```

脚本文件名保留了早期 A3 名称，但会根据日志标量处理 G1 Stage 1 和 Stage 2。Stage 1
训练会从自适应抽取的动作中间相位开始，而且参考平均推进量不等于物理控制周期，因此不要
只用训练中的 average episode length 判断成功。应同时检查 completion fraction、终止原因、
body/joint 跟踪误差，并进行固定速度、首帧开始的全动作评估。

Stage 1 全动作评估：

```bash
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Tracking-G1-Stage1-RandomDt \
  --checkpoint <Stage1日志目录>/model_0.pt <Stage1日志目录>/model_9999.pt \
  --output <Stage1日志目录>/validation/rollouts.json
```

检查标注击球时刻的球拍位置、速度、法向和时间偏差：

```bash
uv run --no-sync python scripts/tools/diagnose_a3_strike.py \
  --task Mjlab-PingPong-Tracking-G1-Stage1-RandomDt \
  --checkpoint <Stage1日志目录>/model_9999.pt \
  --output <Stage1日志目录>/validation/strike_diagnostics.json
```

Stage 2 使用相同种子遍历全部动作，并为每段动作重复多个来球：

```bash
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --checkpoint <Stage2日志目录>/model_9999.pt \
  --tracker-file <Stage2日志目录>/tracker.pt \
  --repeats 4 \
  --output <Stage2日志目录>/validation/rollouts.json
```

接球验收应分别查看机器人失败率、击球率、击球后过网率、有效回球率、成功回球条件下的
落点误差、速度分布和 residual 大小。`ball_finished` 也会在合法落台时触发，因此不能把它
直接当作失败。评估报告会按正手/反手标签分别汇总结果。

速度与 residual 消融使用同一个 checkpoint、tracker、种子和来球分布：

```bash
# 固定参考速度，仅保留学习到的关节 residual
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --checkpoint <Stage2日志目录>/model_9999.pt \
  --tracker-file <Stage2日志目录>/tracker.pt \
  --repeats 4 --fixed-speed \
  --output <Stage2日志目录>/validation/fixed_speed.json

# 去掉 residual，仅保留学习到的参考速度
uv run --no-sync python scripts/tools/evaluate_a3_pingpong.py \
  --task Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --checkpoint <Stage2日志目录>/model_9999.pt \
  --tracker-file <Stage2日志目录>/tracker.pt \
  --repeats 4 --zero-residual \
  --output <Stage2日志目录>/validation/zero_residual.json
```

同时添加 `--fixed-speed --zero-residual` 可以得到冻结 Stage 1 跟踪器基线；添加
`--arrival-time-jitter 0.08` 可以测试 ±80 ms 来球时间偏差。

## 继续训练

Stage 1 示例：

```bash
uv run --no-sync train Mjlab-PingPong-Tracking-G1-Stage1-RandomDt \
  --agent.resume True \
  --agent.load-run <已有日志目录名称> \
  --agent.load-checkpoint model_9999.pt \
  --agent.max-iterations 5000 \
  --agent.run-name g1_stage1_continued
```

Stage 2 恢复时必须继续提供原 tracker。runner 会在恢复模型和优化器之前检查 tracker 哈希：

```bash
uv run --no-sync train Mjlab-PingPong-Receive-G1-Stage2-Adaptive \
  --env.actions.joint-pos.tracker-file <原Stage2日志目录>/tracker.pt \
  --agent.resume True \
  --agent.load-run <已有日志目录名称> \
  --agent.load-checkpoint model_9999.pt \
  --agent.max-iterations 5000 \
  --agent.run-name g1_stage2_continued
```

查看曲线：

```bash
uv run --no-sync tensorboard --logdir logs/rsl_rl/g1_pingpong_tracking_stage1_random_dt
uv run --no-sync tensorboard --logdir logs/rsl_rl/g1_pingpong_receive_stage2
```

## 当前局限

- 动作库固定为当前 43 段右手挥拍；评估同一动作库不能证明对未见动作或不同机器人参数的泛化。
- Stage 2 使用冻结的 Stage 1 JIT，无法在接球训练中联合更新低层跟踪器。
- 小球模型没有旋转和 Magnus 效应，也忽略轻质球对机器人的动力学反作用。
- 来球由空中发球机弹道生成，不要求先在机器人一侧台面反弹，因此不是完整比赛规则引擎。
- 球拍接触使用每个物理子步的球拍中心位移和当前法向做扫掠近似，没有连续插值球拍转动。
- 球拍固定安装在右手，本任务没有实现左右手在线切换。
- 当前实现没有论文回球分支中的 MVAE 自回归运动生成器，也不代表真实机器人部署已经完成。
- 资产来源包没有附带单独的许可证说明；本地 provenance 文件只记录来源和哈希，不新增许可声明。

扩大环境数量或延长训练前，应先运行小规模 preflight，并观察 GPU 显存、有限数值、机器人
失败终止和球事件指标。若显存不足，逐步降低 `--env.scene.num-envs`，不要通过删除观测、
tracker 哈希检查或数据 schema 检查来换取启动成功。
