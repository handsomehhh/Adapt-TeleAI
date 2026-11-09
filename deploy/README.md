# G1 乒乓球 Python 部署

HOPE G1 当前仿真入口统一见 [仿真测试与回看指令](SIMULATION_TESTS.md)：包含策略指令评测、策略＋planner 接球评测、连续 MuJoCo 演示、真机日志回看及 planner 诊断。

常用命令：[基础 play 与 MuJoCo 测试](PLAY_AND_MUJOCO.md) ·
[真机部署](REAL_DEPLOYMENT.md) · [MuJoCo 详细说明](MUJOCO_TESTING.md)。

G1 hitter 实机的机器人、球轨迹与规划目标同屏显示，以及同步日志回放，见
[实机 Viser 可视化](hope_g1_hitter/LIVE_VISUALIZATION.md)。

2026-10-15 最新复核：已精确复现无扰动接球后摔倒，当前版本不具备通用真机部署验证结论。具体原因、代码修复与仍失败的对照见 [FALL_REVIEW_20261015.md](FALL_REVIEW_20261015.md)。
## 在 train 同源 play 中验证完整站立—挥拍—站立

2026-10-13 实测：固定 home=19、seed=42，每个动作一个独立环境，Tele-GMR
`model_29999.pt` 通过 **29/43**，Hard `model_59998.pt` 通过 **27/43**。
Tele-GMR 的 14 个失败中，3 个在挥拍、11 个在回位；Hard 的 16 个失败中，
挥拍和回位各 8 个。**两个模型全库均未通过完整流程验收。**

这里运行的是 `mjlab.scripts.play` 的 `ManagerBasedRlEnv`（与 train 同源），
不是 deploy 的原生 CPU MuJoCo。新增 `--standing-cycle True` 只调整验证调度：
tracker 冻结 home 首帧并至少准备 3 秒、连续站稳 1 秒，随后 Stage 2 执行
完整动作，tracker 回到 home 首帧并至少回位 3 秒、连续站稳 1 秒，最后再
连续稳定保持 5 秒。切换参考不重置机器人位姿、速度和 tracker 动作历史。
此模式关闭原先会自动 reset 的回合终止条件，改为实测失稳与阶段超时判失败。
它使用训练的动作输出，不加入 deploy 的 PREPARE 目标限幅。

站稳门限：最大关节速度 0.1 rad/s，最大躯干角速度分量 0.1 rad/s，倾斜
10°，anchor 速度 0.05 m/s，anchor 高度误差 0.1 m。倾斜超过 45°、高度
误差超过 0.25 m、非有限值、准备/回位超过 15 秒，或最终保持失稳均失败。
完整测试总时限 40 秒。Hard 恢复 checkpoint 的迭代数 59998，来球难度为
1.0，避免停留在难度渐增课程的起点。

以下命令在仓库根目录执行；均已运行并生成完整视频。动作 19 两模型均通过，
Tele-GMR 为 13.04 秒，Hard 为 13.60 秒。这个例子证明该动作能完成流程，
不代表全库、连续多球或其他随机种子也通过。

```bash
.venv/bin/python -m mjlab.scripts.play Mjlab-PingPong-Receive-G1-TeleGMR-Stage2-Adaptive \
  --checkpoint-file logs/rsl_rl/g1_pingpong_receive_stage2_tele_gmr/2026-10-04_22-19-39_g1_tele_gmr_stage2_30k_4096_seed42/model_29999.pt \
  --standing-cycle True --cycle-home 19 --cycle-motions 19 --cycle-seed 42 \
  --device cuda:0 --debug-vis False \
  --cycle-output logs/play_cycle_validation/tele_motion19 \
  --video-headless True --video-length 2000 --video-width 960 --video-height 720 \
  --video-folder logs/play_cycle_validation/tele_motion19

.venv/bin/python -m mjlab.scripts.play Mjlab-PingPong-Receive-G1-TeleGMR-Hard-Stage2-Adaptive \
  --checkpoint-file logs/rsl_rl/g1_pingpong_receive_stage2_tele_gmr_hard/2026-10-06_01-43-07_g1_tele_gmr_hard_stage2_30k_4096_warmstart_seed42/model_59998.pt \
  --standing-cycle True --cycle-home 19 --cycle-motions 19 --cycle-seed 42 \
  --device cuda:0 --debug-vis False \
  --cycle-output logs/play_cycle_validation/hard_motion19 \
  --video-headless True --video-length 2000 --video-width 960 --video-height 720 \
  --video-folder logs/play_cycle_validation/hard_motion19
```

输出目录含 `summary.json`、逐控制周期的 `trace.npz` 和
`rl-video-step-0.mp4`。上面的录制命令自动结束，不开启交互 viewer。
如需实时窗口且不保存视频，删除全部 `--video-*` 参数并添加
`--viewer native`。完整周期按实时速度播放，完成或关闭窗口后退出；中途关闭
不会把未完成的试验记为通过。窗口模式仍保存 summary 和轨迹数据。
复测全库：将 `--cycle-motions 19` 改为 `--cycle-motions all`，删除视频参数，
使用新的 `--cycle-output` 目录。无需指定 `--num-envs`，自动创建 43 个环境。
原始全库结果位于 `logs/play_cycle_validation/tele_all_seed42/summary.json`
和 `logs/play_cycle_validation/hard_all_seed42/summary.json`。
批量与单动作的随机采样序列不同，因此挥拍时长可能略有差异。

### 每个动作重复验证

在上述命令中使用 `--cycle-motions all --cycle-seed 42 --cycle-repeats 10`，
删除视频参数，并设置新的 `--cycle-output`，即可对每个动作执行 10 次独立
试验，使用种子 42–51。每次试验重新初始化；每次内部的站立、挥拍和回位
保持物理连续。动作起始姿态仍固定为 home=19，随机变化来自训练来球采样，
没有额外引入地面摩擦、机器人初始姿态或外力扰动。

本次批量输出根目录为 `logs/play_cycle_repeated_20261013`，两个模型分别保存
在 `tele/seed_42` 至 `tele/seed_51`、`hard/seed_42` 至 `hard/seed_51`。
每份 summary 同时记录 checkpoint/tracker 的 SHA256 和实际来球参数。
汇总命令会核验试验完整性及通过动作的逐步站稳指标，再输出每动作统计与图：

```bash
.venv/bin/python scripts/tools/summarize_standing_cycles.py \
  logs/play_cycle_repeated_20261013 --first-seed 42 --repeats 10
```

完整动作通过、击中球、有效回球分别统计；“10/10”仅表示这 10 次样本全部
通过，不能据此视为任意来球、连续多球或真机条件下的成功保证。

2026-10-13 共完成 **860 次**试验（两模型各 43 动作 × 10 次）；已独立检查
所有通过记录的阶段顺序、站稳指标和最后 5 秒保持，且每动作均有 10 个不同
的来球样本。结果文件为根目录下 `summary.json`、`per_motion.csv`、
`pass_counts.png`。两个模型使用各自的训练来球分布，Hard 分布更宽；以下
结果不能解释为同一来球难度下的模型优劣比较。

| 模型 | 完整流程通过 | 10/10 动作数 | 挥拍失败 | 回位失败 | 最后保持失败 |
| --- | --- | --- | --- | --- | --- |
| Tele-GMR | 294/430（68.4%） | 27 | 27 | 108 | 1 |
| Hard | 224/430（52.1%） | 5 | 113 | 93 | 0 |

两者均无准备阶段失败。逐动作通过次数如下；编号从 0 开始。

| 通过次数 | Tele-GMR 动作编号 | Hard 动作编号 |
| --- | --- | --- |
| 10/10 | 0,1,5,11,14,15,16,17,19,20,22,23,24,25,26,27,29,30,31,32,33,34,37,38,39,40,41 | 22,26,29,33,34 |
| 9/10 | — | 1,3,11,16,20,40,41 |
| 8/10 | — | 0,19,25,27 |
| 7/10 | 6 | 15,36 |
| 6/10 | 3 | 5,23,30,38,39 |
| 5/10 | 9 | 17,37 |
| 4/10 | — | 18,24 |
| 3/10 | 7 | 31,32,35 |
| 2/10 | 42 | 2,7,9 |
| 1/10 | 4 | 12,42 |
| 0/10 | 2,8,10,12,13,18,21,28,35,36 | 4,6,8,10,13,14,21,28 |

共同 10/10 的完整动作是 **22,26,29,33,34**；如果同时要求有效回球也每次
成功，则共同满足的是 **33,34**。原单次演示动作 19 在本批次 Tele-GMR 为
10/10，Hard 为 8/10（种子 44、45 在挥拍中失败）。

本批次 Hard 的 seed=42 为 26/43，与旧单轮记录 27/43 相差动作 32。
逐步轨迹显示起始状态一致，随后产生微小状态差，最终放大为回位结果差异；
因此固定 seed 也不应被理解为接触仿真结果逐位确定。旧单轮记录仍保留。

## 部署入口

在仓库根目录运行。直接加载训练 checkpoint、配套 tracker、训练配置和原始
motion 文件，不需要导出部署包。MuJoCo 与真机共用 `RallyController`；默认
MuJoCo 仿真不启动 ROS、UDP 或 Unitree DDS 订阅。

接球调度参考 [HOPE-TeleAI main，67a1a72](https://github.com/handsomehhh/HOPE-TeleAI/tree/67a1a720e985b01ec157821c9809e63bcbb13d9a)
的 `hope_deploy` 和随机来球流程。实现全部为 Python，不编译或调用 HOPE 的
A3 C++ 部署程序。采用本项目 G1 的两阶段模型与观测定义，不能替换为 A3
的 31 关节策略或球拍速度/法向输入。

## 部署增益

MuJoCo/real 执行 Kp/Kd 在 `g1_pingpong/pd_gains.py` 中逐关节写出，不从训练配置或 AMP metadata 提取。`configs/real.yaml` 的 `policy_pd` 与 `stable_policy.pd` 分别控制乒乓球和 START/AMP 的 scale。数值、覆盖规则与日志位置见 [PD_GAINS.md](PD_GAINS.md)。

## 先运行与真机一致的 MuJoCo 流程

最新完整指令及 R 重置、P 暂停、N 单步操作见 [MUJOCO_TESTING.md](MUJOCO_TESTING.md)。

`--sim-mode full`（默认）与 `standing` 现在直接调用真机的
`deploy.backends.unitree.run_real`。两者使用同一套 START/A/UP/DOWN 处理、
AMP、首帧准备、接管过渡、Stage1/Stage2 推理、动作历史、PD 和接球回位逻辑。
仿真适配器只提供物理世界、传感器、时钟及按钮，不直接修改策略状态。

```zsh
.venv/bin/python -m deploy.g1_pingpong.deploy_agent \
  --env mujoco --sim-mode full \
  --config deploy/configs/g1_tele_gmr.yaml \
  --real-config deploy/configs/real.yaml \
  --mocap-config deploy/configs/mocap_g1_live.yaml \
  --rally-config deploy/configs/mujoco.yaml \
  --motion-index 19 --motion-select auto --motion-allow 19,24 \
  --episodes 3 --duration 120 --seed 901 \
  --output deploy/recordings/mujoco_shared
```

默认自动依次按下 START、模拟落地后 A、稳定策略保持 3 秒后 UP；进入
乒乓球策略接管后即开始独立周期发球，不等待站稳或回位，
完成指定次数后保持 5 秒并停止。持续物理过程只初始化一次，回位不重置机器人。
自动操作模拟吊装与落地；进入 AMP 前解除支撑，之后不修改根位姿或速度。

手动可视化时增加 `--render --sim-controls manual --duration 600`，在 MuJoCo
窗口中按 **Enter=START，空格/P=暂停或继续，L=放下机器人，A=稳定策略，W=UP/乒乓球策略，
F=单球，T=连续发球，S=DOWN/稳定策略，R=重置，P=暂停，N=单步，B/X=停止，Q=退出**。按 A 前先按 L，等待落地。
只有橙色物理球可视化，不依赖旧版候选点标记。

只测试切换、保持且不发球，将 `--sim-mode full` 改成 `--sim-mode standing`。
`--sim-startup ground` 可从地面开始；默认 `suspended` 对应吊起摆姿后放下。
`--sim-policy-wait` 默认 0、只控制首球延迟；`--sim-feed-interval` 控制独立发球周期，默认 2 秒。手动按 T 开关连续发球，F 可随时发单球。默认物理步长 1ms，策略 20ms，
模拟命令应用 2ms，动捕位置输入 10ms。虚拟时钟不模拟真实主机计算耗时。

详细指令、对照指标和边界见
[SHARED_DEPLOYMENT_FLOW_20261015.md](SHARED_DEPLOYMENT_FLOW_20261015.md)。
当前共享流程在种子 901/902/903 下各完成 3 次挥拍、3 次回位、3 次仿真击球；
这些结果仅覆盖当前模型、模拟传感器与执行器条件。
旧版简化 `run_rally` 的 100 拍白名单属于历史结果，不能直接当作新版完整流程验证。
`--sim-mode episode` 保留为训练单回合评估，不含真机启动/AMP 切换。

## 完整接球流程

```text
WAIT → POSITIONING → POSE_HOLD → TO_STABLE → STABLE
     → POLICY_POSTURE → TO_POLICY → WAIT_BALL → EXECUTE → RECOVER → READY → WAIT_BALL
```

START 进入固定 PD 姿态；A 接入导入的 AMP 稳定策略；UP 平滑准备 home 第 0 帧，
再按 `real.yaml` 的 ramp/hold 进入乒乓球。DOWN 可返回 AMP。
`WAIT_BALL` 在 trace 中仍叫 `ARMED`，当前 `readiness_mode: timed` 下没有新增的实测站稳门控。

- 等待和回位：只运行 Stage1 tracker，冻结 home 第 0 帧；home 世界参考在首次接管时冻结。
- 触发挥拍：只依据估计球位置、速度和实际机器人状态选动作，不向控制器传入发球器采样的动作 ID。
- 挥拍：Stage1 基础动作加 Stage2 残差与参考速度，直到动作片段末尾。
- 回位：固定回位时间及等待计数与真机相同，保留控制历史和连续物理状态；不因仿真方便重置姿态。

两阶段是该 checkpoint 的训练结构；AMP 是进入乒乓球前独立的第三个策略。
球源默认是仿真物理球，经过同一位置估计器；`--ball-source mocap` 可改用外部
桥接提供的球位置，此时打开 UDP，机器人 root 仍取自 MuJoCo。
默认 `--ball-source sim` 不连接 ROS、UDP 或机器人 DDS。

真机保护随 `real.yaml` 使用；MuJoCo 按最新要求仅记录倾角/关节范围/速度等物理超限，策略和物理继续运行，不自动 HALT 或冻结。快速振荡自动停控仍关闭。
默认 Kp/Kd 和动作尺度不变。仿真日志中的延迟来自虚拟时间，不是硬件延迟测量。

## checkpoint 与动作切换

`--checkpoint` / `--policy` 覆盖普通 YAML 中的路径，可直接加载 `model_N.pt`
或已有 JIT 文件。默认从对应 run 找 `tracker.pt`、`params/env.yaml` 和
`params/agent.yaml`，没有复制/打包步骤。

```bash
uv run --no-sync python -m deploy.g1_pingpong.deploy_agent \
  --env mujoco --config deploy/configs/g1_tele_gmr.yaml \
  --checkpoint logs/rsl_rl/g1_pingpong_receive_stage2_tele_gmr/2026-10-04_22-19-39_g1_tele_gmr_stage2_30k_4096_seed42/model_28000.pt \
  --motion-index 5 --motion-allow 5,13 --episodes 10 --render
```

模型或配置搬到其他目录时，用 `--tracker`、`--env-config`、`--agent-config`
和 `--motion-dir` 指定原文件。跨任务切换需选匹配的 task/config；同一 run
内换 checkpoint 无需修改其他路径。

| 参数 | 含义 |
| --- | --- |
| `--motion-index 5` | 初始及回位 home 动作，fixed 模式也执行此动作 |
| `--motion-select auto` | 根据预测击球点在候选库中选择动作 |
| `--motion-select fixed` | 始终使用 home 动作 |
| `--motion-allow 5,13` | 候选 ID 白名单；ID 按训练保存的 motion 列表排序 |
| `--motion-allow all` | 开放整个动作库；功能支持，但尚未全部通过连续回位验证 |
| `--motion-file PATH` | 使用单个外部动作，同时设置 `--motion-index 0 --motion-allow 0` |
| `--episodes 10` | 要求完成并稳定回位的挥拍次数；不是每拍重置的 episode |
| `--duration 120` | 仿真秒数上限，或外部输入/真机的运行时间上限 |
| `--seed 42` | 原生随机来球的可复现种子 |
| `--sim-config PATH` | 接球调度、候选动作、触发门限、随机来球参数 |
| `--ball-source sim` | 原生物理随机球；默认值 |
| `--ball-source mocap` | 从明确配置的 ROS/UDP 位置输入感知球 |

修改 `configs/mujoco.yaml` 的 `feed.speed_range`、`lateral_speed_range`、
`intercept_jitter` 可调整来球分布。CLI 默认 home 已统一为 19，共享配置
默认白名单为 `[19,24]`；也可显式指定相同参数。`--rally-config` 与旧参数
`--sim-config` 等价，两种后端读取同一份配置。换候选集也会改变随机发球区域，不能把不同白名单的
同种子结果理解为完全相同来球的对照。

默认 `--motion-select auto`。单独指定 `--motion-index 19` 只改变 home，
挥拍仍从配置的候选库选择。要明确测试动作 19 本身，应指定
`--motion-select fixed --motion-index 19`；固定模式的候选就是 `[19]`，不受
自动模式白名单影响。启动日志会分别打印 `home/prepare/return`、
`swing_select` 和 `swing_candidates`。

## 输出与验收

当前 full/standing 的 `--output DIR` 生成与真机同结构的 `trace.jsonl`、
`summary.json`，以及启用的 `lowstate/`、`command_writes/` 二进制日志和
`mocap_packets.jsonl`。summary 额外记录 simulation 配置、生成球数、
`simulated_hits`、完成/回位次数与 `passed`；不沿用旧版 `swings[].success` 字段。

full 自动模式的 `passed=true` 表示经过策略接管并完成指定回位次数及最后保持，
没有触发控制异常；不表示每球击中，也不是额外的实测站稳门控。standing 和
手动模式仅表示已进入策略且未发生控制异常，不要求完成发球次数。
超时未完成指定回位次数判失败；异常时停止控制，错误和故障快照写入日志。手动可视化在策略结束后继续阻尼物理，可按 R 重试或 Q 退出；物理阈值超限本身不会结束仿真策略。不再使用旧版 HALT 后持续仿真及 `post_halt_max_tilt_degrees` 字段。

仿真时间、模拟反馈和模拟命令应用均明确标记，不作为真实网络/执行器延迟测量。

## 历史简化流程的长序列筛选

以下筛选工具仍运行旧 `run_rally`，其结果不能替代当前共享真机流程的验证。

批量长序列测试使用相同的 `run_rally`，每条序列独立初始化一次，序列内
不重置机器人。例如复跑本轮 43 动作、每动作三个种子的 50 拍筛选：

```bash
uv run --no-sync python -m deploy.tools.screen_rally \
  --config deploy/configs/g1_tele_gmr.yaml \
  --cases deploy/recordings/long_screen_20261013/isolated_cases.json \
  --output deploy/recordings/long_screen_rerun --workers 4
```

`--cases` 为 JSON 列表，每项包含 `name`、`motions`、`seed`，以及可选的
`home`（默认 19）、`swings`（默认 50）、`duration`（默认 1200 秒）。
每项最终保持 5 秒；输出目录必须不存在。报告保留每个动作实际选中和回位
序列、动作切换次数、物理击球数、失败原因、配置副本和代码哈希。
任何一项未通过则最终退出码为 1，但仍运行其余测试并保留结果。
不能仅凭候选集合中的 ID 认定动作已覆盖，必须查看实际 `sequence`。

## 动捕与真机接口

本轮首先验证原生 MuJoCo。动捕输入和 DDS 后端保留，尚未在当前现场验证；
MuJoCo 默认运行不需要连接这些设备。

```text
Avatar-Pro/青瞳 VRPN
  → 外部 vrpn_mocap（仅发布指定球/机器人 topic）
  → Python relay（坐标变换）
  → /ball/point + /P1/pose
  → UDP bridge 或直接 ROS 2 订阅
  → BallEstimator / ReceivePlanner / RallyController
  → MuJoCo 或 Unitree SDK2
```

`configs/mocap.yaml` 配置实际刚体 topic、坐标变换、pelvis marker 偏移、
时间戳及 UDP 地址。relay 接收 `PoseStamped`，输出球 `PointStamped` 和
机器人 `PoseStamped`；同时保留 `/poses` 供兼容工具使用。
现有 HOPE `vrpn_to_pose` 输出可以复用，但需要核对消息类型和坐标系。

本项目采用 G1 训练世界坐标：桌面高 `z=0.76 m`，X 朝对手；HOPE 的桌面
坐标 `z=0` 必须显式变换后输入。不能直接把 HOPE 的数字填入击球面配置。
本接球控制器使用 Python planner，不直接消费 A3 的 `/racket/command`。

### 接入之前的 HOPE 四终端流程

终端 1、2 保留现有 VRPN 和 `vrpn_to_pose`。终端 3 改为本项目的桥接；
终端 4 运行 G1 Python 部署。旧 A3 planner、AimRT、EtherCAT、elink 和 A3
部署命令不参与这条 G1 路径。

```text
终端1：192.168.1.101:3883 → 过滤后的两个 VRPN topic
终端2：vrpn_to_pose → /poses + 实际机器人的 /P1/pose（或 /P2/pose）
终端3：mocap bridge → 本机 UDP 127.0.0.1:15050（统一 G1 坐标）
终端4：BallEstimator → ReceivePlanner → RallyController → G1 SDK2
```

终端 1、2、3 均使用以下 ROS 环境；此前记录的 `ROS_LOCALHOST_ONLY=23`
应改为 **0**。SDK2 的 `real.yaml:domain_id=0` 与 ROS 的 domain 23 独立。

```bash
export HOPE_WS=/home/lxz/HOPE/hope_ws  # 改为这台主机现有工作区
source /opt/ros/humble/setup.zsh
source "$HOPE_WS/install/setup.zsh"
export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

终端 1（保留源头过滤，不能只在部署进程中过滤海量话题）：

```bash
cd "$HOPE_WS"
ros2 launch ./src/vrpn_mocap/vrpn_mocap/launch/client.launch.yaml \
  server:=192.168.1.101 port:=3883 \
  allowed_topics:="/vrpn_mocap/MCAvatar/pose7,/vrpn_mocap/MCAvatar/pose30310"
```

终端 2：

```bash
ros2 launch vrpn_to_pose vrpn_to_pose.launch.py robot_side:=P2
```

终端 3（在本项目根目录）：

```bash
cd /home/lhy/table-tennis/Adapt-TeleAI
uv run --no-project --python /usr/bin/python3 python -m deploy.io.mocap bridge \
  --config deploy/configs/mocap_hope.yaml
```

这里使用与 Humble 匹配、已安装 `numpy` / `PyYAML` 的 ROS Python；不要求
策略环境安装 `rclpy`。桥接只订阅配置中的球和机器人两个话题，best-effort、
depth=1，保留原采样时间，拒绝过期、错误 frame、缺失球索引和非有限位置。
可将同一配置的 `transport` 改为 `ros2` 直接订阅；两种方式共用转换函数。

`mocap_hope.yaml` 明确列出了需要与现场转换器对应的项目：

- 默认 `/poses` 是 `PoseArray`，球为 `poses[0]`，与本机 HOPE planner 的默认
  约定一致；旧 `vrpn_to_pose` 源码不在本机仓库中，现场必须核对数组顺序。
- 默认机器人话题为记录中的 `/P1/pose`、类型 `PoseStamped`。`robot_side:=P2`
  不足以推断机器人输出话题；如果实际为 `/P2/pose`，修改 `source.root_topic`。
- `source.frame` 必须匹配消息 header。若球改用 `/ball/point`，设置
  `source.ball_topic: /ball/point`、`source.ball_type: point_stamped`。
- 模板平移 `[0,0,0.76]` 仅在 HOPE 桌面 z=0、XY 已对齐时适用。训练桌边
  x=0.65/3.39 m，X 朝对手；应根据桌面标定修改同一个旋转/平移，同时作用于
  球与机器人。已经输出 G1 地面坐标时用单位变换，不能再加一次 0.76。
- `root_marker_to_pelvis` 及姿态偏移只用于尚未转成 pelvis 的原始刚体。
  上游已经输出 pelvis 时用零偏移、单位旋转，不能重复补偿安装偏移。

桥接输出已经是 G1 坐标，UDP 接收端只做刚体到 pelvis 的偏移，不再重复
世界坐标转换。不要同时运行旧 relay 和 HOPE bridge 去转换同一份数据。

终端 4：先只读检查 G1 状态与动捕，无电机命令：

```bash
export G1_IFACE=你的PCI网卡名  # ip -br addr 中直连 G1 的网卡，不是 IP 字符串
uv run --no-sync python -m deploy.g1_pingpong.deploy_agent \
  --env real --config deploy/configs/g1_tele_gmr.yaml \
  --mocap-config deploy/configs/mocap_hope.yaml \
  --rally-config deploy/configs/mujoco.yaml \
  --network-interface "$G1_IFACE" \
  --motion-index 19 --motion-select auto --motion-allow 19,24 \
  --duration 60 --output deploy/recordings/real_readonly
```

实际进入控制时，在同一命令中加入 `--enable-motors`，延长 `--duration`，
使用新的输出目录。按 **START** 进入 `POSITIONING`，只用 PD 平滑摆到
225 `mjlab-prior/deploy_lhy` AMP 模型的默认站姿，并保持 `POSE_HOLD`。
放稳到地面后按 **A**，进入 AMP 零速度稳定策略；站稳后按 **UP**，平滑
切入乒乓球策略，过渡关节目标、PD 增益和动作准备姿态，固定 ramp/hold 完成后进入
`WAIT_BALL` 自动接球（trace 中仍叫 `ARMED`，不增加实测站稳门控）。**DOWN** 平滑返回稳定策略；**SELECT/B/X** 停止。
详细的三个终端指令和手柄操作见 [REAL_DEPLOYMENT.md](REAL_DEPLOYMENT.md)。

真机输出也支持目录形式的 `trace.jsonl`、`summary.json`，保留选动作、
固定 home、挥拍结束、回位事件与拒绝原因。旧 `.jsonl` 文件路径仍兼容。
真机按 `--duration` 或手柄停止；`--episodes` 和最终保持验收属于 MuJoCo。
只读过程不计算或发送策略目标，summary 不宣称通过真机运动验收。

`real.yaml` 的 `stop_policy: hope_67a1a72` 仅在真机入口生效，对齐公开
HOPE-TeleAI `67a1a72` 的停止行为；MuJoCo 保持原有配置与退出规则。
生效值记录在 `summary.json` 的 `real_runtime_config`，不是直接改写 YAML
里的旧阈值：反馈超时 300ms、策略倾斜阈值 35°；动捕 200ms 后保持最后
位置，最多再容忍 500ms，期间姿态回退到 IMU。故障后锁定纯阻尼
（Kp=0、Kd=乒乓球策略执行 Kd），IO 正常时持续发送，不自动恢复策略；
SELECT/B/X、Ctrl-C 或时长到期结束程序。底层通信失败时不能保证阻尼送达。
不再因 50ms 循环迟到、额外的实测角度/速度/跳变阈值或姿态交接收敛
超时退出；NaN/Inf 和无效输入仍触发保护。可设 `stop_policy: legacy`
恢复原退出行为。G1 特有的模式检查仍保留，独立发送线程的指令超时
设为 300ms，避免控制器失联后无限重发旧位置目标。

HOPE 使用锁定的球拍目标，本项目 Stage2 还需要球状态：挥拍中球观测
丢失时保留已锁定的接球计划，暂时由 Stage1 tracker 跟踪参考动作，
不把外推球当作实测输入交给 Stage2。最后有效球及原始时间戳仅供外推
诊断，不伪造新观测，也不因此立即停控。首次进入策略前仍需
至少一帧有效根部动捕，不能凭空构造真机世界坐标。

省略 `--duration` 仍是 120 秒；真机可用 `--duration inf` 持续运行。
上述对齐不改变显式时长上限，也不更改 PD、策略输出或关节映射。

外部 ROS 环境使用 `ROS_DOMAIN_ID=23`、`ROS_LOCALHOST_ONLY=0`，例如：

```bash
# 在已 source ROS 的终端运行，不要求策略 .venv 安装 ROS。
uv run --no-project --python /usr/bin/python3 python -m deploy.io.mocap relay --config deploy/configs/mocap.yaml
uv run --no-project --python /usr/bin/python3 python -m deploy.io.mocap bridge --config deploy/configs/mocap.yaml
```

如果现场转换包已发布正确类型、已标定的 `/ball/point` 和 `/P1/pose`，只
运行 bridge。将 ball-source 改为 mocap 可在 MuJoCo 用同一个控制器验证外部
来球；球轨迹由外部输入决定，此模式不用于证明实际回球的物理闭环。

真机配置位于 `configs/real.yaml`，Unitree SDK2 可通过
`bash deploy/scripts/install_sdk2.sh` 安装；原生 MuJoCo 无此依赖。`--env real`
默认只读，只有显式 `--enable-motors` 才创建命令出口。真机使用 START
进行默认站姿 PD 摆姿、A 进入 AMP 稳定策略、UP 进入乒乓球策略、DOWN 返回稳定策略、SELECT/B/X 停止；第一次允许后，稳定回位再自动接下一球。

不启动任何通道的预检：

```bash
uv run --no-sync python -m deploy.tools.preflight \
  --config deploy/configs/g1_tele_gmr.yaml
```

真机网络、消息周期、坐标标定和电机响应尚需现场验证，仿真通过不覆盖这些
环节。本次没有启动真实动捕订阅或发送真机命令。

## 文件结构与诊断工具

| 目录/文件 | 职责 |
| --- | --- |
| `g1_pingpong/deploy_agent.py` | 单一 CLI，直接选择 checkpoint |
| `g1_pingpong/rally.py` | 仿真、真机共用的接球调度 |
| `g1_pingpong/standing.py` | 冻结首帧反馈、实测稳定门限 |
| `g1_pingpong/motions.py` | 动作库、击球点映射、可选回位参考混合 |
| `g1_pingpong/perception.py` | 球速度估计、反弹预测、时序锁定 |
| `g1_pingpong/core.py` / `policy.py` / `configuration.py` | 原训练观测/动作契约、直接模型加载 |
| `g1_pingpong/simulation.py` / `feeder.py` | 唯一连续 MuJoCo 循环、物理随机来球 |
| `backends/` | MuJoCo 与 Unitree SDK2 状态/控制接口 |
| `io/` | 动捕 relay、ROS/UDP 输入 |
| `tools/` | 预检、标定、同输入对拍、独立回合评估 |
| `tests/` | 调度、数据契约、输入时间戳、模拟 DDS 的回归测试 |

独立单回合评估保留为 `--sim-mode episode`，可配合 `--all-motions` 做训练
指标对照；这种模式会按回合 reset，不能代替 full 连续流程验收。
训练输入对拍和批量评估入口为 `python -m deploy.tools.validate_runs`，
合成接口自检为 `python -m deploy.tools.selfcheck --output NEW_DIR`。

正式 real 入口的离线闭环检查（使用真实 checkpoint、MuJoCo 反馈和 HOPE
消息，不开启 SDK/ROS 通道、不发送机器人命令）：

```bash
uv run --no-sync python -m deploy.tools.check_real_loop \
  --config deploy/configs/g1_tele_gmr.yaml \
  --output deploy/recordings/real_loop_offline_check
```

该检查包含三种子连续接球、无球、只读、动捕/球/DDS 中断和停止请求，
使用虚拟时钟，仅验证控制与数据契约，不代替真实设备实时性测试。

已删除重复的 `full_mujoco.py`、`live_mujoco.py`、旧独立 live 验收入口及
`bundles/` 导出副本。原 checkpoint、数据集及已有 `recordings/` 均保留。

```bash
uv run --no-sync pytest deploy/tests -q
uv run --no-sync ruff check deploy
uv run --no-sync ruff format --check deploy
uv run --no-sync ty check deploy --config 'src.include = ["deploy"]' --ignore unresolved-import
```

最后一条的 import 忽略仅用于环境中可选 ROS/SDK 依赖。现场依赖是否可用
应另用 `preflight --require-real --network-interface 实际网卡名` 检查。
