# HOPEPingPongMultiMotionG1 部署

独立随机目标指令评测使用新入口 `deploy.benchmarks.racket_command.run`，详见
[随机 racket command 测试](../benchmarks/racket_command/README.md)。它按固定平面网格
采样位置、朝向、速度与 TTS，范围由动作及训练随机化离线导出；采样器只读取域JSON，
不从选中的 motion 反推目标。支持 MuJoCo 物理球接触/反弹展示与能力热力图。
独立随机来球的第二阶段使用 `deploy.benchmarks.receive.run`，见
[中心区域接球测试](../benchmarks/receive/README.md) 和
[首批实际MuJoCo结果](../benchmarks/receive/RESULTS_20261021.md)。
MuJoCo 连续部署使用本任务入口 `deploy.hope_g1.deploy_agent`，不调用上述评测环境。
它保留手动进入策略的流程，运行中机器人状态、policy 历史和回位过程连续保留。
旧配置 `rally_mujoco.yaml` 的 `motion_plane` 仅保留为历史回归，需显式指定。

真机按终端拆分的启动命令见 [REAL_DEPLOYMENT_QUICKSTART.md](REAL_DEPLOYMENT_QUICKSTART.md)。

本任务使用独立入口 `deploy.hope_g1.deploy_agent` 和 `HopeController`，直接加载
HOPE 的 `model_18000.pt`，不经过 mjlab Stage1 tracker 或 Stage2 actor。
原任务继续使用 `deploy.g1_pingpong.deploy_agent` 和 `deploy/configs/`。

共享部分是 G1 资产、DDS/UDP、AMP、遥控器接管、保护、记录和 MuJoCo IO。
任务配置全部位于本目录的 `configs/`；默认记录写入
`deploy/recordings/hope_g1/<env>_<时间>/`，summary 明确记录 task、controller 和 checkpoint SHA256。

默认使用 A30808 planner → RacketCommand → motion selector → actor，使用 raw 目标模式。
MuJoCo 配置在部署进程内运行相同 Python 核心；真机配置使用本仓库 `ros2_planner/`
下的独立 ROS planner 包，由 `/poses` 输入并发布 `/racket/command`，部署端经 ROS→UDP 桥接消费命令。
此前物理配置下的本机三种子验收为 **30/30 碰拍、30/30 越网后首次落对方台面**；
该历史结果不代表下述 2026-10-21 A3 参数统一后的表现。
进入策略的流程保持原样；历史 A3 对齐证据见 [A3_ALIGNMENT_RESULT.md](A3_ALIGNMENT_RESULT.md)。

## 当前 planner（A30808）

MuJoCo 部署直接调用 `hope_planner_optimized_2_A30808` 的原始 Stage 1/2/3 Python 核心。
真机部署不在控制进程计算 planner：本仓库 ROS 包中的独立节点订阅 `/poses` 并发布
`/racket/command`，现有 `deploy.io.mocap bridge` 再把完整 command 转发到控制进程的本机 UDP 接收器。
因此球输入、planner 输出和机器人动作可以分开观察，command→motion selector→actor 流程保持一致。

部署 planner 使用与 A30808 相同的击球平面：`x=0 m`，发布框的 X min/max 同为 `0 m`；
球桌、球坐标和目标落点不做平移或机器人专属重算。其余 Stage 1/2/3 参数来自
`configs/hope_planner_optimized_2.yaml`。真机使用本仓库的 ROS 2 package/launch 启动 planner，
不修改独立 HOPE workspace。上游逐文件对齐记录见
[`A30808_PROVENANCE.json`](A30808_PROVENANCE.json)。
`a3_planner` 仅作为旧命令行/配置别名，运行时也会转到 A30808 核心。

## 当前坐标与物理配置（2026-10-21）

控制器、真实动捕、球预测、球拍目标均使用 **HOPE 桌面世界系**：近端左角为原点，
桌面 X=[0,2.74]、Y=[-1.525,0]、Z=0；地面Z=-0.76。
真实 `/P1/pose` 与 `/poses[0]` 已经是 P2 换边后的这个系，不再加0.76m。
MuJoCo 物理模型仍保持地面Z=0，通过 `[-0.65,-0.7625,-0.76]` 的模拟动捕桥接
进入 HOPE 系。球和根位置一起转换，速度、朝向和电机指令不变。
馈球参考以逆变换进入物理模型；球桌可视化与物理碰撞仍处于同一 MuJoCo 系。

- `configs/hope_planner_optimized_2.yaml`：A30808 ROS planner 的完整上游参数。
- `configs/a3_planner.yaml`：历史 A3 launch 配置快照，仅用于对照。
- `configs/rally.yaml`：真机入口，与 A30808 planner 对齐：击球平面X=0，回球目标为[2.4,-0.7,0]。
- `configs/rally_mujoco_random.yaml`：默认连续 MuJoCo 部署，全区域独立随机来球，planner 参数读取 hope_planner_optimized_2.yaml。
- `configs/rally_mujoco.yaml`：历史 motion_plane 回归配置；不会作为默认部署配置。
- `configs/rally_a3_reference.yaml`：保留的历史配置入口，planner 仍使用 A30808 参数。

当前 A30808 planner 参数见 `configs/hope_planner_optimized_2.yaml`：目标落点、飞行时间、
drag、桌面反弹系数、球拍恢复系数、积分步长及 Y/Z 发布范围均沿用 ROS planner 配置。
planner 击球平面和发布 X 框沿用 A30808 的 `0 m`。执行 motion 的启动
TTS 上限仍由 `rally*.yaml` 中的 selector 配置控制（正式值0.5s），它不改 planner 预测。
旧的现场标定记录仍保存在 [REAL_PHYSICS_CALIBRATION.json](REAL_PHYSICS_CALIBRATION.json)，
其参数不再作为当前默认。`.200` HOPE 仓库仅只读，没有同步本地修改。
planner 自己使用 A30808 Stage 1 的30点二次拟合窗口、10点启动阈值及其源码中的反弹/清窗规则；
部署的 `BallEstimate` 保留给旧控制路径，不作为 A30808 的 planner 输入。球参数配置不改 actor 或 PD。
训练物理下的30/30不代表真实球物理下的同等回球率，更不代表真机验收。

## 主机分工与迁移

本机 `/home/lhy/table-tennis/Adapt-TeleAI` 负责 MuJoCo；
`.200` 的 `/home/lxz/deploy_g1_pingpong` 负责真机。

本机已迁入 `artifacts/hope_20261012_18000/`：`model_18000.pt`、
`params/env.yaml`、`params/agent.yaml`、22 个 NPZ 动作及 `hit_type.json`。
`migration_manifest.json` 保存来源和逐文件 SHA256；26 个文件均已校验。
另需 `robot/g1_29dof_pingpong_fast.urdf`（同目录 manifest 保存 SHA256），
用于重建训练碰撞几何和开启自碰撞，不能漏同步。
本机 AMP 和 G1 资产也已与 `.200` 核对一致。本机测试不依赖 SSH、`.200`、ROS 或 Isaac。
大型资源位于已忽略的 `artifacts/` 目录，不纳入代码提交。

配置按主机分开：`configs/task.yaml`、`configs/task_real.yaml` 均使用部署仓库内相对路径。
`.200` 的同步范围仅限 `/home/lxz/deploy_g1_pingpong`，
不修改 `/home/lxz/HOPE`，部署也不再依赖该目录。
2026-10-20 真机同步与操作步骤见 [REAL_DEPLOYMENT.md](REAL_DEPLOYMENT.md)。
入口在 check/MuJoCo 时默认前者，real 时默认后者；也可显式传 `--config`。
检查 `.200` 配置时应显式使用 `--config deploy/hope_g1/configs/task_real.yaml`。

## 本机 MuJoCo 测试

先在本机检查资源（不连接机器人）：

```zsh
cd /home/lhy/table-tennis/Adapt-TeleAI
.venv/bin/python -m deploy.hope_g1.deploy_agent --env check
```

动作库严格遵循保存配置的顺序，共 22 个动作，合法索引为 0..21；
本任务的 19 是 `0804_2_kjp-026.npz`，不能沿用旧任务的 19/24 组合。
默认 home=0（`0623_kjp-008.npz`），固定动作；检查输出包含完整索引表。

打开手动测试窗口：

场景显示蓝色球桌、球网和白色边线；桌面、球网尺寸直接读取同一份
所选 rally 配置的 ball 参数。默认视角同时覆盖机器人和球桌，
可以在 `viewer` 配置中修改 lookat、distance、elevation、azimuth。

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --render --sim-controls manual \
  --rally-config deploy/hope_g1/configs/rally_mujoco_random.yaml \
  --motion-index 0 --motion-select auto --motion-allow all --duration 600 \
  --seed 7 \
  --output "deploy/recordings/hope_g1/continuous_$(date +%Y%m%d_%H%M%S)"
```

在 MuJoCo 窗口依次操作（手动模式必须按 L 放下机器人）：

1. **ENTER**：START，从悬挂姿态进入 `POSITIONING`，等待 `POSE_HOLD`。
2. **L**：1.5s 平滑降低支撑，等待终端显示 `support released`。
3. **A**：切换 AMP，等待 `STABLE`。
4. **W**：相当于真机方向键 UP，上身准备后进入 `TO_POLICY` 和 `WAIT_BALL`。
5. 进入接球策略后**自动连续随机发球**，默认每 4s 一球，观察 `EXECUTE → RECOVER → WAIT_BALL`。
   **T** 暂停/恢复连续发球（已经飞出的球继续运动）；**F** 额外发一球。
6. **S** 返回 AMP；**B/X** 停控；**R** 重置；**Q** 退出。
   SPACE/P 暂停或继续，N 单步。

所有 deploy MuJoCo 测试窗口使用相同的播放速度键：`[` 或 `-` 减半，
`]` 或 `+` 加倍，反斜杠 `\` 恢复1倍。范围为1/16～8倍；先点击仿真窗口
再按键，当前倍速显示在窗口/终端。变速只调整墙钟等待，不改物理步长、
policy频率、TTS、传感器时间戳或发球间隔的仿真秒数。原来已开启的进程需重新启动。
这也适用于独立command评测、接球评测及真机球轨迹驱动的人形测试。

自动流程测试不需要按键；加 `--render` 可以观看同一自动流程：

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --motion-index 0 --motion-select auto \
  --target-source hope_planner_optimized_2 --a3-target-mode raw \
  --sim-controls auto --motion-allow all --episodes 10 --duration 70
```

`--standing-only` 可检查 AMP → HOPE 首帧保持，不发球。
配置 `--min-return-rate` 后，退出码 0 要求球先越网、首次落在对方台面，达到指定比例，
同时完成回位且没有检测到失去站立。`--min-hit-rate` 只检查碰拍比例，不能替代有效回球验收。
summary 分别记录 `flow_passed`、`hit_rate`、`legal_return_rate`、`simulated_legal_returns`，
并在 `ball_returns` 中保存逐球碰撞前后速度、过网点、第一次桌面高度交点和出球轨迹。
未配置比例门槛时只验收流程。历史 motion_plane 测试的成功率不能用于说明全区域随机来球能力。
每次运行保存 `rally_config.yaml`，记录命令行覆盖后的实际配置。

默认 `feed.mode=random_launch`，由 `deploy.hope_g1.random_feeder.RandomLaunchFeeder`
独立采样初始球位置和速度，只根据过网、我方半台一次反弹以及穿过配置区域筛选来球。
不读取 motion、planner 输出或策略击球结果，不反向重建轨迹。

`configs/rally_mujoco_random.yaml` 中：

- `feed.plane_x=-0.25`，`feed.y_range=[-1.51225,0.00200]`、
  `feed.z_range=[-0.16225,0.71278]`（hope_table，桌面 Z=0）。
  对应此前裁剪最低 Z 层后的完整包络，`feed.grid=[8,5]`，窗口按同一配置画无文字网格。
  包络不是策略成功保证；其中也可能有训练支持较少或没有支持的位置。
- 每批随机球覆盖全部 40 个网格，每格 `feed.balls_per_cell=3`，打乱顺序连续发送，
  一批耗尽后重新随机生成；随机种子由 `--seed` 指定。
- `feed.launch` 控制初始位置和速度范围；`feed.interval_seconds=4` 控制发球间隔，
  调度不等待动作完成或回位，未接到的球不会自动重试。
- planner 击球平面为 `a3.hit_plane_x=0`，发球采样平面仍为 `feed.plane_x=-0.25`。
  A30808 的回球目标读取 `hope_planner_optimized_2.yaml` 中的 `[2.4,-0.7,0]`；
  `ball.target_xy` 是旧配置字段，不覆盖 A30808 参数。

发球器、部署 IO 和控制器只共用通用采样/动力学代码，不导入 `deploy.benchmarks`。
评测模块保留自己的配置、快照重置、计分和报表，仍使用单独入口。
运行记录默认在 `deploy/recordings/hope_g1/mujoco_<时间>/`；
`summary.json` 的 `ball_returns[].feed` 保存实际随机初始状态和所属网格。
机器人不会逐球重置；仅操作者主动按 **R** 才重启整场部署。

## .200 真机测试

在 `.200` 上操作。2026-10-19 本次只读检查时，`eno1` 是 NO-CARRIER 且没有 IPv4，
DDS 报 `eno1: does not match an available interface`，未创建电机控制发布器。
应先开启机器人并接通 `eno1` 网线，再检查机器人网段。
已确认 `.200` 保存了 `g1-direct-eno1` 配置，地址为 `192.168.123.210/24`。
接线后若未自动恢复地址，由操作者执行：

```zsh
sudo nmcli connection up g1-direct-eno1
ip -brief address show eno1
ping -I eno1 -c 3 192.168.123.164
```

不要把当前动捕网络的 USB 网卡 `enx98fc84e70fe2` 当作机器人网口。
本次没有修改任何网络配置。

再检查当前 ROS 节点；已有的进程保持运行，不要重复启动：

```zsh
source /opt/ros/humble/setup.zsh
export ROS_DOMAIN_ID=23 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 node list
ros2 topic info /P1/pose
ros2 topic echo /P1/pose --once --qos-reliability best_effort
```

上游 VRPN/转换器由既有动捕系统提供，本任务不修改或管理 `.200` 的 HOPE 工作区。
只确认已有 `/P1/pose`、`/poses` 输出。需要时在部署目录使用 ROS→UDP 桥接：

```zsh
source /opt/ros/humble/setup.zsh
export ROS_DOMAIN_ID=23 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd /home/lxz/deploy_g1_pingpong
/usr/bin/python3 -m deploy.io.mocap bridge \
  --config deploy/hope_g1/configs/mocap.yaml
```

本任务的 `configs/mocap.yaml` 与已标定的 live 链路保持一致：
`/P1/pose`（map）、`/poses`（world）→ `127.0.0.1:15050`。
旧 bridge 必须退出并使用本任务配置重启：归一化 UDP 帧现为 hope_table，旧 world 帧会被拒收。

终端 4：先检查 `.200` 的策略配置：

```zsh
cd /home/lxz/deploy_g1_pingpong
.venv/bin/python -m deploy.hope_g1.deploy_agent --env check \
  --config deploy/hope_g1/configs/task_real.yaml
```

不加 `--enable-motors` 时仅接收/记录：

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env real --network-interface eno1 --duration 10
```

确认输出 `root_valid=true`、`ball_valid=true`，没有时间戳或坐标系错误。
等待只读进程结束后，再启动电机控制。

真机控制命令（由操作者手动执行）：

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env real --network-interface eno1 \
  --motion-index 0 --motion-select auto --motion-allow all \
  --target-source hope_planner_optimized_2 --a3-target-mode raw --ball-source mocap \
  --duration 600 --enable-motors
```

依次按 START，等待 `POSE_HOLD`；按 A，等待 `STABLE`；确认机器人站稳后按 UP，
经过 `POLICY_POSTURE → TO_POLICY → WAIT_BALL` 后再送球。
挥拍后自动回位，等待下一球；DOWN 返回 AMP；SELECT/B/X 停控。
真机不使用仿真窗口的 L 键。与旧任务电机控制入口、UDP monitor 不能同时运行。
自动选动作可用 `--motion-select auto --motion-allow all` 或本任务的有效索引列表。
真机记录默认在 `deploy/recordings/hope_g1/real_<时间>/`。

## 推理和运行约定

- 175 维输入、29 维动作，ELU MLP，50 Hz，归一化为
  `(obs - saved_mean) / (saved_std + 0.01)`；独立读取 actor，忽略 critic。
- 顺序、默认姿态、默认关节速度、动作 scale、动作文件及训练回位 75 帧来自保存配置。
  部署通过 `rally*.yaml` 的 `return_steps: 10` 覆盖接球回位等待。
  执行 PD 沿用公共显式增益表；加载时核对其与训练 PD 一致，允许独立配置中的显式 scale。
- 动捕根位姿按配置做逐源样本 pose EMA / 跳变处理，按源时间戳差分、限速及速度 EMA，
  使用与位置同一源样本的速度，再旋转到 pelvis 系；首次为零。
  这是真机状态估计，不等同于 Isaac 提供的无噪声速度。
- 关节参考包含位置和速度；anchor 使用完整相对旋转；机身目标和拍面位置使用 yaw-heading 系，
  期望拍面速度/法向在物理世界中用于规划，送入 actor 前逆变换回训练 world；
  击球后的剩余时间保留负值。按 A3 在切换挥拍动作时按当前 torso 做纯 yaw rebase，
  等球/回位的 home 变换固定，不重置真实机器人状态。
- A30808 planner 同时生成接触点、物理拍速、击球面法向及 deadline。
  raw 模式中物理拍速不翻转；反手只翻转击球面法向，映射为本地 +Y。
  目标在挥拍开始时锁定，后续丢球不换目标/actor；回位使用固定 home。
- 部署采用固定 world home、等球时冻结首帧、挥拍结束后保持 home 10 帧（0.2s）再接球。
  `a3.select_settle_ticks: 10` 与回位窗口重叠，不额外再等 0.2s。
  这是部署调度；训练中的随机换动作和目标扰动不会在部署时随机执行。
  raw actor action 连续保存，AMP 退出/重新进入时清零。

当前验证覆盖 CPU 推理、观测接口和 MuJoCo 接球。完整核对与修复记录见
[TRAIN_PLAY_AUDIT.md](TRAIN_PLAY_AUDIT.md)。真机稳定性与击球效果需另行验收。

对标前的问题记录见 [A3_DEPLOY_COMPARISON.md](A3_DEPLOY_COMPARISON.md)，
完整修正和最新结果见 [A3_ALIGNMENT_RESULT.md](A3_ALIGNMENT_RESULT.md)。
`--target-source reference` 仅用于明确的旧参考目标基线；`--a3-target-mode envelope`
选择 A3 的包络模式。默认 raw 与两者分开记录，不自动回退或补偿动作。

## 2026-10-19 验证记录

以下较早记录的 0 次接触已复现并修复：参考 world/绝对向量、动作库变换和机身目标耦合
完成修正后，三个种子合计 30/30 次模拟碰撞命中；名义跟踪组合阈值为 22/22 通过。
这些接触数据不表示有效回球。新增按参考击球平面发球并要求来球过网后一次落台的测试：
seed=42、7、123 各 10 球，合计碰拍 30/30，有效回球 0/30，覆盖 16 个参考动作，
均完成挥拍回位且未检测到失去站立。当前策略在该来球分布下不能稳定有效回球。
原球物理、PD、checkpoint 和训练目标构造均未为这项测试修改；详细证据见核对报告。

- 持久迁移后在本机重新测试：默认配置直接加载仓库内 checkpoint 和动作，
  自动选中动作 8、17、13，三次挥拍/回位全部完成；最大倾角 12.62°，
  最低 pelvis 高度 0.750m，没有检测到失去站立，接触次数为 0。
  记录在本机 `deploy/recordings/hope_g1/local_migrated_validation_20261019/`。
- 本机图形窗口启动正常，本次窗口流程在 POLICY_POSTURE 时 HALT 结束；
  完整流程结论来自上述无窗口自动测试，图形交互流程由操作者按文档执行。
- `.200` 的策略配置检查通过；本次只读 DDS 检查受 `eno1` 无 carrier/IP 阻塞，
  没有进行电机控制或修改网络。记录目录为
  `deploy/recordings/hope_g1/real_readonly_20261019_migration/`。
- 本地部署测试：147 项通过；`.200` 的新任务及共享运行时测试：17 项通过。
- 与 `.200` 原 HOPE 环境的 `ActorCritic` + `EmpiricalNormalization` 对照
  990 个输入（含仿真实际观测），最大动作绝对误差为 0。
- `.200` 自动选择模式完成动作 21、19、10 的三次挥拍及回位；
  最大倾角 15.45°，最低 pelvis 高度 0.734m，没有检测到失去站立。
  本次模拟接触次数为 0，因此此结果只证明部署流程与站立保持，不能作为击球效果验收。
- 目标机记录：`deploy/recordings/hope_g1/validation_20261019_auto/`，
  包含 `summary.json`、`trace.jsonl` 和 `policy_parity.json`。
- 验证没有启动真机电机输出，也没有改动原训练目录或 checkpoint。

## 真机日志回看

独立的 MuJoCo 回看工具可读取已有真机日志目录，显示实测机器人姿态、球观测与 racket command，支持变速、暂停、逐帧和时间跳转：

```bash
.venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_20261022_121714
```

完整参数、按键和还原范围见 [真机日志回看说明](../tools/REAL_LOG_VIEWER.md)。

