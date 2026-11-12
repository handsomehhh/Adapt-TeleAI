# G1 extended hitter：10700 + 原生 G1 planner

本分支默认使用 `2026-11-03_15-50-58/model_10700.pt`，任务为
`unitree_g1_hitter_pingpong_extended_mjlab`。部署入口仍是
`python -m deploy.hope_g1_hitter.deploy_agent`。

默认 checkpoint 使用 Git LFS 存储。新电脑克隆后先安装 Git LFS，并在仓库中执行：

```bash
git lfs install --local
git lfs pull
```

`model_10700.pt` 下载后约 143 MiB；不要将 Git 中的 LFS 指针文本当作模型文件。
模型内容及 `manifest.json` 中的 SHA256 保持不变。

## 已确认的配置

| 项目 | 当前设置 |
| --- | --- |
| 规划模式 | 原生 G1 planner，关闭横向可达性拒绝；保留训练高度范围和有界速度优化 |
| 球飞行与桌面 | 自动选择：MuJoCo 使用 1v1 原生参数；实机使用 A30808 当前生效的标定 |
| 球拍碰撞求解 | 仿真采用 1v1 拟合碰撞；实机采用 A30808 法向恢复系数 0.6；均保留 G1 有界速度优化 |
| 目标落点 | (2.055, -0.7625)，回球飞行时间 0.5 s |
| 配置站位 | (-0.65, -0.7625)，用于初始目标站位和规划 X 平面 |
| 两次击球之间 | 正手随挥 1.3 s、反手 1.2 s 后保留上次击球目标站位；新击球计划可立即接管 |
| 正反手 | 优先上次一侧，不可达时尝试另一侧；该站位默认从正手开始 |
| 根部姿态 | 完整动捕骨盆姿态；IMU 角速度转换到骨盆局部坐标系 |
| 球拍几何 | 新训练 XML，全套质量、惯量、碰撞与网格；腕部到参考点 [0.242, 0.003, 0] m |

坐标使用已转换好的机器人侧近端桌角：桌面 Z=0，球台 X=[0,2.74]、
Y=[-1.525,0]，机器人朝 +X。`/poses[0]` 是球心，`/P1/pose` 是骨盆。
ROS 的 `world` / `map` 只是沿用现场的 frame 标签，配置不再执行 P2 翻转或球网原点偏移。
训练/1v1 的 Z 以地面为零，因此球拍位置框与输出之间转换 0.76 m；速度与法向不平移。
骨盆相对目标和站位误差保持世界轴，不旋转到身体坐标系。

104 维输入顺序为骨盆局部角速度、29 维关节位置差、29 维关节速度、29 维上次原始动作、
局部重力投影、世界 XY 朝向、站位误差、球拍目标相对骨盆位置、世界目标速度、剩余击球时间。
actor 输出经过训练的 action scale 和 0.9 倍关节软限位解码，50 Hz 执行。
击球时刻后继续传负的剩余时间，随挥时长按动作片段计算：
`(frames - 1 - strike_frame) * control_dt`。默认 model_10700 正手为 1.3 s、反手为 1.2 s，
与 1v0 一致。随挥结束后保留上次击球的目标站位，不回首次 UP 位置或默认站位。
首次进入及 DOWN → UP 重新进入 policy 时，目标站位初始化为配置值。

空闲时球拍目标为保留站位加当前正/反手位置框中心偏移；Z 使用地面/球台高度转换，
目标速度为速度框中心，剩余击球时间固定为该侧 windup。动作历史在随挥结束时保留。
空闲状态显示 `IDLE`；新计划可在随挥或空闲时立即接管，不等待机器人到位。
`completed` 统计完成的随挥，保留的通用运行时字段 `recovered` 为 0，不再统计回位。
MuJoCo 自动测试的 `--episodes` 完成判据使用随挥完成次数，其他控制器仍按原回位判据。

ROS 规划反馈的 `station`、仿真内置 planner 输入和 policy 的 `goal_base_xy` 均使用
当前目标站位。规划器不再因横向不可达而拒绝击球目标；仍将规划站位 Y 截断在
[-1.25, -0.35] m 与当前目标站位 ±0.35 m 的交集内，原击球目标 Y 保持不变。
部署接收端同步取消站位 Y 范围、站位 Y 步长和相对击球位置 Y 的拒绝检查；
仍校验 X、训练高度、训练球拍速度、时效及消息契约。实测骨盆位置仍用于构造
policy 的站位误差和球拍相对位置观测。正反手选择仍优先上次一侧。

此变更同时适用于共享部署链路的实机 ROS planner 和 MuJoCo 内嵌 planner，
不修改训练仓库的 1v1/2v2。更新时需同时 pull `Adapt-Deploy`（本仓库部署副本）
和同级 `g1_planner_ros2`，然后重启 planner 和部署进程；仅更新 planner 会被旧接收端拦截。

## 自动选择物理模型

原启动命令无需增加参数：`--env mujoco` 自动选 `sim`，`--env real` 自动选 `real`；
`--env check` 检查实机配置，不连接机器人。MuJoCo 内嵌的是独立 `g1_planner_ros2` 仓库中的同一个
planner 包，规划器、模拟球飞行/桌面反弹、拍球响应共同使用 `profiles.py` 的参数。

| 参数 | 仿真（原生 1v1） | 实机（当前 A30808 配置） |
| --- | --- | --- |
| 空气阻力 k | 0.1261 | 0.096 |
| 桌面水平/垂直反弹系数 | 0.64 / 0.9215 | 0.816019 / 0.862769 |
| 球质量 | 3.4 g | 2.7 g |
| 球拍法向恢复 | 0.759 × exp(-0.0441 × 法向相对速度绝对值) | 常数 0.6 |
| 切向冲量 / 旋转升力 | 使用原生 1v1 模型 | 关闭，与旧实机模型一致 |

独立包内置 2026-11-04 的 A30808 标定快照，保留原始文件字节与参数 hash，
运行不再依赖 HOPE-TeleAI。来源记录在 planner 包的 `CALIBRATION.json` 中。
`planner_g1.yaml` 的 `real_config` 可指定现场 YAML；ROS planner 也应配置相同内容。
修改后重启两端。旧 HOPE 仓库中的标定修改不会自动同步到独立包。
启动报告打印模式、参数、来源路径和 SHA256。

ROS 节点从 `/g1/planner/state` 的 `runtime_mode` 自动识别，独立于 `use_sim_time`。
反馈和命令同时携带 `physics_sha256`，避免两端使用不同标定；切换模式会清空旧球轨迹。
未收到有效模式时等待反馈。G1 的正反手、可达范围、站位和目标落点规则保持一致。

## ROS planner 与订阅

新包位于同级仓库：
`../g1_planner_ros2/src/hope_planner_g1`。
两个新消息在该工作区的 `hope_msgs` 中，原来的 `RacketCommand` 保留。

```text
/poses (PoseArray, 球心)
  -> hope_planner_g1
  -> /g1/racket/command (G1RacketCommand)
  -> deploy.io.mocap bridge -> UDP 15050 -> HitterController

HitterController 的目标站位/正反手/启用状态
  -> UDP 15051 -> 同一个 bridge
  -> /g1/planner/state (G1PlannerState) -> hope_planner_g1
```

反馈是控制器的**目标站位**，不是当前动捕骨盆位置。部署端直接采用 planner 选出的
side、station、position、velocity，不再强制反手、二次裁剪速度或重算站位。
planner 无有效反馈时不规划，反馈超时为 0.15 s。启用 policy 会生成新的 session ID；
旧 session、重复消息、错误 policy contract 都不会作为新的击球目标使用。
新一轮无可达解不会中断已接受目标的随挥。

`strike_time` 是绝对 ROS/球观测时钟的击球时刻，bridge 用当前 ROS 时间换算为本机
monotonic 截止时间；求解、ROS 和 UDP 耗时不会延后击球截止时间。
因此球源和 planner 必须处在同一 ROS 时钟域，bridge 与部署程序必须同机运行。

在具备 ROS 2 的机器上，使用现场原有 ROS 环境和 ROS_DOMAIN_ID：

```bash
cd /home/yiranwang/TeleHuman/g1_planner_ros2
# 先 source 现场 ROS 2 的 setup.bash
colcon build --symlink-install --packages-up-to hope_planner_g1
source install/setup.bash
ros2 launch hope_planner_g1 hope_planner.launch.py
```

独立仓库包含 `hope_msgs`，迁移保持所有消息定义不变。构建后重启 bridge/planner，
客户端 source 新仓库的安装环境；不需要旧 A30808 包。

另一个终端使用 ROS 对应的系统 Python 启动 bridge（需 numpy、yaml）：

```bash
cd /home/yiranwang/TeleHuman/Adapt-TeleAI
source /home/yiranwang/TeleHuman/g1_planner_ros2/install/setup.bash
/usr/bin/python3 -m deploy.io.mocap bridge \
  --config deploy/hope_g1_hitter/configs/mocap.yaml
```

部署环境检查与启动命令：

```bash
cd /home/yiranwang/TeleHuman/Adapt-TeleAI
.venv/bin/python -m deploy.hope_g1_hitter.deploy_agent --env check
.venv/bin/python -m deploy.hope_g1_hitter.deploy_agent --env real --duration 120
```

第二条沿用原部署入口的默认无电机输出模式；现场实际输出仍使用原有的
`--enable-motors` 开关和手柄 START → A（AMP）→ UP（击球 policy）流程。
这次代码修改没有执行实机电机输出。

## 离线验证

MuJoCo、实机和 check 模式都自动从部署仓库（Adapt-TeleAI 或 Adapt-Deploy）位置解析
`../g1_planner_ros2/src/hope_planner_g1`，优先加载这里的源码，无需预先安装 planner。
路径基于代码文件位置，不依赖当前终端目录，也不包含机器用户名。
换电脑时保持两个仓库并列，安装部署环境的 NumPy/SciPy/PyYAML 等依赖即可。
找不到该目录会明确报错，不会悄悄使用其他已安装版本；启动报告中的
`planner_source` 显示实际来源；内嵌模式同时保留 `embedded_planner_source`。

实机 ROS planner 节点仍单独启动；deploy 导入共享物理参数无需再安装包或设置 `PYTHONPATH`。
离线测试：

```bash
.venv/bin/python -m pytest \
  deploy/validation/hitter_physics_profile_checks.py \
  deploy/validation/hitter_contact_checks.py \
  deploy/validation/hitter_planner_checks.py \
  deploy/validation/g1_upstream_parity_checks.py -q -s
.venv/bin/python -m deploy.hope_g1_hitter.deploy_agent \
  --env mujoco --sim-controls auto --sim-startup suspended \
  --duration 32 --episodes 2 --seed 42
```

跨仓库对照测试读取兄弟目录 PingPong；可通过 `PINGPONG_REPO` 和
`G1_PLANNER_PACKAGE` 覆盖路径。实机推理不依赖训练仓库。
`uv sync` 清理额外包后，内嵌 MuJoCo 仍会直接加载相邻仓库；实机/check 需恢复上述安装。

自动物理模式更新后：41 项测试通过，包含 sim → real → sim 的节点回调切换、
标定文件读取、模拟球与 planner 参数一致性、100 组旧 A30808 球拍公式对照。
seed=42 的 32 秒无渲染测试完成两次挥拍及恢复，4 个来球、3 次接触、2 次合法回球；
仅用于启动流程验证，不作为成功率评估。详见
[`hitter_physics_profile_result.json`](../validation/hitter_physics_profile_result.json)。

2026-11-04 验证结果（28 项测试通过，详见 `../validation/g1_planner_migration_result.json`）：

- 8 seeds × 40 来球 × 2 组物理参数，640 例规划与原 1v1 一致。
  原仿真参数：52 正手 / 74 反手 / 194 无解；实机标定参数：66 / 80 / 174。
- 100 组随机状态的 104 维观测、actor 输出和动作解码对照通过。
- 20 组随机姿态的拍面正运动学，以及机器人质量、惯量和碰撞几何与训练 XML 一致。
- 消息时间转换、双侧执行、会话隔离、负时间随挥、站位回归及异常输入回归测试。
- 默认悬吊启动：完成两次击球周期，3 个来球、2 次接触、0 次合法回球，未出现倾倒告警。
  这是修正仿真拍球模型前的历史 smoke，不代表当前成功率。后续发球诊断与修正结果见
  [发球实验](../validation/hitter_feed_audit_report.md)。当前 G1 仿真使用与 planner 相同的
  拍球响应，并传播碰撞产生的旋转及 Magnus 力；碰撞检测仍是 swept disk，非完整 1v1 场景复制。
- 额外 `--sim-startup ground` 测试在进入击球 policy 前的静态 PD 等待期倾倒；
  不将该启动模式标记为通过。
- 本机没有 ROS 2，因此未运行 colcon 构建、真实 ROS 通信或实机测试。

policy 接口保持一致；当前部署额外放开横向拒绝。当前仿真物理参数采用 1v1 原生值；
实机采用 A30808 标定，因此实机规划数值可能与仿真不同。
有界求解可能有残差；请求的落点是目标，不等于保证过网或上台。

模型与 XML/mesh/动作元数据集中在 `artifacts/hitter_extended_20261103_10700/`，
`manifest.json` 记录文件 SHA256。planner 和部署使用相同 `contract.json`，更换模型时
需要重新导出并同步 `hope_planner_g1/config/policy_contract.json`。

## “发一球”与拒接

发球来自独立随机 launch bank，只保证过网、己方一次反弹及穿过
X=-0.22、Y=[-1.45,-0.5]、Z=[-0.1,0.39] 的网格；没有用 policy/planner 的
可达性过滤。当前默认站位为 (-0.65,-0.7625)，规划站位单次横移仍截断为 0.35 m；
后续规划基于控制器反馈的当前目标站位 Y，规划 X 保持配置值。横向超限不再拒接；
高度、物理轨迹和时间等检查仍可能拒绝来球。

默认配置 `feed.continuous_on_policy_entry: true`：进入 policy 后会自动每 4 s 发球。
按钮 F 会立即新建球并替换当前球；希望逐球测试时，先按 T，确认界面的
`Continuous feed (T)` 为 OFF，再按 F。F 需要已经进入击球 policy。
界面的 `Planner` 行显示估计/规划状态和不可达原因。

每次模拟新建球会重置两个位置估计器及新 planner 的反弹计数，保持 policy 动作历史、
正反手状态和随挥。实机仍从真实观测间断/方向变化识别新轨迹，不接收模拟发球事件。

离线重现按钮发球实验：

```bash
.venv/bin/python -m deploy.validation.hitter_feed_audit \
  --seed 42 --balls 12 --output /tmp/g1_feed_current_42
# 对照旧碰撞/轨迹重置行为，空气和桌面仍使用当前仿真参数：
.venv/bin/python -m deploy.validation.hitter_feed_audit \
  --seed 42 --balls 12 --legacy-contact --keep-track --output /tmp/g1_feed_legacy_42
```

这些脚本调用相同启动入口和 F 按钮回调，以无渲染方式自动执行 START/放下/A/UP，
关闭连续发球后每 4 s 按 F，不更改采样、planner 或 policy 目标。
历史 36 球实验使用更新前的实机空气/桌面参数；当前模式自动切换后的成功率不能沿用该结果。

## 实机链路诊断记录

实机 `deploy.hope_g1_hitter.deploy_agent --env real ...` 默认自动开启独立 ROS 记录进程，
不需要 `--live-view`，原有动捕、转换、planner 和桥接启动命令保持不变。
配置为 `configs/chain_recording.yaml`；`--chain-record-config` 可覆盖配置，
`--no-chain-record` 可关闭。系统 ROS Python 与策略虚拟环境隔离；仅子进程 source
`/opt/ros/humble/setup.bash` 和并列 `g1_planner_ros2/install/setup.bash`。
ROS_DOMAIN_ID 优先沿用当前终端，未设置时使用配置中的 23。

每次运行目录新增 `ros_chain/`：

- `events.jsonl`：原始 VRPN 球话题、`/poses`、planner 指令、控制器反馈，保留重复数据、
  整数 ROS 时间戳、发布者 GID、DDS 时间戳和接收时间。
- `metadata.json` / `config.yaml`：话题及采集配置；`process.log` / `vrpn_tap.log`：进程日志。
- `coverage.json`：各话题数量、缺失话题及是否真正收到 VRPN 原始时间。
- `launcher_summary.json`：记录进程退出码及是否被强制结束。
- `summary.json`：写盘错误、队列丢弃和未写完数量；这些计数不代表能检测到所有 DDS 丢包。

lxz 的 Humble Python 不提供逐消息 publisher GID，记录为 `null`；发布者列表和 endpoint
GID 会定期从 ROS 图保存。其他环境若提供逐消息 GID，则直接保留。

独立 VRPN tap 默认复用 `~/HOPE/hope_ws/install/vrpn_mocap/lib/vrpn_mocap/client_node`，
连接配置中的服务器，在独立命名空间记录原始 `msg_time` 和本机接收时间，不发布 `/poses`，
不改动或重启原动捕进程。它是一条额外的 VRPN 连接，不能等同于原客户端实际收到的序列。
缺少该程序时仍记录四类主话题，但明确标记 tap 不可用；没有 ROS 环境而导致记录器启动
失败时，部署入口在进入控制循环前报错，可修复环境或显式用 `--no-chain-record`。

仅记录、不加载策略或连接电机的检查命令：

```bash
cd /home/lxz/Adapt-Deploy
.venv/bin/python -m deploy.tools.record_mocap_chain \
  --output deploy/recordings/mocap_chain_check --duration 30
```

每次使用新的输出目录。planner 自身还在 `~/.ros/g1_planner_audit/` 记录实际消费的消息、
估计窗口和指令，以 `session_id`、`command_id`、整数源时间戳关联本次日志。
须重启更新后的 planner 才有这一部分记录；独立订阅日志不能替代它。
