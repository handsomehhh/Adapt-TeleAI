# HOPE G1 真机部署：完整启动流程与历史问题汇总

更新：2026-10-22。适用机器：`192.168.152.200`；部署目录：`/home/lxz/deploy_g1_pingpong`。

> 当前独立 planner 的 package/build/source/launch 流程已迁到本仓库 `deploy/hope_g1/ros2_planner/`，
> 参数使用本仓库 `deploy/hope_g1/configs/hope_planner_optimized_2.yaml`。本文后续旧链路图中
> “内置 planner”描述已过时；planner 的 `x_hit` 已按 A30808 设为 `0.0`，取代下文历史记录中的
> `-0.25`。当前操作命令以 [REAL_DEPLOYMENT_QUICKSTART.md](REAL_DEPLOYMENT_QUICKSTART.md) 为准。

本文整理已有部署问题、当前正式配置及所有需要启动的组件。启动命令由操作者执行；编写本文时没有重启节点、运行电机控制或修改 `/home/lxz/HOPE`。ROS 软件仍使用该目录中已有的安装环境和 launch 文件；读取、source、运行这些文件不需要修改其源码。

## 1. 本次日志应该如何理解

日志先显示：

```text
no PoseStamped received yet: ball=..., robot=...
```

随后只显示：

```text
no PoseStamped received yet: ball=/vrpn_mocap/MCAvatar/pose30310
```

含义是：

- `vrpn_to_pose` 已启动，已创建订阅，但起初两路都没有收到数据。
- 后面机器人输入至少收到过一条消息，所以机器人从告警中消失。
- 球输入仍未收到第一条 `PoseStamped`，优先检查原始球 topic、球的动捕识别和 VRPN 发布。
- 该告警按“启动后是否收到过消息”判断；机器人不再出现在告警中，并不能证明它当前仍持续更新。必须用频率、时间戳和 UDP 有效性检查确认。

设置 `ROS_DOMAIN_ID=23` 是必要的环境配置，但不能让上游未识别、未发布的球凭空出现。下游 relay 和控制器都不能代替上游产生动捕数据。

## 2. 必须运行的完整链路

```text
动捕服务器 / Avatar-Pro / CMTracker
  │ VRPN，当前服务器 192.168.1.101:3883
  ▼
vrpn_mocap 客户端
  ├─ /vrpn_mocap/MCAvatar/pose4       机器人原始 PoseStamped
  └─ /vrpn_mocap/MCAvatar/pose30310   球原始 PoseStamped
  ▼
vrpn_to_pose
  │ 毫米转米、robot_side=P2 换边
  ├─ /P1/pose       机器人归一化位姿
  ├─ /poses        球 PoseArray，当前球在 poses[0]
  └─ /ball/point   同一球的 PointStamped 辅助输出
  ▼
deploy.io.mocap bridge
  │ 当前配置订阅 /P1/pose 和 /poses[0]
  │ UDP 127.0.0.1:15050，归一化 world_frame=hope_table
  ▼
deploy.hope_g1.deploy_agent --env real
  │ 球状态估计 → 内置 A30808 ROS planner → motion 选择 → G1 policy
  │ 同时管理 START / AMP / UP / 回位 / 停控
  ▼
Unitree SDK2 / DDS → eno1 → G1
```

| 组件 | 是否需要 | 说明 |
|---|---|---|
| 动捕服务器与球/机器人跟踪 | 必须 | 在动捕主机确认相机、刚体/标记、VRPN 输出均工作 |
| `vrpn_mocap` 客户端 | 必须 | 把 VRPN 数据变为 ROS 原始话题 |
| `vrpn_to_pose` | 必须 | 毫米转米、P2 换边和标准话题输出 |
| `deploy.io.mocap bridge` | 必须 | 将标准 ROS 位姿发送到部署控制器的 UDP 接口 |
| A30808 `hope_planner_optimized_2` ROS 节点 | 必须 | 独立订阅 `/poses`，发布 `/racket/command` |
| `deploy.hope_g1.deploy_agent --env real` | 必须 | 真机任务唯一控制入口；消费 planner command，执行选择器和 policy |
| UDP monitor | 启动前临时使用 | 检查结束后退出，不能和控制器同时占用接收端口 |
| `--env check`、preflight、只读接收 | 启动前检查 | 不是需要一直挂着的控制节点 |
| 独立 AMP 控制进程、A3 C++ deploy | 当前链路不需要 | 不要额外运行另一个电机控制程序 |
| Isaac `scripts/play.py`、MuJoCo 发球器、benchmark | 真机不需要 | 属于训练或仿真测试 |
| 额外 TF 节点 | 当前这条转换链路不要求 | `map/world` 标签不会在下游自动触发 TF 转换；数值坐标仍必须实际标定正确 |

同一个组件只启动一份。已有正常进程时复用，不要重复启动两个 VRPN 客户端、两个转换器或两个 UDP bridge。

## 3. 终端环境与两套网络

### 3.1 每个运行 ROS 命令的终端都执行

以下使用 zsh：

```zsh
source /opt/ros/humble/setup.zsh
source /home/lxz/HOPE/hope_ws/install/setup.zsh

export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

- 环境变量只影响本终端之后启动的进程，不会自动更新已经运行的旧节点。
- 使用 bash 时改为对应的 `setup.bash`，不要混用 shell 脚本。
- 部署控制器使用 `.venv/bin/python`；ROS→UDP 桥接使用 `/usr/bin/python3`，因为系统 ROS 的 `rclpy` 不一定安装在部署 venv 中。
- 不必先激活 venv 才运行 ROS。使用本文的绝对 Python 路径即可。

### 3.2 动捕网络与机器人网络分别检查

| 链路 | 本次配置 | 检查内容 |
|---|---|---|
| 动捕服务器 → VRPN 客户端 | `192.168.1.101:3883` | 服务器可达、VRPN 服务启用、目标被识别 |
| ROS 节点间发现 | `ROS_DOMAIN_ID=23` | 上下游进程使用一致环境；QoS 兼容 |
| ROS → 部署 | 本机 UDP `127.0.0.1:15050` | bridge 有数据发送，接收端只有一份 |
| 部署 → 机器人 | `eno1`，SDK DDS domain `0` | 网口有 carrier/IP、机器人 LowState 可接收 |

ROS domain `23` 与 Unitree DDS domain `0` 是不同接口的配置，不需要强行改成相同值。

本次只读核对到的 VRPN 客户端确实使用服务器 `192.168.1.101`、端口 `3883`。现场地址有变化时，以实际动捕服务器为准。

## 4. 按顺序启动

### 第一步：动捕主机与目标识别

在动捕软件中确认：

1. 相机和重建正常；球与机器人目标确实在有效视野内。
2. VRPN 输出已开启。
3. 当前机器人传感器编号为 `4`，球为 `30310`。这些是当前绑定，不是永远固定的物理身份。
4. 扫场、重建目标或重新标定后，核对编号、全局原点、坐标轴及单位。

如果扫场/标定改变了世界原点或坐标轴，需要重新对齐球桌坐标；如果全局坐标保持不变，也应复核桌面、边线和机器人位姿。不要仅凭“之前标定过”跳过数值检查。

### 第二步：终端 A，启动 VRPN → ROS 客户端

先执行 3.1 的 ROS 环境，然后运行：

```zsh
ros2 launch /home/lxz/HOPE/hope_ws/src/vrpn_mocap/vrpn_mocap/launch/client.launch.yaml \
  server:=192.168.1.101 \
  port:=3883 \
  allowed_topics:=/vrpn_mocap/MCAvatar/pose4,/vrpn_mocap/MCAvatar/pose30310
```

该 launch 已设置 `multi_sensor=true`，`allowed_topics` 是完整输出 topic 名称的逗号分隔白名单。保留终端运行。

检查终端中执行：

```zsh
ros2 topic list
ros2 topic info /vrpn_mocap/MCAvatar/pose4 -v
ros2 topic info /vrpn_mocap/MCAvatar/pose30310 -v

ros2 topic echo /vrpn_mocap/MCAvatar/pose4 \
  --once --qos-reliability best_effort
ros2 topic echo /vrpn_mocap/MCAvatar/pose30310 \
  --once --qos-reliability best_effort

timeout 10s ros2 topic hz /vrpn_mocap/MCAvatar/pose4
timeout 10s ros2 topic hz /vrpn_mocap/MCAvatar/pose30310
```

`timeout` 到期结束频率检查属于预期行为。两路都要有实际消息和持续更新的时间戳；topic 名存在不是通过标准。

球无数据时：先在动捕软件中移动球，确认可识别；再核对球编号。若需要发现当前所有编号，先在原终端 Ctrl-C 退出这一份 VRPN 客户端，再省略白名单参数重启：

```zsh
ros2 launch /home/lxz/HOPE/hope_ws/src/vrpn_mocap/vrpn_mocap/launch/client.launch.yaml \
  server:=192.168.1.101 \
  port:=3883
```

省略 `allowed_topics` 表示不限制输出。发现实际球 topic 后，同时更新 VRPN 白名单和下一步 relay 的 `ball_input_topic`。不要再额外并行启动一份客户端。

### 第三步：终端 B，启动坐标转换与标准话题输出

先执行 3.1 的 ROS 环境，然后运行：

```zsh
ros2 launch vrpn_to_pose vrpn_to_pose.launch.py \
  robot_side:=P2 \
  robot_input_topic:=/vrpn_mocap/MCAvatar/pose4 \
  ball_input_topic:=/vrpn_mocap/MCAvatar/pose30310
```

保留终端运行。预期日志为正确的输入话题、`position scale to metres: 0.001`。

在检查终端中验证：

```zsh
ros2 topic echo /P1/pose --once --qos-reliability best_effort
ros2 topic echo /poses --once --qos-reliability best_effort
ros2 topic echo /ball/point --once --qos-reliability best_effort

timeout 10s ros2 topic hz /P1/pose
timeout 10s ros2 topic hz /poses
```

说明：

- `robot_side:=P2` 表示输入数据按 P2 侧换边，输出机器人 topic 仍是 `/P1/pose`；不是应改订阅 `/P2/pose`。
- 当前球输出在 `/poses` 的 `poses[0]`，这才是部署桥接实际订阅的球输入。
- `/ball/point` 是同一球的辅助检查输出，不需要再启动一个程序产生它。
- 球和机器人均已由毫米转米，下游不能再乘一次 `0.001`。
- 机器人四元数必须有效；零四元数会导致转换器丢弃机器人位姿。

### 第四步：终端 C，启动 ROS → UDP bridge

先执行 3.1 的 ROS 环境，然后运行：

```zsh
cd /home/lxz/deploy_g1_pingpong

/usr/bin/python3 -m deploy.io.mocap bridge \
  --config deploy/hope_g1/configs/mocap.yaml
```

保留终端运行。预期输出包含：

```text
HOPE ROS topics -> UDP ('127.0.0.1', 15050)
```

先用下面的只读命令检查是否已有同类进程：

```zsh
pgrep -af 'deploy.io.mocap bridge'
```

已有一份使用同一配置且能正常输出数据的 bridge 时，不重复启动。如果修改了其订阅、标定或帧配置，应在所属终端退出旧进程后重新启动；修改 YAML 不会自动更新旧进程。

### 第五步：终端 D，临时验证 UDP 数据

```zsh
cd /home/lxz/deploy_g1_pingpong

.venv/bin/python -m deploy.io.transport monitor \
  --config deploy/hope_g1/configs/mocap.yaml \
  --duration 10
```

检查：

- `accepted` 持续增长，`rejected` 不持续增长。
- 机器人可见时 `root_valid=true`；球被识别且采样足够时 `ball_valid=true`。
- 球运动时，位置和估计随之更新，而不是保持旧值。

monitor 结束后再启动控制器。monitor 与控制器都是 UDP 15050 的接收者，不能同时运行。

### 第六步：机器人网口与部署静态检查

```zsh
cd /home/lxz/deploy_g1_pingpong

ip -brief address show eno1
ip link show eno1
nmcli connection show g1-direct-eno1
```

若现场仍使用已有 `g1-direct-eno1` 配置，接通网线并开机后，可由操作者启用：

```zsh
sudo nmcli connection up g1-direct-eno1
```

已有配置的地址约定为本机 `192.168.123.210/24`，既有机器人连通性检查地址为 `192.168.123.164`；以实际机器人网络配置为准：

```zsh
ip -brief address show eno1
ping -I eno1 -c 3 192.168.123.164

.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env check \
  --config deploy/hope_g1/configs/task_real.yaml \
  --rally-config deploy/hope_g1/configs/rally.yaml

.venv/bin/python -m deploy.tools.preflight \
  --require-real \
  --real-config deploy/hope_g1/configs/real.yaml \
  --mocap-config deploy/hope_g1/configs/mocap.yaml \
  --network-interface eno1
```

`--env check` 检查资产和配置。preflight 检查依赖、SDK 导入和网口是否存在，不保证网口有 carrier、收到 LowState 或动捕当前有效；ping 也不能替代 SDK 数据检查。

2026-10-22 最近一次只读核对时 `eno1` 为 DOWN。这是当时的观测，不是运行本文命令之后的实时判断。

### 第七步：只读接收检查，不输出电机命令

确保 UDP monitor 已结束，bridge 仍在运行：

```zsh
cd /home/lxz/deploy_g1_pingpong

.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env real \
  --config deploy/hope_g1/configs/task_real.yaml \
  --real-config deploy/hope_g1/configs/real.yaml \
  --mocap-config deploy/hope_g1/configs/mocap.yaml \
  --rally-config deploy/hope_g1/configs/rally.yaml \
  --network-interface eno1 \
  --ball-source mocap \
  --motion-index 0 --motion-select auto --motion-allow all \
  --target-source hope_planner_optimized_2 --a3-target-mode raw \
  --duration 10 \
  --output "deploy/recordings/hope_g1/readonly_$(date +%Y%m%d_%H%M%S)"
```

本条没有 `--enable-motors`，检查机器人状态接收、动捕和通信报错，不代表已执行过击球策略。等待其退出，再运行正式控制器。

### 第八步：正式真机部署

保持动捕服务器、终端 A/B/C 运行。只运行一个电机控制程序；本入口内部加载 AMP，不额外启动独立 AMP。

```zsh
cd /home/lxz/deploy_g1_pingpong

.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env real \
  --config deploy/hope_g1/configs/task_real.yaml \
  --real-config deploy/hope_g1/configs/real.yaml \
  --mocap-config deploy/hope_g1/configs/mocap.yaml \
  --rally-config deploy/hope_g1/configs/rally.yaml \
  --network-interface eno1 \
  --ball-source mocap \
  --motion-index 0 --motion-select auto --motion-allow all \
  --target-source hope_planner_optimized_2 --a3-target-mode raw \
  --duration 600 --enable-motors \
  --output "deploy/recordings/hope_g1/real_$(date +%Y%m%d_%H%M%S)"
```

遥控器顺序：

1. **START**：进入定位，等待 `POSE_HOLD`。
2. 操作者完成实际放下/支撑操作；**A**：进入稳定控制，等待 `STABLE`。
3. 站稳后按 **UP**：`POLICY_POSTURE → TO_POLICY → WAIT_BALL`。
4. 到 `WAIT_BALL` 再送球；控制器自动规划、选动作、执行、回位并继续等球。
5. **DOWN** 返回 AMP；**SELECT/B/X** 或 Ctrl-C 停控并进入相应阻尼退出流程。

A 应在进入 `POSE_HOLD` 后重新按下，不要从定位阶段一直按住。真机不使用 MuJoCo 的 L/F/T 键；来球由真实发球提供，也没有仿真每4秒发球的调度。

## 5. 启动前的坐标与数值核对

当前部署使用 `hope_table`，单位米：

| 含义 | 数值 |
|---|---|
| 机器人侧桌面端线 | X=0 |
| X 正方向 | 从机器人侧指向对手 |
| 球网 | X=1.37 |
| 对方端线 | X=2.74 |
| 桌面宽度范围 | Y∈[-1.525,0] |
| 桌面高度 | Z=0 |
| 理论静放球心 | Z≈0.02 |
| 正式击球平面 | X=-0.25，近端桌边向机器人侧25cm |
| 回球目标 | [2.055,-0.7625,0]，对方半台中心 |

- `X=-0.25` 是球桌世界系的位置，不是机器人身体前方固定25cm。
- P2 换边在当前转换器中为 `x'=2.74-x`、`y'=-1.525-y`、`z'=z`，下游不再重复换边。
- `map`、`world`、`hope_table` 字符串不会自动修复错误的数值坐标。
- MuJoCo 的 `[-0.65,-0.7625,-0.76]` 是仿真地面系到桌面系的桥接，不要额外叠加到已经归一化的真实动捕数据上。
- 机器人 pelvis 接近桌面高度时，其 Z 接近0可能合理；不能看到 Z 小就统一加0.76。
- 静放球的 Z 若长期明显偏离0.02，需要复核桌面原点、标记定义和单位。以前曾观测到约-0.0033，属于历史残差，不是应长期照抄的标定值。
- 当前下游 `root_marker_to_pelvis=[0,0,0]`；必须确认上游目标确实代表所需根位姿。不要根据单个姿态猜一个偏置。
- 球拍安装、marker 或机器人刚体定义变化后，应分别复核对应变换；球桌世界标定不能代替这些安装参数。

## 6. 常见故障与处理顺序

| 现象 | 可能的问题 | 优先检查/处理 |
|---|---|---|
| 两个原始 topic 都没数据 | 动捕服务器/VRPN客户端未启动、服务器地址错误、ROS环境不一致 | 先确认服务器，再确认终端A、topic发布者和两路原始echo |
| 只有球一直 `no PoseStamped received yet` | 球未识别/遮挡、编号变化、白名单未包含实际球topic、输入类型不匹配 | 检查动捕软件中的球，再核对原始球topic和白名单；不要先重启policy |
| 原始topic有数据，relay仍收不到 | 输入topic/类型/QoS或进程环境不一致 | `topic info -v`，确认 PoseStamped；新开检查终端使用相同ROS环境 |
| `/P1/pose` 没有输出 | 原始机器人消息未到、四元数无效、转换器丢弃 | 看原始机器人echo及转换器 `dropping /P1/pose` 日志 |
| `/P1/pose` 有，`/poses` 没有 | 球分支没有收到有效原始消息 | 沿原始球 → relay球输出逐级检查 |
| `ros2 topic list` 看得到但 hz/echo无数据 | 只存在订阅、发现信息残留、发布者不发新样本等 | 查publisher数量与实际消息，不把topic名称当作有效性证明 |
| 已设置 domain 仍无数据 | 旧进程环境未更新，或问题本来在上游识别/发布 | 比较实际运行节点的环境；在其所属终端重启受影响组件即可，不必全部重启 |
| UDP monitor accepted=0 | bridge未启动、ROS输入没数据、配置/端口不一致 | 先看标准ROS输出，再看bridge日志、UDP地址与端口 |
| UDP rejected增长 | frame、时间戳、坐标数值或配置不匹配 | 对照 `mocap.yaml` 和错误详情；不要关闭检查来掩盖问题 |
| Address already in use / UDP异常 | monitor和控制器同时运行，或重复接收进程 | 退出临时monitor或旧控制器，再启动唯一接收端 |
| `Robot mocap pose expired` | 根位姿没有持续更新、延迟/间断超过期限 | 检查原始机器人、转换后机器人、UDP整条链路；topic存在不够 |
| 后面几球没有反应 | 球未被持续识别、时间戳过期、估计未就绪、仍在回位或时间/可达性门限拒绝 | 查该次 `trace.jsonl` 的 ball_age、ball_position、decision、state；不能直接认定policy没能力 |
| `[Reader] take sample error` | 底层采样/通信异常，需结合来源日志判断 | 查原始输入与节点日志；单独这一行不足以归因policy |
| `eno1` DOWN/NO-CARRIER、没有LowState | 网线、机器人电源、网口连接或地址问题 | 查carrier/IP和SDK状态；动捕网络正常不代表机器人网络正常 |
| DDS `does not match an available interface` | 网口名称/状态与配置不符 | 核对 `eno1` 和实际系统接口，按实际网口配置 |
| `dquote>` | shell双引号未闭合 | Ctrl-C取消未完成输入，重新粘贴完整命令；输出路径末尾要有闭合双引号 |
| 命令换行后参数没接上 | 反斜杠后还有空格或注释 | 每个续行反斜杠必须是该行最后一个字符 |

频率检查不要求从“出现一条消息”直接推断链路合格。当前根/球接收过期阈值约0.1秒，源数据年龄阈值0.15秒；即使平均频率足够，持续丢帧或较大延迟也会失效。实际门限以 `configs/mocap.yaml` 为准。

## 7. 之前已经处理过的部署问题

| 问题 | 当前状态 |
|---|---|
| 混用旧任务入口与HOPE G1 checkpoint/motion编号 | HOPE任务使用 `deploy.hope_g1.deploy_agent`；动作0..21，不能沿用旧任务编号24 |
| 原先仿真按motion反推来球，导致能力被高估 | 正式连续MuJoCo使用独立 `random_launch`；真机直接用实测球 |
| 世界系、毫米/米、P2换边重复或遗漏 | 统一为hope_table；转换器完成毫米→米与P2换边，下游不重复做 |
| 把参考动作世界向量直接作为训练世界向量送入actor | 已按参考重定位映射处理；不能把约84.5°的参考旋转简单理解为球桌也旋转了 |
| 远端仍使用旧球动力学参数 | 2026-10-21正式部署已与本机MuJoCo的A3参数对齐 |
| 回球目标位于台面侧边界 | 正式目标采用对方半台中心[2.055,-0.7625,0]，保留原目标对照配置 |
| 独立policy评测与连续部署混淆 | 评测可逐球恢复快照；用户可视化/真机部署走连续控制，二者入口和环境分离 |
| 用碰拍率代替严格指令跟踪或有效回球率 | 三项分别统计，不互相替代；历史motion关联发球成绩不能当独立随机测试成绩 |

正式真机配置保留：`max_command_time_to_strike=0.5`、`max_start_frame=29`、`target_mode=raw`、自动选择全部22个动作。实验性短TTS第38帧放宽没有合入正式部署。

## 8. 当前仍存在的能力限制

这些问题与“节点没启动/球没数据”应分开排查：

- A3原预测器在有限桌面外仍限制球的最低预测高度，低于桌面的球可能得到偏高的击球指令；尚未修复。
- 反弹后拟合可能尚未收敛就触发并锁定指令；已有单球较大预测误差记录。
- motion主要按参考击球位置距离选择，没有联合评估朝向、速度和当前姿态下的准备时间。
- 约0.2秒准备时间的实验碰拍明显下降；正式部署保留更早的启动帧限制。
- MuJoCo当前没有球旋转/Magnus模型，固定约4秒发球也不代表所有实际对打节奏。
- 文件同步、离线检查和MuJoCo通过，不代表真机已经完成新的击球验收。

核对依据：

- [2026-10-21同步记录](REAL_SYNC_RESULT_20261021.json)：142个文件，正式配置和资产对齐，远端离线验证通过。
- [当前部署需求审计](../recordings/hope_g1/deployment_requirements_audit_20261021/report.html)：低球、预测及动作选择的边界。
- [连续回球目标对照](../recordings/hope_g1/return_target_comparison_20261021/report.html)：同一145球的两个回球目标。
- [TTS配对测试](../recordings/hope_g1/tts_sweep_20261021/report.html)：三档各40球。

后面三个链接是本机测试记录；未同步全部录制目录到`.200`，远端打开文档时可能没有对应图表文件。

## 9. 最短执行顺序

```text
确认动捕目标/标定和机器人网络
→ 动捕服务器开启VRPN
→ 启动一份vrpn_mocap，确认两路原始PoseStamped
→ 启动一份vrpn_to_pose，确认/P1/pose和/poses
→ 启动独立A30808 planner（/poses → /racket/command）
→ 启动一份ROS→UDP bridge（同时转发mocap和/racket/command）
→ UDP monitor检查并退出
→ 资产/SDK检查与只读接收，退出检查进程
→ 启动唯一G1真机控制器
→ START → POSE_HOLD → A → STABLE → UP → WAIT_BALL → 送球
```
