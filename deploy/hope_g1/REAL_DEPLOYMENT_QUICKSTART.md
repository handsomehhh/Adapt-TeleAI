# HOPE G1 真机启动指令（.200）

主机网口：机器人直连 `eno1`（当前 IP `192.168.123.165`）；交换机 `192.168.1.23`；路由器 `192.168.31.210`。
ROS 环境统一使用 `ROS_DOMAIN_ID=23`、`ROS_LOCALHOST_ONLY=0`。
机器人刚体当前 ID 为 `1`，原始话题为 `/vrpn_mocap/MCAvatar/pose1`；
重新创建刚体后，终端 1 的 `allowed_topics` 和终端 2 的 `robot_input_topic` 必须同时更新。
球刚体仍为 `30310`。

接球等待已恢复下午测试配置：挥拍结束后保持 home 75 帧（约 1.5 秒）；
`a3.select_settle_ticks: 25`（约 0.5 秒），与回位等待重叠。

## 终端 1：VRPN 原始数据

```zsh
cd /home/lxz/HOPE/hope_ws
source /opt/ros/humble/setup.zsh
source install/setup.zsh

export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch ./src/vrpn_mocap/vrpn_mocap/launch/client.launch.yaml \
  server:=192.168.1.101 port:=3883 \
  allowed_topics:=/vrpn_mocap/MCAvatar/pose1,/vrpn_mocap/MCAvatar/pose30310
```

## 终端 2：VRPN → 标准 ROS topics

```zsh
source /opt/ros/humble/setup.zsh
source /home/lxz/HOPE/hope_ws/install/setup.zsh

export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch vrpn_to_pose vrpn_to_pose.launch.py \
  robot_side:=P2 \
  robot_input_topic:=/vrpn_mocap/MCAvatar/pose1 \
  ball_input_topic:=/vrpn_mocap/MCAvatar/pose30310
```

## 终端 3：独立 A30808 planner

```zsh
source /opt/ros/humble/setup.zsh
source /home/lxz/HOPE/hope_ws/install/setup.zsh

export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd /home/lxz/deploy_g1_pingpong
export DEPLOY_G1_PINGPONG_ROOT=/home/lxz/deploy_g1_pingpong
colcon --log-base deploy/ros2_log build \
  --base-paths deploy/hope_g1/ros2_planner \
  --packages-select hope_g1_deploy_planner \
  --build-base deploy/ros2_build \
  --install-base deploy/ros2_install \
  --symlink-install
source deploy/ros2_install/setup.zsh
ros2 launch hope_g1_deploy_planner hope_planner.launch.py
```

顺序与 HOPE workspace 一致：先 source ROS 和已有接口 workspace，再在本仓库执行
`colcon build --symlink-install`、source 本仓库 install，最后 `ros2 launch`。
构建、安装、日志均写入本仓库的 `deploy/ros2_build`、`deploy/ros2_install`、
`deploy/ros2_log`；planner 核心和参数分别使用本仓库的
`deploy/hope_g1/hope_planner_optimized_2/` 与
`deploy/hope_g1/configs/hope_planner_optimized_2.yaml`。HOPE workspace 只作为 ROS
接口依赖被 source，不执行构建或修改。

## 终端 4（仅启动 G1 deploy 时需要）：ROS → deploy UDP bridge

仅检查前三个 ROS 节点和 topics 时，不用启动本 bridge。G1 deploy 控制进程通过 UDP 接收动捕和 `/racket/command`。

```zsh
source /opt/ros/humble/setup.zsh
source /home/lxz/HOPE/hope_ws/install/setup.zsh

export ROS_DOMAIN_ID=23
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

cd /home/lxz/deploy_g1_pingpong
/usr/bin/python3 -m deploy.io.mocap bridge \
  --config deploy/hope_g1/configs/mocap.yaml
```

## 终端 5：检查 ROS topics

```zsh
source /opt/ros/humble/setup.zsh
source /home/lxz/HOPE/hope_ws/install/setup.zsh
export ROS_DOMAIN_ID=23 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 topic list
timeout 10s ros2 topic hz /P1/pose
timeout 10s ros2 topic hz /poses
ros2 topic info /racket/command -v
timeout 10s ros2 topic echo /racket/command --once
```

`/racket/command` 在 planner 生成有效命令后才发布。

## 终端 6：检查 deploy 接收（10 秒，只读）

```zsh
cd /home/lxz/deploy_g1_pingpong
.venv/bin/python -m deploy.io.transport monitor \
  --config deploy/hope_g1/configs/mocap.yaml --duration 10
```

确认 `root_valid`、`ball_valid` 和 `planner_command_valid` 为 true 后退出 monitor。

## 终端 7：真机只读检查

```zsh
cd /home/lxz/deploy_g1_pingpong

.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env real \
  --config deploy/hope_g1/configs/task_real.yaml \
  --real-config deploy/hope_g1/configs/real.yaml \
  --mocap-config deploy/hope_g1/configs/mocap.yaml \
  --rally-config deploy/hope_g1/configs/rally.yaml \
  --network-interface eno1 --ball-source mocap \
  --motion-index 0 --motion-select auto --motion-allow all \
  --target-source hope_planner_optimized_2 --a3-target-mode raw \
  --duration 10 \
  --output "deploy/recordings/hope_g1/real_readonly_$(date +%Y%m%d_%H%M%S)"
```

## 终端 8：真机运行

```zsh
cd /home/lxz/deploy_g1_pingpong

.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env real \
  --config deploy/hope_g1/configs/task_real.yaml \
  --real-config deploy/hope_g1/configs/real.yaml \
  --mocap-config deploy/hope_g1/configs/mocap.yaml \
  --rally-config deploy/hope_g1/configs/rally.yaml \
  --network-interface eno1 --ball-source mocap \
  --motion-index 0 --motion-select auto --motion-allow all \
  --target-source hope_planner_optimized_2 --a3-target-mode raw \
  --duration 600 --enable-motors \
  --output "deploy/recordings/hope_g1/real_$(date +%Y%m%d_%H%M%S)"
```

`START → POSE_HOLD → A → STABLE → UP → WAIT_BALL`。停止：`SELECT/B/X` 或 `Ctrl-C`。

## 停止所有进程

依次在终端 8、4、3、2、1 按 `Ctrl-C`。

## 真机日志在 MuJoCo 中回放（离线）

在 .200 的桌面终端运行，将 `real_YYYYMMDD_HHMMSS` 替换为实际录制目录：

```zsh
cd /home/lxz/deploy_g1_pingpong
.venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_YYYYMMDD_HHMMSS \
  --paused --rate 0.5
```

该入口按日志恢复实测关节和根位姿，不运行 policy、不连接 ROS/DDS、不发送机器人指令。
需要 `summary.json`、`trace.jsonl`；球和 planner 接收记录来自 `mocap_packets.jsonl`。
绿色点/箭头：收到的有效 planner command 的位置/法向；蓝色箭头：目标速度；
橙色：实测球；浅蓝框：command 所在 x-hit 平面；紫色：估计击球点或实测穿平面点；
青色：同一事件时刻的拍心位置。紫色/青色/金色连线对比球–command、球–拍心、拍心–command。
左上角显示事件类型、XYZ、事件时间、剩余 TTS 和三组距离（cm）。
默认 `--command-view locked`：在 `motion_selected` 时显示机器人实际接受的 command，
位置/法向/速度/deadline 固定，直到该次回位完成；后续 planner 候选不会改变绿色目标。
如需检查原始 planner 连续更新，显式添加 `--command-view received`：收到有效候选即可显示，
新候选替换旧候选，无效候选清除，deadline 后最多保留2秒。该视图不代表机器人执行目标。
没有选择事件的旧日志回退到挥拍期间的已接受 trace 目标，不推断未知接收时刻。

穿平面位置由相邻实测球点沿 +X→-X 穿越 command 的 X 时线性插值；
不跨超过 50ms 的缺测区间、不外推。使用本机接收时间，第二个观测到达后才显示，
因此它是动捕观测的穿平面估计，不是 planner 预测或仿真球轨迹。
球在目标平面前被击回时，按来球/出球 X 方向反转和与重建有限拍面的接近程度，
标记 `ESTIMATED HIT (ball center)`，无需球继续穿平面。检测使用反向前20–60ms、
反向后20–70ms观测拟合速度，并允许3cm几何重建容差；需要后续观测后才显示。
它是球心击球位置估计，不能替代接触传感器，不能用于确认实际碰撞或评分。
没有触球证据但球穿过平面时显示 `MEASURED CROSSING (contact unconfirmed)`，
并插值该时刻的拍心位置比较；不因单纯反向就认定球拍击中。
没有穿平面且无法判定触球、或动捕缺测时保持 waiting，不补造交点。
标记保留至本次回位结束，倒退回放不会提前显示未来事件。

| 按键 | 操作 |
|---|---|
| Space | 暂停/继续 |
| R | 从头播放，保留当前倍速 |
| `[` / `-` | 减速，最低 0.0625 倍 |
| `]` / `+`（主键盘 `=`） | 加速，最高 8 倍 |
| `\` | 恢复 1 倍速 |
| ← / → | 后退/前进 1 秒 |
| `,` / `.` | 上一/下一机器人记录帧，并暂停 |
| N / P | 下一/上一状态或动作阶段事件 |
| Home / End | 跳到开头/结尾 |

关闭窗口退出；鼠标操作继续使用 MuJoCo 原生视角控制。
只有 SSH、没有桌面显示时，可检查日志或导出视频：

```zsh
.venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_YYYYMMDD_HHMMSS --inspect
MUJOCO_GL=egl .venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_YYYYMMDD_HHMMSS \
  --video /tmp/real_replay.mp4 --rate 0.5
```
