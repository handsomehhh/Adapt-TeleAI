# HOPE G1 真机部署（2026-10-21）

当前正式真机配置 `configs/rally.yaml` 与连续 MuJoCo 的
`configs/rally_mujoco_random.yaml` 共用相同的 planner、动作选择、球/球桌参数：
planner 击球平面X=0m，回球目标[2.4,-0.7,0]，最大启动TTS=0.5s，最晚起始帧29。
保留真机 `real_a3` profile、DDS/动捕及保护配置；不运行仿真发球器。
0.2s短TTS实验中放宽到第38帧启动的设置未合入真机。
后文注明2026-10-20日期的观测属于历史状态，不代表当前连接状况。

目标目录：`.200:/home/lxz/deploy_g1_pingpong`。只同步部署代码和资源，不修改
`/home/lxz/HOPE`，不依赖该目录。入口与旧 Stage2 任务分开。

使用本机 MuJoCo 验证通过的 checkpoint、22 动作、训练配置、URDF、
A3 raw planner、HOPE桌面坐标、源采样率根姿态/速度滤波及反弹后球速重估。
真机由动捕提供球与根位姿，不运行仿真发球器。保留 START → A → UP 进入策略流程。
同步文件列表、SHA256 和旧文件备份位于目标目录 `.backups/hope_g1_<时间>/`。

## 1. 资源与硬件接口检查

以下在 .200 的部署目录执行：

```zsh
cd /home/lxz/deploy_g1_pingpong
.venv/bin/python -m deploy.hope_g1.deploy_agent --env check \
  --config deploy/hope_g1/configs/task_real.yaml
.venv/bin/python -m deploy.tools.preflight --require-real \
  --real-config deploy/hope_g1/configs/real.yaml \
  --mocap-config deploy/hope_g1/configs/mocap.yaml --network-interface eno1
ip -brief address show eno1
ping -I eno1 -c 3 192.168.123.164
```

preflight 检查依赖和网口存在，不代表已接收到机器人 LowState。
2026-10-20 同步前 eno1 为 DOWN，无 IPv4。先开机接线；若已有的网络配置
没有自动连接，可由操作者运行 `sudo nmcli connection up g1-direct-eno1`，
预期地址 `192.168.123.210/24`。本次同步不修改网络设置。

## 2. 动捕（单独终端）

先由现有动捕系统提供 `/P1/pose` 和 `/poses`，再检查；topic list 出现名称不代表有数据。

```zsh
source /opt/ros/humble/setup.zsh
export ROS_DOMAIN_ID=23 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic echo /P1/pose --once --qos-reliability best_effort
ros2 topic echo /poses --once --qos-reliability best_effort
```

启动更新后的 ROS→UDP bridge 并保持终端运行；若旧 bridge 正在运行，先 Ctrl-C 退出，确保只留一份：

```zsh
cd /home/lxz/deploy_g1_pingpong
/usr/bin/python3 -m deploy.io.mocap bridge \
  --config deploy/hope_g1/configs/mocap.yaml
```

另一终端检查 10 秒：

```zsh
cd /home/lxz/deploy_g1_pingpong
.venv/bin/python -m deploy.io.transport monitor \
  --config deploy/hope_g1/configs/mocap.yaml --duration 10
```

要求 root_valid 持续为 true；球在视野中移动时 ball_valid 为 true，rejected 不持续增加。
monitor 结束后才能启动控制器，两者占用同一 UDP 15050 接收端口。
这个 bridge 也会订阅 `/racket/command`，并将 ROS planner 的完整命令送到控制进程；
它仍是控制器 UDP 15050 的唯一接收端。
输入沿用已有标定：root=/P1/pose（map），ball=/poses（world，slot 0）。
两个 frame 标签不做 TF 查询。数值约定为 HOPE 桌面世界系：近端左角为原点，
X 指向对手，桌面X=[0,2.74]、Y=[-1.525,0]、Z=0，地面Z=-0.76。
球心放在桌面时理论Z约0.02m；当前实测约-0.0033m，约2.3cm残余偏差需要物理标定复核。
站立 pelvis 高度接近桌面时，发布Z接近0是合理的，不给球或机器人单独加0.76。
保留操作者提供的下游零marker偏置；仍需确认刚体定义，不从单个位姿猜标定。

本轮 UDP 归一化帧从 world 改为 hope_table。旧 bridge 要 Ctrl-C 退出后，
使用上述 hope_g1/configs/mocap.yaml 重启，不能复用运行中的旧桥接。
2026-10-21 更新：本地 rally.yaml、rally_mujoco.yaml 和 rally_a3_reference.yaml 的
球/球桌参数已统一为 `.200` A3 launch 实际配置；不再默认使用旧的G1现场标定值。
后文2026-10-20的“训练物理/真机标定物理”结果是历史证据，不代表当前配置。
来源及对拍见 [A3_PLANNER_PARITY_20261021.md](A3_PLANNER_PARITY_20261021.md)。

## 3. 独立 ROS planner（单独终端）

ROS planner 订阅 `/poses` 并发布 `/racket/command`。在 deploy 仓库构建、source 并启动本仓库 ROS 包：

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

流程与 HOPE workspace 一致，但 package、planner 核心、参数文件对应本 deploy 仓库：
source ROS 和已安装的 HOPE ROS 接口 workspace，在本仓库用 `colcon build --symlink-install`
构建，source 本仓库 install，再启动 planner。节点订阅 `/poses` 并发布
`/racket/command`；核心来自 `deploy/hope_g1/hope_planner_optimized_2/`，参数来自
`deploy/hope_g1/configs/hope_planner_optimized_2.yaml`。构建输出全部放在 `deploy/`，
不构建或修改 `/home/lxz/HOPE`。手动调参只改 deploy 下的 YAML，然后重新 build 并重启。

在另一个 ROS 终端确认话题：

```zsh
source /opt/ros/humble/setup.zsh
source /home/lxz/HOPE/hope_ws/install/setup.zsh
export ROS_DOMAIN_ID=23 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic info /racket/command -v
ros2 topic echo /racket/command --once
```

只有来球满足 planner 发布条件时才会出现 racket command；没有消息时，同时检查 planner
启动日志中的 `publish_intercept_box` 和 `/poses` 是否持续更新。

## 4. 只读接收（不输出电机指令）

```zsh
cd /home/lxz/deploy_g1_pingpong
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env real --network-interface eno1 --ball-source mocap \
  --motion-index 0 --motion-select auto --motion-allow all \
  --target-source hope_planner_optimized_2 --a3-target-mode raw --duration 10 \
  --output "deploy/recordings/hope_g1/real_readonly_$(date +%Y%m%d_%H%M%S)"
```

等待机器人 LowState 成功连接，确认没有根位姿过期、DDS 或时间戳报错。
若 eno1 没有 carrier/IP 或无 LowState，先解决连接。此步骤和离线测试均不能替代真机验收。

## 5. 电机控制（操作者手动执行）

关闭其它机器人控制程序；保持桥接运行。首次测试使用支撑并由操作者监护。

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

START → 等待 POSE_HOLD → A → 等待 STABLE → 确认站稳后 UP →
POLICY_POSTURE → TO_POLICY → WAIT_BALL，再送球。自动选动作、挥拍并回位。
DOWN 返回 AMP；SELECT/B/X 或 Ctrl-C 停控并进入阻尼流程。
真机无需 MuJoCo 的 L/F/T 键，来球由真实发球提供。动作 ID 为0..21，不能使用旧任务的24。

本机三种子30/30有效回球是仿真结果；同步不代表已经通过真机稳定性或击球验收。

## 对齐后的测试命令

训练球物理回归（默认 rally_mujoco.yaml）：

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --motion-select auto --motion-allow all \
  --episodes 10 --duration 70 --min-return-rate 0.8
```

使用与真机完全相同的规划和球物理参数（仅物理模型仍是仿真）：

```zsh
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --rally-config deploy/hope_g1/configs/rally.yaml \
  --motion-select auto --motion-allow all \
  --episodes 10 --duration 70 --min-return-rate 0.8
```

两条命令均可加 `--render`。训练物理三种子30/30有效回球；
真机标定物理seed42为10/10碰拍、9/10有效回球，全部完成回位，无站立丢失。
这些是模拟接触结果，不是实际机器人击球验收。
标定参数只使用原真实记录前3球的飞行/反弹速度比，后3球留出验证：
穿平面位置误差中位数4.5cm、最大5.8cm；时间误差中位数8.6ms、最大77ms。
静态球的2.3cm高度残差、旋转/接触模型误差和更多不同来球仍需现场验证。

重放真实日志（不连接机器人或ROS，不输出电机命令）：

```zsh
.venv/bin/python -m deploy.hope_g1.replay_real \
  --summary deploy/recordings/hope_g1/real_20261020_140656/summary.json \
  --ball-samples deploy/recordings/hope_g1/real_20261020_140656/mocap_packets.jsonl \
  --output deploy/recordings/hope_g1/hope_table_replay.json
```

报告里的原触发时刻来自旧运行；新planner在该时刻可能还在等待<=0.5s窗口，
不能把离线位置误差当成新控制器已经实际碰拍或成功回球。
