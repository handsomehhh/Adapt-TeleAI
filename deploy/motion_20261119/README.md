# 2026-11-19 动作库双机模式

`motion_20261119` 是与 `hope_g1_hitter` 并列的完整 policy/planner 启动模式。复刻依据是 lxz 的 `/home/lxz/deploy_g1_pingpong/deploy/recordings/hope_g1/real_20261119_095258/`：该次运行完成 288 次挥拍及回位。

## 启动目录与命令

以下每个终端都在 **lxz** 执行，先进入同一个目录：

```bash
cd /home/lxz/g1_duel/mocap_duel
```

先停止已有 host、P1、P2 控制进程，再选择模式；原有网络隔离启停方式见相邻 [mocap_duel README](../../../mocap_duel/README.md)。本模式使用同一组网络配置，切换模式不需要改变网卡、IP 或网络命名空间。

```bash
# 终端 A：VRPN、坐标转换、一个共享规划工作进程、两路 bridge、集中 rosbag
python3 scripts/run.py host --config config/site.yaml --mode motion_20261119

# 终端 B：P1，eno1，pose1，不翻转；各自遥控器控制接管
python3 scripts/run.py actor --config config/site.yaml --mode motion_20261119 --robot P1 --enable-motors

# 终端 C：P2，enx6c1ff7eb0f25，pose5，主机翻转
python3 scripts/run.py actor --config config/site.yaml --mode motion_20261119 --robot P2 --enable-motors

# 终端 D：原有统一画面，显示球、球台、两台机器人；无需填写模式
python3 scripts/run.py viewer --config config/site.yaml
```

统一画面还显示黄色的 planner 来球预测线，以及 P1 蓝色/P2 橙色的半透明球拍目标。预测直接来自历史 Stage 2 求解时保留的采样，未额外拟合；球拍目标来自各自控制器实际采用的 `goal_position`，不把数组中心候选当作实际选中目标。虚影仅表示位置，姿态沿用实测姿态。网页可分别开关，超过 0.2 秒的观测/预测隐藏。更新预测功能后重启终端 A 和 D；命令及网络配置不变。

`--enable-motors` 授权输出控制指令。省略它时 actor 只读反馈。沿用每台机器人的 START → A → UP 接管、DOWN 返回站立、SELECT/B/X 停止操作；两台独立处理故障和停用。新模式始终记录统一 viewer 所需的实测状态，使用终端 D 显示，不使用 actor 的独立 `--live-view`。

默认仍为原 hitter 模式。切回时停止 A/B/C，再从上述 host/actor 命令移除 `--mode motion_20261119`，或改为 `--mode hitter`。两个模式不能混用；启动器检查正在运行的固定角色，UDP 另外核对模式与 P1/P2。日志自动建立，进程之间不需要手工 runid。

## 忠实保留的部分

- 根据部署基线 `04385c17d978e03802bf3e6578efbc22dc719563` 加该次归档的 `deploy_g1_pingpong.diff` 和 untracked 归档恢复源码；不是从今天的原工作区直接复制。
- 原 `model_60000.pt`（SHA256 `75485485106fd222d3af390447b6857b102131f094927bb3eb3779cebe4964ea`）、训练参数、`motions_compact162` 中的 162 条动作、原训练 URDF、原 AMP 站立策略。`ASSETS.json` 记录全部文件哈希，启动时校验。
- 原 175 维观测、`(obs - mean) / (std + 0.01)`、29 维原始动作历史、关节顺序和 PD。保留 pelvis 源频率滤波/速度估计及 IMU 融合。动作参考使用原 `a3_live` 规则：初次选择按实测 anchor 对齐，执行中保留参考变换，回到冻结 home。
- `distance_command_array` 的完整候选数组选择、时间窗、15 tick settle、原 motion 起始帧、连续目标更新、关闭 live motion reselection、击球后 30 帧跟踪、10 帧回位、`base_target_transition: smooth`、关闭 base compensation。
- 原 ROS 命令数组的 compact 顺序、center_index、逐候选 valid/工作区过滤、发布时间相对 TTS、bridge monotonic 解码。没有替换为 hitter 的单个目标协议。
- 原 DDS 子进程、遥控器状态机和停止行为。

源码在 `reference/` 中隔离，避免改变旧 hitter 依赖的模块。`SOURCE.json` 记录归档来源和文件哈希。相对原源码，仅重定位 Python 包名，并在 `reference/backends/unitree.py` 添加可选观测记录回调、输入工厂注入，以及已有的双机 DDS 日志文件名隔离；没有改动控制状态转换。

## 双机适配边界

planner 在相邻 `g1_planner_ros2` 的 `hope_planner_g1.motion_20261119` 中。一个 ROS 节点接收全部球位置，一个独立工作进程计算历史 Stage 1–3。按实际球 X 方向确认接球侧：P1 收负 X 来球，P2 收正 X 来球；P2 先转到自己的桌面坐标，再运行相同历史算法。换边会清空上一段拟合和求解缓存，丢弃尚未发布的旧结果，只给当前接球侧发布数组。停用一侧不会阻止另一侧规划。

球估计由主机转发，两台 actor 不做自己的球轨迹拟合；原来单机 actor 辅助球估计器被共享缓存替代。该模式的 175 维观测与 command-array 选择均不读取这个辅助球估计来生成目标。原策略已接受的挥拍仍由原控制器完成，不增加强制中断。

planner 使用该次**实际保存**的配置：落点 `(2.3, -0.8)`、回球时间 `0.65 s`、中心平面 `x=-0.2 m`，前方 15 个/后方 20 个附加平面、间距 `0.02 m`，共 36 个候选平面。摩擦反弹模型、阻力 `0.096`、水平系数 `0.816019`、竖直系数 `0.862769`、球拍系数 `0.7`、Stage 2 `2 ms` 和 Stage 3 `RK4 / 25 ms` 均保留。rally 中的历史 ball 参数也原样保存；实机目标由外部 planner 配置决定，未擅自将两处历史值统一。

## 验证与文件同步

本机与 lxz 均有 `artifacts/`。按仓库原有规则，大型权重和动作文件不进入 Git；代码同步之外需要同步该目录，哈希不符会拒绝启动。

在 **lxz 的 Adapt-Deploy 目录**，以下命令只做离线检查，不连接机器人：

```bash
cd /home/lxz/g1_duel/Adapt-Deploy
.venv/bin/python -m deploy.motion_20261119.deploy_agent --env check
PYTHONPATH=.:../g1_planner_ros2/src/hope_planner_g1 .venv/bin/python -m pytest -q deploy/motion_20261119/validation ../g1_planner_ros2/src/hope_planner_g1/test/test_motion_20261119.py
```

在 **lxz 的 g1_duel 目录**，合成链路测试固定使用 domain 74、localhost、UDP 回环，不连接 VRPN 或机器人 SDK：

```bash
cd /home/lxz/g1_duel
DUEL_ROS_DOMAIN_ID=74 DUEL_ROS_LOCALHOST_ONLY=1 bash mocap_duel/scripts/ros_env.sh /usr/bin/python3 g1_planner_ros2/src/hope_planner_g1/test/ros_motion_duel_smoke.py
```

验证包含 70 帧原日志的观测/推理/关节目标与 FK、原数组动作选择及连续更新、原来球 252 帧对应 26 组规划黄金结果（两侧分别验证）、换边/过期清空、模式/侧标识、统一 viewer 状态记录，以及两侧 ROS→UDP 链路。实机对打效果需现场验收。
