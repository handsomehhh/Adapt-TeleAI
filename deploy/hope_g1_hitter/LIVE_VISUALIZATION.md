# 实机实时可视化与同步记录

实时查看器随 deploy 启动，在独立进程中运行 Viser、辅助球预测和日志记录。
它使用 deploy 已收到的动捕和 DDS 实测状态，不另外订阅机器人 DDS，不发送控制指令。
原有动捕转发、转换、planner、UDP bridge 的启动方式不变。

在 lxz 的 `~/Adapt-Deploy` 中，给原有实机部署命令增加 `--live-view`：

```bash
cd /home/lxz/Adapt-Deploy
.venv/bin/python -m deploy.hope_g1_hitter.deploy_agent \
  --env real --network-interface eno1 --enable-motors \
  --duration 600 \
  --live-view --viewer web --web-host 0.0.0.0 --web-port 8081
```

打开 `http://192.168.152.200:8081`。无需浏览器连接即可运行和录制。
启动时自动加载并列的 `~/g1_planner_ros2`，无需额外设置 `PYTHONPATH`。
手柄 START/A/UP/DOWN 与停止按键语义不变。关闭浏览器不会停止机器人。
查看器的退出和异常不会请求机器人停止；部署退出、机器人 IO 关闭后才清理查看器。

**只读验证**：去掉 `--enable-motors`，其余参数保持不变，即可读取机器人状态并显示，
不发送电机指令。未启用策略时 planner 可能没有有效击打目标，但辅助球轨迹预测仍工作。
只记录、不启动网页用 `--live-record-only` 替代 `--live-view`。
这些参数用于 `--env real`；MuJoCo 仿真仍用原有 `--render --viewer web`。

## 显示与数据来源

- 机器人：本次 checkpoint 对应的完整模型，骨盆位置/姿态使用控制循环实际观测，
  29 个关节使用 DDS 实测 `q`，不是 policy 的 `target_q`。可展开关节角数值表。
- 球台：本次部署场景里的球台。模型、球、规划目标统一对齐到 HOPE 桌面坐标。
- 绿色：桥接后、与 deploy 一致的原始动捕球观测，最近 5 秒。
- 青色/蓝色：复用并列 `g1_planner_ros2` 的 `TrajectoryMonitor`，使用本次实机标定，
  15 Hz 生成辅助预测，青色为最新预测（RGB 26/217/255、世界宽度 0.008 m，与 PingPong 一致），
  蓝色保留本段首次预测。不预测球拍、球网碰撞。三条轨迹各有独立开关。
- 橙色扁圆：最新辅助预测中记录的球台反弹点；半径 0.065 m、半厚 0.008 m、
  不透明度 0.85，可独立关闭。回放只读已有反弹点，不重新预测；球观测过期时隐藏。
- 绿色半透明骨盆：控制器当前的 `goal_base_xy`；金黄色半透明球拍：控制器当前的
  `goal_position`。复用保存模型的几何，颜色和透明度与 PingPong 1v0/1v1 目标虚影一致
  （骨盆 RGB 38/255/115、不透明度 0.35；球拍 RGB 255/191/26、不透明度 0.45）。
  骨盆保留实测高度，两个虚影都保留实测几何姿态，仅显示位置目标。
- 虚影反映实际传给 policy 的目标，包含随挥和空闲时的目标变化；不使用最后收到的
  ROS command 推测当前目标。原紫色点已由虚影替代，可用“显示骨盆 / 球拍目标虚影”开关隐藏。
  辅助球预测不会用于控制或替代 planner 的击打求解。
- 已锁定的指令、目标关节角、原始根部姿态等（控制循环有提供时）同时保存在记录里。

场景仅更新运动学，没有 `mj_step` 或物理 rollout。骨盆/关节状态缺失或超过 0.2 秒、
球观测超过 0.1 秒时隐藏实时模型/球，显示过期状态。
虚影要求本帧有有效机器人观测、`policy=pingpong` 及两个有限的位置目标；数据过期、
策略退出或目标字段缺失时隐藏，不沿用上一帧目标。

新增记录的 metadata 版本为 2，保存虚影几何名称；state 事件补充 `goal_base_xy`、
`policy` 和 `swing_active`。需要重启 deploy 才会开始记录新字段。
旧日志如果缺少这些记录，不补算骨盆目标，也不退回紫色点兼容显示；原始模型与球轨迹仍可回放。
保留的蓝色参考轨迹是历史预测，不表示当前仍有效。

实时 Live View 与日志回放共用 PingPong 风格的中文三标签页：

- **控制**：紧凑状态、日志选择与播放、任务叠加开关、相机跟随/视场角/重置。
- **可视化**：实测关节角、坐标系、惯性、关节、骨架连线及凸包等模型显示控件。
- **分组**：几何体、参考点、关节等 MuJoCo 分组开关。

默认采用 1v0 单机器人斜视近景：距离 4.8 m、方位角 145°、俯视 20°，
观察中心为球台地面坐标 (0.65, -0.76, 0.9) m；按记录的坐标平移转换。
鼠标仍可旋转和缩放，“重置相机”回到默认视角。接触力等未记录的动力学数据不提供显示控件。

## 日志与回放

默认写在本次 deploy 输出目录的 `visualization/` 下，启动时打印完整路径：

```text
deploy/recordings/hope_g1_hitter/real_YYYYmmdd_HHMMSS/
  trace.jsonl                   # 原有部署日志
  visualization/
    metadata.json               # 模型、关节顺序、坐标平移、物理参数及校验值
    scene.mjb                   # 本次实际部署场景，包含网格
    calibration.yaml            # 本次实机物理标定快照
    events.jsonl                # 原始观测、机器人帧、收到的指令、当时生成的预测
    summary.json                # 事件数量、预测误差统计
    transport_summary.json      # 队列接收/丢弃数量及工作进程退出状态
```

指定 `--output /path/to/new_run` 可选目录；不会覆盖已有 `visualization/`。
原来的 `--output xxx.jsonl` 格式则记录在 `xxx_diagnostics/visualization/`。
日志不自动轮转或清理。可视化队列上限 2048，拥塞时丢弃观测事件而不等待控制循环；
应检查 `transport_summary.json` 的 dropped/forced_stop 和工作进程退出状态。
显示失败会写 `display_error.txt` 并继续后台记录；工作进程失败写 `worker_error.txt`。

回放命令（将路径替换为实际打印的目录）：

```bash
cd /home/lxz/Adapt-Deploy
.venv/bin/python -m deploy.tools.real_viewer \
  --replay /home/lxz/Adapt-Deploy/deploy/recordings/hope_g1_hitter/real_YYYYmmdd_HHMMSS/visualization \
  --web-host 0.0.0.0 --web-port 8081
```

也可以传入所有实机记录的父目录，在浏览器的“运行记录（最新在前）”下拉列表中选择一行：

```bash
cd /home/lxz/Adapt-Deploy
.venv/bin/python -m deploy.tools.real_viewer \
  --replay-dir /home/lxz/Adapt-Deploy/deploy/recordings/hope_g1_hitter \
  --web-host 127.0.0.1 --web-port 8081
```

目录模式扫描直接子目录中的 `visualization/`，列出同时包含 `metadata.json`、
`scene.mjb`、`events.jsonl` 的记录，默认打开名称排序最新的一条。
“刷新记录列表 / 重新读取当前日志”可发现新记录，也可重新读取仍在录制的日志；
每次读取时固定文件长度，播放期间进度条不会随源文件增长。
`--replay` 和 `--replay-dir` 二选一，原来的单目录命令仍然支持。

回放默认暂停，点击“播放”开始；支持暂停、前进一帧（20 ms）、调速、从头回放，结束后保留最后画面。
“回放进度（秒）”可拖动或输入秒数，前进或后退后自动暂停；机器人、动捕轨迹、
当时预测和规划目标一起恢复到选定时刻，点击“播放”继续。
进度条中的深色区间表示连续至少 1 秒没有有效球观测；球样本超过 0.1 秒未更新即为不可见，
短暂丢帧不标记。下方显示区间数量和总时长，可拖动跳过，但不会自动跳过或删除日志。
日志开头、结尾的长时间无球也会标记；切换记录或重新读取日志时重算。
通用 Viser 日志回放界面的“回放时间（秒）”同样显示这些区间。
切换记录会加载该记录自身的模型和坐标配置，时间归零并暂停，浏览器地址不变。
回放启动时建立每 5 秒的状态检查点，不需要把全部事件常驻内存，也不重新预测轨迹。
`--autoplay` 自动播放，Ctrl-C 关闭回放。回放不连接 ROS、机器人，不加载 policy，
不需要 planner 包，不重新计算预测，不修改源日志。复制整个 `visualization/` 即可搬移；
二进制场景建议使用 `metadata.json` 中记录的同版本 MuJoCo 打开。

同步采用同一主机 monotonic 时间上的因果保持：显示该时刻之前已经收到的最新值，
不使用未来采样插值。原始动捕时间戳、机器人观测年龄和本机接收时间都被保留；
这不是不同传感器硬件采样时刻完全一致的承诺。

旧的未完成 `deploy/tools/live_viewer.py` 已删除，改用以上入口。

本机回归检查：
` .venv/bin/python -m pytest deploy/validation/live_visualization_checks.py -q `。
检查覆盖队列拥塞、源时间戳、过期状态、子进程记录、保存预测的回放和模型坐标。
另用实际部署模型与合成传感器数据完成 Chrome 显示验证；这不代表完成实机联调。
