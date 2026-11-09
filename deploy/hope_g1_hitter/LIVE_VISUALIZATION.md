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
- 橙色/蓝色：复用并列 `g1_planner_ros2` 的 `TrajectoryMonitor`，使用本次实机标定，
  15 Hz 生成辅助预测，蓝色保留本段首次预测。不预测球拍、球网碰撞。
- 紫色：deploy 实际收到的有效 planner 击打目标。辅助预测不等于 planner 指令，
  不会用于控制或替代 planner 的击打求解。
- 已锁定的指令、目标关节角、原始根部姿态等（控制循环有提供时）同时保存在记录里。

场景仅更新运动学，没有 `mj_step` 或物理 rollout。骨盆/关节状态缺失或超过 0.2 秒、
球观测超过 0.1 秒时隐藏实时模型/球，显示过期状态；收到的规划目标超过 0.15 秒隐藏。
保留的蓝色参考轨迹是历史预测，不表示当前仍有效。

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

回放默认暂停，取消“暂停回放”后播放；支持调速、从头播放，结束后保留最后画面。
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
