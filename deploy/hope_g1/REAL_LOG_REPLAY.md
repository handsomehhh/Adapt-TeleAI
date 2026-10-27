# 真机日志在 MuJoCo 中回放（离线）

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
