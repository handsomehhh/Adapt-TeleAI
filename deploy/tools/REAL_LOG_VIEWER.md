# 真机日志可视化回看

入口：`deploy.tools.view_real_log`。独立于部署、策略评测及 planner 重算工具，直接把真机日志中的实测状态放入 MuJoCo 场景。不会连接 ROS/DDS、加载策略网络或发送机器人指令。

## 启动

在项目根目录执行，使用本机带桌面的终端：

```bash
.venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_20261022_121714
```

`--recording`（别名 `--log-dir`）是唯一必需参数，替换成要查看的日志目录即可。已从 .200 只读复制 `real_20261022_121714` 和 `real_20261022_121834` 到本机；未修改远端 HOPE 文件。

从第一次挥拍前开始，以四分之一速度暂停打开：

```bash
.venv/bin/python -m deploy.tools.view_real_log \
  --log-dir deploy/recordings/hope_g1/real_20261022_121714 \
  --start 30 --rate 0.25 --paused
```

时间从第一帧可用机器人状态起算，单位为秒。回看完整过程包括 WAIT、进入策略、挥拍、恢复等日志中记录的阶段。运行期间可自由调整相机视角。

| 按键 | 功能 |
|---|---|
| 空格 | 暂停 / 继续 |
| `R` | 从日志开头重新播放，保留当前倍速和相机视角 |
| `[` / `-` | 速度减半，最低 0.0625 倍 |
| `]` / `+` | 速度加倍，最高 8 倍 |
| `\` | 恢复 1 倍速 |
| 左 / 右方向键 | 后退 / 前进 1 秒 |
| `,` / `.` | 上 / 下一帧机器人实测状态，并暂停 |
| `P` / `N` | 上 / 下一处状态、阶段或 motion 变化 |
| Home / End | 跳至开始 / 结束 |

播放结束自动暂停，按 `R` 可从头重新播放，仍可回退、逐帧查看。按键需要 MuJoCo 场景窗口获得焦点。

## 显示内容与还原范围

- 机器人：实测 `q`、`root_pos`、`root_quat`，按日志的 `robot_command_contract.joint_names` 映射。不是目标关节角，也不重新计算动作。
- 紫色半透明机器人：进入击球策略后，显示该次策略输入 observation 中的参考关节姿态（HOPE 的 `actor_obs` 前 29 维，兼容 132 维 `tracker_obs`）。每帧将参考机器人的 pelvis 位置和完整朝向对齐到实测 pelvis，以便比较肢体姿态；不展示参考根轨迹的偏移。暂停、逐帧、倒退及视频导出均同步更新。
- 右侧「状态与误差」新增 `Motion file (observation)`、`Motion frame` 和 `Reference robot`：文件名、当前第几帧 / 总帧数，以及日志中的零起始 index。HOPE 待球和回位时显示实际输入的 home motion 第 1 帧（index 0），挥拍时显示当前参考帧。日志没有记录帧号时明确显示 unknown，不根据回放时间推算。
- 离开策略、缺失或无效 observation 时隐藏透明机器人。104 维 hitter 策略没有参考 motion 输入，显示 `hitter observation has no reference motion`；不将其陀螺仪或实测关节误当参考姿态。参考显示直接来自日志，无需原始 motion 文件或 checkpoint。
- Base 目标：使用同一日志帧中的 `goal_base_xy`（即策略的 target base position）。黄色地面靶圈标出目标站位，白圈是实测 pelvis 的地面投影，黄色箭头从当前位置指向目标，长度就是实际水平差距。为避免球桌遮挡，在当前 pelvis 高度也绘制目标圈和水平箭头，并用虚线连接到地面投影。上方标记的高度只是显示平面，不代表目标高度；日志只提供 XY。圈的大小仅为标记尺寸，不代表到位容差。
- 右侧同时显示当前/目标 XY（桌面世界坐标，米）、水平距离（厘米）、`target - current` 的 ΔX/ΔY，以及按当前 pelvis 航向解释的 forward/backward、left/right 偏差。目标随待球、挥拍、回位及时间跳转更新；不受 `--command-view` 影响。HOPE 和 104 维 hitter 均支持，缺失目标或退出策略时隐藏标记并显示 `--`。
- 橙色球：`mocap_packets.jsonl` 中原始、已转换到 `hope_table` 的实测球位置；小橙点表示最近 0.4 秒内已收到的观测。
- 绿色点和箭头：机器人已接收并执行的 planner racket command 目标位置、法向；蓝色箭头表示目标速度（画图缩放 0.06 秒）。仅在日志含 `raw_racket_command` 且处于 EXECUTE/SWING 时显示，挥拍结束即隐藏；发球前的 home goal、待球目标及未接受的 planner 预测均不显示。没有已接收指令时，TTS 显示 `--`。这些是指令，不是实测球拍。
- 界面：日志名、当前时间、倍速、暂停状态、state、phase、motion、球的桌面坐标和机器人帧龄。
- 球桌：从日志 `rally_runtime_config.ball` 还原尺寸，使用该次记录的 `simulation_world_translation` 转换到 MuJoCo 地面场景。

机器人和球都使用本机单调时钟对齐：机器人 `time`、球包 `arrival_time`。球只显示当前已经到达的最近样本，不使用未来数据插值，超过 0.1 秒未收到新包则隐藏当前球。机器人按最近实测帧保持，不补算缺失动作；帧龄在界面显示。球包到达时间并非相机曝光时间，因此回看保留了原日志的传输延迟，不宣称可以从画面单独判定真实碰拍时刻。

这是记录状态的可视化还原，不会重新积分碰撞动力学，也不会生成反事实回球轨迹；原日志未记录的数据无法恢复。坐标标定偏差也会原样保留。当前支持具备完整关节顺序和 world 元数据的 HOPE G1 日志，未知坐标系会报错。

## 输入文件

| 文件 | 用途 |
|---|---|
| `summary.json`，必需 | 关节顺序、坐标转换、球桌参数 |
| `trace.jsonl`，必需 | 机器人实测状态、执行阶段及 racket command |
| `mocap_packets.jsonl`，可选 | 原始球观测；缺失时仅显示机器人和指令目标 |

读取时建立内存快照；若日志仍在写入，新增数据需要重启回看工具才能载入。最后一条未写完的 JSON 行会提示并跳过。

只查看文件统计，无需图形窗口：

```bash
.venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_20261022_121714 --inspect
```

## 导出视频

```bash
MUJOCO_GL=egl .venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_20261022_121714 \
  --start 30 --end 35 --rate 0.25 --fps 30 \
  --video deploy/recordings/hope_g1/review/first_swing.mp4
```

`--rate` 同样决定视频速度：5 秒日志以 0.25 倍速导出约 20 秒视频。视频含场景和观测标记；交互窗口额外显示状态文字。
