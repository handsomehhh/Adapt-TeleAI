# 乒乓球测试的网页可视化

所有原有 `--render` 测试入口默认使用 Viser 网页显示，与 mjlab 网页 viewer
共用 `mjviser.ViserMujocoScene`。MuJoCo、policy、planner 和评分仍在 Python 后端运行。
模型状态、球、网格、command 和误差图形直接来自原测试，网页不重新计算物理或评分。
本次迁移不修改真机控制参数、checkpoint、PD 或 planner 算法。

## 启动

推荐先启动统一测试入口：

```bash
.venv/bin/python -m deploy.tools.web_platform
```

打开 `http://127.0.0.1:8070`，选择测试模式，再点击“启动测试”。
页面提供四种模式：球拍指令评测、完整 planner 接球评测、真机回放、连续手动仿真。
三维画面内嵌在右侧，也可以独立打开。底层仍调用下列原有入口。

- 指令评测可以指定 Y/Z，或选择已有冻结 suite；X、朝向、速度和 TTS 沿用原采样配置。
  当前使用纯 command 跟踪模式，不生成物理球。
- 接 planner 模式仍走在线观测、估计和当前 planner 流程，沿用现有 HopeReceiveAdapter。
- 回放自动列出本地 `deploy/recordings/hope_g1` 的日志，可以指定起始秒数，默认暂停。
- 可以设置数量、种子、倍速；高级设置支持 command adapter 和现有 YAML 配置路径。
  其他框架的 policy 仍需实现对应 adapter，并准备其依赖；网页不会自动转换 obs 或模型。
- 页面显示完整启动参数、运行日志和退出状态，结束后可以打开本次及历史报告。
  评测输出保存在 `deploy/recordings/web_runs/<时间戳>_<模式>/`，并保留 `launcher.log`。
- 一次运行一个任务；切换前先停止当前任务。关闭页面不会停止后台任务；
  停止平台进程会同时停止它启动的测试，不影响其他独立进程。

平台默认使用 **8070**，三维 viewer 使用 **8071**。端口被占用时可以修改：

```bash
.venv/bin/python -m deploy.tools.web_platform --port 8070 --viewer-port 8071
```

远程运行统一入口时，需要同时转发两个端口：

```bash
ssh -N -L 8070:127.0.0.1:8070 -L 8071:127.0.0.1:8071 用户名@服务器
```

下面是直接启动单个场景的方法，仍然可用。

从仓库根目录执行，下列入口默认在 `http://127.0.0.1:8080` 提供页面。
每次以终端打印的实际 URL 为准。一次运行对应一个后端场景；多个浏览器共享播放控制。

真机日志回放：

```bash
.venv/bin/python -m deploy.tools.view_real_log \
  --recording deploy/recordings/hope_g1/real_20261027_173921 \
  --start 67.5 --paused
```

不接 planner，直接评测球拍指令：

```bash
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --point -0.8 0.3 --directions 1 --speed-levels 1 --tts-levels 1 \
  --command-only --render
```

接 planner，完整接球流程：

```bash
.venv/bin/python -m deploy.benchmarks.receive.run --limit 3 --render
```

连续 MuJoCo 部署流程及手动控制：

```bash
.venv/bin/python -m deploy.hope_g1.deploy_agent \
  --env mujoco --render --sim-controls manual --duration 600
```

原来的 task/rally/runtime/mocap 配置参数、冻结 suite、seed、adapter 和输出目录参数
继续有效。不添加 `--render` 的仿真评测继续无界面运行。回放的 `--inspect` 和
`--video` 继续走原有检查/离屏输出路径，不创建网页服务器。

## 网页操作

网页支持原快捷键：Space 播放/暂停，`,` / `.` 逐帧，左右方向键前后1秒，
N/P 切换回放阶段，Home/End 跳到开头/结尾，R 重新播放，`[` / `]` 或
`-` / `+` 调整速度，`\` 恢复1倍速。连续仿真的 Enter、L/A/W/S/F/T/N/R/B/X/Q
等按键仍按原定义运行。统一入口外层页面和三维画面内都可使用；输入框打字、
组合快捷键和长按重复事件不会触发测试指令。若焦点停在输入框或按钮上，
请先点击三维画面空白处。

回放的左右箭头只跳转时间，不移动镜头。侧边“每次跳转帧数”可手动输入整数，
使用“后退 / 前进”两个按钮按实测机器人记录的帧索引跳转并暂停；
逗号 / 句号快捷键仍每次跳转1帧，前后1秒按钮保留。
“设置倍速”可直接输入数值，“当前倍速”同步显示实际速度，包括键盘调整后的值。
控制栏按播放、按帧定位、时间与阶段分区，可分别折叠。数字输入框使用大字号，
帧数旁提供独立的“减少 −1 / 增加 +1”，倍速提供“减半 / 正常 / 加倍”；
不再依赖数字框右侧的小箭头。重播、视角和结束操作放在折叠区内。

播放区显示“播放中 / 已暂停”，按钮随状态切换为“暂停播放 / 继续播放”。
回放的“TTS 时序”显示 command 接收时刻、初始 TTS、经过时间、deadline 和
结束倒计时（过期后为超时秒数）。在锁定执行视图中，以选中 command 的接收时间
为开始，以记录的 deadline 为结束；不是整段动作或回位结束时间。
实测事件可用后，显示估计击球或穿平面时刻相对 deadline 的提前/滞后毫秒数。
这些数值随日志时钟推进，暂停冻结、倒退重新计算；缺失记录时显示未知。
173921 录制约50帧/秒，
1秒约50帧；这里的帧数不等于网页渲染帧数或动捕球采样数。

- **控制**：播放/暂停、倍速、重置视角、结束查看/停止测试。
- **回放专用**：逐帧、前后1秒、阶段跳转、从头播放、拖动时间轴；时间轴数值可以直接编辑。
- **连续仿真专用**：START、悬挂下放、AMP、UP/DOWN、单球、连续发球、单步、重置和停止动作。
- **状态与误差**：保留当前测试的状态、目标、TTS、接受原因和误差。
- **场景显示**：机器人/球桌的几何分组、相机和显示选项。

真实日志回放继续使用锁定执行 command：绿色为目标，蓝色为目标速度，橙色为当前球。
紫色是估计击球点或实测穿平面点，青色是同一事件时刻的拍心，连线比较三者。
是否估计触球、缺测和反向跳转的处理与原回放一致。

网页按钮将操作排队，交由原仿真/回放线程执行。浏览器掉帧和网络延迟只影响显示；
不改变物理步长、观测时间戳、deadline 或评分。关闭浏览器不会自动停止后台任务，
可以重新打开；使用网页停止按钮或终端 Ctrl-C 结束。

## 网页地址与原生窗口

所有可视化入口增加以下选项：

```text
--viewer web          网页模式，默认
--viewer native       使用原来的 MuJoCo 窗口和键盘操作
--web-port 8080       网页端口；同时跑多个场景时使用不同端口
--web-host 127.0.0.1  监听地址
--web-no-wait         不等待首个浏览器连接，直接开始测试
```

默认等浏览器首次连接后才开始，防止短测试在打开页面前已经结束。等待浏览器时
不推进仿真；连接后原测试按其原有流程继续。批量无界面评测及静态 HTML 报告保持原样，
不为每个并行 worker 创建网页服务器。

远程服务器推荐通过 SSH 转发端口。在自己的电脑执行，例如：

```bash
ssh -N -L 8080:127.0.0.1:8080 用户名@运行测试的服务器
```

然后在自己电脑访问 `http://127.0.0.1:8080`。通过跳板机时在 SSH 命令中添加 `-J`。

## 实现与验证

- `deploy/viewer.py`：共用网页后端、调试几何转换和控制事件队列；保留原生后端。
- `deploy/tools/web_platform.py` / `web_platform.html`：统一网页入口、子进程生命周期、
  日志和报告访问；只允许启动现有 MuJoCo 测试及回放入口。
- 原测试执行器：只替换 viewer 创建入口，继续使用原来的 `user_scn`、状态测量和评分。
- `deploy/benchmarks/planner_snapshot.py`：评测专用的可复制 planner 数值状态。
  修复当前线程化 planner 无法深拷贝的问题；在观测送达的虚拟时刻同步执行当前核心，
  不读取未来球轨迹。接球 adapter 使用当前 planner 的诊断字段并转接实测位置观测。
  真机的 ROS/后台线程实现不变。

运行专项验证：

```bash
.venv/bin/python -m pytest deploy/validation/web_viewer_checks.py -q
.venv/bin/python -m pytest deploy/validation/web_platform_checks.py -q
```

专项检查覆盖几何位置/尺寸、透明度、控制线程、待处理时间轴跳转、planner 状态独立性，
以及同一冻结指令在网页/无界面执行时得到完全相同的结果和逐帧记录。
真实 checkpoint 的 command 和 receive 模式也分别进行了网页/无界面对照。
