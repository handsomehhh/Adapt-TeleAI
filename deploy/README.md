# G1 AdaPT 部署验证与模块测试

从仓库根目录运行。参考相邻 `Adapt-TeleAI/deploy` 的浏览器控制、MuJoCo
后端、模块测试、记录和回放结构重新实现；不导入其中的 HOPE/A3 控制器、
动作选择器、残差策略或真机启动流程。

该目录是 **CPU 原生 MuJoCo 仿真验证平台**。策略、MVAE、29 关节伺服配置、
机器人资产、观测构造、球动力学与估计器来自当前 G1 AdaPT 实验。
CLI 会屏蔽 CUDA，默认不使用 GPU，也不连接机器人 SDK、DDS 或 ROS。
训练后台服务与此平台独立。

## 浏览器查看第一轮完整策略

```bash
bash deploy/run.sh --config deploy/configs/round1.json
```

打开终端打印的地址，默认 `http://127.0.0.1:8085`。初始暂停，取消“暂停”即可运行。
启动时会立即打印仓库与解释器路径，再打印模型和网页加载进度；加载运行库需要数秒。
不会自动弹出本地仿真窗口。已有后台查看器时，新开测试请增加 `--port 8086`，
并打开新端口对应的网页，或先停止后台查看器再使用默认端口。
需要在关闭终端后保留查看器时，使用独立用户服务：

```bash
bash deploy/service.sh start
bash deploy/service.sh status
bash deploy/service.sh stop
```

停止此服务只结束查看器。服务日志在 `deploy/recordings/service/viewer.log`。
远程主机可在本机使用 `ssh -N -L 8085:127.0.0.1:8085 用户名@服务器` 转发后打开。
入口优先使用 `ADAPT_PYTHON` 指定的解释器，其次本仓库 `.venv`，再其次相邻
`Adapt-TeleAI/.venv`（仅复用依赖，不使用其源码）；否则交给项目 `uv run`。

页面提供模块切换、暂停、单步、重置、单球、独立/连续来球、显示倍速、手动参考
速度和 32 维 latent、参考骨架、实际/参考拍心、拍面法向、预测球轨迹和落点目标。
显示当前 Planner、alpha、跟踪误差、击球/过网/落台/摔倒状态和误差曲线。
所有控制事件在主仿真线程执行；显示倍速只改变墙钟调度，物理步长始终 0.005 s，
控制周期始终 0.02 s。

网页先点击三维画面获取焦点，再使用 **Space 暂停/继续**、**R 从头重置并暂停**；
输入框中打字不会触发快捷键，长按不会连续触发，Ctrl/Cmd+R 保留浏览器刷新行为。
快捷键与页面按钮共用主线程事件队列；回放模式中 R 回到第一帧。

物理仿真固定 **200 Hz**，策略控制固定 **50 Hz**，画面独立以 **60 Hz 为目标**发送。
“播放倍速”默认 **1× 真实时间**；慢放只改变墙钟调度，仍执行相同的 0.02 s 控制步。
画面在相邻已执行状态之间插值（1× 时有最多一个控制周期的显示延迟），
不会将插值状态送给策略、球接触检测或记录，也不会在 reset / motion 切换时跨段插值。
“动作速度 alpha”会改变参考动作推进及策略输入，与观看慢放不同；对比默认效果时两者
均保持 1。高层 Planner 模式下，alpha 由 Planner 输出，手动 alpha 控件禁用。
状态区显示服务器画面发送率与实测仿真/墙钟速度；发送率不等同于浏览器实际绘制率。
主机短时卡顿只允许有限步数追赶，不会在暂停、切换倍速或长时间阻塞后突然快进。

### Tracker 完整 motion 对比

```bash
bash deploy/run.sh --config deploy/configs/round1.json --mode tracker --port 8086
```

打开 `http://127.0.0.1:8086`。原色完整机器人执行 Tracker 控制，绿色完整机器人
显示 NPZ 根姿态和全部 29 个关节的参考动作（包含球拍），初始帧也会立即显示。
默认并排；可以切换“重叠”、调节参考透明度、隐藏参考机器人或骨架。
并排只是把参考图像沿 Y 轴平移 1.4 m，误差与策略输入仍使用原始坐标。

“Tracker motion”面板提供：

- **选择 motion**：区分训练快照中的“训练原始动作”和“物理修正动作”；
  显式传入 `--motion /绝对路径/动作.npz` 时，也会列出其所在目录的其他动作。
  单独从 CLI 启动 Tracker 默认选训练原始动作；高层模块默认从修正动作库随机选择
  初始动作，符合训练时的起点分布。显式 `--motion` 可以固定一个起点。
- **上一个 / 下一个 motion**：在当前来源目录内循环选择。
- **motion 时间**：拖动到指定参考时间，同时将实际机器人重置到该帧，清空控制历史，
  并暂停；取消暂停后测试从该姿态开始执行的效果。拖动不会回放之前的实际物理轨迹，
  要查看已执行过程应使用录像回放模式。
- **重置机器人与参考**：当前 motion 从头开始，两台机器人同步回到起点并暂停。
- **循环 motion**：默认关闭，播完整段后暂停；打开后每轮从头重置机器人再执行，
  不把动作末帧与首帧当作连续的物理运动。手动参考速度控制动作进度。

切换动作复用已加载 Tracker；motion 控件只在 Tracker 模式启用。记录保留每帧
motion 路径、参考时间和段编号，`summary.json` 记录切换/重置位置，回放可跨动作。
这些操作不会改变训练数据或训练服务。

## 单独验证每个模块

| `--mode` | 执行内容 | 必需模型 |
| --- | --- | --- |
| `tracker` | NPZ 参考动作 → Tracker → 29 关节目标 → 动力学 | Tracker |
| `mvae` | 手动 latent/速度 → MVAE，运动学预览 | MVAE |
| `mvae_tracker` | 手动 latent/速度 → MVAE → Tracker → 动力学 | MVAE、Tracker |
| `rally` | 始终调用 Rally → MVAE → Tracker | Rally 及其冻结依赖 |
| `standing` | 无球观测 → Standing → MVAE → Tracker | Standing 及其冻结依赖 |
| `hierarchy` | 按球观测有效性切换两个 Planner | 四个模型 |
| `ball` | 固定机器人姿态，测试发球、动力学、位置估计和轨迹预测 | 无策略模型 |
| `replay` | 原样显示记录的机器人/球状态，不重算物理与评分 | 记录文件 |

`mvae` 和 `ball` 的机器人是运动学显示，不能据此判定机器人站立稳定；
测试生成动作的物理可执行性应使用 `mvae_tracker`。
`rally`/`standing` 是独立测试高层策略，但物理执行仍需其配套的 MVAE 和 Tracker。

```bash
# 同一套第一轮模型，按模块查看
bash deploy/run.sh --config deploy/configs/round1.json --mode tracker
bash deploy/run.sh --config deploy/configs/round1.json --mode mvae
bash deploy/run.sh --config deploy/configs/round1.json --mode mvae_tracker
bash deploy/run.sh --config deploy/configs/round1.json --mode rally
bash deploy/run.sh --config deploy/configs/round1.json --mode standing
bash deploy/run.sh --config deploy/configs/round1.json --mode ball

# 当前第二轮 Tracker；无需等其他模块完成
bash deploy/run.sh --config deploy/configs/round2_tracker.json
```

CLI 可以覆盖 JSON 设置。`--run` 读取训练 `status.json` 并定位各阶段模型；
尚在训练的阶段会找最新已保存 checkpoint。也可指定 `--bundle`、`--motion`、
`--tracker`、`--mvae`、`--rally`、`--standing`。
Tracker 接受训练 checkpoint 或 JIT；Planner 接受训练 checkpoint，使用其成对 JIT。
Planner 会校验其冻结依赖哈希；混用其他轮次的 Tracker/MVAE 会被拒绝。
训练 checkpoint 使用 PyTorch 原格式，只加载自己信任的本地文件。

## 无界面批量测试

```bash
# 所有七种模块各运行 250 控制步
bash deploy/run.sh --config deploy/configs/round1.json --suite --viewer headless \
  --steps 250 --output deploy/recordings/round1_suite

# 持续测试，直到 Ctrl+C；默认不设步数上限
bash deploy/run.sh --config deploy/configs/round1.json --viewer headless

# 明确指定 120 秒的连续回归测试，结束后保留完整记录
bash deploy/run.sh --config deploy/configs/round1.json --viewer headless \
  --scenario continuous --steps 6000 --output deploy/recordings/round1_continuous
```

默认 `continuous` 保留机器人、生成器与动作历史；不因击球、单球评分结束、4 秒
时限或摔倒而自动重置或暂停。评分结束后继续运行 Standing，保留球的后续解析轨迹，
直到球落地或静止，再等候 0.5 s 后发下一球。机器人摔倒后继续执行并记录后续状态，
自动发球等待机器人恢复站立；当前策略没有经过从地面起身训练，可能无法恢复。
手动 R 可以重置。球的评分仍使用训练的终止判据（包含 2.5 s 单球超时），
终止评分不会终止后续物理演示，也不会把失败改记为成功。

如果 Planner / MVAE / Tracker 出现非有限输出，页面明确标记策略数值失败，
锁存错误并保持最后有效的关节目标，继续记录后续物理状态；不会自动恢复网络、
重置机器人或把这段保持动作算成策略成功。R 重置后才重新启用策略。
`control_fault_events` 记录时间与错误，后续帧 `policy_output_valid=false`。
`summary.json` 的 `completed=true` 仅表示执行和记录正常结束，不能代表策略通过测试。

`--scenario independent` 用于原评估协议对照，每球结束后重置机器人和生成器，
不能用这个模式的平均击球率证明连续恢复稳定。连续记录中的 `fall_events` 和
`ever_fallen` 覆盖单球评分结束后发生的摔倒；`trial_rates` 仅统计单球评分阶段。
重置/切换模块是显式用户操作。实际球来自独立合法发球库，接触使用执行后的拍面，
绝不使用 MVAE 参考拍面。策略输入只使用带噪声/丢帧/延迟的位置估计；
可通过 `--noise`、`--dropout`、`--delay-steps` 和 `--seed` 设置测试条件。

输出默认为 `deploy/recordings/<时间戳>/`，不会覆盖已有输出目录。
`summary.json` 记录模式、物理/感知配置、模型路径/哈希、逐球结果和统计；
`trace.npz` 与 `chunks/trace_*.npz` 合起来记录每个控制步的 qpos/qvel、目标关节、参考姿态、球、拍面、
latent/速度、132 维 Tracker 观测和 309 维 Planner 观测。
观测有效标志说明对应模式是否真正使用该网络，未使用项以零填充。
浏览器“保存记录”保存当前全部记录；正常停止或 Ctrl-C 也会保存。
启动时先保存模型和运行元数据。轨迹与摘要以临时文件完整写完后原子替换，
最终退出保存期间忽略重复 Ctrl-C，避免留下部分压缩文件；运行中的快照标记
`completed=false`，正常退出保存后才标记执行完成。
默认 `--max-record-steps 0`，不因记录长度暂停。每 1000 步分段写盘，内存只保留
最近不足 2000 步；`--record-chunk-steps` 可调整分段大小。回放会按 `summary.json`
中的 `trace_chunks` 顺序自动拼接全部分段和当前 `trace.npz`，请保留整个记录目录。
只有显式设置非零 `--max-record-steps` 才会达到该步数后暂停。
错误写入 `error.json`，本次记录标记未完成。

## 回放

```bash
bash deploy/run.sh --mode replay --trace deploy/recordings/round1_suite/hierarchy/trace.npz
```

通过回放帧滑块跳转、暂停、单步和显示倍速检查具体动作。回放不需要模型权重，
但需要原记录的 `summary.json` 和相同机器人资产。它不会重新执行策略或评分。

## 验证与边界

```bash
uv run python -m pytest -q deploy/validation/test_platform.py
uv run python -m pytest -q deploy/validation/test_playback.py
uv run python -m deploy.validation.web_smoke
uv run python -m deploy.validation.playback_smoke
uv run python -m deploy.validation.training_parity
```

本机没有项目 `.venv` 时，可像 `run.sh` 一样，通过
`uv run --no-project --python ../Adapt-TeleAI/.venv/bin/python python -m pytest ...`
使用已有依赖。

专项测试使用真实第一轮模型，覆盖各模块运行、缺少其他模块时的独立加载、
冻结依赖拒绝、与 `HierarchicalController` 的观测/关节目标/状态反馈一致性、
逐帧回放一致性、显示刷新不改变物理轨迹。真实 HTTP 测试覆盖完整策略页面和回放页面。
模块权重不存在时，模型相关测试会跳过，不能据此宣称已验证真实模型。
`playback_smoke` 使用真实 Tracker 分别运行 1× 和 0.25× 六秒，检查墙钟速度与
独立画面发送率，并将测量结果写入 `deploy/recordings/pacing_validation_*/`。

原生 CPU MuJoCo 与训练的 MuJoCo-Warp 在接触求解和子步运动学刷新时序上存在
数值差异；部署现在对齐训练的球观测更新顺序和子步派生姿态读取时序。
机身角速度使用世界 `cvel` 旋转到机身 link 坐标，避免误用
`mj_objectVelocity(BODY, local=1)` 返回的惯性坐标系分量。
`training_parity` 会用同一个初始姿态、无噪声来球，对比真实 CPU 训练环境与原生
部署链的观测、目标关节和物理状态，避免只比较两套可能共享错误的手写推理代码。
本平台验证部署控制链，不能直接把结果当作训练同源评分。
训练同源最终指标仍使用 `scripts/evaluate_g1_final.py`。
当前保持解析球模型，球与机器人之间没有反向力反馈；未新增旋转球物理。
没有硬件网络或电机输出，不能把平台运行成功当作实机部署通过。

## 文件结构

`artifacts.py`：模型解析与依赖核验；`control.py`：分模块推理；
`world.py`：原生 MuJoCo；`session.py`：控制/感知/来球/评分与记录；
`viewer.py`：浏览器场景与事件队列；`replay.py`：原样回放；
`recording.py`：有界内存记录与分段保存；
`validation/`：契约与实际模型验证。
