c# 随机 racket command 测试

网页可视化已接入：原命令增加 `--render`，浏览器打开终端打印的地址。默认 `--viewer web`，
`--viewer native` 保留桌面窗口。详见 [网页可视化](../../WEB_VISUALIZATION.md)。

当前默认进行第一阶段的 MuJoCo command 跟踪测试：在冻结的训练支持域内采样位置、
拍面法向、速度和 TTS，交给现有 motion 选择器和 policy，检查指定 TTS 时刻的实际球拍状态。
不生成球，不调用球状态估计器或 A3 轨迹/回球 planner，也不依赖碰球或落台结果评分。

默认同时启用 `rally_constraints.enabled: true`，把command限制在正常对打的物理范围：
先在该固定位置生成与策略无关的合法来球速度样本（先越网、己方半台一次反弹），
再按理想球拍碰撞模型检查采样法向/速度能否使球越网并首次落到对方半台。
落点距边界至少10cm，球底过网净空至少3cm，出球速度2..12m/s。
拍面法向与速度仍在导出的联合训练支持域内，不调用回球planner生成command。
多档速度从满足条件的范围选取，默认至少相差0.3m/s，不再强制使用1.5/4.5两端。
同方向组的不同速度/TTS共享同一合法来球速度条件；TTS仍独立使用0.28/0.50s。
这是采样前的物理任务定义，不读取策略执行或命中结果，不修改实际动作。
所有条件及模型验证依据保存在 `suite.config.rally_sampling_audit` 中，回放时复核。
未找到足够合法组合的点位/方向单独记录采样状态，不解释为证明不可达或policy失败。

第一阶段仍使用现有选择器，因为该actor需要参考motion。报告保留全部目标作为分母，
单列选择器接受率、所选motion是否位于导出的联合训练支持域，以及实际执行误差。
这不能解释成脱离参考motion的纯神经网络能力上限。

第二阶段“独立随机发球 → 球估计 → planner → command → 选择器/policy”现有独立入口
`deploy.benchmarks.receive.run`，见 [中心区域接球测试](../receive/README.md)。
历史“按command构造物理球”的诊断模式不等于第二阶段。

## 运行

### 按指定 TTS 分组的网格指标

完整区域、充分采样的入口（推荐）：

```zsh
MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/tts-inductor TORCHINDUCTOR_COMPILE_THREADS=1 \
.venv/bin/python -m deploy.benchmarks.racket_command.batch_run \
  --config deploy/benchmarks/racket_command/grid_tts_full.yaml \
  --workers 20 --output deploy/recordings/racket_command/full_grid_run
```

原始8×6网格只去掉最下方一层，保留其余边界，实际8×5=40格。覆盖完整Y范围
Y=[-1.5123,0.0020]m、Z=[-0.16225,0.7128]m；每格每档50个均匀随机位置target，
不跳过训练支持域的空洞。
有位置支持时在该位置的联合训练域中抽法向/速度；无位置支持时仍保留该位置，
从完整冻结域抽向量。额外记录`command_in_training_support`，不据此过滤目标。
`spatial_coverage: full_rectangle`是全区域压力测试，不将训练域外失败解释为域内跟踪能力。

TTS取0.20/0.30/0.40/0.50/0.60s，
共5档、10,000次。当前选择器最大TTS=0.5s；超出启动窗口的指令如实拒绝并计入分母。
每格每档50次时，成功率的95% Wilson区间最宽约±13.4个百分点。
每档重复相同位置/法向/速度target；物理来球仍由该条command独立构造。
接受样本执行4.5s评测回位，已拒绝样本执行至TTS+0.25s，覆盖全部deadline诊断窗口。

并行仅发生在独立MuJoCo进程间，不连接真机；每次仍恢复完整准备状态。
球物理加速用`torch.compile`编译原有核函数，保持相同球/桌/网/移动拍面碰撞模型，
自动化测试对照原始逐步实现的轨迹与事件。启动需要本地C++编译工具链。
为控制报告大小，完整批量配置不保存所有周期trace或球轨迹；`results.jsonl`仍保存每次
指令、选择决定、deadline实际状态、误差、碰拍与落台记录。逐进程journal也保留。

`report.html`为独立交互报告，直接打开即可切换TTS和指标，悬停显示置信区间、有效n、
支持域比例；`tts_overview.png`为趋势图，`overview_*.png`为各档网格对照图，
`tts_grids/tts_*.png`每档一张全指标图。另有`tts_summary.csv`与逐格`tts_grids/summary.csv`。
`report.json`只嵌入汇总，逐样本证据保存在JSONL，避免HTML加载数万条轨迹。
中断后以相同`--output`加`--resume`继续：校验冻结suite，合并已落盘journal，跳过已完成trial。

先跑粗网格：TTS=0.3/0.4/0.5s，Y–Z 划分为3×2，每个有效格随机采样3个
位置/法向/速度 target。同一组 target 在三档 TTS 重复执行，合计54条。

```zsh
MUJOCO_GL=egl .venv/bin/python -m deploy.benchmarks.racket_command.run \
  --config deploy/benchmarks/racket_command/grid_tts.yaml \
  --tts-values 0.3 0.4 0.5 --samples-per-cell 3 \
  --output deploy/recordings/racket_command/tts_grid_run
```

`grid_tts.yaml` 中的 `grid`、`samples_per_cell` 可增加密度。这里每格采样数指
**每个 TTS** 的数量；各档重复同一批 target，不根据策略成功或拒绝重采样。
该配置启用历史物理球诊断模式，不调用来球估计或回球planner；这不是第二阶段接球测试。
物理来球无法构造时如实记录，目标仍执行。只测指令跟踪可加 `--command-only`。
`tts_values` 与原 `sweep` 互斥；CLI指定列表会移除默认sweep。支持域以外的TTS也不
自动裁剪，选择器拒绝仍在分母，具体接受范围见adapter配置。

输出 `tts_grids/tts_00_0.3s.png` 等，每档一张含多指标的二维网格图，颜色尺度跨TTS一致。
`tts_grids/summary.csv`、`summary.json` 和 `report.json` 中的 `tts_grids` 提供逐格数据：

- 计划/已测数量、选择器接受率、指定时刻联合达标率、含回位的完整成功率。
- 物理碰拍率、合法回球率，分母为该格该TTS的全部已测target。
- 指定TTS的拍心到target误差、法向与速度误差；物理模式另外提供拍心到球心距离。
- 实际碰拍时刻的拍心到球心距离，仅碰拍样本有值；它不是球面到拍面的间隙。
- 误差的有效样本数、均值、中位数、P95、标准差；未覆盖deadline的不计算该时刻误差。

拒绝仍计失败；未执行样本不进入成功率分母，缺数据的格显示灰色。
实际碰拍距离取物理步记录；deadline距离对控制周期的拍心/球心位置线性插值，
提前碰拍后球已反弹时，deadline距离反映当时的真实分离，不能解释为碰撞间隙。

从仓库根目录运行。固定一个精确点位，在相同位置测试3组方向×3档速度×3档TTS，共27条：

```zsh
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --point -0.8 0.2 --directions 3 --speed-levels 3 --tts-levels 3 --render \
  --output deploy/recordings/racket_command/fixed_point_demo
```

`--point`接收Y、Z，X由击球平面确定，本例目标始终为[-0.25,-0.8,0.2]m。
每个方向组合内，拍面法向与速度方向固定；完整交叉速度和TTS，而不是每次同时换位置。
3档速度在各方向对应的合法速度范围内选取；3档TTS为0.28/0.39/0.50s。
方向在该点的训练联合支持域内抽样，再经过上述正常对打条件筛选。
每个样本仍恢复同一准备状态。一个点的所有组合连续测完后才换下一个点；点内顺序可随机。

窗口显示实体机器人、球桌与 Y–Z 网格，金色圆盘/箭头表示目标拍面/法向，紫色箭头
表示目标速度，青色表示实际拍心与轨迹，红点和连线表示目标时刻的拍心与位置误差。
默认只显示目标/实际球拍与网格，不显示球。网格颜色表示指定TTS时刻的三项联合达标率。
阈值默认：位置≤5cm、拍面法向≤15°、速度向量误差≤0.5m/s。不能用更早或更晚的
最近位置替代指定时刻；最近位置时间误差只作诊断。完整成功率另外要求完成回位且无失败。

以下仅说明历史 `evaluation_mode: command_ball_diagnostic`：橙色乒乓球及尾迹来自真实积分状态，包含重力、二次阻力、有限桌面反弹、球网、地面和
实际运动拍面的扫掠碰撞，每 1ms 更新。接触后由实际拍面速度、法向与恢复系数产生
出球速度，不求解或修正出球落点，不强制碰拍。当前统一使用 A3 参数：球半径2cm、
拍面模型半径7.5cm、drag=0.1537、桌面水平/竖直系数=0.69/1.01、球拍恢复系数=0.88。
物理参数与G1部署planner一致；历史suite仍保留各次实验自己的参数。
`evaluation_mode: policy_tracking` 是当前默认；`--command-only` 可将历史suite切换为第一阶段评测。
`--hide-ball` 只影响历史物理诊断的显示，不等于禁用球物理。
左上角显示 trial、网格、TTS 倒计时和选择结果。
网格边界与采样使用同一份测试配置；无需也不接受 `--grid` 参数。
新测试使用配置中的实际划分（默认从 8×6 中裁掉最低 Z 行，实际为 8×5），回放 `--suite` 时
使用该次实验保存的范围和划分，不会套用当前默认值。三维场景中的网格不显示编号、
坐标刻度或轴名称；黄色边框和窗口中的 Y/Z 区间标明当前测试格，无有效采样区域的格子显示灰色。
Space/P 暂停，Q/Escape 退出；`--playback-rate 0.5` 慢速，2 为双速。
退出时保存已完成结果，并明确标记 partial/interrupted。

完整默认评测：原始 8×6 格由 `crop_bottom_z_rows: 1` 裁掉最低一行，保留其余格子的
尺寸与边界。当前 Z 范围为 [-0.16225, 0.71278]m（相对桌面），实际为 8×5 格。
每个有效网格只随机选一次固定点，然后测试5方向×2速度×2TTS。
当前39格具有训练位置支持，最多780条；正常对打条件可能减少可用点位/方向数量，
最终数量以 `sampling_summary` 为准。同一网格的目标位置完全一致。设为 0 可恢复原始完整范围。
裁剪只影响新生成的测试序列；历史 `--suite` 回放保持其原始区域。

```zsh
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --output deploy/recordings/racket_command/grid_seed42
```

每次使用新的输出目录，防止覆盖结果。配置位于 `default.yaml`；`--config` 可替换。
`--samples-per-cell N`可改为每个固定点N组随机方向/速度/TTS组合（不做完整交叉），
系统交叉测试建议使用`--directions`、`--speed-levels`、`--tts-levels`。
如需复现此前每条命令都随机位置的采样方式，显式使用`--position-sampling random`。
默认域来自 `deploy/hope_g1/configs/command_domain.json`：将各 motion 的参考击球点
加训练的每轴±0.25m扰动，再与固定 X=-0.25m 平面相交，取21个非空区域的并集。
当前外包框 Y=[-1.5123,0.0020]m、Z=[-0.3373,0.7128]m；只在并集内采样，
不把外包框空洞当成失败。重叠区域做概率校正，保证每格内位置在并集上均匀分布。
随后在支持该位置的区域内采样完整速度/法向组合：参考速度方向保持，3m/s按
[0.5,1.5]缩放，即1.5–4.5m/s；参考法向每分量加±0.5扰动后归一化，独立于速度。
TTS=[0.28,0.50]s，由strike frame、max_start_frame和现有启动窗口共同确定。
使用训练已达到的curriculum scale=1.0，来源与参数均写入域文件。
这是理论训练支持域，不等于每个目标都已被策略证明可达；不按实际成功或选择器结果过滤。
旧矩形压力测试仍可显式使用 `--unrestricted`，其结果不得解释为训练域内能力。

域导出与运行时采样分开；若改变home/world摆放或训练随机化参数，先重新导出：

```zsh
.venv/bin/python -m deploy.hope_g1.export_command_domain --scale 1.0
```

导出只用动作/训练参数和准备姿态的固定世界变换，不用测试成绩。纯采样器读取JSON快照，
不会把来源motion ID传给选择器；suite嵌入域数据，后续回放不需要重新读取动作来生成目标。

也可以先独立生成目标，完全不加载模型或动作库：

```zsh
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --generate-only --output deploy/recordings/racket_command/suite_seed42
.venv/bin/python -m deploy.benchmarks.racket_command.run \
  --suite deploy/recordings/racket_command/suite_seed42/suite.json \
  --render --output deploy/recordings/racket_command/replay_seed42
```

相同 suite 可用于不同 checkpoint、motion 集合或 agent adapter 对照：
`--task-config` 指定 checkpoint/资源配置，`--motion-allow 0,1,2` 限制选择器候选，
`--adapter package.module:Class` 替换完整 agent 实现。以上选项都不影响目标生成。
套件以 SHA256 固定全部指令、顺序与测试配置。

截图使用 `--capture-first 3`，会保存前三条实际执行样本在目标时刻、实际碰拍和首次落台时的画面。
无显示器时可以使用 `MUJOCO_GL=egl`；截图不改变评测结果，精确时刻指标仍由插值计算。

## 边界与执行条件

- `sampling.py` 只依赖配置、NumPy 和公共 command 数据结构；不导入 motion、selector、policy。
- `interfaces.CommandAgent` 是测试环境与被测 agent 的唯一操作接口。
- `HopeCommandAdapter` 位于 `deploy/hope_g1`，连接现有 `select_racket_command` 和
  `HopeActorController`。选择器仍按位置距离评分，保留原门槛；未为了测试优化选择规则。
- 公共 `deploy.racket_command.RacketCommand` 包含 position、normal、velocity、deadline。
  `HopeController.step(..., racket_command=...)` 直接接收指令；原真机/球预测入口不变。
- 只在每次批量测试开始时，通过原有悬挂下放、AMP、UP 流程准备一次，此阶段没有球。
  每条指令前恢复同一份完整 MuJoCo integration state 和独立 controller/policy 历史；
  不跨样本沿用上次挥拍状态，不在挥拍中重置、加外力或纠正关节动作。
- 每条目标只提交一次。被拒绝时不重新挑目标，不悄悄等待到启动窗口再提交。
- 当前 HOPE 选择器上限 TTS=0.5s、max_start_frame=29；修改采样范围可能增加拒绝率，
  拒绝原因如实报告。基座 XY 目标沿用原 command-to-goal 语义，不额外生成基座轨迹。
- 使用同一 MuJoCo 机器人、碰撞模型、原 policy PD 和 actor。评测使用理想 MuJoCo
  位姿/速度，不模拟传感器时延和噪声，不代表真机性能。
- 以下两项只用于历史物理球诊断，第一阶段不执行：球动力学与原部署共用核函数，接触模型是移动圆盘，忽略拍边、整机其它部位、旋转与Magnus效应；
  球碰撞不反向施加关节冲量（与原轻球模型一致）。这不是完整可变形球拍/球的接触模型。
  接触速度使用拍心平移速度，未包含偏心接触点的角速度修正；每球只响应首次拍面碰撞。
- 来球用独立随机数流生成，验证过网和己方一次反弹历史、有限桌面与预期接触点；
  不满足条件的command仍照常测试，另记物理来球不可构造，不重采样目标或算成成功。
  t=0可处于来球飞行中段，先前过网/反弹历史记录在feed中；这允许短TTS与合理球速共存。
  物理球随后只正向积分，不在TTS瞬移，也不在碰撞后修正轨迹。

## 指标与输出

目标时刻为指令下发后的 TTS。位置和速度向量在线性插值后的精确时刻计算，法向
插值后归一化。法向比较圆盘几何朝向，正反面等价（abs(dot)）；原 policy 的
backhand 法向符号约定保持不变。速度按带方向的完整向量比较。

第一阶段主指标 `target_pass`：选择器接受，且在指定TTS时刻位置误差≤5cm、
拍面误差≤15°、速度向量误差≤0.5m/s，三项同时满足。
另列 `success`：主指标满足，并完成回位且无失败。回位指原控制器完成 return 流程，不额外要求 measured 稳态门槛。
另记录±0.25s 内最接近目标的时刻偏差、是否≤40ms，作为诊断，不能代替 deadline 评分。
每条执行 4.5s；提前结束/失稳导致未覆盖 deadline 时，直接判失败，不用终止位置外推。

- `suite.json`：完整预生成目标、seed、配置、校验和。
- `results.jsonl`：逐样本目标、motion、接受/拒绝原因、误差与成功标志，边测边写。
- `traces/trial_*.json`：逐控制周期实际拍心位置/法向/速度、相位和站立诊断。
- `preparation/`：进入策略的原有完整日志，无发球。
- `adapter_metadata.json`：checkpoint、控制周期、PD、坐标转换、选择器门槛等。
- `report.json`：机器可读汇总；`capability.png`：六幅二维热力图。
- 第一阶段不生成 `physical_ball.png`，逐样本结果不包含碰球或回球评分。
- `difficulty.png`：TTS、速度、朝向难度分组曲线，以及 motion 选择分布。
- `point_summary.json`：每个精确固定点的计划/已测数量、指定时刻达标及完整成功数量。
- `points/point_*.png`：每个点按方向分列的速度–TTS矩阵，分别展示联合达标、位置/法向/速度误差；单点模式另保存`point_sweeps.png`。
- 第一阶段的 `ball_summary.enabled=false`，没有回球率分母，不将“没有球”记为失败。
- `selected_motion_exported_support`：所选motion是否支持该command的导出训练范围；
  `command_supported_motions`列出符合联合位置/法向/速度支持域的候选，仅作诊断，不反馈修改选择结果。
- `report.html`：无需网络的交互图，可按 TTS、速度、朝向难度、motion 过滤，
  网格显示样本数，鼠标查看指标；同时列出失败/拒绝原因。
  “固定点位：速度×TTS”区域可选择点位和方向，查看同一位置的组合测试结果。

综合成功率包含所有已执行目标，拒绝也在分母；同时列接受后的成功率。
未执行目标不计入已评测分母，报告明确列 scheduled/evaluated 和 complete/partial 状态。
误差热力图只汇总实际覆盖 deadline 的样本；以综合成功率判断覆盖能力时不能忽略灰格。
随机球拍指令不保证其速度/法向与来球组合本身能产生合法回球，所以物理回球率用于
观察整个场景结果，不能替代指令跟踪能力指标，也不能通过出球修正提高分数。
