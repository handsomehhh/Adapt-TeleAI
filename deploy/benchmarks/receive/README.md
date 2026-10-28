# 第二阶段：独立随机来球、planner与策略执行

网页可视化已接入：原命令增加 `--render`，浏览器打开终端打印的地址。默认 `--viewer web`，
`--viewer native` 保留桌面窗口。详见 [网页可视化](../../WEB_VISUALIZATION.md)。

入口 `deploy.benchmarks.receive.run`。与第一阶段 `racket_command.run` 分开：

```text
随机球初始位置/速度 → 1ms球动力学
                    ↓ 180Hz位置样本
              实际二次拟合估计器
                    ↓
              原A3 Stage 2/3 planner
                    ↓ racket command
              现有motion选择器 / G1 actor
                    ↓
              实际MuJoCo球拍碰撞 / 回球
```

没有从motion、racket command或策略成绩反推来球。初始p/v均匀随机，采样前只筛选：
合法越网、己方半台一次反弹、进入指定接球区域。进入区域的真值交点/时间仅用于分格与
离线误差评估，不进入控制器。planner只获得估计器拟合的球状态，不能读取物理球真值速度。

## 默认中心区域

`center.yaml`：X=-0.25m，Y=[-1.000,-0.525]m，Z=[0.12,0.42]m，桌面Z=0。
3列×2行、每格5球，共30球。初始球位于对面半台：X=[1.7,2.5]m、
Y=[-1.05,-0.475]m、Z=[0.30,0.85]m；Vx=[-4.5,-2.8]m/s、
Vy=[-0.35,0.35]m/s、Vz=[-1.0,0.5]m/s。均在筛选前独立采样。

每条球轨迹在推理前冻结，球/桌面/球拍参数沿用与A3统一的rally配置。球仅在trial初始
赋值，之后自由积分并与实际拍面碰撞，不瞬移、不追随参考动作、不修正出球速度/落点。
实际球拍碰撞采用现有有限移动圆盘核函数；不模拟旋转、Magnus、拍边与整机其它部位碰球。

在现有START/AMP/UP流程准备后，每球恢复同一份机器人状态及controller/actor历史，
测试一球的规划、执行及回位；第一球命令接受后不再触发第二次挥拍，避免追打地面反弹。
默认无观测噪声/延迟，位置以180Hz送入现有估计器；不是模拟真实现场所有传感器误差。

## 启动

使用A3实际配置中的原回球目标[2.5,0,0]：

```bash
.venv/bin/python -m deploy.benchmarks.receive.run \
  --render \
  --output "deploy/recordings/receive/center_$(date +%Y%m%d_%H%M%S)"
```

标准对打的对方半台中心回球目标：

```bash
.venv/bin/python -m deploy.benchmarks.receive.run \
  --return-target 2.055 -0.7625 \
  --render \
  --output "deploy/recordings/receive/center_return_$(date +%Y%m%d_%H%M%S)"
```

`--return-target`是显式任务落点输入，不修改A3算法、物理参数、选择器或policy。
不提供时保留原A3配置；实际目标记录于metadata和每条command的diagnostic。
原A3目标Y=0在台边，当前球模型按球心判台内，毫米级边界误差也可能被判出界；
两组结果要分开报告，不把中心落点对照冒充原配置。

回放本次相同30球，检查对方半台中心回球：

```bash
.venv/bin/python -m deploy.benchmarks.receive.run \
  --suite deploy/recordings/receive/center_suite_20261021/suite.json \
  --return-target 2.055 -0.7625 \
  --render \
  --output "deploy/recordings/receive/center_replay_$(date +%Y%m%d_%H%M%S)"
```

窗口显示实际物理球（橙色）、实际拍心轨迹（青色）、锁定的planner目标（金色）以及
中心接球网格；命令锁定前不显示虚构目标。Space/P暂停、Q/Escape退出。
`--playback-rate 2`加快显示，`--limit N`只运行前N球、报告标记partial。
`--capture-first N`保存前N球发球、锁定command、deadline、碰拍与首次落点截图。

其他配置参数：`--config`、`--seed`、`--balls-per-cell`、`--task-config`、
`--rally-config`、`--runtime-config`、`--mocap-config`。`--generate-only`仅生成球序列，
不加载策略/机器人；该阶段使用批量球物理，因此允许加载Torch。

## 指标

- 接受率：planner生成并经选择器接受的command。
- 碰拍率：实际移动球拍与物理球的碰撞记录。
- 有效回球：碰拍后越网、没有触网、第一次桌面高度交点位于对方台内；落地后再弹回不算。
- 规划位置/时间误差：锁定command与无球拍干预的来球真值穿平面位置/时刻对比。
- command跟踪：指定deadline时刻的实际位置、法向、速度误差，独立于碰球/回球分数。
  球可以在deadline前后碰拍回球有效，但仍不满足严格的command跟踪阈值。
- 无失稳与回位统计单独记录。全部已测球在分母内，不因失败换样本。

`suite.json`保存球序列、物理与区域配置及SHA256；`results.jsonl`边测边写，
`traces/`保存位置观测、planner候选/拒绝原因及实际球拍轨迹；`report.json`机器可读，
`report.html`展示逐球结果/截图，`capability.png`按真实来球位置展示热力图。
`metadata.json`记录真实落点目标、估计器与策略运行参数，不连接真机。

可选离线诊断：

```bash
.venv/bin/python -m deploy.benchmarks.receive.audit \
  --recording deploy/recordings/receive/center_e2e_20261021
```

该诊断用理想球拍按已锁定command的局部直线位置/速度/法向执行，输出另一个
`ideal_paddle_audit.json`，用于检查command的物理效果；绝不反馈给实际测试，也不改写真实成绩。
