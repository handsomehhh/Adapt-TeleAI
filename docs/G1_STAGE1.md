# 独立 G1 Stage 1 变速跟踪

任务 ID：`Mjlab-G1-PingPong-Stage1`。实现位于 `src/mjlab/tasks/g1_stage1/`，
使用独立 29 自由度 G1 右手球拍资产。原网球任务保留在 `adapt_tennis/`，两者不互相导入。

当前训练资产已更新为 `g1_29dof_pingpong_fast_extended_27mm.urdf` 的 MuJoCo 转换版：
31 个碰撞体、单盒脚底、更新后的拍柄/惯量，拍面中心相对右腕为 `[0.242, 0.003, 0]` m。
资产源文件、网格和校验记录均保存在独立 `unitree_g1_pingpong/` 内。

## 数据与训练目标

输入为离线 G1 全身动作 NPZ。默认读取 `dataset/g1_stage1/motions/`，
也可以通过 `--env.commands.motion.motion-directory` 指定其他动作目录。
支持带关节/body 名称的 NPZ，以及显式配置旧 BFS schema 的无名称格式。
默认数据为复制的 43 段 Tele-GMR 动作，来源和逐文件哈希记录于
`dataset/g1_stage1/manifest.json`，不依赖原工作目录的软链接。

物理步长 0.005 秒，4 个子步组成 0.02 秒控制周期。每步参考时间随机推进
0.01–0.04 秒，策略观察对应时间增量；参考关节用线性插值、四元数用 SLERP。
PPO 奖励和折扣按固定物理时间计算。保留已有 Stage 1 的身体/腕/挥拍跟踪、
关节限位、全身动作变化及额外腰部动作变化惩罚，不包含接球奖励。

## 从头训练

在当前仓库根目录安装环境后运行：

```bash
uv sync
uv run train Mjlab-G1-PingPong-Stage1 \
  --gpu-ids '[0]' \
  --env.scene.num-envs 4096 \
  --agent.resume False \
  --agent.max-iterations 30000 \
  --agent.save-interval 2000 \
  --agent.run-name g1_stage1_fresh
```

小规模检查可将环境数改成 16、迭代数改成 2、保存间隔改成 1。
默认 `resume=False`，不会加载原 Stage 1 或网球 checkpoint。
数据 NPZ 不随 Git 同步；在其他机器需另外复制数据目录或显式指定数据路径。

## 输出与播放

每次运行保存配置、TensorBoard 日志和 `model_N.pt`，并自动导出
`jit/modeljit_N.pt`。JIT 包含输入归一化，输入为 132 维 actor 观测，
输出为 29 维关节动作。checkpoint 还保存动作采样失败统计。

```bash
uv run play Mjlab-G1-PingPong-Stage1 \
  --checkpoint-file logs/rsl_rl/g1_stage1/<运行目录>/model_N.pt \
  --motion-directory dataset/g1_stage1/motions \
  --num-envs 1
```

播放关闭训练噪声，并从动作首帧以固定速度执行。
小规模检查只验证加载、仿真、PPO 更新、保存和导出，不等于长期收敛或真机验收。
MVAE 与上层 planner 属于后续独立工作，本阶段无需等待它们。
