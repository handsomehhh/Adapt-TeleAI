# AdaPT：原始网球与 G1 乒乓球迁移

本分支保留原始 AdaPT 网球，并在独立目录实现 G1 乒乓球的变速 tracker、
MVAE、接球/站立 planner、评估和推理导出流程。原网球任务不依赖这些新增模块。

| 内容 | 实现位置 | 数据与模型 |
| --- | --- | --- |
| 官方原始网球 | `src/mjlab/tasks/adapt_tennis/` | `dataset/player1/`、`dataset/player2/`、`ckpts/player1/` |
| 独立 G1 Stage 1 | `src/mjlab/tasks/g1_stage1/` | `dataset/g1_stage1/motions/`；训练输出 `logs/rsl_rl/g1_stage1/` |
| G1 AdaPT 接球框架 | `src/mjlab/tasks/g1_adapt/` | 物理修正数据、MVAE、冻结 tracker、接球/站立 planner |

两者使用独立的动作命令、奖励、配置和 runner，不互相导入任务实现。
原网球任务目录与官方提交 `615b42f8edf8d78b1c8a9601673c70ea67e45ee9`
逐文件一致。共享 mjlab 仿真框架保留关节限位张量和 Warp 驱动查询两处兼容修正；
PPO、原网球资产、示例数据和预训练模型恢复为该官方基线。

G1 Stage 1 从头训练，不读取旧 checkpoint，不依赖球、MVAE、接球 Stage 2 或部署。
详见 [G1 Stage 1 使用说明](docs/G1_STAGE1.md) 和
[G1 AdaPT 完整训练流程、论文对照及已知差异](docs/G1_ADAPT.md)。
新增框架已做短训练和接口验证，尚未训练出能打球的完整模型，也不包含硬件控制程序。
原 A3、旧 G1 接球和 HOPE 部署扩展不作为新框架的依赖。

以下为保留的官方原始 README：

---

# AdaPT: Adaptive Motion Planning and Tracking
<!-- Badges -->
[![](https://img.shields.io/badge/Website-%F0%9F%9A%80-yellow)](https://humanoidtennis.github.io/AdaPT)
[![](https://img.shields.io/badge/Youtube-🎬-red)](https://www.youtube.com/watch?v=mEF-YTn-ksU)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-purple.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-2608.20087-brown)](https://arxiv.org/abs/2608.20087)
<!-- [![](https://img.shields.io/badge/Bilibili-📹-blue)](https://www.bilibili.com/video/XXXX) -->
This is the official PyTorch implementation of the paper "[**Towards Professional Tennis Styles for Humanoid Robots with Adaptive Motion Planning and Tracking**](https://arxiv.org/abs/2608.20087)" by

[Tao Huang](https://taohuang13.github.io/), [Ruofei Liu](https://github.com/lrf23), [Xuchen Tang](https://github.com/tangxch), [Xinyin Zhang](https://github.com/audu-123), [Junli Ren](https://renjunli99.github.io/), [Huayi Wang](https://why618188.github.io/), [Feiyu Jia](https://trap-1.github.io/), [Yukai Qi](https://www.linkedin.com/in/yukaiqi), [Kangning Yin](https://yinkangning0124.github.io/), [Weishuai Zeng](https://zengweishuai.github.io/), [Lipeng Chen](https://lipeng-chen.github.io/), Xi Li, Ting Wu, [Kailin Li](https://kailinli.top/), [Ruoli Dai](https://cn.linkedin.com/in/tristan-ruoli-dai-b2369330), [Jingbo Wang](https://wangjingbo1219.github.io/), [Lei Han](https://leihan.org/), [Jiangmiao Pang](https://oceanpang.github.io/)

<p align="left">
  <img width="98%" src="docs/images/teaser.jpg" alt="AdaPT teaser" style="box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.3); border-radius: 4px;">
</p>

## 📑 Table of Contents
- [🔥 News](#-news)
- [🛠️ Installation Instructions](#-installation-instructions)
- [🤖 Run AdaPT on Unitree G1](#-run-adapt-on-unitree-g1)
- [✉️ Contact](#-contact)
- [🏷️ License](#-license)
- [🎉 Acknowledgments](#-acknowledgments)
- [📝 Citation](#-citation)

## 🔥 News
- \[2026-08\] We release the training code for the first stage of our adaptive serve tracking.
- \[2026-08\] We release the [paper](https://arxiv.org/abs/2608.20087) and [demos](https://humanoidtennis.github.io/AdaPT/) of AdaPT.

## 🛠️ Installation Instructions
Clone this repository:
```bash
git clone https://github.com/noitom-robotics/AdaPT
cd AdaPT
```

Create the virtual environment and install dependencies with [uv](https://docs.astral.sh/uv/):

If `uv` is not installed yet, run one of the following:

```bash
# Recommended (Linux / macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

Then create the project environment and install all dependencies:

```bash
uv sync
```

This creates a `.venv` directory in the project root and resolves packages from `uv.lock`. On Linux, PyTorch is installed with CUDA 12.x support by default.

### Error Catching
Regarding potential installation errors, please refer to [this document](docs/ERROR.md) for solutions.

## 🤖 Run AdaPT on Unitree G1
### Policy Training
```bash
uv run train Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt \
  --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file dataset/player1/p1_serve.npz \
  --agent.run_name adapt_stage1_p1 \
  --agent.save-interval 2000 \
  --agent.max-iterations 28000
```

After training, you may play the resulted checkpoints, and we also provide the pretrained checkpoint:
```bash
uv run play Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt \
  --checkpoint-file ckpts/player1/model_24000.pt \
  --motion-file dataset/player1/p1_serve.npz \
  --racket-hand left
```

### Train & Play Command Arguments

- **`--racket-hand`**: Use `left` for `player1` (left-handed athlete) and `right` for `player2` (right-handed athlete). This selects the racket MJCF and hit-arm reward terms.
- **Hit-arm keyframes**: Tune `HIT_ARM_KEYFRAME_TIMES_S` in [`stage1_tracking_env_cfg.py`](src/mjlab/tasks/adapt_tennis/stage1_tracking_env_cfg.py) to match your motion clip (e.g., `3.4` s for `player1`, `1.84` s for `player2`).

## ✉️ Contact
For any questions, please feel free to email liurf23@mail2.sysu.edu.cn or taou.cs13@gmail.com. We will respond to it as soon as possible.

## 🏷️ License
This repository is released under the Apache-2.0 license. See [LICENSE](LICENSE) for additional details.

## 🎉 Acknowledgments
This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

* [mjlab](https://github.com/mujocolab/mjlab): The foundation for training and visualizing codes. 
* [AdaMimic](https://github.com/InternRobotics/AdaMimic): Motion speed adaptive and keyframe tracking algorithm implementation.

## 📝 Citation

If you find our work useful, please consider citing:

```
@article{huang2026adapt,
  title={Towards Professional Tennis Styles for Humanoid Robots with Adaptive Motion Planning and Tracking},
  author={Tao Huang, Ruofei Liu, Xuchen Tang, Xinyin Zhang, Junli Ren, Huayi Wang, Feiyu Jia, Yukai Qi, Kangning Yin, Weishuai Zeng, Lipeng Chen, Xi Li, Ting Wu, Kailin Li, Ruoli Dai, Jingbo Wang, Lei Han, Jiangmiao Pang},
  journal={arXiv preprint arXiv:2608.20087},
  year={2026},
  url={https://arxiv.org/abs/2608.20087}
}
```
