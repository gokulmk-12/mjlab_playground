# mjlab playground

A collection of tasks built with [mjlab](https://github.com/mujocolab/mjlab), starting with ports from [MuJoCo Playground](https://playground.mujoco.org/).

## Tasks

| Task ID | Robot | Description | Preview |
|---------|-------|-------------|---------|
| `Mjlab-Getup-Flat-Unitree-Go1` | Unitree Go1 | Fall recovery on flat terrain | ![Go1 getup](https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/go1_getup_teaser.gif) |
| `Mjlab-Getup-Flat-Booster-T1` | Booster T1 | Fall recovery on flat terrain | ![T1 getup](https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/t1_getup_teaser.gif) |

## Setup

```bash
uv sync
```

## Training

```bash
uv run train --task Mjlab-Getup-Flat-Unitree-Go1 --num_envs 4096
```
