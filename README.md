# mjlab playground

A collection of tasks built with [mjlab](https://github.com/mujocolab/mjlab), starting with ports from [MuJoCo Playground](https://playground.mujoco.org/).

## Tasks

| Task ID | Robot | Description | Preview |
|---------|-------|-------------|---------|
| `Mjlab-Getup-Flat-Unitree-Go1` | Unitree Go1 | Fall recovery on flat terrain | <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/go1_getup_teaser.gif" width="200"/> |
| `Mjlab-Getup-Flat-Booster-T1` | Booster T1 | Fall recovery on flat terrain | <img src="https://raw.githubusercontent.com/mujocolab/mjlab_playground/assets/t1_getup_teaser.gif" width="200"/> |

## Getting Started

```bash
git clone https://github.com/mujocolab/mjlab_playground.git && cd mjlab_playground
uv sync
```

Train a task:

```bash
uv run train <task-id> --num_envs 4096
```

Play back a trained policy:

```bash
uv run play <task-id>
```

## Citation

If you use this repository in your research, consider citing mjlab:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

This repository is released under an [Apache-2.0 License](LICENSE).
