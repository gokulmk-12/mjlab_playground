"""Curriculum functions for the getup task."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
  import torch
  from mjlab.envs import ManagerBasedRlEnv


class TerminationCurriculumStage(TypedDict):
  step: int
  params: dict[str, float]


def termination_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  termination_name: str,
  stages: list[TerminationCurriculumStage],
) -> dict[str, torch.Tensor]:
  """Update a termination term's params based on training steps.

  Each stage specifies a ``step`` threshold and a ``params`` dict.
  When ``env.common_step_counter`` reaches a stage's ``step``, the
  params are applied. Later stages take precedence.
  """
  del env_ids
  term_cfg = env.termination_manager.get_term_cfg(termination_name)
  for stage in stages:
    if env.common_step_counter >= stage["step"]:
      if "params" in stage:
        term_cfg.params.update(stage["params"])
  return term_cfg.params
