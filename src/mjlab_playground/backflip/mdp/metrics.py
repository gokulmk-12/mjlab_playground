from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab_playground.backflip.mdp.stage_manager import (
    BackflipStageManager, STAGE_LAND, STAGE_AIR
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

class flip_success:
    """Binary success — 1 if robot completed full flip and reached land stage."""
    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        self._succeeded = torch.zeros(env.num_envs, device=env.device)
    
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._succeeded[:] = 0.0
        else:
            self._succeeded[env_ids] = 0.0
    
    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        stage_mgr: BackflipStageManager = env.extras["stage_manager"]

        in_land = stage_mgr.in_stage(STAGE_LAND)
        full_turn = stage_mgr.is_one_turn
        succeeded = (in_land & full_turn).float()
        self._succeeded = torch.maximum(self._succeeded, succeeded)
        return self._succeeded

class max_stage_reached:
    """Track the maximum stage index reached per episode."""
    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        self._max_stage = torch.zeros(env.num_envs, device=env.device)
    
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._max_stage[:] = 0.0
        else:
            self._max_stage[env_ids] = 0.0
    
    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        stage_mgr: BackflipStageManager = env.extras["stage_manager"]
        self._max_stage = torch.maximum(
            self._max_stage, stage_mgr.stage.float()
        )
        return self._max_stage

def air_stage_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Fraction of envs currently in or past air stage."""
    stage_mgr: BackflipStageManager = env.extras["stage_manager"]
    return (stage_mgr.stage >= STAGE_AIR).float()

def land_stage_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Fraction of envs currently in land stage."""
    stage_mgr: BackflipStageManager = env.extras["stage_manager"]
    return stage_mgr.in_stage(STAGE_LAND).float()