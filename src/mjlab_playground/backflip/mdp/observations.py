from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from mjlab.sensor import ContactSensor
from mjlab_playground.backflip.mdp.stage_manager import BackflipStageManager

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.found is not None
    return (sensor_data.found > 0).float()

def backflip_stage_one_hot(env: ManagerBasedRlEnv) -> torch.Tensor:
    """One-hot encoded current stage. Shape (N, 5).

    Reads from env.extras['stage_manager'] — must be updated before
    observations are computed each step.
    """
    return env.extras["stage_manager"].stage_one_hot

def backflip_is_half_turn(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Whether robot has passed the halfway point of the flip. Shape (N, 1)."""
    return env.extras["stage_manager"].is_half_turn.float().unsqueeze(-1)

def backflip_is_one_turn(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Whether robot has completed full rotation. Shape (N, 1)."""
    return env.extras["stage_manager"].is_one_turn.float().unsqueeze(-1)

def backflip_turn_flags(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Half-turn and full-turn flags concatenated. Shape (N, 2)."""
    stage_mgr: BackflipStageManager = env.extras["stage_manager"]
    half = stage_mgr.is_half_turn.float().unsqueeze(-1)  # (N, 1)
    full = stage_mgr.is_one_turn.float().unsqueeze(-1) 
    return torch.cat([half, full], dim=-1)

def base_height(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Base z height in world frame. Shape (N, 1)."""
    entity: Entity = env.scene[entity_name]
    return entity.data.root_link_pos_w[:, 2: 3]
