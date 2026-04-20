from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab_playground.backflip.mdp.stage_manager import (
    BackflipStageManager,
    STAGE_AIR
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor.contact_sensor import ContactSensor

def trunk_contact_termination(env: ManagerBasedRlEnv, sensor_name: str, settle_steps: int = 10, force_threshold: float = 10.0) -> torch.Tensor:
    """Terminate when trunk contacts the ground.
    
    Skips the first settle_steps to avoid terminating immediately on episode start if the robot spawns slightly clipped.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    past_settle = env.episode_length_buf > settle_steps

    if sensor.data.force_history is not None:
        force_mag = torch.norm(sensor.data.force_history, dim=-1) # [B, N, H]
        return (force_mag > force_threshold).any(dim=-1).any(dim=-1) & past_settle # [B]
    
    assert sensor.data.found is not None
    trunk_contact = torch.any(sensor.data.found, dim=-1)
    return past_settle & trunk_contact

def landing_without_flip_termination(env: ManagerBasedRlEnv, feet_sensor_name: str) -> torch.Tensor:
    """Terminate if robot lands during air stage without completing half turn.
    Prevents the policy from learning to just fall and land without flipping.
    """
    stage_mgr: BackflipStageManager = env.extras["stage_manager"]
    sensor: ContactSensor = env.scene[feet_sensor_name]
    
    assert sensor.data.found is not None
    any_foot = torch.any(sensor.data.found, dim=-1)

    in_air = stage_mgr.in_stage(STAGE_AIR)
    no_half_turn = ~stage_mgr.is_half_turn
    return in_air & any_foot & no_half_turn