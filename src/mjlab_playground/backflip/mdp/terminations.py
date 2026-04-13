from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab_playground.backflip.mdp.stage_manager import (
    BackflipStageManager,
    STAGE_AIR
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

def trunk_contact_termination(env: ManagerBasedRlEnv, sensor_name: str, settle_steps: int = 10) -> torch.Tensor:
    """Terminate when trunk contacts the ground.
    
    Skips the first settle_steps to avoid terminating immediately on episode start if the robot spawns slightly clipped.
    """
    sensor = env.scene[sensor_name]
    trunk_contact = sensor.data.found[:, :, 0].any(dim=-1)
    past_settle = env.episode_length_buf > settle_steps
    return past_settle & trunk_contact

def landing_without_flip_termination(env: ManagerBasedRlEnv, feet_sensor_name: str) -> torch.Tensor:
    """Terminate if robot lands during air stage without completing half turn.
    Prevents the policy from learning to just fall and land without flipping.
    """
    stage_mgr: BackflipStageManager = env.extras["stage_manager"]
    sensor = env.scene[feet_sensor_name]
    foot_contacts = sensor.data.found[:, :, 0]
    any_foot = foot_contacts.any(dim=-1)

    in_air = stage_mgr.in_stage(STAGE_AIR)
    no_half_turn = ~stage_mgr.is_half_turn
    return in_air & any_foot & no_half_turn