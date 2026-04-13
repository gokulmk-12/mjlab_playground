from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab_playground.backflip.mdp.stage_manager import BackflipStageManager

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

def init_stage_manager(env: ManagerBasedRlEnv, feet_sensor_name: str, settle_steps: int = 10, sit_to_jump_height: float = 0.22) -> None:
    """Instantiate the stage manager and store it in env.extras."""
    stage_mgr = BackflipStageManager(
        env=env, settle_steps=settle_steps, sit_to_jump_height=sit_to_jump_height
    )
    env.extras["stage_manager"] = stage_mgr
    env.extras["feet_sensor_name"] = feet_sensor_name

def reset_stage_manager(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Reset stage manager state for the given envs."""
    if "stage_manager" not in env.extras:
        return
    
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env.extras["stage_manager"].reset(env_ids)

def update_stage_manager(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, entity_name: str) -> None:
    """Update stage transitions and integrate pitch. Called every step."""
    if "stage_manager" not in env.extras:
        return
    
    stage_mgr: BackflipStageManager = env.extras["stage_manager"]
    entity: Entity = env.scene[entity_name]
    feet_sensor_name: str = env.extras["feet_sensor_name"]

    base_height = entity.data.root_link_pos_w[:, 2]
    body_z      = entity.data.projected_gravity_b

    elapsed_time = env.episode_length_buf * env.step_dt

    sensor = env.scene[feet_sensor_name]
    foot_contacts = sensor.data.found[:, :, 0]

    stage_mgr.update(
        env=env,
        foot_contacts=foot_contacts,
        base_height=base_height,
        body_z=body_z,
        elapsed_time=elapsed_time,
    )