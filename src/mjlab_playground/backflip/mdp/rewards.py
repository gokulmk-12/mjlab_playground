from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab_playground.backflip.mdp.stage_manager import (
    STAGE_AIR, STAGE_JUMP, STAGE_LAND, STAGE_SIT, STAGE_STAND, BackflipStageManager
)
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor.contact_sensor import ContactSensor

def base_height_reward(env: ManagerBasedRlEnv, stand_height: float = 0.35, sit_height: float = 0.20, jump_height_cap: float = 0.5, entity_name: str = "robot") -> torch.Tensor:
    """Stage-wise base height reward.

    Stand/Land : -|pz - stand_height|   penalize deviation from standing height
    Sit        : -|pz - sit_height|     penalize deviation from crouch height
    Jump/Air   : 1(pz <= cap) * pz      reward upward motion up to cap
    """
    stage_mgr: BackflipStageManager = env.extras["stage_manager"]
    entity: Entity = env.scene[entity_name]
    pz = entity.data.root_link_pos_w[:, 2]

    stand_land  = stage_mgr.in_stage(STAGE_STAND) | stage_mgr.in_stage(STAGE_LAND)
    sit         = stage_mgr.in_stage(STAGE_SIT)
    jump_air    = stage_mgr.in_stage(STAGE_JUMP) | stage_mgr.in_stage(STAGE_AIR)

    reward      = torch.zeros(env.num_envs, device=env.device)
    reward      = torch.where(stand_land, -torch.abs(pz - stand_height), reward)
    reward      = torch.where(sit, -torch.abs(pz - sit_height), reward)
    reward      = torch.where(jump_air, (pz <= jump_height_cap).float() * pz, reward)

    return reward

def base_velocity_reward(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Stage-wise base velocity reward.

    Stand/Sit/Land : -(vx² + vy² + ωz²)   penalize planar motion and yaw
    Jump/Air       : -1_turn * ωy           penalize pitch rate once turn is done
                                            (encourages stopping spin after 360°)
    """
    stage_mgr: BackflipStageManager = env.extras["stage_manager"]
    entity: Entity = env.scene[entity_name]

    lin_vel_b = entity.data.root_link_lin_vel_b
    ang_vel_b = entity.data.root_link_ang_vel_b

    vx, vy = lin_vel_b[:, 0], lin_vel_b[:, 1]
    wz, wy = ang_vel_b[:, 2], ang_vel_b[:, 1]

    one_turn = stage_mgr.has_completed_turn.float()

    stationary_penalty = -(vx ** 2 + vy ** 2 + wz ** 2)
    rotation_penalty   = -one_turn * wy

    stand_sit_land = (
        stage_mgr.in_stage(STAGE_STAND) | stage_mgr.in_stage(STAGE_SIT) | stage_mgr.in_stage(STAGE_LAND)
    )
    jump_air = stage_mgr.in_stage(STAGE_JUMP) | stage_mgr.in_stage(STAGE_AIR)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward = torch.where(stand_sit_land, stationary_penalty, reward)
    reward = torch.where(jump_air, rotation_penalty, reward)

    return reward

def base_balance_reward(env: ManagerBasedRlEnv, entity_name: str = "payload") -> torch.Tensor:
    """Stage-wise base balance reward.

    Stand/Sit/Land : -angle(zB, zW)           penalize tilting from upright
    Jump/Air       : -|angle(yB, zW) - π/2|   penalize deviation from
                                               y-axis perpendicular to world z
                                               (keeps flip in sagittal plane)
    """
    stage_mgr: BackflipStageManager = env.extras["stage_manager"]
    entity: Entity = env.scene[entity_name]

    body_z = entity.data.projected_gravity_b
    upright_penalty = -torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0))

    sagittal_penalty = -torch.abs(
        torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - torch.pi / 2.0
    )

    stand_sit_land = (
        stage_mgr.in_stage(STAGE_STAND) | stage_mgr.in_stage(STAGE_SIT) | stage_mgr.in_stage(STAGE_LAND)
    )
    jump_air = stage_mgr.in_stage(STAGE_JUMP) | stage_mgr.in_stage(STAGE_AIR)

    reward = torch.zeros(env.num_envs, device=env.device)
    reward = torch.where(stand_sit_land, upright_penalty, reward)
    reward = torch.where(jump_air, sagittal_penalty, reward)

    return reward

def style_penalty(env: ManagerBasedRlEnv, entity_name: str = "robot") -> torch.Tensor:
    """Penalize deviation from default joint positions across all stages.
    -Σ (qj - qj_default)²
    """
    entity: Entity = env.scene[entity_name]
    joint_pos = entity.data.joint_pos
    default_joint_pos = entity.data.default_joint_pos
    return -torch.sum((joint_pos - default_joint_pos) ** 2, dim=-1)

def foot_contact_sequence_penalty(env: ManagerBasedRlEnv, feet_sensor_name: str) -> torch.Tensor:
    """Penalize incorrect takeoff sequence — only active in Jump stage.

    Cost fires if rear feet leave the ground before front feet.
    1(|I_foot_rear| == 0) in jump stage only.
    """
    stage_mgr: BackflipStageManager = env.extras["stage_manager"]
    in_jump = stage_mgr.in_stage(STAGE_JUMP)

    sensor: ContactSensor = env.scene[feet_sensor_name]
    assert sensor.data.found is not None
    foot_contacts = sensor.data.found

    rear_contact = torch.logical_or(foot_contacts[:, 2], foot_contacts[:, 3]) # RL | RR
    rear_left_ground = ~ rear_contact

    # Cost: rear feet off ground during jump = bad sequence
    penalty = in_jump.float() * rear_left_ground.float()
    return -penalty

def body_contact_penalty(env: ManagerBasedRlEnv, body_sensor_name: str) -> torch.Tensor:
    """Penalize any non-foot body link contacting the ground. All stages.
    1(|I_body| > 0)
    """
    sensor = env.scene[body_sensor_name]
    body_contact = sensor.data.found[:, :, 0].any(dim=-1)
    return -body_contact.float()

