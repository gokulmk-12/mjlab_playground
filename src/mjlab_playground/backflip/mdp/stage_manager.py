"""Stage manager for the backflip task.

Stages:
    0 - Stand:  upright, stable
    1 - Sit:    crouch / load rear legs
    2 - Jump:   front feet leave ground first
    3 - Air:    all feet off ground, rotating
    4 - Land:   any foot touches ground again

Transitions (all condition-based, per environment):
    Stand -> Sit:   always at episode start after settle_steps
    Sit   -> Jump:  base height < sit_to_jump_height
    Jump  -> Air:   all four feet off ground
    Air   -> Land:  any foot contacts ground
    Land  -> done:  episode ends (no further transition)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

STAGE_STAND = 0
STAGE_SIT   = 1
STAGE_JUMP  = 2
STAGE_AIR   = 3
STAGE_LAND  = 4
NUM_STAGES  = 5

class BackflipStageManager:
    def __init__(self, env: ManagerBasedRlEnv, settle_steps: int = 10, sit_to_jump_height: float = 0.22, max_stand_time: float = 5.0):
        n, d = env.num_envs, env.device

        self.settle_steps       = settle_steps
        self.sit_to_jump_height = sit_to_jump_height

        self.stage: torch.Tensor = torch.zeros(n, dtype=torch.long, device=d)

        self.is_half_turn: torch.Tensor = torch.zeros(n, dtype=torch.bool, device=d)
        self.is_one_turn: torch.Tensor  = torch.zeros(n, dtype=torch.bool, device=d)

        # start_time: random delay before Stand->Sit transition sampled at reset, uniform [0, max_stand_time]
        self.start_time: torch.Tensor = torch.zeros(n, dtype=torch.float32, device=d)
        self.max_stand_time: float = max_stand_time

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset stage and pitch tracking for the given env ids."""
        self.stage[env_ids]         = STAGE_STAND
        self.is_half_turn[env_ids]  = False
        self.is_one_turn[env_ids]   = False

        self.start_time[env_ids]    = torch.rand(len(env_ids), device=self.start_time.device) * self.max_stand_time

    def _update_turn_flags(self, body_z: torch.Tensor) -> None:
        """body_z = projected_gravity_b = quat_rotate_inverse(quat, world_z). Shape (N, 3)."""
        # Half turn: robot is upside down — body z points negative x and negative z
        new_half = self.is_half_turn | (
            (body_z[:, 0] < 0) & (body_z[:, 2] < 0)
        )
        # Full turn: had half turn AND back to upright orientation
        new_one = self.is_one_turn | (
            new_half & (body_z[:, 0] >= 0) & (body_z[:, 2] >= 0)
        )
        self.is_half_turn = new_half
        self.is_one_turn = new_one
    
    def in_stage(self, stage_id: int) -> torch.Tensor:
        """Boolean mask of envs currently in the given stage. Shape (N,)."""
        return self.stage == stage_id
    
    @property
    def stage_one_hot(self) -> torch.Tensor:
        """One-hot encoding of current stage. Shape (N, NUM_STAGES)."""
        return torch.nn.functional.one_hot(self.stage, num_classes=NUM_STAGES).float()
        
    def _advance_stages(self, env: ManagerBasedRlEnv, foot_contacts: torch.Tensor, base_height: torch.Tensor, body_z: torch.Tensor, elapsed_time: torch.Tensor) -> None:
        """Apply stage transition conditions, in order."""
        any_foot = foot_contacts.any(dim=-1)
        all_feet_off = ~any_foot

        # Stand -> Sit: past random start time AND hasn't started turning yet
        stand_to_sit = (self.stage == STAGE_STAND) & (elapsed_time > self.start_time) & ~self.is_half_turn

        # Sit -> Jump: base drops below threshold AND majority of feet on ground
        foot_mean = foot_contacts.float().mean(dim=-1)
        sit_to_jump = (self.stage == STAGE_SIT) & (base_height < self.sit_to_jump_height) & (foot_mean >= 0.0)

        # Jump -> Air: all feet leave ground
        jump_to_air = (self.stage == STAGE_JUMP) & all_feet_off

        # Air -> Land: any foot contact AND has done half turn
        air_to_land = (self.stage == STAGE_AIR) & any_foot & self.is_half_turn
        
        # Apply in reverse order N->0
        self.stage = torch.where(air_to_land, torch.full_like(self.stage, STAGE_LAND), self.stage)
        self.stage = torch.where(jump_to_air, torch.full_like(self.stage, STAGE_AIR), self.stage)
        self.stage = torch.where(sit_to_jump, torch.full_like(self.stage, STAGE_JUMP), self.stage)
        self.stage = torch.where(stand_to_sit, torch.full_like(self.stage, STAGE_SIT), self.stage)

    @property
    def has_completed_turn(self) -> torch.Tensor:
        """True if full 360° rotation detected. Shape (N,) bool."""
        return self.is_one_turn
    
    def update(self, env: ManagerBasedRlEnv, foot_contacts: torch.Tensor, base_height: torch.Tensor, body_z: torch.Tensor, elapsed_time: torch.Tensor) -> None:
        self._update_turn_flags(body_z)
        self._advance_stages(env, foot_contacts, base_height, body_z, elapsed_time)
