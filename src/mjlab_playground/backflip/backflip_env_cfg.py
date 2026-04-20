import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import mjlab_playground.backflip.mdp as mdp

def make_backflip_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create backflip task configuration."""

    ## ------ Observations ------------
    actor_terms: dict[str, ObservationTermCfg] = {
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "prev_action": ObservationTermCfg(
            func=mdp.last_action
        ),
        "stage": ObservationTermCfg(
            func=mdp.backflip_stage_one_hot,
        ),
        "turn_flags": ObservationTermCfg(
            func=mdp.backflip_turn_flags,
        )
    }

    critic_terms: dict[str, ObservationTermCfg] = {
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
        ),
        "actions": ObservationTermCfg(
            func=mdp.last_action,
        ),
        "stage": ObservationTermCfg(
            func=mdp.backflip_stage_one_hot,
        ),
        "turn_flags": ObservationTermCfg(
            func=mdp.backflip_turn_flags,
        ),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
        ),
        "base_height": ObservationTermCfg(
            func=mdp.base_height,
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    observations: dict[str, ObservationGroupCfg] = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=10,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        )
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.25,
            use_default_offset=True
        )
    }

    events: dict[str, EventTermCfg] = {
        "init_stage_manager": EventTermCfg(
            func=mdp.init_stage_manager,
            mode="reset",
            params={
                "feet_sensor_name": "feet_ground_contact",
                "settle_steps": 10,
                "sit_to_jump_height": 0.22,
            }
        ),
        "reset_stage_manager": EventTermCfg(
            func=mdp.reset_stage_manager,
            mode="reset"
        ),
        "update_stage_manager": EventTermCfg(
            func=mdp.update_stage_manager,
            mode="step"
        ),
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.02),    
                "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    rewards: dict[str, RewardTermCfg] = {
        # --- reward terms ---
        "base_height": RewardTermCfg(
            func=mdp.base_height_reward,
            weight=1.0,
            params={
                "stand_height": 0.35, 
                "sit_height": 0.20, 
                "jump_height_cap": 0.5, 
                "entity_name": "robot",
            }
        ),
        "base_velocity": RewardTermCfg(
            func=mdp.base_velocity_reward,
            weight=1.0,
            params={
                "entity_name": "robot",
            }
        ),
        "base_balance":  RewardTermCfg(
            func=mdp.base_balance_reward,
            weight=1.0,
            params={
                "entity_name": "robot",
            },
        ),
        "energy": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-2.5e-4,
        ),
        "style": RewardTermCfg(
            func=mdp.style_penalty,
            weight=0.1,
            params={
                "entity_name": "robot",
            },
        ),

        # --- cost/penalty terms ---
        "foot_contact_sequence": RewardTermCfg(
            func=mdp.foot_contact_sequence_penalty,
            weight=1.0,
            params={
                "feet_sensor_name": "feet_ground_contact"
            },
        ),
        "dof_joint_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-1.0,
        ),
        "dof_joint_vel": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.001
        ),

        "action_rate": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.01,
        ),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "trunk_ground_contact": TerminationTermCfg(
            func=mdp.trunk_contact_termination,
            params={
                "sensor_name": "trunk_ground_touch",
                "settle_steps": 10,
            }
        ),
        "landing_without_flip": TerminationTermCfg(
            func=mdp.landing_without_flip_termination,
            params={
                "feet_sensor_name": "feet_ground_contact"
            }
        ),
    }

    metrics: dict[str, MetricsTermCfg] = {
        # primary success metric — did the robot complete a full flip
        "flip_success": MetricsTermCfg(
            func=mdp.flip_success,
        ),
        "stage_reached": MetricsTermCfg(
            func=mdp.max_stage_reached,
        ),
        "air_rate": MetricsTermCfg(
            func=mdp.air_stage_rate,
        ),
        "land_rate": MetricsTermCfg(
            func=mdp.land_stage_rate,
        ),
        "mean_action_acc": MetricsTermCfg(
            func=mdp.mean_action_acc,
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                terrain_type="plane"
            ),
            num_envs=10,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands={},
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=20.0
    )