"""Booster T1 getup environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_playground.asset_zoo.robots.booster_t1.t1_constants import get_t1_robot_cfg
from mjlab_playground.getup import mdp
from mjlab_playground.getup.getup_env_cfg import make_getup_env_cfg

# From reference t1_getup.py and confirmed via qpos[2]=0.665 at home keyframe.
# Waist body is 0.1155m below trunk (XML offset), so 0.665 - 0.1155 ≈ 0.55.
_TORSO_HEIGHT = 0.67
_WAIST_HEIGHT = 0.55


def booster_t1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Booster T1 getup task configuration."""
  cfg = make_getup_env_cfg()

  cfg.scene.entities = {"robot": get_t1_robot_cfg()}

  cfg.sim.njmax = 200
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"

  # Self-collision sensor.
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="Trunk", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (self_collision_cfg,)

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name},
  )

  # Torso + waist height. Waist reward prevents "sitting on booty" local minimum
  # where torso is high but waist (pelvis) stays near ground.
  cfg.rewards["torso_height"].params["desired_height"] = _TORSO_HEIGHT
  cfg.rewards["waist_height"] = RewardTermCfg(
    func=mdp.body_height_reward,
    weight=1.0,
    params={
      "desired_height": _WAIST_HEIGHT,
      "asset_cfg": SceneEntityCfg("robot", body_names=("Waist",)),
    },
  )
  cfg.metrics["getup_success"].params["desired_height"] = _TORSO_HEIGHT

  # Per-joint posture std: tight hips, medium knees/ankles, loose arms/waist.
  cfg.rewards["posture"].params["std"] = {
    r".*_Hip_Roll": 0.08,
    r".*_Hip_Yaw": 0.08,
    r".*_Hip_Pitch": 0.12,
    r".*_Knee_Pitch": 0.15,
    r".*_Ankle_Pitch": 0.2,
    r".*_Ankle_Roll": 0.2,
    r"(Waist|AAHead_yaw|Head_pitch|.*_Shoulder.*|.*_Elbow.*)": 0.5,
  }

  cfg.viewer.body_name = "Trunk"

  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("Trunk",)
  )

  foot_geom_names = ("left_foot_collision", "right_foot_collision")
  cfg.events["geom_friction_slide"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
      "operation": "abs",
      "axes": [0],
      "ranges": (0.3, 1.5),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_spin"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [1],
      "ranges": (1e-4, 2e-2),
      "shared_random": True,
    },
  )
  cfg.events["foot_friction_roll"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=foot_geom_names),
      "operation": "abs",
      "distribution": "log_uniform",
      "axes": [2],
      "ranges": (1e-5, 5e-3),
      "shared_random": True,
    },
  )

  cfg.events["reset_fallen_or_standing"].params["fall_height"] = 1.2
  cfg.actions["joint_pos"].settle_steps = 50  # 1s at 50Hz

  cfg.curriculum = {}

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.events["reset_fallen_or_standing"].params["fall_probability"] = 1.0

  return cfg


if __name__ == "__main__":
  cfg = booster_t1_getup_env_cfg()

  import mujoco.viewer as viewer
  from mjlab.scene import Scene

  scene = Scene(cfg.scene, "cpu")
  viewer.launch(scene.spec.compile())
