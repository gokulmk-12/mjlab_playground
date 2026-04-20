from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_playground.asset_zoo.robots.unitree_go2.go2_constants import get_go2_robot_cfg
from mjlab_playground.backflip.backflip_env_cfg import make_backflip_env_cfg

def unitree_go2_backflip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_backflip_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.impratio = 10

    cfg.scene.entities = {"robot": get_go2_robot_cfg()}

    foot_names = ("FR", "FL", "RR", "RL")
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    trunk_head_ground_cfg = ContactSensorCfg(
        name="trunk_ground_touch",
        primary=ContactMatch(mode="geom", entity="robot", pattern=("base1_collision", "base2_collision", "base3_collision")),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
        trunk_head_ground_cfg
    )

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = 0.25

    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
    
    return cfg

if __name__ == "__main__":
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import ViserPlayViewer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = unitree_go2_backflip_env_cfg()
    env = ManagerBasedRlEnv(cfg, device=device)

    class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
            del obs
            return torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    
    policy = PolicyZero()
    ViserPlayViewer(env, policy).run() # type: ignore