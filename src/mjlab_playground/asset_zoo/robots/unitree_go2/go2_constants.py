import mujoco
from pathlib import Path
import mujoco.viewer as viewer

from mjlab.entity.entity import Entity
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

GO2_SOURCE: Path = Path(__file__).parent
GO2_XML: Path = (
    GO2_SOURCE / "xml" / "go2.xml"
)
assert GO2_XML.exists()

def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    return spec

## Actuator Configs for the GO2 Robot
GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*hip_.*",
    ),
    stiffness=20.0,
    damping=0.5,
    effort_limit=23.5,
    armature=0.01,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=20.0,
  damping=0.5,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=20.0,
  damping=0.5,
  effort_limit=45,
  armature=0.02,
)

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.34),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,
        "RR_thigh_joint": 1.0,
        ".*calf_joint": -1.5,
        ".*R_hip_joint": 0.1,
        ".*L_hip_joint": -0.1,
    },
    joint_vel={".*": 0.0},
)

_foot_regex = "^[FR][LR]_foot_collision$"
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6, ),
    solimp=(0.9, 0.95, 0.023),
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_foot_regex: 6, ".*_collision": 1},
    priority={_foot_regex: 1},
    friction={_foot_regex: (1, 5e-3, 5e-4)},
    solref=(0.01, 1),
)

GO2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        GO2_ACTUATOR_HIP,
        GO2_ACTUATOR_THIGH,
        GO2_ACTUATOR_CALF,
    ),
    soft_joint_pos_limit_factor=0.9
)

def get_go2_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=GO2_ARTICULATION,
    )

if __name__ == "__main__":
    robot = Entity(get_go2_robot_cfg())
    robot.write_xml(Path("assets/scene.xml"))
    viewer.launch(robot.spec.compile())