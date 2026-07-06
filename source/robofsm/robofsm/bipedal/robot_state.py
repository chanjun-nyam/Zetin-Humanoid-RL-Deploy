from dataclasses import dataclass, MISSING

import torch as th

from robofsm.fsm import BaseRobotState



@dataclass
class RobotState(BaseRobotState):
    n_qdim: int = MISSING
    q_names: list[str] = MISSING

    quat_w: th.Tensor = MISSING
    linvel_b: th.Tensor = MISSING
    angvel_b: th.Tensor = MISSING

    qpos: th.Tensor = MISSING
    qvel: th.Tensor = MISSING
    qpos_trg: th.Tensor = MISSING

    kp: th.Tensor = MISSING
    kd: th.Tensor = MISSING

    qpos_def: th.Tensor = MISSING
    kp_def: th.Tensor = MISSING
    kd_def: th.Tensor = MISSING
