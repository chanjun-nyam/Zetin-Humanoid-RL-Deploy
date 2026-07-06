from dataclasses import dataclass, MISSING
from typing import List

import numpy as np

from robofsm_np.fsm import BaseRobotState


@dataclass
class RobotState(BaseRobotState):
    n_qdim: int = MISSING
    q_names: List[str] = MISSING

    quat_w: np.ndarray = MISSING
    linvel_b: np.ndarray = MISSING
    angvel_b: np.ndarray = MISSING

    qpos: np.ndarray = MISSING
    qvel: np.ndarray = MISSING
    qpos_trg: np.ndarray = MISSING

    kp: np.ndarray = MISSING
    kd: np.ndarray = MISSING

    qpos_def: np.ndarray = MISSING
    kp_def: np.ndarray = MISSING
    kd_def: np.ndarray = MISSING
