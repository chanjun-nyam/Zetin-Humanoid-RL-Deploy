import time

import numpy as np

from robofsm_np.fsm import BaseNode
from .robot_state import RobotState
from .rl_policy import RLPolicy


class HardStopNode(BaseNode[RobotState]):
    def __init__(self, state: RobotState):
        super().__init__(state)

    def on_enter(self):
        pass

    def on_update(self):
        self.state.kp.fill(0)
        self.state.kd.fill(0)

    def on_exit(self):
        pass


class SoftStopNode(BaseNode[RobotState]):
    def __init__(self, state: RobotState, duration: float):
        super().__init__(state)
        self.duration = duration

    def on_enter(self):
        # capture robot state
        s = self.state
        self.qpos_cap = s.qpos.copy()
        s.qpos_trg[...] = s.qpos
        s.kp[...] = s.kp_def
        s.kd[...] = s.kd_def

        # reset timer
        self.frame_t = time.perf_counter()
        self.t = 0.0

    def on_update(self):
        # update timer
        new_frame_t = time.perf_counter()
        dt = new_frame_t - self.frame_t
        self.frame_t = new_frame_t
        self.t += dt

        # set qpos target
        s = self.state
        r = min(self.t / self.duration, 1.0)
        s.qpos_trg[...] = (1.0 - r) * self.qpos_cap + r * s.qpos_def

    def on_exit(self):
        pass


class RLPolicyNode(BaseNode[RobotState]):
    def __init__(self, state: RobotState, rl_policy: RLPolicy, duration: float):
        super().__init__(state)
        self.rl_policy = rl_policy
        self.duration = duration

        self.usr_cmd = np.array([0.5] * 6, dtype=np.float32)
        self.is_walk = np.array(False, dtype=np.float32)

        # preprocess: q-variables indices mapping
        rl_q_names = self.rl_policy.cfg.q_names
        ref_q_names = self.state.q_names
        self.to_q_ref = [rl_q_names.index(x) for x in ref_q_names]
        self.from_q_ref = [ref_q_names.index(x) for x in rl_q_names]

    def set_cmd(self, usr_cmd: np.ndarray, is_walk: np.ndarray):
        self.usr_cmd[...] = usr_cmd
        self.is_walk[...] = is_walk

    def on_enter(self):
        # capture robot state
        s = self.state
        self.qpos_cap = s.qpos.copy()
        s.qpos_trg[...] = s.qpos_def
        s.kp[...] = s.kp_def
        s.kd[...] = s.kd_def

        # reset timer
        self.frame_t = time.perf_counter()
        self.t = 0.0

        # reset rl policy
        self.rl_policy.reset()

    def on_update(self):
        # update timer
        new_frame_t = time.perf_counter()
        dt = new_frame_t - self.frame_t
        self.frame_t = new_frame_t
        self.t += dt

        # compute qpos target: run rl-policy
        s = self.state
        qpos_trg = self.rl_policy.step(
            quat=s.quat_w,
            linvel=s.linvel_b,
            angvel=s.angvel_b,
            qpos=s.qpos[self.from_q_ref],
            qvel=s.qvel[self.from_q_ref],
            is_walk=self.is_walk,
            usr_cmd=self.usr_cmd,
        )[self.to_q_ref]

        # set qpos target
        r = min(self.t / self.duration, 1.0)
        s.qpos_trg[...] = (1.0 - r) * self.qpos_cap + r * qpos_trg

    def on_exit(self):
        pass
