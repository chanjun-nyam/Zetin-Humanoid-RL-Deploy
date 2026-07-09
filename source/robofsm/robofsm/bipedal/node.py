import time

import torch as th

from robofsm.fsm import BaseNode
from .robot_state import RobotState
from .rl_policy import RLPolicy



class _TimerNode(BaseNode[RobotState]):
    def __init__(self, state: RobotState):
        super().__init__(state)

    def on_enter(self):
        self.t0 = time.perf_counter()
        self.t = 0.0

    def on_update(self):
        new_t = time.perf_counter() - self.t0
        self.dt = new_t - self.t
        self.t = new_t



class HardStopNode(_TimerNode):
    def __init__(self, state: RobotState):
        super().__init__(state)

    def on_enter(self):
        super().on_enter()

    def on_update(self):
        super().on_update()
        self.state.kp.zero_()
        self.state.kd.zero_()



class SoftStopNode(_TimerNode):
    def __init__(self, state: RobotState, duration: float):
        super().__init__(state)
        self.duration = duration

    def on_enter(self):
        super().on_enter()
        self.r = 0.0

        # capture robot state
        s = self.state
        self.qpos_cap = s.qpos.clone()
        s.qpos_trg.copy_(s.qpos)
        s.kp.copy_(s.kp_def)
        s.kd.copy_(s.kd_def)

    def on_update(self):
        super().on_update()
        self.r = min(self.r + self.dt / self.duration, 1.0)

        # set qpos target
        s = self.state
        s.qpos_trg.copy_((1.0 - self.r) * self.qpos_cap + self.r * s.qpos_def)



class RLPolicyNode(_TimerNode):
    def __init__(self, state: RobotState, rl_policy: RLPolicy, duration: float):
        super().__init__(state)
        self.rl_policy = rl_policy
        self.duration = duration

        # preprocess: q-variables indices mapping
        rl_q_names = self.rl_policy.cfg.q_names
        ref_q_names = self.state.q_names
        self.to_q_ref = [rl_q_names.index(x) for x in ref_q_names]
        self.from_q_ref = [ref_q_names.index(x) for x in rl_q_names]

    def register_cmd(self, axes: th.Tensor, btns: th.Tensor):
        self.cmd_axes = axes
        self.cmd_btns = btns

    def on_enter(self):
        super().on_enter()
        self.r = 0.0

        # capture robot state
        s = self.state
        self.qpos_cap = s.qpos.clone()
        s.qpos_trg.copy_(s.qpos)
        s.kp.copy_(s.kp_def)
        s.kd.copy_(s.kd_def)

        # reset rl policy
        self.rl_policy.reset()

    def on_update(self):
        super().on_update()
        self.r = min(self.r + self.dt / self.duration, 1.0)

        # compute qpos target: run rl-policy
        s = self.state
        qpos_trg = self.rl_policy.step(
            quat=s.quat_w,
            angvel=s.angvel_b,
            qpos=s.qpos[self.from_q_ref]-s.qpos_def[self.from_q_ref],
            qvel=s.qvel[self.from_q_ref],
            cmd_axes=self.cmd_axes,
            cmd_btns=self.cmd_btns,
        )[self.to_q_ref]

        # set qpos target
        s.qpos_trg.copy_((1.0 - self.r) * self.qpos_cap + self.r * qpos_trg)
