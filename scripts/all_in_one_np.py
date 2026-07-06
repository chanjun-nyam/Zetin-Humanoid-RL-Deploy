import os

# offscreen rendering backend (set before mujoco creates any gl context).
# 'egl' for headless gpu servers, 'osmesa' for cpu-only. override via env var.
os.environ.setdefault('MUJOCO_GL', 'egl')

from dataclasses import dataclass, MISSING
from typing import Callable, Optional
import time

import mujoco as mj
import numpy as np

from robofsm_np.fsm import BaseNode
from robofsm_np.bipedal import (
    RobotState,
    HardStopNode,
    SoftStopNode,
    RLPolicyNode,
    RLPolicy,
    RLPolicyCfg,
)
from web_stream_ import WebStreamServer, WebStreamCfg, SliderCfg


@dataclass
class SimulatorCfg:

    mjcf_path: str = MISSING

    sim_freq: int = MISSING

    rend_freq: int = MISSING

    log_freq: int = MISSING


class Simulator:

    def __init__(self, cfg: SimulatorCfg):
        self.cfg = cfg
        self.rend_decimation = cfg.sim_freq // cfg.rend_freq
        self.log_decimation = cfg.sim_freq // cfg.log_freq

        # rendering / user-command hooks (registered via register_* methods below)
        self._render_callback: Optional[Callable] = None
        self._cmd_callback: Optional[Callable] = None

        # init mujoco
        self.mj_model = mj.MjModel.from_xml_path(self.cfg.mjcf_path)
        self.mj_data = mj.MjData(self.mj_model)

        self.mj_model.opt.timestep = 1.0 / cfg.sim_freq

    def register_render_callback(self, callback: Callable):
        self._render_callback = callback

    def register_cmd_callback(self, callback: Callable):
        self._cmd_callback = callback

    def run(self, node: BaseNode[RobotState]):
        s = node.state

        mj_model = self.mj_model
        mj_data = self.mj_data

        def array(data, dtype=np.float32):
            return np.array(data, dtype=dtype)

        # preprocess: sensor data indices
        sensor_names = ['quat', 'linvel', 'linacc', 'angvel']
        sensor_ids = [mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SENSOR, x) for x in sensor_names]
        sensor_slices = [
            slice(mj_model.sensor_adr[x], mj_model.sensor_adr[x] + mj_model.sensor_dim[x])
            for x in sensor_ids
        ]
        sensor_name2slice = {a: b for a, b in zip(sensor_names, sensor_slices)}

        # buffers
        usr_cmd = [0.5] * 7

        from robofsm_np.utils.buffer import SMABuffer
        vel_buff = SMABuffer.init_like(array([0.0] * 3), (0,), self.cfg.sim_freq)

        # main loop
        step_num = 0

        step_dt = 0.0
        sim_dt = 0.0
        rend_dt = 0.0
        total_dt = 0.0

        while True:
            step_ns = time.perf_counter_ns()
            step_num += 1

            # step mujoco simulator
            mj.mj_step(mj_model, mj_data)

            # arrays (views sharing memory with mujoco buffers, like th.from_numpy)
            s.quat_w = mj_data.sensordata[sensor_name2slice['quat']]
            s.linvel_b = mj_data.sensordata[sensor_name2slice['linvel']]
            s.angvel_b = mj_data.sensordata[sensor_name2slice['angvel']]
            s.qpos = mj_data.qpos[7:]
            s.qvel = mj_data.qvel[6:]

            vel_buff.update(np.concatenate([s.linvel_b[0:2], s.angvel_b[2:3]]))

            # step policy-runner
            t_ns = time.perf_counter_ns()
            node = node.update()
            sim_dt = (time.perf_counter_ns() - t_ns) / 1e9

            # user cmd
            if isinstance(node, RLPolicyNode):
                node.set_cmd(array(usr_cmd[:6]), array(usr_cmd[6] > 0.6, dtype=np.bool_))

            # write action to robot
            qtau = s.kp * (s.qpos_trg - s.qpos) - s.kd * s.qvel
            mj_data.ctrl[:] = qtau

            # render-rate hooks: pull user-command, push rendered frame
            if step_num % self.rend_decimation == 0:
                t_ns = time.perf_counter_ns()
                if self._cmd_callback is not None:
                    usr_cmd = self._cmd_callback()
                if self._render_callback is not None:
                    self._render_callback(mj_model, mj_data)
                rend_dt = (time.perf_counter_ns() - t_ns) / 1e9

            # logging
            if step_num % self.log_decimation == 0:
                print(
                    f'[simulator]\n'
                    f'step-num: {step_num}\n'

                    f'total-util: {total_dt * self.cfg.sim_freq:.4f} | '
                    f'step-util: {step_dt * self.cfg.sim_freq:.4f} | '
                    f'sim-util: {sim_dt * self.cfg.sim_freq:.4f} | '
                    f'rend-util: {rend_dt * self.cfg.sim_freq:.4f}\n'

                    f'quat: {" | ".join([f"{x:6.3f}" for x in s.quat_w])}\n'
                    f'lvel: {" | ".join([f"{x:6.3f}" for x in s.linvel_b])}\n'
                    f'avel: {" | ".join([f"{x:6.3f}" for x in s.angvel_b])}\n'
                    f'avgv: {" | ".join([f"{x:6.3f}" for x in vel_buff.sma])}\n'
                    f'qpos: {" | ".join([f"{x:6.3f}" for x in s.qpos])}\n'
                    f'qvel: {" | ".join([f"{x:6.3f}" for x in s.qvel])}\n'
                    f'qtrg: {" | ".join([f"{x:6.3f}" for x in s.qpos_trg])}\n'
                    f'qtau: {" | ".join([f"{x:6.3f}" for x in qtau])}\n'
                    f'ucmd: {" | ".join([f"{x:6.3f}" for x in usr_cmd])}\n'
                )

            # fix loop delta-time
            step_dt = (time.perf_counter_ns() - step_ns) / 1e9
            while ((time.perf_counter_ns() - step_ns) / 1e9) * self.cfg.sim_freq < 1.0:
                pass
            total_dt = (time.perf_counter_ns() - step_ns) / 1e9


if __name__ == '__main__':
    # q-names
    REF_Q_NAMES = [
        'abad_L_Joint', 'hip_L_Joint', 'knee_L_Joint', 'ankle_L_Joint',
        'abad_R_Joint', 'hip_R_Joint', 'knee_R_Joint', 'ankle_R_Joint',
    ]
    RL_Q_NAMES = [
        'abad_L_Joint', 'abad_R_Joint',
        'hip_L_Joint',  'hip_R_Joint',
        'knee_L_Joint', 'knee_R_Joint',
        'ankle_L_Joint','ankle_R_Joint',
    ]

    SIM_FREQ = 400
    POLICY_FREQ = 50
    REND_FREQ = 50
    LOG_FREQ = 10

    # build fsm graph
    def array(data, dtype=np.float32):
        return np.array(data, dtype=dtype)

    robot_state = RobotState(
        n_qdim=8,
        q_names=REF_Q_NAMES,
        quat_w=array([1, 0, 0, 0]),
        linvel_b=array([0, 0, 0]),
        angvel_b=array([0, 0, 0]),
        qpos=array([0] * 8),
        qvel=array([0] * 8),
        qpos_trg=array([0] * 8),
        kp=array([0] * 8),
        kd=array([0] * 8),
        qpos_def=array([0] * 8),
        kp_def=array([45] * 8),
        kd_def=array([1.5] * 6 + [0.8] * 2),
    )

    rl_policy = RLPolicy(RLPolicyCfg(
        step_freq=SIM_FREQ,
        policy_freq=POLICY_FREQ,
        device='cpu',
        q_names=RL_Q_NAMES,
        qpos_def=robot_state.qpos_def.tolist(),
        n_history=10,
        obs_scale=[0.25, 1.0, 1.0, 0.05, 1.0, 1.0],
        action_scale=[0.5] * 8,
        obs_clip=(-100.0, 100.0),
        action_clip=(-100.0, 100.0),
        vel_cmd_rng=[
            (-1.5, 1.5),
            (-1.0, 1.0),
            (-1.0, 1.0),
        ],
        gait_cmd_rng=[
            (0.8, 1.6),
            (0.4, 0.6),
            (np.pi / 2, np.pi / 2),
        ],
        # model_path='models/tron1_s_flat.onnx',
        model_path='models/tron1_s_rough.onnx',
    ))

    hard_stop_node = HardStopNode(robot_state)
    soft_stop_node = SoftStopNode(robot_state, duration=5.0)
    rl_policy_node = RLPolicyNode(robot_state, rl_policy=rl_policy, duration=0.000001)

    # configure edge
    # hard_stop_node.add_edge()

    # controller configuration
    simulator_cfg = SimulatorCfg(
        mjcf_path='assets/Tron1-S/robot.xml',
        sim_freq=SIM_FREQ,
        rend_freq=REND_FREQ,
        log_freq=LOG_FREQ,
    )

    # build simulator
    simulator = Simulator(simulator_cfg)

    # reachable over an ssh-forwarded tcp port (default 8000).
    web_stream = WebStreamServer(
        cfg = WebStreamCfg(
            host='0.0.0.0',
            port=8000,
            width=600,
            height=400,
            camera='track',
            jpeg_quality=80,
            stream_freq=REND_FREQ,
            sliders=[
                SliderCfg('vx', 0.0, 1.0),
                SliderCfg('vy', 0.0, 1.0),
                SliderCfg('wz', 0.0, 1.0),
                SliderCfg('fq', 0.0, 1.0),
                SliderCfg('rt', 0.0, 1.0),
                SliderCfg('of', 0.0, 1.0),
                SliderCfg('iw', 0.0, 1.0),
            ]
        )
    )

    # register rendering + user-command as callbacks on the simulator
    simulator.register_render_callback(web_stream.render_frame)
    simulator.register_cmd_callback(web_stream.get_cmd)

    # run controller
    # simulator.run(hard_stop_node)
    # simulator.run(soft_stop_node)
    simulator.run(rl_policy_node)
