from dataclasses import dataclass, MISSING
from typing import List
import time

import mujoco as mj
import mujoco.viewer
import torch as th

from policy_runner import PolicyRunner, PolicyRunnerCfg



@dataclass
class SimulatorCfg:

    mjcf_path: str = MISSING

    rendering: bool = MISSING

    sim_freq: int = MISSING

    rend_freq: int = MISSING

    log_freq: int = MISSING

    q_names: List[str] = MISSING

    qpos_default: List[float] = MISSING

    q_kp: List[float] = MISSING

    q_kd: List[float] = MISSING

    policy_runner_cfg: PolicyRunnerCfg = MISSING



class Simulator:

    def __init__(self, cfg: SimulatorCfg):
        self.cfg = cfg
        self.policy_decimation = cfg.policy_runner_cfg.decimation
        self.rend_decimation = cfg.sim_freq // cfg.rend_freq
        self.log_decimation = cfg.sim_freq // cfg.log_freq

        # init policy-runner
        self.policy_runner = PolicyRunner(cfg.policy_runner_cfg)

        # init mujoco
        self.mj_model = mj.MjModel.from_xml_path(self.cfg.mjcf_path)
        self.mj_data = mj.MjData(self.mj_model)
        self.mj_viewer = None
        if cfg.rendering:
            self.mj_viewer = mj.viewer.launch_passive(self.mj_model, self.mj_data)

        self.mj_model.opt.timestep = 1 / cfg.sim_freq


    def run(self):

        mj_model = self.mj_model
        mj_data = self.mj_data
        mj_viewer = self.mj_viewer

        dtype = th.float32
        device = th.device(self.cfg.policy_runner_cfg.device)
        def float_tensor(x):
            return th.tensor(x, dtype=dtype, device=device)

        # preprocess: q-related quantaties
        ref_q_names = [mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(mj_model.njnt)][1:]
        q_names = self.cfg.q_names

        to_q_ref = [q_names.index(x) for x in ref_q_names]
        from_q_ref = [ref_q_names.index(x) for x in q_names]

        # preprocess: sensor data indices
        sensor_names = ['quat', 'gyro', 'acc']
        sensor_ids = [mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SENSOR, x) for x in sensor_names]
        sensor_slices = [
            slice(mj_model.sensor_adr[x], mj_model.sensor_adr[x] + mj_model.sensor_dim[x])
            for x in sensor_ids
        ]
        sensor_name2slice = {a: b for a, b in zip(sensor_names, sensor_slices)}

        # buffers
        qpos_default = float_tensor(self.cfg.qpos_default)
        qpos_trg = float_tensor([0.0] * len(self.cfg.qpos_default))
        q_kp = float_tensor(self.cfg.q_kp)
        q_kd = float_tensor(self.cfg.q_kd)

        # main loop
        step_num = 0

        step_dt = 0.0
        sim_dt = 0.0
        policy_dt = 0.0
        total_dt = 0.0

        while True:
            step_ns = time.perf_counter_ns()
            step_num += 1

            # step mujoco simulator
            mj.mj_step(mj_model, mj_data)

            # tensors
            quat = th.from_numpy(mj_data.sensordata[sensor_name2slice['quat']])
            gyro = th.from_numpy(mj_data.sensordata[sensor_name2slice['gyro']])
            qpos = th.from_numpy(mj_data.qpos[7:])[from_q_ref]
            qvel = th.from_numpy(mj_data.qvel[6:])[from_q_ref]
            usr_cmd = float_tensor([0.0, 0.0, 0.0])

            # step policy-runner
            s = time.perf_counter_ns()
            self.policy_runner.sim_step(quat, gyro, qpos - qpos_default, qvel)
            sim_dt = (time.perf_counter_ns() - s) / 1e9

            # policy step
            if step_num % self.policy_decimation == 0:
                s = time.perf_counter_ns()
                qpos_trg = self.policy_runner.policy_step(usr_cmd) + qpos_default
                policy_dt = (time.perf_counter_ns() - s) / 1e9

            # write action to robot
            qtau = q_kp * (qpos_trg - qpos) - q_kd * qvel
            mj_data.ctrl[:] = qtau[to_q_ref].numpy()

            # rendering
            if step_num % self.rend_decimation == 0:
                if mj_viewer is not None:
                    mj_viewer.sync()

            # logging
            if step_num % self.log_decimation == 0:
                print(
                    f'[controller-log]\n'
                    f'step-num: {step_num}\n'

                    f'total-util: {total_dt * self.cfg.sim_freq:.4f} | '
                    f'step-util: {step_dt * self.cfg.sim_freq:.4f} | '
                    f'sim-util: {sim_dt * self.cfg.sim_freq:.4f} | '
                    f'policy-util: {policy_dt * self.cfg.sim_freq:.4f}\n'

                    f'quat: {" | ".join([f"{x:6.3f}" for x in quat])}\n'
                    f'gyro: {" | ".join([f"{x:6.3f}" for x in gyro])}\n'
                    f'qpos: {" | ".join([f"{x:6.3f}" for x in qpos])}\n'
                    f'qvel: {" | ".join([f"{x:6.3f}" for x in qvel])}\n'
                    f'qtrg: {" | ".join([f"{x:6.3f}" for x in qpos_trg])}\n'
                )

            # fix loop delta-time
            step_dt = (time.perf_counter_ns() - step_ns) / 1e9
            while ((time.perf_counter_ns() - step_ns) / 1e9) * self.cfg.sim_freq < 1.0:
                pass
            total_dt = (time.perf_counter_ns() - step_ns) / 1e9



if __name__ == '__main__':
    # q-names
    USD_Q_NAMES = [
        'abad_L_Joint', 'abad_R_Joint',
        'hip_L_Joint',  'hip_R_Joint',
        'knee_L_Joint', 'knee_R_Joint',
        'ankle_L_Joint','ankle_R_Joint',
    ]

    # controller configuration
    simulator_cfg = SimulatorCfg(
        mjcf_path='robot-description/pointfoot/SF_TRON1B/xml/robot.xml',
        rendering=True,

        sim_freq=500,
        rend_freq=50,
        log_freq=10,

        q_names=USD_Q_NAMES,
        qpos_default=[
            0.0, 0.0,
            0.0, 0.0,
            # 0.13, -0.13, # TODO
            0.0, 0.0,
            0.0, 0.0,
        ],
        q_kp=[45.0] * 8,
        q_kd=[1.5] * 6 + [0.8] * 2,

        policy_runner_cfg=PolicyRunnerCfg(
            decimation=10,
            device='cpu',
            ref_q_names=USD_Q_NAMES,
            q_names=USD_Q_NAMES,
            q_scale=[0.5] * 8,
            n_history=10,
            obs_scale=[0.25, 1.0, 1.0, 0.05, 1.0, 1.0],
            obs_clip=(-5.0, 5.0),
            action_clip=(-5.0, 5.0),
            model_path='models/tron1_0_s_rough.onnx',
            # model_path='models/tron1_0_s_flat_.onnx',
            # model_path='models/tron1_0_s_flat_2.onnx',
        ),
    )

    # setup environment variable (limxsdk-lowlevel requires this)
    import os
    os.environ['ROBOT_TYPE'] = 'SF_TRON1B'

    # run controller
    simulator = Simulator(simulator_cfg)
    simulator.run()
