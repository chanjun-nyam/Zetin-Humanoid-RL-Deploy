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
        sensor_names = ['quat', 'linvel', 'linacc', 'angvel']
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

        usr_cmd = float_tensor([0.0, 0.0, 0.0])

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
            linvel = th.from_numpy(mj_data.sensordata[sensor_name2slice['linvel']])
            angvel = th.from_numpy(mj_data.sensordata[sensor_name2slice['angvel']])
            qpos = th.from_numpy(mj_data.qpos[7:])[from_q_ref]
            qvel = th.from_numpy(mj_data.qvel[6:])[from_q_ref]

            # step policy-runner
            s = time.perf_counter_ns()
            self.policy_runner.sim_step(quat, linvel, angvel, qpos - qpos_default, qvel * 2)
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

                    f'quat: {" | ".join([f"{x:6.3f}" for x in self.policy_runner.quat])}\n'
                    f'linv: {" | ".join([f"{x:6.3f}" for x in self.policy_runner.linvel.sma])}\n'
                    f'angv: {" | ".join([f"{x:6.3f}" for x in self.policy_runner.angvel.sma])}\n'
                    f'qpos: {" | ".join([f"{x:6.3f}" for x in self.policy_runner.qpos])}\n'
                    f'qvel: {" | ".join([f"{x:6.3f}" for x in self.policy_runner.qvel.sma])}\n'
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

    SIM_FREQ = 500
    POLICY_FREQ = 50
    REND_FREQ = 50
    LOG_FREQ = 10

    # controller configuration
    simulator_cfg = SimulatorCfg(
        mjcf_path='assets/Tron1-S/robot.xml',
        rendering=False,

        sim_freq=SIM_FREQ,
        rend_freq=REND_FREQ,
        log_freq=LOG_FREQ,

        q_names=USD_Q_NAMES,
        qpos_default=[
            0.0, 0.0,
            0.0, 0.0,
            # 0.13, -0.13, # TODO
            0.0, 0.0,
            0.0, 0.0,
        ],
        q_kp=[45., 45., 45., 45., 45., 45., 45., 45.],
        q_kd=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.8, 0.8],

        policy_runner_cfg=PolicyRunnerCfg(
            decimation=SIM_FREQ // POLICY_FREQ,
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

    # run controller
    simulator = Simulator(simulator_cfg)
    simulator.run()
