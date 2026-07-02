from dataclasses import dataclass, MISSING
from typing import List
import time
import copy

import torch as th

import limxsdk.robot as sdk_robot
import limxsdk.datatypes as sdk_dtype

from policy_runner import PolicyRunner, PolicyRunnerCfg



@dataclass
class ControllerCfg:

    robot_ip: str = MISSING

    loop_freq: int = MISSING

    log_freq: int = MISSING

    ref_q_names: List[str] = MISSING

    q_names: List[str] = MISSING

    qpos_default: List[float] = MISSING

    q_kp: List[float] = MISSING

    q_kd: List[float] = MISSING

    zero_cmd_norm: float = MISSING

    policy_runner_cfg: PolicyRunnerCfg = MISSING



class Controller:

    def __init__(self, cfg: ControllerCfg):
        self.cfg = cfg
        self.policy_decimation = cfg.policy_runner_cfg.decimation
        self.log_decimation = cfg.loop_freq // cfg.log_freq

        self.n_qdim = len(cfg.ref_q_names)

        # init policy-runner
        self.policy_runner = PolicyRunner(cfg.policy_runner_cfg)

        # init limxsdk
        self.sdk_robot_type = sdk_robot.RobotType.PointFoot
        self.sdk_robot = sdk_robot.Robot(self.sdk_robot_type, is_sim=False)

        if not self.sdk_robot.init(cfg.robot_ip):
            print(f'[limxsdk-lowlevel] robot initialization failed')
            exit()

        # buffers for callback
        self._imu_data = sdk_dtype.ImuData()
        self._robot_state = sdk_dtype.RobotState()
        self._sensor_joy = sdk_dtype.SensorJoy()
        self._diagnostic_value = sdk_dtype.DiagnosticValue()

        self._imu_data.quat = [1.0, 0.0, 0.0, 0.0]
        self._robot_state.q = [0.0] * self.n_qdim
        self._robot_state.dq = [0.0] * self.n_qdim
        self._sensor_joy.axes = [0.0, 0.0, 0.0]

        self._imu_data_cnt = 0
        self._robot_state_cnt = 0
        self._sensor_joy_cnt = 0
        self._diagnostic_value_cnt = 0

        # subscribe callbacks
        self._subscribe_callbacks()

        # logging metadata
        print(
            f'[limxsdk-lowlevel]\n'
            f'motor number: {self.sdk_robot.getMotorNumber()}\n'
        )


    def _subscribe_callbacks(self):

        def imu_data_callback(imu_data: sdk_dtype.ImuData):
            self._imu_data = imu_data
            self._imu_data_cnt += 1
        self.imu_data_callback = imu_data_callback

        def robot_state_callback(robot_state: sdk_dtype.RobotState):
            self._robot_state = robot_state
            self._robot_state_cnt += 1
        self.robot_state_callback = robot_state_callback

        def sensor_joy_callback(sensor_joy: sdk_dtype.SensorJoy):
            self._sensor_joy = sensor_joy
            self._sensor_joy_cnt += 1
        self.sensor_joy_callback = sensor_joy_callback

        def robot_diagnostic_callback(diagnostic_value: sdk_dtype.DiagnosticValue):
            self._diagnostic_value = diagnostic_value
            self._diagnostic_value_cnt += 1
        self.robot_diagnostic_callback = robot_diagnostic_callback

        self.sdk_robot.subscribeImuData(self.imu_data_callback)
        self.sdk_robot.subscribeRobotState(self.robot_state_callback)
        self.sdk_robot.subscribeSensorJoy(self.sensor_joy_callback)
        self.sdk_robot.subscribeDiagnosticValue(self.robot_diagnostic_callback)


    def run(self):
        time.sleep(1.0)
        print(f'[limxsdk-lowlevel] run()!!\n')

        dtype = th.float32
        device = th.device(self.cfg.policy_runner_cfg.device)
        def float_tensor(x):
            return th.tensor(x, dtype=dtype, device=device)

        policy = self.policy_runner

        # preprocess: q-related quantaties
        to_q_ref = [self.cfg.q_names.index(x) for x in self.cfg.ref_q_names]
        from_q_ref = [self.cfg.ref_q_names.index(x) for x in self.cfg.q_names]

        # buffers
        qpos_default = float_tensor(self.cfg.qpos_default)
        gait_theta = float_tensor([0.0, 0.0])

        _robot_cmd = sdk_dtype.RobotCmd()
        _robot_cmd.mode = [0.0] * self.n_qdim
        _robot_cmd.q = [0.0] * self.n_qdim
        _robot_cmd.dq = [0.0] * self.n_qdim
        _robot_cmd.tau = [0.0] * self.n_qdim
        _robot_cmd.Kp = [0.0] * self.n_qdim
        _robot_cmd.Kd = [0.0] * self.n_qdim

        # main loop
        sdk_rate = sdk_robot.Rate(self.cfg.loop_freq)
        step_num = 0

        step_dt = 0.0
        loop_dt = 0.0
        policy_dt = 0.0
        total_dt = 0.0

        # TODO: flags

        while True:
            step_ns = time.perf_counter_ns()
            step_num += 1

            # fix data read from callbacks
            imu_data = copy.deepcopy(self._imu_data)
            robot_state = copy.deepcopy(self._robot_state)
            sensor_joy = copy.deepcopy(self._sensor_joy)
            diagnostic_value = copy.deepcopy(self._diagnostic_value)

            imu_data_cnt = int(self._imu_data_cnt)
            robot_state_cnt = int(self._robot_state_cnt)
            sensor_joy_cnt = int(self._sensor_joy_cnt)
            diagnostic_value_cnt = int(self._diagnostic_value_cnt)

            usr_cmd = [
                sensor_joy.axes[1], # linvel-x
                sensor_joy.axes[0], # linvel-y
                sensor_joy.axes[2], # angvel-z
                1.2, # gait frequency
                0.5, # gait ratio
                th.pi, # gait offset
            ]
            # TODO

            # tensors
            quat = float_tensor(imu_data.quat)
            angvel = float_tensor(imu_data.gyro)
            linvel = th.zeros_like(angvel)
            qpos = float_tensor(robot_state.q)[from_q_ref]
            qvel = float_tensor(robot_state.dq)[from_q_ref]

            # gait
            vel_cmd = float_tensor(usr_cmd[0:3])
            gait_freq = float_tensor(usr_cmd[3])
            gait_ratio = float_tensor(usr_cmd[4])
            gait_offset = float_tensor(usr_cmd[5])

            gait_theta[0].add_((2.0 * th.pi * (1.0 / self.cfg.sim_freq)) * gait_freq)
            gait_theta[1].copy_(gait_theta[0] + gait_offset)
            gait_theta.remainder_(2.0 * th.pi)

            # apply zero-cmd
            is_walk = float_tensor(usr_cmd[0:3]).square().sum(dim=0).sqrt() > self.cfg.zero_cmd_norm
            vel_cmd *= is_walk
            gait_freq *= is_walk

            # step policy-runner
            s = time.perf_counter_ns()
            policy.sim_step(quat, linvel, angvel, qpos - qpos_default, qvel)
            loop_dt = (time.perf_counter_ns() - s) / 1e9

            # policy step
            if step_num % self.policy_decimation == 0:
                command = th.cat([vel_cmd, th.stack([gait_freq, gait_ratio, gait_offset])], dim=-1)
                clock = th.cat([gait_theta.sin(), gait_theta.cos()], dim=-1)

                s = time.perf_counter_ns()
                policy.policy_step(command[0:5], clock * is_walk)
                policy_dt = (time.perf_counter_ns() - s) / 1e9

            # write action to robot
            qpos_trg = qpos_default + policy.action * policy.q_scale
            _robot_cmd.q = qpos_trg[to_q_ref].tolist()
            _robot_cmd.Kp = float_tensor(self.cfg.q_kp)[to_q_ref].tolist()
            _robot_cmd.Kd = float_tensor(self.cfg.q_kd)[to_q_ref].tolist()
            self.sdk_robot.publishRobotCmd(_robot_cmd)

            # TODO: robot flag 에 따른
            self.sdk_robot.setRobotLightEffect(sdk_dtype.LightEffect.STATIC_WHITE)

            # logging
            if step_num % self.log_decimation == 0:
                print(
                    f'[limxsdk-lowlevel]\n'
                    f'step-num: {step_num}\n'

                    f'joy-axes: {" | ".join([f"{x:6.3f}" for x in sensor_joy.axes])}\n'
                    f'joy-buttons: {" | ".join([f"{x}" for x in sensor_joy.buttons])}\n'
                    f'motor-names: {robot_state.motor_names}\n'

                    f'total-util: {total_dt * self.cfg.loop_freq:.4f} | '
                    f'step-util: {step_dt * self.cfg.loop_freq:.4f} | '
                    f'loop-util: {loop_dt * self.cfg.loop_freq:.4f} | '
                    f'policy-util: {policy_dt * self.cfg.loop_freq:.4f}\n'

                    f'cnt: {imu_data_cnt} | {robot_state_cnt} | {sensor_joy_cnt} | {diagnostic_value_cnt}\n'

                    f'quat: {" | ".join([f"{x:6.3f}" for x in quat])}\n'
                    f'lvel: {" | ".join([f"{x:6.3f}" for x in linvel])}\n'
                    f'avel: {" | ".join([f"{x:6.3f}" for x in angvel])}\n'
                    f'qpos: {" | ".join([f"{x:6.3f}" for x in qpos])}\n'
                    f'qvel: {" | ".join([f"{x:6.3f}" for x in qvel])}\n'
                    f'qtrg: {" | ".join([f"{x:6.3f}" for x in qpos_trg])}\n'
                    f'ucmd: {" | ".join([f"{x:6.3f}" for x in usr_cmd])}\n'
                )

            # fix loop delta-time
            step_dt = (time.perf_counter_ns() - step_ns) / 1e9
            sdk_rate.sleep()
            total_dt = (time.perf_counter_ns() - step_ns) / 1e9



if __name__ == '__main__':
    # q-names
    SDK_Q_NAMES = [
        'abad_L_Joint', 'hip_L_Joint', 'knee_L_Joint', 'ankle_L_Joint',
        'abad_R_Joint', 'hip_R_Joint', 'knee_R_Joint', 'ankle_R_Joint',
    ]
    USD_Q_NAMES = [
        'abad_L_Joint', 'abad_R_Joint',
        'hip_L_Joint',  'hip_R_Joint',
        'knee_L_Joint', 'knee_R_Joint',
        'ankle_L_Joint','ankle_R_Joint',
    ]

    LOOP_FREQ = 500
    POLICY_FREQ = 50
    LOG_FREQ = 10

    # controller configuration
    controller_cfg = ControllerCfg(
        robot_ip='127.0.0.1',
        # robot_ip='10.192.1.2',

        loop_freq=LOOP_FREQ,
        log_freq=LOG_FREQ,

        ref_q_names=SDK_Q_NAMES,
        q_names=USD_Q_NAMES,
        qpos_default=[0.0] * 8,
        q_kp=[45., 45., 45., 45., 45., 45., 45., 45.],
        q_kd=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.8, 0.8],

        zero_cmd_norm=0.2,

        policy_runner_cfg=PolicyRunnerCfg(
            decimation=LOOP_FREQ // POLICY_FREQ,
            device='cpu',
            ref_q_names=USD_Q_NAMES,
            q_names=USD_Q_NAMES,
            q_scale=[0.5] * 8,
            n_history=10,
            obs_scale=[0.25, 1.0, 1.0, 0.05, 1.0, 1.0],
            obs_clip=(-100.0, 100.0),
            action_clip=(-100.0, 100.0),
            model_path='models/tron1_s_flat.onnx',
        ),
    )

    # setup environment variable (limxsdk-lowlevel requires this)
    import os
    os.environ['ROBOT_TYPE'] = 'SF_TRON1B'

    # run controller
    controller = Controller(controller_cfg)
    controller.run()
