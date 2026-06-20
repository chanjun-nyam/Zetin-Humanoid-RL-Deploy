from dataclasses import dataclass, MISSING
from typing import List
import copy
import time

import torch

import limxsdk.robot as sdk_robot
import limxsdk.datatypes as sdk_dtype

from policy_runner import PolicyRunner, PolicyRunnerCfg



@dataclass(slots=True)
class ControllerCfg:

    robot_ip: str = MISSING

    loop_freq: int = MISSING

    logging_steps: int = MISSING

    ref_q_names: List[str] = MISSING

    q_names: List[str] = MISSING

    q_kp: List[float] = MISSING

    q_kd: List[float] = MISSING

    policy_runner_cfg: PolicyRunnerCfg = MISSING



class Controller:
    __slots__ = (
        'cfg', 'decimation', 'device',
        'ref_q_names', 'q_names', 'n_qdim',
        'q_map', 'q_map_inv',
        'policy_runner',
        'robot_type', 'robot',
        '_imu_data', '_robot_state', '_sensor_joy', '_diagnostic_value',
        'imu_data_callback', 'robot_state_callback', 'sensor_joy_callback', 'robot_diagnostic_callback',
    )


    def __init__(self, cfg: ControllerCfg):
        self.cfg = cfg

        self.decimation = cfg.policy_runner_cfg.decimation
        self.device = torch.device(cfg.policy_runner_cfg.device)

        self.ref_q_names = self.cfg.ref_q_names
        self.q_names = self.cfg.q_names
        self.n_qdim = len(self.ref_q_names)

        self.q_map = [self.q_names.index(name) for name in self.ref_q_names]
        self.q_map_inv = [self.ref_q_names.index(name) for name in self.q_names]

        # policy runner
        self.policy_runner = PolicyRunner(cfg.policy_runner_cfg)

        # limxsdk robot
        self.robot_type = sdk_robot.RobotType.PointFoot
        self.robot = sdk_robot.Robot(self.robot_type, is_sim=False)

        if not self.robot.init(cfg.robot_ip):
            print(f'[controller-log] robot initialization failed')
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

        # subscribe callbacks
        self._subscribe_callbacks()

        # print metadata
        print(
            f'[limxsdk-lowlevel]\n'
            f'motor number: {self.robot.getMotorNumber()}\n'
            f'motor names: {self.robot.getMotorNames()}\n'
        )


    def _subscribe_callbacks(self):

        def imu_data_callback(imu_data: sdk_dtype.ImuData):
            self._imu_data = imu_data
        self.imu_data_callback = imu_data_callback

        def robot_state_callback(robot_state: sdk_dtype.RobotState):
            self._robot_state = robot_state
        self.robot_state_callback = robot_state_callback

        def sensor_joy_callback(sensor_joy: sdk_dtype.SensorJoy):
            self._sensor_joy = sensor_joy
        self.sensor_joy_callback = sensor_joy_callback

        def robot_diagnostic_callback(diagnostic_value: sdk_dtype.DiagnosticValue):
            self._diagnostic_value = diagnostic_value
        self.robot_diagnostic_callback = robot_diagnostic_callback

        self.robot.subscribeImuData(self.imu_data_callback)
        self.robot.subscribeRobotState(self.robot_state_callback)
        self.robot.subscribeSensorJoy(self.sensor_joy_callback)
        self.robot.subscribeDiagnosticValue(self.robot_diagnostic_callback)


    def _apply_q_map(self, q_arr: List[float], inv: bool = False) -> List[float]:
        return (
            [q_arr[self.q_map[i]] for i in range(self.n_qdim)]
            if not inv else
            [q_arr[self.q_map_inv[i]] for i in range(self.n_qdim)]
        )


    def _float_tensor(self, data):
        return torch.tensor(data, dtype=torch.float32, device=self.device)


    def run(self):
        # logging
        time.sleep(1.0)
        print(f'[controller-log] main loop start\n')

        # TODO
        self.robot.setRobotLightEffect(sdk_dtype.LightEffect.STATIC_WHITE)

        # buffers
        qpos_trg = self._float_tensor([0.0] * self.n_qdim)

        _robot_cmd = sdk_dtype.RobotCmd()
        _robot_cmd.mode = [0.0 for _ in range(self.n_qdim)]
        _robot_cmd.q = [0.0 for _ in range(self.n_qdim)]
        _robot_cmd.dq = [0.0 for _ in range(self.n_qdim)]
        _robot_cmd.tau = [0.0 for _ in range(self.n_qdim)]
        _robot_cmd.Kp = self._apply_q_map(self.cfg.q_kp)
        _robot_cmd.Kd = self._apply_q_map(self.cfg.q_kd)

        # main loop
        rate = sdk_robot.Rate(self.cfg.loop_freq)
        step_num = 0

        total_dt = 0.0
        step_util = 0.0
        sim_util = 0.0
        policy_util = 0.0

        while True:
            step_num += 1
            step_ns = time.perf_counter_ns()

            # fix data read from callbacks
            imu_data = copy.deepcopy(self._imu_data)
            robot_state = copy.deepcopy(self._robot_state)
            sensor_joy = copy.deepcopy(self._sensor_joy)
            diagnostic_value = copy.deepcopy(self._diagnostic_value)

            # robot data tensors
            quat = self._float_tensor(imu_data.quat)
            angvel = self._float_tensor(imu_data.gyro)
            qpos = self._float_tensor(self._apply_q_map(robot_state.q, inv=True))
            qvel = self._float_tensor(self._apply_q_map(robot_state.dq, inv=True))
            usr_cmd = self._float_tensor([
                sensor_joy.axes[1], # linear-x
                sensor_joy.axes[0], # linear-y
                sensor_joy.axes[2], # angular-z
            ])

            # simulation step
            s = time.perf_counter_ns()
            self.policy_runner.sim_step(quat, angvel, qpos, qvel)
            sim_dt = (time.perf_counter_ns() - s) / 1e9

            # policy step
            if step_num % self.decimation == 0:
                s = time.perf_counter_ns()
                qpos_trg = self.policy_runner.policy_step(usr_cmd)
                policy_dt = (time.perf_counter_ns() - s) / 1e9

            # write action to robot
            _robot_cmd.q = self._apply_q_map(qpos_trg.tolist())
            self.robot.publishRobotCmd(_robot_cmd)

            # logging
            if step_num % self.cfg.logging_steps == 1:
                print(
                    f'[controller-log]\n'
                    f'step-num: {step_num}\n'
                    f'joy-axes: {sensor_joy.axes}\n'
                    f'joy-buttons: {sensor_joy.buttons}\n'
                    f'motor-names: {robot_state.motor_names}\n'
                    f'total_util: {total_dt * self.cfg.loop_freq:.4f}\n'
                    f'step_util: {step_util:.3f}, sim_util: {sim_util:.3f}, policy_util: {policy_util:.3f}\n'
                )

            # compute time related variables
            step_dt = (time.perf_counter_ns() - step_ns) / 1e9
            rate.sleep()
            total_dt = (time.perf_counter_ns() - step_ns) / 1e9

            step_util = step_dt / total_dt
            sim_util = sim_dt / total_dt
            if step_num % self.decimation == 0:
                policy_util = policy_dt / total_dt



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

    # controller configuration
    controller_cfg = ControllerCfg(
        robot_ip='127.0.0.1',
        # robot_ip='10.192.1.2',

        loop_freq=500,
        logging_steps=100, # 500 / 100 = 5Hz

        ref_q_names=SDK_Q_NAMES,
        q_names=USD_Q_NAMES,

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
            model_path='models/tron1_0_s_rough.onnx',
        ),
    )

    # setup environment variable (limxsdk-lowlevel requires this)
    import os
    os.environ['ROBOT_TYPE'] = 'SF_TRON1B'

    # run controller
    controller = Controller(controller_cfg)
    controller.run()
