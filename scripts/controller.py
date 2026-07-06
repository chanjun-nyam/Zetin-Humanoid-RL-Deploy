from dataclasses import dataclass, MISSING
from typing import List
import time
import copy

import numpy as np

import limxsdk.robot as sdk_robot
import limxsdk.datatypes as sdk_dtype

from robofsm.fsm import BaseNode
from robofsm.bipedal import (
    RobotState,
    HardStopNode,
    SoftStopNode,
    RLPolicyNode,
    RLPolicy,
    RLPolicyCfg,
)


@dataclass
class ControllerCfg:

    robot_ip: str = MISSING

    loop_freq: int = MISSING

    log_freq: int = MISSING


class Controller:

    def __init__(self, cfg: ControllerCfg):
        self.cfg = cfg
        self.log_decimation = cfg.loop_freq // cfg.log_freq

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
        self._robot_state.q = [0.0] * 8
        self._robot_state.dq = [0.0] * 8
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

    def run(self, node: BaseNode[RobotState]):
        s = node.state

        time.sleep(1.0)
        print(f'[limxsdk-lowlevel] run()!!\n')

        def array(data, dtype=np.float32):
            return np.array(data, dtype=dtype)

        # buffers
        _robot_cmd = sdk_dtype.RobotCmd()
        _robot_cmd.mode = [0.0] * 8
        _robot_cmd.q = [0.0] * 8
        _robot_cmd.dq = [0.0] * 8
        _robot_cmd.tau = [0.0] * 8
        _robot_cmd.Kp = [0.0] * 8
        _robot_cmd.Kd = [0.0] * 8

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

            usr_cmd = [0.5] * 7
            # TODO

            # arrays
            s.quat_w = array(imu_data.quat)
            s.angvel_b = array(imu_data.gyro)
            s.linvel_b = np.zeros_like(s.angvel_b)
            s.qpos = array(robot_state.q)
            s.qvel = array(robot_state.dq)

            # step policy-runner
            t_ns = time.perf_counter_ns()
            node = node.update()
            loop_dt = (time.perf_counter_ns() - t_ns) / 1e9

            # user cmd
            if isinstance(node, RLPolicyNode):
                node.set_cmd(array(usr_cmd[:6]), array(usr_cmd[6] > 0.6, dtype=np.bool_))

            # write action to robot
            _robot_cmd.q = s.qpos_trg.tolist()
            _robot_cmd.Kp = s.kp.tolist()
            _robot_cmd.Kd = s.kd.tolist()
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

                    f'quat: {" | ".join([f"{x:6.3f}" for x in s.quat_w])}\n'
                    f'lvel: {" | ".join([f"{x:6.3f}" for x in s.linvel_b])}\n'
                    f'avel: {" | ".join([f"{x:6.3f}" for x in s.angvel_b])}\n'
                    f'qpos: {" | ".join([f"{x:6.3f}" for x in s.qpos])}\n'
                    f'qvel: {" | ".join([f"{x:6.3f}" for x in s.qvel])}\n'
                    f'qtrg: {" | ".join([f"{x:6.3f}" for x in s.qpos_trg])}\n'
                    f'ucmd: {" | ".join([f"{x:6.3f}" for x in usr_cmd])}\n'
                )

            # fix loop delta-time
            step_dt = (time.perf_counter_ns() - step_ns) / 1e9
            sdk_rate.sleep()
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

    LOOP_FREQ = 400
    POLICY_FREQ = 50
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
        step_freq=LOOP_FREQ,
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
    rl_policy_node = RLPolicyNode(robot_state, rl_policy=rl_policy, duration=5.0)

    # configure edge
    # hard_stop_node.add_edge()

    # controller configuration
    controller_cfg = ControllerCfg(
        robot_ip='127.0.0.1',
        # robot_ip='10.192.1.2',
        loop_freq=LOOP_FREQ,
        log_freq=LOG_FREQ,
    )

    # setup environment variable (limxsdk-lowlevel requires this)
    import os
    os.environ['ROBOT_TYPE'] = 'SF_TRON1B'

    # run controller
    controller = Controller(controller_cfg)
    controller.run(hard_stop_node)
    # controller.run(soft_stop_node)
    # controller.run(rl_policy_node)
