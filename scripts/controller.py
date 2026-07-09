import time
import copy
import math

import torch as th

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
from robofsm.utils.math import quat_apply, vec_dot



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

LOOP_FREQ = 500
POLICY_FREQ = 50
LOG_FREQ = 10

MODEL_PATH = 'models/tron1_s_flat.onnx'



def build_robofsm_graph():
    # build: robot-state
    def tensor(data, dtype=th.float32, device=th.device('cpu')):
        return th.tensor(data, dtype=dtype, device=device)

    robot_state = RobotState(
        n_qdim=len(REF_Q_NAMES),
        q_names=REF_Q_NAMES,

        quat_w=tensor([1, 0, 0, 0]),
        linvel_b=tensor([0, 0, 0]),
        angvel_b=tensor([0, 0, 0]),

        qpos=tensor([0] * 8),
        qvel=tensor([0] * 8),
        qpos_trg=tensor([0] * 8),

        kp=tensor([0] * 8),
        kd=tensor([0] * 8),

        qpos_def=tensor([0] * 8),
        kp_def=tensor([45., 45., 45., 45., 45., 45., 45., 45.]),
        kd_def=tensor([1.5, 1.5, 1.5, 0.8, 1.5, 1.5, 1.5, 0.8]),
    )

    # build: rl-policy
    rl_policy = RLPolicy(RLPolicyCfg(
        step_freq=LOOP_FREQ,
        policy_freq=POLICY_FREQ,
        device='cpu',
        q_names=RL_Q_NAMES,
        qpos_def=robot_state.qpos_def.tolist(),
        n_history=10,
        action_scale=[0.5] * 8,
        obs_clip=(-1e6, 1e6),
        action_clip=(-1e6, 1e6),
        cmd_rng=[
            (-1.5, 1.5), # linvel-x
            (-1.0, 1.0), # linvel-y
            (-1.0, 1.0), # angvel-z
            (0.8, 1.6), # gait freq
            (0.5, 0.5), # gait ratio
            (th.pi, th.pi), # gait offset
        ],
        max_stride=1.2,
        model_path=MODEL_PATH,
    ))

    # build: fsm nodes
    hard_stop_node = HardStopNode(robot_state)
    soft_stop_node = SoftStopNode(robot_state, duration=3.0)
    rl_policy_node = RLPolicyNode(robot_state, rl_policy=rl_policy, duration=1e-6)

    # edge triggers
    axes = {
        'linvel-x': 0.5,
        'linvel-y': 0.5,
        'angvel-z': 0.5,
        'gait-freq': 0.5,
        'gait-ratio': 0.5,
        'gait-offset': 0.5,
    }
    btns = {
        'hard-stop': False,
        'soft-stop': False,
        'rl-run': False,
    }
    cmd = (axes, btns)

    rl_cmd = (
        th.tensor([0.5] * 6, dtype=th.float32, device='cpu'),
        th.tensor([False], dtype=th.bool, device='cpu'),
    )
    rl_policy_node.register_cmd(*rl_cmd)

    # build: fsm edges
    hard_stop_node.add_edge(
        id='soft-stop', ord=0, next=soft_stop_node,
        fn=lambda: btns['soft-stop'] and hard_stop_node.t > 1.0,
    )

    soft_stop_node.add_edge(
        id='emergency hard-stop', ord=0, next=hard_stop_node,
        fn=lambda: btns['hard-stop'],
    )
    soft_stop_node.add_edge(
        id='run rl-policy', ord=1, next=rl_policy_node,
        fn=lambda: btns['rl-run'] and soft_stop_node.t > 3.0,
    )

    rl_policy_node.add_edge(
        id='emergency hard-stop', ord=0, next=hard_stop_node,
        fn=lambda: btns['hard-stop'],
    )
    rl_policy_node.add_edge(
        id='emergency soft-stop', ord=1, next=soft_stop_node,
        fn=lambda: btns['soft-stop'],
    )
    rl_policy_node.add_edge(
        id='tilt soft-stop', ord=2, next=soft_stop_node,
        fn=lambda: float(vec_dot(
            quat_apply(soft_stop_node.state.quat_w, tensor([0, 0, 1])), tensor([0, 0, 1])
        ).item()) < math.cos(math.radians(60.0))
    )

    return hard_stop_node, cmd, rl_cmd



def run_robofsm_graph(
        root_node: BaseNode[RobotState],
        cmd: tuple[dict, dict],
        rl_cmd: tuple[th.Tensor, th.Tensor],
        robot_ip: str,
    ):
    node = root_node
    s = node.state

    cmd_axes, cmd_btns = cmd
    rl_cmd_axes, rl_cmd_btns = rl_cmd

    log_decimation = LOOP_FREQ // LOG_FREQ

    # init robot
    robot = sdk_robot.Robot(
        robot_type=sdk_robot.RobotType.PointFoot,
        is_sim=False,
    )
    if not robot.init(robot_ip):
        print(f'[{__name__}] robot initialization failed')
        exit()

    # logging metadata
    print(
        f'[{__name__}]\n'
        f'motor number: {robot.getMotorNumber()}\n'
    )

    # buffers
    _imu_data = sdk_dtype.ImuData()
    _robot_state = sdk_dtype.RobotState()
    _sensor_joy = sdk_dtype.SensorJoy()
    _diagnostic_value = sdk_dtype.DiagnosticValue()
    _imu_data_new = False
    _robot_state_new = False
    _sensor_joy_new = False
    _diagnostic_value_new = False

    # subscribe callbacks
    def imu_data_callback(x: sdk_dtype.ImuData):
        nonlocal _imu_data, _imu_data_new
        _imu_data = x
        _imu_data_new = True
    def robot_state_callback(x: sdk_dtype.RobotState):
        nonlocal _robot_state, _robot_state_new
        _robot_state = x
        _robot_state_new = True
    def sensor_joy_callback(x: sdk_dtype.SensorJoy):
        nonlocal _sensor_joy, _sensor_joy_new
        _sensor_joy = x
        _sensor_joy_new = True
    def robot_diagnostic_callback(x: sdk_dtype.DiagnosticValue):
        nonlocal _diagnostic_value, _diagnostic_value_new
        _diagnostic_value = x
        _diagnostic_value_new = True

    robot.subscribeImuData(imu_data_callback)
    robot.subscribeRobotState(robot_state_callback)
    robot.subscribeSensorJoy(sensor_joy_callback)
    robot.subscribeDiagnosticValue(robot_diagnostic_callback)

    # main loop
    rate = sdk_robot.Rate(LOOP_FREQ)
    step_num = 0

    step_dt = 0.0
    sim_dt = 0.0
    total_dt = 0.0

    total_utils_buff = [0.0] * LOOP_FREQ

    usr_axes = [0.5] * 6
    usr_btns = [False] * 4

    robot_cmd = sdk_dtype.RobotCmd()
    robot_cmd.mode = [0] * 8
    robot_cmd.q = [0.] * 8
    robot_cmd.dq = [0.] * 8
    robot_cmd.tau = [0.] * 8
    robot_cmd.Kp = [0.] * 8
    robot_cmd.Kd = [0.] * 8

    def tensor(data, dtype=th.float32, device=th.device('cpu')):
        return th.tensor(data, dtype=dtype, device=device)
    def norm_axis(a):
        return min(1.0, max(0.0, 0.5 + 0.5 * float(a)))

    _qtau = tensor([0.] * 8)

    while True:
        step_ns = time.perf_counter_ns()
        step_num += 1

        # latch data read from callbacks
        imu_data = copy.deepcopy(_imu_data)
        robot_state = copy.deepcopy(_robot_state)
        sensor_joy = copy.deepcopy(_sensor_joy)
        diagnostic_value = copy.deepcopy(_diagnostic_value)

        imu_data_new = bool(_imu_data_new)
        _imu_data_new = False
        robot_state_new = bool(_robot_state_new)
        _robot_state_new = False
        sensor_joy_new = bool(_sensor_joy_new)
        _sensor_joy_new = False
        diagnostic_value_new = bool(_diagnostic_value_new)
        _diagnostic_value_new = False

        # tensors
        if imu_data_new:
            s.quat_w = tensor(imu_data.quat)
            s.angvel_b = tensor(imu_data.gyro)
            s.linvel_b = th.zeros_like(s.angvel_b)
        if robot_state_new:
            s.qpos = tensor(robot_state.q)
            s.qvel = tensor(robot_state.dq)
            _qtau = tensor(robot_state.tau)
        if sensor_joy_new:
            usr_axes[0] = norm_axis(sensor_joy.axes[3])
            usr_axes[1] = norm_axis(sensor_joy.axes[2])
            usr_axes[2] = norm_axis(sensor_joy.axes[0])
            usr_axes[3] = norm_axis(sensor_joy.axes[1])
            usr_axes[4] = 0.5
            usr_axes[5] = 0.5
            usr_btns[0] = sensor_joy.buttons[14]
            usr_btns[1] = sensor_joy.buttons[12]
            usr_btns[2] = sensor_joy.buttons[15]

        # command
        cmd_axes = {
            'linvel-x': 0.5,
            'linvel-y': 0.5,
            'angvel-z': 0.5,
            'gait-freq': 0.5,
            'gait-ratio': 0.5,
            'gait-offset': 0.5,
        }
        cmd_axes['linvel-x'] = usr_axes[0]
        cmd_axes['linvel-y'] = usr_axes[1]
        cmd_axes['angvel-z'] = usr_axes[2]
        cmd_axes['gait-freq'] = usr_axes[3]
        cmd_axes['gait-ratio'] = usr_axes[4]
        cmd_axes['gait-offset'] = usr_axes[5]
        cmd_btns['hard-stop'] = usr_btns[0]
        cmd_btns['soft-stop'] = usr_btns[1]
        cmd_btns['rl-run'] = usr_btns[2]

        rl_cmd_axes[0] = cmd_axes['linvel-x']
        rl_cmd_axes[1] = cmd_axes['linvel-y']
        rl_cmd_axes[2] = cmd_axes['angvel-z']
        rl_cmd_axes[3] = cmd_axes['gait-freq']
        rl_cmd_axes[4] = cmd_axes['gait-ratio']
        rl_cmd_axes[5] = cmd_axes['gait-offset']
        rl_cmd_btns[0] = (rl_cmd_axes[0:3] - 0.5).square().sum().sqrt() > 0.1

        # step policy-runner
        t_ns = time.perf_counter_ns()
        node = node.update()
        sim_dt = (time.perf_counter_ns() - t_ns) / 1e9

        # write action to robot
        qtau = s.kp * (s.qpos_trg - s.qpos) - s.kd * s.qvel

        robot_cmd.q = s.qpos_trg.tolist()
        robot_cmd.Kp = s.kp.tolist()
        robot_cmd.Kd = s.kd.tolist()
        robot.publishRobotCmd(robot_cmd)

        if step_num % LOOP_FREQ == 0:
            robot.setRobotLightEffect(sdk_dtype.LightEffect.STATIC_WHITE)
        elif step_num % LOOP_FREQ == LOOP_FREQ // 2:
            robot.setRobotLightEffect(sdk_dtype.LightEffect.STATIC_GREEN)

        # monitoring
        total_utils_buff[step_num % LOOP_FREQ] = total_dt * LOOP_FREQ

        # logging
        if step_num % log_decimation == 0:
            print(
                f'[{__name__}]\n'
                f'step-num: {step_num}\n'

                f'total-util-mean: {sum(total_utils_buff) / LOOP_FREQ:.4f} | '
                f'total-util-max: {max(total_utils_buff):.4f}\n'

                # f'total-util: {total_dt * LOOP_FREQ:.4f} | '
                # f'step-util: {step_dt * LOOP_FREQ:.4f} | '
                # f'sim-util: {sim_dt * LOOP_FREQ:.4f}\n'

                f'quat: {" | ".join([f"{x:6.3f}" for x in s.quat_w])}\n'
                f'lin : {" | ".join([f"{x:6.3f}" for x in s.linvel_b])}\n'
                f'ang : {" | ".join([f"{x:6.3f}" for x in s.angvel_b])}\n'
                f'qpos: {" | ".join([f"{x:6.3f}" for x in s.qpos])}\n'
                f'qvel: {" | ".join([f"{x:6.3f}" for x in s.qvel])}\n'
                f'qtrg: {" | ".join([f"{x:6.3f}" for x in s.qpos_trg])}\n'
                f'kp  : {" | ".join([f"{x:6.3f}" for x in s.kp])}\n'
                f'kd  : {" | ".join([f"{x:6.3f}" for x in s.kd])}\n'
                f'qtau: {" | ".join([f"{x:6.3f}" for x in qtau])}\n'
                f'qtau: {" | ".join([f"{x:6.3f}" for x in _qtau])}\n'

                f'axes: {" | ".join([f"{x:6.3f}" for x in cmd_axes.values()])}\n'
                f'btns: {" | ".join([f"{x:6.3f}" for x in cmd_btns.values()])}\n'
            )

        # fix loop delta-time
        step_dt = (time.perf_counter_ns() - step_ns) / 1e9
        rate.sleep()
        total_dt = (time.perf_counter_ns() - step_ns) / 1e9



if __name__ == '__main__':
    import os
    os.environ['ROBOT_TYPE'] = 'SF_TRON1B'

    run_robofsm_graph(
        *build_robofsm_graph(),
        robot_ip='127.0.0.1',
        # robot_ip='10.192.1.2',
    )
