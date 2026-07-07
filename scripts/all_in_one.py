import os

# offscreen rendering backend (set before mujoco creates any gl context).
# 'egl' for headless gpu servers, 'osmesa' for cpu-only. override via env var.
os.environ.setdefault('MUJOCO_GL', 'egl')

import time

import mujoco as mj
import torch as th

from robofsm.fsm import BaseNode
from robofsm.bipedal import (
    RobotState,
    HardStopNode,
    SoftStopNode,
    RLPolicyNode,
    RLPolicy,
    RLPolicyCfg,
)
from web_stream import WebStreamServer, WebStreamCfg, SliderCfg, ToggleCfg



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

LOOP_FREQ = 300
POLICY_FREQ = 50
REND_FREQ = 50
LOG_FREQ = 10

MJCF_PATH = 'assets/Tron1-S/robot.xml'



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
        kd_def=tensor([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.8, 0.8]),
    )

    # build: rl-policy
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
        cmd_rng=[
            (-1.5, 1.5), # linvel-x
            (-1.0, 1.0), # linvel-y
            (-1.0, 1.0), # angvel-z
            (0.8, 1.6), # gait freq
            (0.4, 0.6), # gait ratio
            (th.pi/2, th.pi/2), # gait offset
        ],
        # model_path='models/tron1_s_flat.onnx',
        model_path='models/tron1_s_rough.onnx',
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
        fn=lambda: btns['rl-run'] and soft_stop_node.t > 5.0,
    )

    rl_policy_node.add_edge(
        id='emergency hard-stop', ord=0, next=hard_stop_node,
        fn=lambda: btns['hard-stop'],
    )
    rl_policy_node.add_edge(
        id='emergency soft-stop', ord=1, next=soft_stop_node,
        fn=lambda: btns['soft-stop'],
    )

    return hard_stop_node, cmd, rl_cmd



def run_robofsm_graph(
        root_node: BaseNode[RobotState],
        cmd: tuple[dict, dict],
        rl_cmd: tuple[th.Tensor, th.Tensor],
        _axes_callback,
        _btns_callback,
        _render_callback,
    ):
    node = root_node
    s = node.state

    cmd_axes, cmd_btns = cmd
    rl_cmd_axes, rl_cmd_btns = rl_cmd

    rend_decimation = LOOP_FREQ // REND_FREQ
    log_decimation = LOOP_FREQ // LOG_FREQ

    # init mujoco
    mj_model = mj.MjModel.from_xml_path(MJCF_PATH)
    mj_model.opt.timestep = 1.0 / LOOP_FREQ
    mj_data = mj.MjData(mj_model)

    # preprocess: sensor data indices
    sensor_names = ['quat', 'linvel', 'linacc', 'angvel']
    sensor_ids = [mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SENSOR, x) for x in sensor_names]
    sensor_slices = [
        slice(mj_model.sensor_adr[x], mj_model.sensor_adr[x] + mj_model.sensor_dim[x])
        for x in sensor_ids
    ]
    sensor_name2slice = {a: b for a, b in zip(sensor_names, sensor_slices)}

    # main loop
    step_num = 0

    step_dt = 0.0
    sim_dt = 0.0
    rend_dt = 0.0
    total_dt = 0.0

    total_utils_buff = [0.0] * LOOP_FREQ

    usr_axes = [0.5] * 6
    usr_btns = [False] * 4

    while True:
        step_ns = time.perf_counter_ns()
        step_num += 1

        if usr_btns[3]:
            mj_data = mj.MjData(mj_model)
        # step mujoco simulator
        mj.mj_step(mj_model, mj_data)

        # tensors
        s.quat_w = th.from_numpy(mj_data.sensordata[sensor_name2slice['quat']])
        s.linvel_b = th.from_numpy(mj_data.sensordata[sensor_name2slice['linvel']])
        s.angvel_b = th.from_numpy(mj_data.sensordata[sensor_name2slice['angvel']])
        s.qpos = th.from_numpy(mj_data.qpos[7:])
        s.qvel = th.from_numpy(mj_data.qvel[6:])

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
        mj_data.ctrl[:] = qtau.numpy()

        # render-rate hooks: pull user-command, push rendered frame
        if step_num % rend_decimation == 0:
            t_ns = time.perf_counter_ns()
            usr_axes = _axes_callback()
            usr_btns = _btns_callback()
            _render_callback(mj_model, mj_data)
            rend_dt = (time.perf_counter_ns() - t_ns) / 1e9

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
                # f'sim-util: {sim_dt * LOOP_FREQ:.4f} | '
                # f'rend-util: {rend_dt * LOOP_FREQ:.4f}\n'

                f'quat: {" | ".join([f"{x:6.3f}" for x in s.quat_w])}\n'
                f'lin : {" | ".join([f"{x:6.3f}" for x in s.linvel_b])}\n'
                f'ang : {" | ".join([f"{x:6.3f}" for x in s.angvel_b])}\n'
                f'qpos: {" | ".join([f"{x:6.3f}" for x in s.qpos])}\n'
                f'qvel: {" | ".join([f"{x:6.3f}" for x in s.qvel])}\n'
                f'qtrg: {" | ".join([f"{x:6.3f}" for x in s.qpos_trg])}\n'
                f'kp  : {" | ".join([f"{x:6.3f}" for x in s.kp])}\n'
                f'kd  : {" | ".join([f"{x:6.3f}" for x in s.kd])}\n'
                f'qtau: {" | ".join([f"{x:6.3f}" for x in qtau])}\n'

                f'axes: {" | ".join([f"{x:6.3f}" for x in cmd_axes.values()])}\n'
                f'btns: {" | ".join([f"{x:6.3f}" for x in cmd_btns.values()])}\n'
            )

        # fix loop delta-time
        step_dt = (time.perf_counter_ns() - step_ns) / 1e9
        while ((time.perf_counter_ns() - step_ns) / 1e9) * LOOP_FREQ < 1.0:
            pass
        total_dt = (time.perf_counter_ns() - step_ns) / 1e9



if __name__ == '__main__':
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
                SliderCfg('linvel-x', 0.0, 1.0),
                SliderCfg('linvel-y', 0.0, 1.0),
                SliderCfg('angvel-z', 0.0, 1.0),
                SliderCfg('gait-freq', 0.0, 1.0),
                SliderCfg('gait-ratio', 0.0, 1.0),
                SliderCfg('gait-offset', 0.0, 1.0),
            ],
            toggles=[
                ToggleCfg('hard-stop'),
                ToggleCfg('soft-stop'),
                ToggleCfg('rl-run'),
                ToggleCfg('reset-sim'),
            ]
        )
    )

    run_robofsm_graph(
        *build_robofsm_graph(),
        web_stream.get_cmd,
        web_stream.get_toggles,
        web_stream.render_frame,
    )
