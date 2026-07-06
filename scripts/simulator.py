# Copyright information
#
# © [2024] LimX Dynamics Technology Co., Ltd. All rights reserved.
#
# Web-stream variant: renders the MuJoCo scene off-screen and serves it over
# HTTP (see web_stream.py) instead of opening a local mujoco.viewer window.
# The limxsdk robot I/O (command subscribe + state/IMU publish) is unchanged.

import os

# Off-screen rendering backend — MUST be set before mujoco creates a GL context.
# 'egl' for a headless GPU server, 'osmesa' for CPU-only. Override via env var.
os.environ.setdefault('MUJOCO_GL', 'egl')

import sys
import time
from functools import partial

import mujoco

import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes

# The web rendering module you provided.
from web_stream import WebStreamServer, WebStreamCfg, SliderCfg


class SimulatorMujoco:
    def __init__(self, asset_path, joint_sensor_names, robot, web_cfg=None,
                 quat_sensor_names=None, gyro_sensor_names=None, acc_sensor_names=None,
                 reset_keyframe=None):
        self.robot = robot
        self.joint_sensor_names = joint_sensor_names
        self.joint_num = len(joint_sensor_names)

        # Load the MuJoCo model and data from the specified XML asset path
        self.mujoco_model = mujoco.MjModel.from_xml_path(asset_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

        self.dt = self.mujoco_model.opt.timestep     # Simulation timestep
        self.fps = 1.0 / self.dt                     # Physics steps per second

        # IMU sensor-name candidates. Different MJCFs name the IMU sensors
        # differently (pointfoot: quat/gyro/acc; some humanoids: quat/angvel/
        # linacc). We try each candidate in order and use the first that exists.
        self.quat_sensor_names = quat_sensor_names or ['quat', 'orientation', 'imu_quat']
        self.gyro_sensor_names = gyro_sensor_names or ['gyro', 'angvel', 'imu_gyro', 'angular_velocity']
        self.acc_sensor_names  = acc_sensor_names  or ['acc', 'linacc', 'accelerometer', 'imu_acc', 'linear_acceleration']

        # Print what sensors the model actually has (handy when names mismatch).
        available = [mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                     for i in range(self.mujoco_model.nsensor)]
        print(f"[simulator] model sensors: {available}")

        # ---- Reset keyframe selection ----------------------------------------
        # If the MJCF defines keyframes, reset() puts the robot back into one
        # (a proper standing pose) instead of the bare zero-pose default. You
        # can force a specific keyframe by name via reset_keyframe.
        self.reset_key_id = -1
        if self.mujoco_model.nkey > 0:
            if reset_keyframe is not None:
                self.reset_key_id = mujoco.mj_name2id(
                    self.mujoco_model, mujoco.mjtObj.mjOBJ_KEY, reset_keyframe)
            if self.reset_key_id == -1:
                # try common names, else fall back to the first keyframe
                for name in ['home', 'stand', 'default', 'init']:
                    kid = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_KEY, name)
                    if kid != -1:
                        self.reset_key_id = kid
                        break
                if self.reset_key_id == -1:
                    self.reset_key_id = 0
        key_desc = (f"keyframe #{self.reset_key_id}"
                    if self.reset_key_id >= 0 else "zero-pose (mj_resetData)")
        print(f"[simulator] reset target: {key_desc}")

        # ---- Web stream (replaces the passive mujoco.viewer) -----------------
        # WebStreamServer starts an HTTP server in a background thread and
        # exposes render_frame()/get_cmd()/consume_reset(). We render from THIS
        # (sim) thread, because the GL context created inside render_frame is
        # thread-local.
        self.web_cfg = web_cfg if web_cfg is not None else WebStreamCfg()
        self.web_stream = WebStreamServer(self.web_cfg)

        # Render at the configured stream frequency, not every physics step
        # (off-screen render + JPEG encode is far too slow to run every step).
        self.rend_decimation = max(1, int(round(self.fps / self.web_cfg.stream_freq)))

        # Latest user-command vector from the web sliders (see note in run()).
        self.user_cmd = self.web_stream.get_cmd()

        # Initialize robot command data with default values
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for _ in range(self.joint_num)]
        self.robot_cmd.q = [0. for _ in range(self.joint_num)]
        self.robot_cmd.dq = [0. for _ in range(self.joint_num)]
        self.robot_cmd.tau = [0. for _ in range(self.joint_num)]
        self.robot_cmd.Kp = [0. for _ in range(self.joint_num)]
        self.robot_cmd.Kd = [0. for _ in range(self.joint_num)]

        # Initialize robot state data with default values
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for _ in range(self.joint_num)]
        self.robot_state.q = [0. for _ in range(self.joint_num)]
        self.robot_state.dq = [0. for _ in range(self.joint_num)]

        # Initialize IMU data structure
        self.imu_data = datatypes.ImuData()

        # Set up callback for receiving robot commands in simulation mode
        self.robotCmdCallbackPartial = partial(self.robotCmdCallback)
        self.robot.subscribeRobotCmdForSim(self.robotCmdCallbackPartial)

    # Callback function for receiving robot command data
    def robotCmdCallback(self, robot_cmd: datatypes.RobotCmd):
        self.robot_cmd = robot_cmd

    def _resolve_sensor(self, candidates):
        """Return (name, id, adr, dim) of the first candidate that exists, else (None, -1, -1, 0)."""
        for name in candidates:
            sid = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid != -1:
                return name, sid, int(self.mujoco_model.sensor_adr[sid]), int(self.mujoco_model.sensor_dim[sid])
        return None, -1, -1, 0

    def reset(self):
        """Restore the simulation to its initial state (keyframe or zero-pose)."""
        if self.reset_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.mujoco_model, self.mujoco_data, self.reset_key_id)
        else:
            mujoco.mj_resetData(self.mujoco_model, self.mujoco_data)
        # clear actuator commands so the last PD target doesn't fling the robot
        self.mujoco_data.ctrl[:] = 0.0
        # recompute derived quantities (sensordata, etc.) for the new state
        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
        print("[simulator] reset")

    def run(self):
        frame_count = 0
        self.rate = Rate(self.fps)  # Maintain the loop at the physics rate

        model = self.mujoco_model
        data = self.mujoco_data

        # Resolve IMU sensors ONCE, validating each one. If a name is missing,
        # mj_name2id returns -1 and sensor_adr[-1] would silently point at the
        # last sensor -> out-of-bounds reads. So we fail loudly instead.
        quat_name, quat_id, quat_adr, quat_dim = self._resolve_sensor(self.quat_sensor_names)
        gyro_name, gyro_id, gyro_adr, gyro_dim = self._resolve_sensor(self.gyro_sensor_names)
        acc_name,  acc_id,  acc_adr,  acc_dim  = self._resolve_sensor(self.acc_sensor_names)

        missing = []
        if quat_id == -1: missing.append(f"quat (tried {self.quat_sensor_names})")
        if gyro_id == -1: missing.append(f"gyro (tried {self.gyro_sensor_names})")
        if acc_id  == -1: missing.append(f"acc  (tried {self.acc_sensor_names})")
        if missing:
            available = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                         for i in range(model.nsensor)]
            raise RuntimeError(
                "IMU sensor(s) not found: " + "; ".join(missing) + ".\n"
                f"Available sensors in this model: {available}.\n"
                "Pass the correct names via quat_sensor_names / gyro_sensor_names / acc_sensor_names."
            )

        print(f"[simulator] IMU mapping -> quat:'{quat_name}'({quat_dim}) "
              f"gyro:'{gyro_name}'({gyro_dim}) acc:'{acc_name}'({acc_dim})")

        try:
            while True:
                # Handle a reset requested from the web UI before stepping.
                if self.web_stream.consume_reset():
                    self.reset()

                # Step the MuJoCo physics simulation
                mujoco.mj_step(model, data)

                # Update robot state and apply PD control from the robot command
                for i in range(self.joint_num):
                    self.robot_state.q[i] = data.qpos[i + 7]
                    self.robot_state.dq[i] = data.qvel[i + 6]
                    self.robot_state.tau[i] = data.ctrl[i]

                    data.ctrl[i] = (
                        self.robot_cmd.Kp[i] * (self.robot_cmd.q[i] - self.robot_state.q[i]) +
                        self.robot_cmd.Kd[i] * (self.robot_cmd.dq[i] - self.robot_state.dq[i]) +
                        self.robot_cmd.tau[i]
                    )

                # Timestamp + publish robot state
                self.robot_state.stamp = time.time_ns()
                self.robot.publishRobotStateForSim(self.robot_state)

                # Extract IMU data (orientation, gyro, acceleration)
                self.imu_data.quat[0] = data.sensordata[quat_adr + 0]
                self.imu_data.quat[1] = data.sensordata[quat_adr + 1]
                self.imu_data.quat[2] = data.sensordata[quat_adr + 2]
                self.imu_data.quat[3] = data.sensordata[quat_adr + 3]

                self.imu_data.gyro[0] = data.sensordata[gyro_adr + 0]
                self.imu_data.gyro[1] = data.sensordata[gyro_adr + 1]
                self.imu_data.gyro[2] = data.sensordata[gyro_adr + 2]

                self.imu_data.acc[0] = data.sensordata[acc_adr + 0]
                self.imu_data.acc[1] = data.sensordata[acc_adr + 1]
                self.imu_data.acc[2] = data.sensordata[acc_adr + 2]

                # Timestamp + publish IMU data
                self.imu_data.stamp = time.time_ns()
                self.robot.publishImuDataForSim(self.imu_data)

                # ---- Render via the web module (replaces viewer.sync) --------
                if frame_count % self.rend_decimation == 0:
                    # Pull the latest slider values. In this SDK-driven sim the
                    # actual joint control comes from the external robot process
                    # (robotCmdCallback), so these sliders are NOT used for
                    # control by default — they are just available as a user
                    # command channel. Forward them to the robot here if you
                    # want a virtual joystick (SDK-specific call).
                    self.user_cmd = self.web_stream.get_cmd()

                    # Push a freshly rendered frame to all connected browsers.
                    self.web_stream.render_frame(model, data)

                frame_count += 1
                self.rate.sleep()  # Keep the loop at the physics rate
        except KeyboardInterrupt:
            print("\n[simulator] stopped.")


if __name__ == '__main__':
    os.environ['ROBOT_TYPE'] = 'SF_TRON1B'
    # Create a Robot instance of the PointFoot type
    robot = Robot(RobotType.PointFoot, True)

    # Default IP address for the robot
    robot_ip = "127.0.0.1"

    # Initialize the robot with the provided IP address
    if not robot.init(robot_ip):
        sys.exit()

    # Robot model XML path based on the robot type (same layout as before)
    model_path = 'assets/Tron1-S/robot.xml'

    # Joint sensor names by robot family (unchanged)
    if False:
        joint_sensor_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "wheel_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "wheel_R_Joint",
        ]
    elif True:
        joint_sensor_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint", "ankle_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint", "ankle_R_Joint",
        ]
    else:
        joint_sensor_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint",
        ]

    # Web stream configuration (open http://<host>:8000 in a browser).
    # camera='' -> free camera. Set camera='track' only if robot.xml defines it.
    web_cfg = WebStreamCfg(
        host='0.0.0.0',
        port=8000,
        width=640,
        height=480,
        camera='',
        jpeg_quality=80,
        stream_freq=50,
        sliders=[
            SliderCfg('vx', -1.5, 1.5),
            SliderCfg('vy', -1.0, 1.0),
            SliderCfg('wz', -1.0, 1.0),
        ],
    )

    # Create and run the web-streamed MuJoCo simulator.
    # If your model names the IMU sensors differently, override here, e.g.:
    #   gyro_sensor_names=['angvel'], acc_sensor_names=['linacc']
    # To reset to a specific keyframe: reset_keyframe='home'
    simulator = SimulatorMujoco(model_path, joint_sensor_names, robot, web_cfg)
    simulator.run()
