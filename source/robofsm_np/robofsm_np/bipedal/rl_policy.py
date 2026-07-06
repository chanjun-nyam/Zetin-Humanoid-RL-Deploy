from dataclasses import dataclass, MISSING
from typing import List, Tuple

import numpy as np
import onnxruntime as ort

from robofsm_np.utils.buffer import HistoryBuffer, SMABuffer
from robofsm_np.utils.math import quat_apply, quat_conj


@dataclass
class RLPolicyCfg:

    step_freq: int = MISSING

    policy_freq: int = MISSING

    device: str = MISSING

    q_names: List[str] = MISSING

    qpos_def: List[float] = MISSING

    n_history: int = MISSING

    obs_scale: List[float] = MISSING

    action_scale: List[float] = MISSING

    obs_clip: Tuple[float, float] = MISSING

    action_clip: Tuple[float, float] = MISSING

    vel_cmd_rng: List[Tuple[float, float]] = MISSING

    gait_cmd_rng: List[Tuple[float, float]] = MISSING

    model_path: str = MISSING


class RLPolicy:

    def __init__(self, cfg: RLPolicyCfg):
        self.cfg = cfg

        if cfg.step_freq % cfg.policy_freq != 0:
            raise ValueError('`step_freq` must be divisible by `policy_freq`.')
        self.decimation = cfg.step_freq // cfg.policy_freq

        self.step_idx = 0

        # initialize onnx model
        self._init_onnx()

        # initialize arrays/buffers
        self._init_buff()

    def _init_onnx(self):
        # onnx inference session options
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        opts.enable_cpu_mem_arena = True
        opts.enable_mem_pattern = True
        opts.enable_mem_reuse = True

        providers = ['CPUExecutionProvider']

        # make onnx inference session
        self.session = ort.InferenceSession(self.cfg.model_path, sess_options=opts, providers=providers)

        # log onnx model metadata
        print(
            '[{}: onnx input metadata]\n'.format(__name__) +
            ''.join([
                '{}, {}, {}\n'.format(inp.name, inp.shape, inp.type)
                for inp in self.session.get_inputs()
            ])
        )
        print(
            '[{}: onnx output metadata]\n'.format(__name__) +
            ''.join([
                '{}, {}, {}\n'.format(out.name, out.shape, out.type)
                for out in self.session.get_outputs()
            ])
        )

    def _init_buff(self):
        # dimension values
        n_qdim = len(self.cfg.q_names)
        n_obs_t = 6 + n_qdim * 3
        n_obs_priv = 3

        obs_dims = [3, 3, n_qdim, n_qdim, n_qdim, 3]

        array = self._array

        # robot data
        self.QUAT_IDENTITY = array([1, 0, 0, 0])
        self.GRAVITY_DIR_W = array([0, 0, -1])
        self.qpos_def = array(self.cfg.qpos_def)

        self.root_quat_w = self.QUAT_IDENTITY.copy()
        self.root_linvel_b = SMABuffer.init_like(array([0, 0, 0]), (0,), self.decimation)
        self.root_angvel_b = SMABuffer.init_like(array([0, 0, 0]), (0,), self.decimation)
        self.gravity_dir_b = self.GRAVITY_DIR_W.copy()
        self.qpos = array([0] * n_qdim)
        self.qvel = SMABuffer.init_like(self.qpos, (0,), self.decimation)

        # command
        self.vel_cmd_rng = array(self.cfg.vel_cmd_rng)
        self.gait_cmd_rng = array(self.cfg.gait_cmd_rng)

        self.is_walk = array(False, dtype=np.bool_)
        self.vel_cmd = array([0, 0, 0])
        self.gait_cmd = array([0, 0.5, np.pi])  # TODO

        # gait
        self.gait_clock = array(0)
        self.gait_clock_signal = array([0, 0, 0, 0])

        # others
        self.obs_scale = array(sum([[s] * obs_dims[i] for i, s in enumerate(self.cfg.obs_scale)], []))
        self.action_scale = array(self.cfg.action_scale)

        self.action = array([0] * n_qdim)
        self.obs_hist = HistoryBuffer.init_like(array([0] * n_obs_t), (0,), self.cfg.n_history)

    def _reset_buff(self):
        array = self._array

        # robot data
        self.root_quat_w[...] = self.QUAT_IDENTITY
        self.root_linvel_b.reset(())
        self.root_angvel_b.reset(())
        self.gravity_dir_b[...] = self.GRAVITY_DIR_W
        self.qpos.fill(0)
        self.qvel.reset(())

        # command
        self.is_walk.fill(0)
        self.vel_cmd.fill(0)
        self.gait_cmd[...] = array([0, 0.5, np.pi])  # TODO

        # gait
        self.gait_clock.fill(0)
        self.gait_clock_signal.fill(0)

        # others
        self.action.fill(0)
        self.obs_hist.reset(())

    def step(
        self,
        quat: np.ndarray,
        linvel: np.ndarray,
        angvel: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        is_walk: np.ndarray,
        usr_cmd: np.ndarray,
    ) -> np.ndarray:
        # update-buff: robot data
        self.root_quat_w[...] = quat
        self.root_linvel_b.update(linvel)
        self.root_angvel_b.update(angvel)
        self.gravity_dir_b[...] = quat_apply(quat_conj(self.root_quat_w), self.GRAVITY_DIR_W)
        self.qpos[...] = qpos
        self.qvel.update(qvel)

        # update-buff: command
        usr_cmd = np.clip(usr_cmd, 0.0, 1.0)

        self.is_walk[...] = is_walk
        self.vel_cmd[...] = (
            self.vel_cmd_rng[:, 0]
            + usr_cmd[0:3] * np.squeeze(np.diff(self.vel_cmd_rng, n=1, axis=-1), axis=1)
        )
        self.gait_cmd[...] = (
            self.gait_cmd_rng[:, 0]
            + usr_cmd[3:6] * np.squeeze(np.diff(self.gait_cmd_rng, n=1, axis=-1), axis=1)
        )

        # update-buff: gait
        # TODO
        self.gait_clock += (2.0 * np.pi * (1.0 / self.cfg.step_freq)) * self.gait_cmd[0]
        np.remainder(self.gait_clock, 2.0 * np.pi, out=self.gait_clock)

        gait_theta = np.tile(self.gait_clock, 2).copy()
        gait_theta[0] += 0.0
        gait_theta[1] += self.gait_cmd[2]
        self.gait_clock_signal[...] = np.concatenate([np.sin(gait_theta), np.cos(gait_theta)])

        # apply `is_walk`
        self.vel_cmd *= self.is_walk
        # self.gait_cmd *= self.is_walk
        self.gait_cmd[0] *= self.is_walk  # TODO
        self.gait_clock_signal *= self.is_walk

        # policy step
        if self.step_idx % self.decimation == self.decimation - 1:
            self._policy_step()

        # compute output: target qpos
        qpos_trg = self.qpos_def + self.action_scale * self.action

        self.step_idx = (self.step_idx + 1) % self.cfg.step_freq
        return qpos_trg

    def _policy_step(self):
        # compute observation array
        self.obs_hist.update(
            np.concatenate([
                self.root_angvel_b.sma,
                self.gravity_dir_b,
                self.qpos,
                self.qvel.sma,
                self.action,
            ], axis=-1) * self.obs_scale[:-3]
        )
        obs_priv = np.concatenate([
            self.root_linvel_b.sma,
        ], axis=-1) * self.obs_scale[-3:]

        obs_tensor = np.concatenate([
            self.obs_hist.buff.reshape(-1),
            self.vel_cmd,
            self.gait_cmd[0:2],
            self.gait_clock_signal,
            obs_priv,
        ], axis=-1)
        np.clip(obs_tensor, self.cfg.obs_clip[0], self.cfg.obs_clip[1], out=obs_tensor)
        obs_tensor[-3:] = 0.0  # TODO: implement linvel estimator

        # run onnx session
        session_input = {'input': np.expand_dims(obs_tensor, axis=0).astype(np.float32, copy=False)}
        session_output = self.session.run(['output'], session_input)

        # compute action array
        self.action[...] = np.squeeze(session_output[0], axis=0)
        np.clip(self.action, self.cfg.action_clip[0], self.cfg.action_clip[1], out=self.action)

    def reset(self):
        self._reset_buff()
        self.step_idx = 0

    def _array(self, data, **kwargs):
        opts = {'dtype': np.float32}
        opts.update(kwargs)
        return np.array(data, **opts)
