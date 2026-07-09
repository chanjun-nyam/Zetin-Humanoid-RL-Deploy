from dataclasses import dataclass, MISSING
from typing import List, Tuple

import torch as th
import onnxruntime as ort

from robofsm.utils.buffer import HistoryBuffer, SMABuffer
from robofsm.utils.math import vec_norm, quat_apply, quat_conj



@dataclass
class RLPolicyCfg:

    step_freq: int = MISSING

    policy_freq: int = MISSING

    device: str = MISSING

    q_names: List[str] = MISSING

    qpos_def: List[float] = MISSING

    n_history: int = MISSING

    action_scale: List[float] = MISSING

    obs_clip: Tuple[float, float] = MISSING

    action_clip: Tuple[float, float] = MISSING

    cmd_rng: List[Tuple[float, float]] = MISSING

    max_stride: float = MISSING

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

        # initialize tensors/buffers
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
            f'[{__name__}: onnx input metadata]\n' +
            ''.join([
                f'{inp.name}, {inp.shape}, {inp.type}\n'
                for inp in self.session.get_inputs()
            ])
        )
        print(
            f'[{__name__}: onnx output metadata]\n' +
            ''.join([
                f'{out.name}, {out.shape}, {out.type}\n'
                for out in self.session.get_outputs()
            ])
        )


    def _init_buff(self):
        # dimension values
        n_qdim = len(self.cfg.q_names)
        n_obs_t = 6 + n_qdim * 3
        n_obs_priv = 3

        tensor = self._tensor

        # robot data
        self.QUAT_IDENTITY = tensor([1, 0, 0, 0])
        self.GRAVITY_DIR_W = tensor([0, 0, -1])
        self.qpos_def = tensor(self.cfg.qpos_def)

        self.root_quat_w = self.QUAT_IDENTITY.clone()
        self.root_angvel_b = SMABuffer.init_like(tensor([0, 0, 0]), (0,), self.decimation)
        self.gravity_dir_b = self.GRAVITY_DIR_W.clone()
        self.qpos = tensor([0] * n_qdim)
        self.qvel = SMABuffer.init_like(self.qpos, (0,), self.decimation)

        # command
        self.cmd_rng = tensor(self.cfg.cmd_rng)

        self.is_walk = tensor(False, dtype=th.bool)
        self.cmd = tensor([0, 0, 0, 0, 0, 0])

        # gait
        self.gait_freq = tensor(0)
        self.gait_clock = tensor(0)
        self.gait_ratio = tensor(0)
        self.gait_theta = tensor([0, 0])

        # others
        self.action_scale = tensor(self.cfg.action_scale)

        self.action = tensor([0] * n_qdim)
        self.obs_hist = HistoryBuffer.init_like(tensor([0] * n_obs_t), (0,), self.cfg.n_history)


    def _reset_buff(self):
        # robot data
        self.root_quat_w.copy_(self.QUAT_IDENTITY)
        self.root_angvel_b.reset(())
        self.gravity_dir_b.copy_(self.GRAVITY_DIR_W)
        self.qpos.zero_()
        self.qvel.reset(())

        # command
        self.is_walk.zero_()
        self.cmd.zero_()

        # gait
        self.gait_freq.zero_()
        self.gait_clock.zero_()
        self.gait_ratio.zero_()

        # others
        self.action.zero_()
        self.obs_hist.reset(())


    def step(
            self,
            quat: th.Tensor,
            angvel: th.Tensor,
            qpos: th.Tensor,
            qvel: th.Tensor,
            cmd_axes: th.Tensor,
            cmd_btns: th.Tensor,
        ) -> th.Tensor:
        # update-buff: robot data
        self.root_quat_w.copy_(quat)
        self.root_angvel_b.update(angvel)
        self.gravity_dir_b.copy_(
            quat_apply(quat_conj(self.root_quat_w), self.GRAVITY_DIR_W)
        )
        self.qpos.copy_(qpos)
        self.qvel.update(qvel)

        # update-buff: command
        cmd_axes = cmd_axes.clip(0.0, 1.0)

        self.is_walk.copy_(cmd_btns[0])
        self.cmd.copy_(
            self.cmd_rng[:,0] + cmd_axes * self.cmd_rng.diff(n=1, dim=-1).squeeze(1)
        )
        self.cmd.mul_(self.is_walk.unsqueeze(0))

        # update-buff: gait
        # TODO
        self.gait_freq.copy_(self.cmd[3])
        # clamp required stride distance
        req_linvel = vec_norm(self.cmd[0:2])
        self.gait_freq.copy_(th.where(
            req_linvel / self.gait_freq.clip(min=1e-6) > self.cfg.max_stride,
            req_linvel / self.cfg.max_stride,
            self.gait_freq,
        ))

        self.gait_clock.add_((2.0 * th.pi * (1.0 / self.cfg.step_freq)) * self.gait_freq)
        self.gait_clock.remainder_(2.0 * th.pi)

        self.gait_ratio.copy_(self.cmd[4])

        self.gait_theta[0].copy_(self.gait_clock)
        self.gait_theta[1].copy_(self.gait_clock + self.cmd[5])
        self.gait_theta.remainder_(2.0 * th.pi)

        # policy step
        if self.step_idx % self.decimation == self.decimation - 1:
            self._policy_step()

        # compute output: target qpos
        qpos_trg = self.qpos_def + self.action_scale * self.action

        self.step_idx = (self.step_idx + 1) % self.cfg.step_freq
        return qpos_trg


    def _policy_step(self):
        # compute observation tensors/buffers
        obs_hist_t = th.cat([
            self.root_angvel_b.sma * 0.25,
            self.gravity_dir_b,
            self.qpos,
            self.qvel.sma * 0.05,
            self.action,
        ], dim=-1)

        self.obs_hist.update(obs_hist_t)
        obs_hist = self.obs_hist.buff # (n_history, n_obs_hist_t)

        obs_cmd = th.cat([
            self.cmd[0:3],
            self.gait_freq.unsqueeze(0),
            self.gait_ratio.unsqueeze(0),
            self.gait_theta.sin() * self.is_walk.unsqueeze(0),
            self.gait_theta.cos() * self.is_walk.unsqueeze(0),
        ], dim=-1)

        # final observation
        observation = {
            'obs_hist_t': obs_hist_t,
            'obs_hist': obs_hist,
            'obs_cmd': obs_cmd,
        }
        observation = th.cat([
            observation['obs_hist'].view(-1),
            observation['obs_cmd'],
        ], dim=-1).clip(*self.cfg.obs_clip) # TODO

        # run onnx session
        session_input = {'input': observation.unsqueeze(0).numpy()}
        session_output = self.session.run(['output'], session_input)

        # compute action tensor
        self.action.copy_(th.from_numpy(session_output[0]).squeeze(0))
        self.action.clip_(*self.cfg.action_clip)


    def reset(self):
        self._reset_buff()
        self.step_idx = 0


    def _tensor(self, data, **kwargs):
        opts = {'dtype': th.float32, 'device': self.cfg.device}
        opts.update(kwargs)
        return th.tensor(data, **opts)
