from dataclasses import dataclass, MISSING
from typing import List, Tuple

import torch as th
import onnxruntime as ort

from policy_runner.utils import HistoryBuffer, SMABuffer
from policy_runner.math_utils import (
    quat_apply,
    quat_conj,
)



@dataclass(slots=True)
class PolicyRunnerCfg:

    decimation: int = MISSING

    device: str = MISSING

    ref_q_names: List[str] = MISSING

    q_names: List[str] = MISSING

    q_scale: List[float] = MISSING

    n_history: int = MISSING

    obs_scale: List[float] = MISSING

    obs_clip: Tuple[float, float] = MISSING

    action_clip: Tuple[float, float] = MISSING

    model_path: str = MISSING



class PolicyRunner:
    __slots__ = (
        'cfg', 'device',
        '_buff_initialized',
        'session',
        'n_qdim', 'n_obs_t', 'n_obs_priv',
        'QUAT_IDENTITY', 'GRAV_W',
        'quat', 'linvel', 'angvel', 'qpos', 'qvel', 'action',
        'obs_t', 'obs_priv', 'obs_hist',
        'q_scale',
    )


    def __init__(self, cfg: PolicyRunnerCfg):
        self.cfg = cfg
        self.device = th.device(cfg.device)

        # initialize onnx model
        self._init_onnx()

        # lazy initialization for buffers
        self._buff_initialized = False


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
            f'[onnx input metadata]\n' +
            ''.join([
                f'{inp.name}, {inp.shape}, {inp.type}\n'
                for inp in self.session.get_inputs()
            ])
        )
        print(
            f'[onnx output metadata]\n' +
            ''.join([
                f'{out.name}, {out.shape}, {out.type}\n'
                for out in self.session.get_outputs()
            ])
        )


    def _init_buff(self, n_qdim: int):
        # numbers
        self.n_qdim = n_qdim
        self.n_obs_t = 6 + n_qdim * 3
        self.n_obs_priv = 3

        def _zeros(*args, dtype=th.float32, device=self.device):
            return th.zeros(size=args, dtype=dtype, device=device)

        # robot data
        self.QUAT_IDENTITY = th.tensor([1, 0, 0, 0], dtype=th.float32, device=self.device)
        self.GRAV_W = th.tensor([0, 0, -1], dtype=th.float32, device=self.device)

        self.quat = self.QUAT_IDENTITY.clone()
        self.linvel = _zeros(3)
        self.angvel = _zeros(3)
        self.qpos = _zeros(self.n_qdim)
        self.qvel = _zeros(self.n_qdim)
        self.action = _zeros(self.n_qdim)

        self.linvel = SMABuffer.init_like(self.linvel, (0,), self.cfg.decimation)
        self.angvel = SMABuffer.init_like(self.angvel, (0,), self.cfg.decimation)
        self.qvel = SMABuffer.init_like(self.qvel, (0,), self.cfg.decimation)

        # observation
        self.obs_t = _zeros(self.n_obs_t)
        self.obs_priv = _zeros(self.n_obs_priv)
        self.obs_hist = HistoryBuffer.init_like(self.obs_t, (0,), self.cfg.n_history)

        # q-scale tensor
        idx_map = [self.cfg.q_names.index(name) for name in self.cfg.ref_q_names]
        self.q_scale = th.tensor(
            [self.cfg.q_scale[idx_map[i]] for i in range(len(idx_map))],
            dtype=th.float32, device=self.device,
        )


    def sim_step(
            self,
            quat: th.Tensor,
            linvel: th.Tensor,
            angvel: th.Tensor,
            qpos: th.Tensor,
            qvel: th.Tensor,
        ):
        # lazy initialization
        if not self._buff_initialized:
            self._init_buff(n_qdim=qpos.shape[-1])
            self._buff_initialized = True

        self.quat.copy_(quat)
        self.qpos.copy_(qpos)

        self.linvel.update(linvel)
        self.angvel.update(angvel)
        self.qvel.update(qvel)


    def _compute_observation(self, command: th.Tensor, clock_sin: th.Tensor, clock_cos: th.Tensor) -> th.Tensor:
        linvel = self.linvel.sma
        angvel = self.angvel.sma
        qpos = self.qpos
        qvel = self.qvel.sma
        action = self.action
        grav_b = quat_apply(quat_conj(self.quat), self.GRAV_W)

        # update observation tensors/buffers
        self.obs_t.copy_(
            th.cat([
                angvel * self.cfg.obs_scale[0],
                grav_b * self.cfg.obs_scale[1],
                qpos * self.cfg.obs_scale[2],
                qvel * self.cfg.obs_scale[3],
                action * self.cfg.obs_scale[4],
            ], dim=-1)
        )
        self.obs_priv.copy_(
            th.cat([
                linvel * self.cfg.obs_scale[5],
            ], dim=-1)
        )
        self.obs_hist.update(self.obs_t)

        # compute final observation tensor
        obs_tensor = th.cat([
            self.obs_hist.buff.view(-1),
            command,
            clock_sin,
            clock_cos,
            self.obs_priv,
        ], dim=-1)
        obs_tensor[-3:] = 0.0 # TODO

        return obs_tensor


    def policy_step(self, command: th.Tensor, clock_sin: th.Tensor, clock_cos: th.Tensor) -> th.Tensor:
        # compute observation tensor
        obs_tensor = self._compute_observation(command, clock_sin, clock_cos)
        obs_tensor.clip_(*self.cfg.obs_clip)

        # run onnx session
        session_input = {'input': obs_tensor.unsqueeze(0).numpy()}
        session_output = self.session.run(['output'], session_input)

        # compute action tensor
        self.action.copy_(th.from_numpy(session_output[0]).squeeze(0))
        self.action.clip_(*self.cfg.action_clip)

        return self.action * self.q_scale


    def clear_buff(self):
        # robot data
        self.quat.copy_(self.QUAT_IDENTITY)
        self.qpos.zero_()
        self.action.zero_()

        self.linvel.reset(())
        self.angvel.reset(())
        self.qvel.reset(())

        # observation
        self.obs_t.zero_()
        self.obs_priv.zero_()
        self.obs_hist.reset(())
