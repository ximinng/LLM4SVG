from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList

import numpy as np

from typing import Set, Tuple

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def first(arr):
    return arr[0]

def get_module_device(m: Module):
    return next(m.parameters()).device

def inplace_copy(tgt: Tensor, src: Tensor, *, auto_move_device = False):
    if auto_move_device:
        src = src.to(tgt.device)

    tgt.copy_(src)

def inplace_lerp(tgt: Tensor, src: Tensor, weight, *, auto_move_device = False):
    if auto_move_device:
        src = src.to(tgt.device)

    tgt.lerp_(src, weight)

# algorithm 2 in https://arxiv.org/abs/2312.02696

def sigma_rel_to_gamma(sigma_rel):
    t = sigma_rel ** -2
    return np.roots([1, 7, 16 - t, 12 - t]).real.max().item()

class KarrasEMA(Module):
    """
    exponential moving average module that uses hyperparameters from the paper https://arxiv.org/abs/2312.02696
    can either use gamma or sigma_rel from paper
    """

    def __init__(
        self,
        model: Module,
        sigma_rel: float | None = None,
        gamma: float | None = None,
        ema_model: Module | None = None,           # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        update_every: int = 100,
        frozen: bool = False,
        param_or_buffer_names_no_ema: Set[str] = set(),
        ignore_names: Set[str] = set(),
        ignore_startswith_names: Set[str] = set(),
        allow_different_devices = False,              # if the EMA model is on a different device (say CPU), automatically move the tensor
        move_ema_to_online_device = False             # will move entire EMA model to the same device as online model, if different
    ):
        super().__init__()

        assert exists(sigma_rel) ^ exists(gamma), 'either sigma_rel or gamma is given. gamma is derived from sigma_rel as in the paper, then beta is dervied from gamma'

        if exists(sigma_rel):
            gamma = sigma_rel_to_gamma(sigma_rel)

        self.gamma = gamma
        self.frozen = frozen

        self.online_model = [model]

        # ema model

        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = deepcopy(model)
            except Exception as e:
                print(f'Error: While trying to deepcopy model: {e}')
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        for p in self.ema_model.parameters():
            p.detach_()

        # parameter and buffer names

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if torch.is_floating_point(param) or torch.is_complex(param)}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if torch.is_floating_point(buffer) or torch.is_complex(buffer)}

        # tensor update functions

        self.inplace_copy = partial(inplace_copy, auto_move_device = allow_different_devices)
        self.inplace_lerp = partial(inplace_lerp, auto_move_device = allow_different_devices)

        # updating hyperparameters

        self.update_every = update_every

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        # whether to manage if EMA model is kept on a different device

        self.allow_different_devices = allow_different_devices

        # whether to move EMA model to online model device automatically

        self.move_ema_to_online_device = move_ema_to_online_device

        # init and step states

        self.register_buffer('initted', torch.tensor(False))
        self.register_buffer('step', torch.tensor(0))

    @property
    def model(self):
        return first(self.online_model)

    @property
    def beta(self):
        return (1. - 1. / (self.step.item() + 1.)) ** (1. + self.gamma)

    def eval(self):
        return self.ema_model.eval()

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(ma_params.data, current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(ma_buffers.data, current_buffers.data)

    def copy_params_from_ema_to_model(self):
        copy = self.inplace_copy

        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            copy(current_params.data, ma_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            copy(current_buffers.data, ma_buffers.data)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True))

        self.update_moving_average(self.ema_model, self.model)

    def iter_all_ema_params_and_buffers(self):
        for name, ma_params in self.get_params_iter(self.ema_model):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                continue

            yield ma_params

        for name, ma_buffer in self.get_buffers_iter(self.ema_model):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                continue

            yield ma_buffer

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        if self.frozen:
            return

        # move ema model to online model device if not same and needed

        if self.move_ema_to_online_device and get_module_device(ma_model) != get_module_device(current_model):
            ma_model.to(get_module_device(current_model))

        # get some functions and current decay

        copy, lerp = self.inplace_copy, self.inplace_lerp
        current_decay = self.beta

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                copy(ma_params.data, current_params.data)
                continue

            lerp(ma_params.data, current_params.data, 1. - current_decay)

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                copy(ma_buffer.data, current_buffer.data)
                continue

            lerp(ma_buffer.data, current_buffer.data, 1. - current_decay)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

# post hoc ema wrapper

# solving of the weights for combining all checkpoints into a newly synthesized EMA at desired gamma
# Algorithm 3 copied from paper, redone in torch

def p_dot_p(t_a, gamma_a, t_b, gamma_b):
    t_ratio = t_a / t_b
    t_exp = torch.where(t_a < t_b , gamma_b , -gamma_a)
    t_max = torch.maximum(t_a , t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio ** t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den

def solve_weights(t_i, gamma_i, t_r, gamma_r):
    rv = lambda x: x.double().reshape(-1, 1)
    cv = lambda x: x.double().reshape(1, -1)
    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
    b = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
    return torch.linalg.solve(A, b)

class PostHocEMA(Module):

    def __init__(
        self,
        model: Module,
        sigma_rels: Tuple[float, ...] | None = None,
        gammas: Tuple[float, ...] | None = None,
        checkpoint_every_num_steps: int = 1000,
        checkpoint_folder: str = './post-hoc-ema-checkpoints',
        checkpoint_dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        super().__init__()
        assert exists(sigma_rels) ^ exists(gammas)

        if exists(sigma_rels):
            gammas = tuple(map(sigma_rel_to_gamma, sigma_rels))

        assert len(gammas) > 1, 'at least 2 ema models with different gammas in order to synthesize new ema models of a different gamma'
        assert len(set(gammas)) == len(gammas), 'calculated gammas must be all unique'

        self.gammas = gammas
        self.num_ema_models = len(gammas)

        self._model = [model]
        self.ema_models = ModuleList([KarrasEMA(model, gamma = gamma, **kwargs) for gamma in gammas])

        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        self.checkpoint_every_num_steps = checkpoint_every_num_steps
        self.checkpoint_dtype = checkpoint_dtype
        self.ema_kwargs = kwargs

    @property
    def model(self):
        return first(self._model)

    @property
    def step(self):
        return first(self.ema_models).step

    @property
    def device(self):
        return self.step.device

    def copy_params_from_model_to_ema(self):
        for ema_model in self.ema_models:
            ema_model.copy_params_from_model_to_ema()

    def copy_params_from_ema_to_model(self):
        for ema_model in self.ema_models:
            ema_model.copy_params_from_ema_to_model()

    def update(self):
        for ema_model in self.ema_models:
            ema_model.update()

        if not (self.step.item() % self.checkpoint_every_num_steps):
            self.checkpoint()

    def checkpoint(self):
        step = self.step.item()

        for ind, ema_model in enumerate(self.ema_models):
            filename = f'{ind}.{step}.pt'
            path = self.checkpoint_folder / filename

            pkg = deepcopy(ema_model).to(self.checkpoint_dtype).state_dict()
            torch.save(pkg, str(path))

    def synthesize_ema_model(
        self,
        gamma: float | None = None,
        sigma_rel: float | None = None,
        step: int | None = None,
    ) -> KarrasEMA:
        assert exists(gamma) ^ exists(sigma_rel)
        device = self.device

        if exists(sigma_rel):
            gamma = sigma_rel_to_gamma(sigma_rel)

        synthesized_ema_model = KarrasEMA(
            model = self.model,
            gamma = gamma,
            **self.ema_kwargs
        )

        synthesized_ema_model

        # get all checkpoints

        gammas = []
        timesteps = []
        checkpoints = [*self.checkpoint_folder.glob('*.pt')]

        for file in checkpoints:
            gamma_ind, timestep = map(int, file.stem.split('.'))
            gammas.append(self.gammas[gamma_ind])
            timesteps.append(timestep)

        step = default(step, max(timesteps))
        assert step <= max(timesteps), f'you can only synthesize for a timestep that is less than the max timestep {max(timesteps)}'

        # line up with Algorithm 3

        gamma_i = torch.tensor(gammas, device = device)
        t_i = torch.tensor(timesteps, device = device)

        gamma_r = torch.tensor([gamma], device = device)
        t_r = torch.tensor([step], device = device)

        # solve for weights for combining all checkpoints into synthesized, using least squares as in paper

        weights = solve_weights(t_i, gamma_i, t_r, gamma_r)
        weights = weights.squeeze(-1)

        # now sum up all the checkpoints using the weights one by one

        tmp_ema_model = KarrasEMA(
            model = self.model,
            gamma = gamma,
            **self.ema_kwargs
        )

        for ind, (checkpoint, weight) in enumerate(zip(checkpoints, weights.tolist())):
            is_first = ind == 0

            # load checkpoint into a temporary ema model

            ckpt_state_dict = torch.load(str(checkpoint), weights_only=True)
            tmp_ema_model.load_state_dict(ckpt_state_dict)

            # add weighted checkpoint to synthesized

            for ckpt_tensor, synth_tensor in zip(tmp_ema_model.iter_all_ema_params_and_buffers(), synthesized_ema_model.iter_all_ema_params_and_buffers()):
                if is_first:
                    synth_tensor.zero_()

                synth_tensor.add_(ckpt_tensor * weight)

        # return the synthesized model

        return synthesized_ema_model

    def __call__(self, *args, **kwargs):
        return tuple(ema_model(*args, **kwargs) for ema_model in self.ema_models)
