import os
import torch
import torch.nn as nn
from typing import NamedTuple
import math

from scripts.transcoder.kernels import TritonDecoderAutograd
import logging


class ForwardOutput(NamedTuple):
    fvu: torch.Tensor
    """Fraction of variance unexplained."""

    auxk_loss: torch.Tensor
    """AuxK loss."""

    multi_topk_fvu: torch.Tensor
    """Multi-TopK FVU."""


class Transcoder(nn.Module):
    """Transcoder for training
    """
    def __init__(self, n_inputs: int, n_outputs:int, expansion: int, k_value: int, dead_feature_threshold: int, device: torch.device) -> None:
        """
        :param n_inputs: dimensionality of the original data (residual stream token dimension)
        :param expansion: expansion factor for latent dimension
        :param k_value: number of non-zero elements to keep in latent dim
        :param dead_feature_threshold: number of forward passes without activation after which a latent unit is considered dead
        :param device: torch device (e.g. cuda)
        """
        super().__init__()
        # dynamic parameters
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_latents = int(expansion * self.n_inputs)
        self.k = k_value
        self.dead_feature_threshold = dead_feature_threshold
        self.device = device
        
        self.register_buffer("activation_frequency", torch.zeros(self.n_latents, dtype=torch.int32, device=device))
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.int32, device=device))
        self.register_buffer("last_fired_step", torch.full((self.n_latents,), -1, dtype=torch.int32, device=device))

        # layers
        self.encoder: nn.Module = nn.Linear(self.n_inputs, self.n_latents, bias=False, device=device)
        self.enc_bias = nn.Parameter(torch.zeros(self.n_latents, device=device))
        self.W_dec  = nn.Parameter(torch.empty(self.n_latents, self.n_outputs, device=device))
        self.dec_bias = nn.Parameter(torch.zeros(self.n_outputs, device=device))
        
        with torch.no_grad():
            if n_inputs == n_outputs:
                    self.W_dec.copy_(self.encoder.weight)
            else:
                # build an orthonormal basis from encoder weights
                Q, _ = torch.linalg.qr(self.encoder.weight, mode="reduced")
                m = self.n_outputs
                if Q.shape[1] >= m:
                    self.W_dec.copy_(Q[:, :m])
                else:
                    extra = torch.randn(self.n_latents, m - Q.shape[1], device=self.device)
                    Q_full, _ = torch.linalg.qr(torch.cat([Q, extra], dim=1), mode="reduced")
                    self.W_dec.copy_(Q_full[:, :m])

        # static parameters
        self.register_buffer("epsilon", torch.tensor(1e-8, dtype=torch.float32, device=device))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: pre activation latents (shape: [batch, n_latents])
        """
        return self.encoder(x) + self.enc_bias

    def activation(self, pre_act: torch.Tensor, k: int) -> torch.Tensor:
        """
        :param pre_act: pre-activation latents features (shape: [batch, n_latents])
        :param k: number of 
        :return:    values of latents features (shape: [batch, k])
                    indices of the latents features (shape: [batch, k])
        """
        topk = torch.topk(pre_act, k=k, dim=-1)
        return topk.values, topk.indices

    def decode(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        :param values:  the top-k latent activations (shape: [batch, k])
        :param indices: their positions in the full latent vector (shape: [batch, k])
        :return: outputs (shape: [batch, n_outputs])
        """
        return TritonDecoderAutograd.apply(indices, values, self.W_dec.T) + self.dec_bias

    def total_variance(self, y: torch.Tensor) -> torch.Tensor:
        """
        :param y: target data (shape: [batch, n_outputs])
        :return: total variance of the target 
        """
        y_mean = y.mean(dim=0, keepdim=True)
        return (y - y_mean).pow(2).sum()

    def reconstruction_loss(self, latents_pre_act: torch.Tensor, y: torch.Tensor, total_variance: float) -> torch.Tensor:
        """
        :param total_variance: total variance of the target 
        :param latents_pre_act: pre activation latents (shape: [batch, n_latents])
        :param y: target data (shape: [batch, n_outputs])
        :return: reconstruction loss normalize as fraction of unexplained variance
        """
        values, indices = self.activation(latents_pre_act, k=self.k)

        if not self.training:
            self.update_activation_frequency(indices)

        out = self.decode(values, indices)

        error = (y - out)
        l2_loss = error.pow(2).sum()
        epsilon = self.epsilon.to(dtype=y.dtype)
        fvu = l2_loss / (total_variance + epsilon)

        return fvu

    @torch.no_grad()
    def update_dead_count(self, indices: torch.Tensor) -> None:
        """
        :param indices: positions of latent that fired (shape: [batch, k])
        """
        indices_flat = indices.view(-1)
        unique_idxs = torch.unique(indices_flat)
        self.global_step.add_(indices.size(0))
        self.last_fired_step[unique_idxs] = self.global_step

    @torch.no_grad()
    def update_activation_frequency(self, indices: torch.Tensor) -> None:
        """
        :param indices: positions of latent that fired (shape: [batch, k])
        """
        indices_flat = indices.reshape(-1)
        counts = torch.bincount(indices_flat, minlength=self.n_latents)
        self.activation_frequency.add_(counts)

    @torch.no_grad()
    def reset_activation_frequency(self) -> None:
        """
        """
        self.activation_frequency.zero_()

    @torch.no_grad()
    def dead_mask(self) -> torch.Tensor:
        """
        :return: mask where 1 is dead latents and 0 is living latents (shape: [n_latents])
        """
        return (self.global_step - self.last_fired_step) > self.dead_feature_threshold

    def auxk_loss(self, latents_pre_act: torch.Tensor, y: torch.Tensor, total_variance: float) -> torch.Tensor:
        """
        :param total_variance: total variance of the target 
        :param latents_pre_act: pre activation latents (shape: [batch, n_latents])
        :param y: target data (shape: [batch, n_outputs])
        :return: auxiliary loss normalize as fraction of unexplained variance and scale by...
        """
        k_aux = self.n_inputs // 2

        dead_mask = self.dead_mask()
        num_dead = int(dead_mask.sum().item())

        if num_dead > 0:
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # exclude living latents of the loss
            neg_inf = torch.finfo(latents_pre_act.dtype).min
            auxk_latents = torch.where(dead_mask[None], latents_pre_act, neg_inf)

            auxk_values, auxk_indices = self.activation(auxk_latents, k=k_aux)

            auxk_out = self.decode(auxk_values, auxk_indices)
            auxk_loss = (y - auxk_out).pow(2).sum()
            epsilon = self.epsilon.to(dtype=y.dtype)
            auxk_loss = scale * auxk_loss / (total_variance + epsilon)

            return auxk_loss
        else:
            return y.new_zeros(())

    def multi_topk_loss(self, latents_pre_act: torch.Tensor, y: torch.Tensor, total_variance: float) -> torch.Tensor:
        """
        :param total_variance: total variance of the target 
        :param latents_pre_act: pre activation latents (shape: [batch, n_latents])
        :param y: target data (shape: [batch, n_outputs])
        :return: multi top k normalize as fraction of unexplained variance and scale by 1/8
        """
        multi_topk_k = min(self.k * 4, self.n_latents)

        multi_topk_values, multi_topk_indices = self.activation(latents_pre_act, k=multi_topk_k)

        if self.training:
            self.update_dead_count(multi_topk_indices)

        multi_topk_out = self.decode(multi_topk_values, multi_topk_indices)
        epsilon = self.epsilon.to(dtype=y.dtype)
        multi_topk_fvu = (y - multi_topk_out).pow(2).sum() / (total_variance + epsilon)

        return multi_topk_fvu

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> ForwardOutput:
        """
        :param x: input data (shape: [batch, n_inputs])
        :param y: target data (shape: [batch, n_outputs])
        :return:    reconstruction loss normalize as fraction of unexplained variance
                    auxiliary loss normalize as fraction of unexplained variance and scale by...
                    multi top k normalize as fraction of unexplained variance and scale by 1/8
        """
        latents_pre_act = self.encode(x)

        # total variance of target
        total_variance = self.total_variance(y)

        # reconstruction loss
        fvu = self.reconstruction_loss(latents_pre_act, y, total_variance)

        # dead latents loss
        auxk_loss = self.auxk_loss(latents_pre_act, y, total_variance)

        # multi_topk loss
        multi_topk_fvu = self.multi_topk_loss(latents_pre_act, y, total_variance)

        return ForwardOutput(fvu, auxk_loss, multi_topk_fvu)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd[prefix + "k"] = self.k
        return sd
