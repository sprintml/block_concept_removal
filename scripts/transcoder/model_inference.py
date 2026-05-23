from typing import Callable

import torch
import torch.nn as nn

from scripts.transcoder.kernels import triton_sparse_dense_matmul

class Transcoder(nn.Module):
    """Transcoder for inference

    Implements:
        latents = activation(encoder(x) + enc_bias)
        outputs = decoder(latents) + dec_bias
    """
    def __init__(self, n_inputs: int, n_outputs:int, expansion: int, k_value: int) -> None:
        """
        :param n_inputs: dimensionality of the original data (residual stream token dimension)
        :param expansion: expansion factor for latent dimension
        :param k_value: expectancy of the number of non-zero elements to keep in latent dim
        """
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_latents = int(expansion * self.n_inputs)

        self.encoder: nn.Module = nn.Linear(self.n_inputs,self.n_latents, bias=False)
        self.enc_bias = nn.Parameter(torch.zeros(self.n_latents))
        self.W_dec  = nn.Parameter(torch.empty(self.n_latents, self.n_outputs))
        self.dec_bias = nn.Parameter(torch.zeros(self.n_outputs))
        self.k = k_value
        self.idxs = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: pre activation latents (shape: [batch, n_latents])
        """
        return self.encoder(x) + self.enc_bias

    def activation(self, pre_act: torch.Tensor) -> torch.Tensor:
        """
        :param pre_act: pre-activation latents features (shape: [batch, n_inputs])
        :param return:    values of latents features (shape: [batch, k])
                    indices of the latents features (shape: [batch, k])
        """
        topk = torch.topk(pre_act, k=self.k, dim=-1)
        return topk.values, topk.indices

    def decode(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        :param values:  the top-k latent activations (shape: [batch, k])
        :param indices: their positions in the full latent vector (shape: [batch, k])
        :return: outputs (shape: [batch, n_inputs])
        """
        return triton_sparse_dense_matmul(indices, values, self.W_dec) + self.dec_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input data (shape: (B, T, n_inputs) or (N, n_inputs))
        :return: output data (same rank with last dim = n_outputs)
        """
        orig_dtype = x.dtype

        if x.dim() == 3:
            B, T, D = x.shape
            assert D == self.n_inputs, f"expected last dim {self.n_inputs}, got {D}"
            x2d = x.reshape(B * T, D)
            latents_pre = self.encode(x2d)
            values, indices = self.activation(latents_pre)
            out2d = self.decode(values, indices)
            return out2d.view(B, T, self.n_outputs).to(orig_dtype)

        elif x.dim() == 2:
            N, D = x.shape
            assert D == self.n_inputs
            latents_pre = self.encode(x)
            values, indices = self.activation(latents_pre)
            out2d = self.decode(values, indices)
            return out2d.to(orig_dtype)

        else:
            raise ValueError(f"Unexpected shape {tuple(x.shape)}")

    def get_latents(self, x: torch.Tensor):
        """
        :param x: input data (shape: [batch, n_inputs])
        :return: values of latents features (shape: [batch, k])
                indices of the latents features (shape: [batch, k])
        """
        latents_pre_act = self.encode(x)
        values, indices = self.activation(latents_pre_act)
        return values, indices

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor], strict: bool = True) -> "Transcoder":
        # model shape and expansion factor
        n_latents, n_inputs = state_dict["encoder.weight"].shape
        n_outputs = state_dict["dec_bias"].shape[0]
        expansion_factor = n_latents // n_inputs

        # retrieve k
        k_value = int(state_dict.pop("k"))

        # call constructor
        transcoder = cls(n_inputs, n_outputs, expansion_factor, k_value)

        # remove buffer
        state_dict.pop("activation_frequency", None)
        state_dict.pop("global_step", None)
        state_dict.pop("last_fired_step", None)
        state_dict.pop("epsilon", None)

        # load remaining state dict
        transcoder.load_state_dict(state_dict, strict=strict)
        return transcoder

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd[prefix + "k"] = self.k
        return sd


