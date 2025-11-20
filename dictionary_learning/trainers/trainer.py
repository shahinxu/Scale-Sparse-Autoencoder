class SAETrainer:
    def __init__(self, seed=None):
        self.seed = seed
        self.logging_parameters = []

    def update(self, 
               step, # index of step in training
               activations, # of shape [batch_size, d_submodule]
        ):
        pass # implemented by subclasses

    def get_logging_parameters(self):
        stats = {}
        for param in self.logging_parameters:
            if hasattr(self, param):
                stats[param] = getattr(self, param)
            else:
                print(f"Warning: {param} not found in {self}")
        return stats
    
    @property
    def config(self):
        return {
            'wandb_name': 'trainer',
        }

import torch
from typing import Callable, Optional
import einops

class ConstrainedAdam(torch.optim.Adam):
    def __init__(
        self, params, constrained_params, lr: float, betas: tuple[float, float] = (0.9, 0.999)
    ):
        super().__init__(params, lr=lr, betas=betas)
        self.constrained_params = list(constrained_params)

    def step(self, closure=None):
        with torch.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with torch.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=0, keepdim=True)


@torch.no_grad()
def set_decoder_norm_to_unit_norm(
    W_dec_DF: torch.nn.Parameter, activation_dim: int, d_sae: int
) -> torch.Tensor:

    D, F = W_dec_DF.shape

    assert D == activation_dim
    assert F == d_sae

    eps = torch.finfo(W_dec_DF.dtype).eps
    norm = torch.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


@torch.no_grad()
def remove_gradient_parallel_to_decoder_directions(
    W_dec_DF: torch.Tensor,
    W_dec_DF_grad: torch.Tensor,
    activation_dim: int,
    d_sae: int,
) -> torch.Tensor:

    D, F = W_dec_DF.shape
    assert D == activation_dim
    assert F == d_sae

    normed_W_dec_DF = W_dec_DF / (torch.norm(W_dec_DF, dim=0, keepdim=True) + 1e-6)

    parallel_component = einops.einsum(
        W_dec_DF_grad,
        normed_W_dec_DF,
        "d_in d_sae, d_in d_sae -> d_sae",
    )
    W_dec_DF_grad -= einops.einsum(
        parallel_component,
        normed_W_dec_DF,
        "d_sae, d_in d_sae -> d_in d_sae",
    )
    return W_dec_DF_grad


def get_lr_schedule(
    total_steps: int,
    warmup_steps: int,
    decay_start: Optional[int] = None,
    resample_steps: Optional[int] = None,
    sparsity_warmup_steps: Optional[int] = None,
) -> Callable[[int], float]:
    if decay_start is not None:
        assert resample_steps is None, (
            "decay_start and resample_steps are currently mutually exclusive."
        )
        assert 0 <= decay_start < total_steps, "decay_start must be >= 0 and < steps."
        assert decay_start > warmup_steps, "decay_start must be > warmup_steps."
        if sparsity_warmup_steps is not None:
            assert decay_start > sparsity_warmup_steps, (
                "decay_start must be > sparsity_warmup_steps."
            )

    assert 0 <= warmup_steps < total_steps, "warmup_steps must be >= 0 and < steps."

    if resample_steps is None:

        def lr_schedule(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps

            if decay_start is not None and step >= decay_start:
                return (total_steps - step) / (total_steps - decay_start)

            return 1.0
    else:
        assert 0 < resample_steps < total_steps, "resample_steps must be > 0 and < steps."

        def lr_schedule(step: int) -> float:
            return min((step % resample_steps) / warmup_steps, 1.0)

    return lr_schedule


def get_sparsity_warmup_fn(
    total_steps: int, sparsity_warmup_steps: Optional[int] = None
) -> Callable[[int], float]:

    if sparsity_warmup_steps is not None:
        assert 0 <= sparsity_warmup_steps < total_steps, (
            "sparsity_warmup_steps must be >= 0 and < steps."
        )

    def scale_fn(step: int) -> float:
        if not sparsity_warmup_steps:
            return 1.0
        else:
            return min(step / sparsity_warmup_steps, 1.0)

    return scale_fn