import torch as t
from ..trainers.trainer import (
    SAETrainer, 
    set_decoder_norm_to_unit_norm, 
    remove_gradient_parallel_to_decoder_directions
)
from ..dictionary import JumpReluAutoEncoder, StepFunction
from collections import namedtuple

class JumpReluTrainer(SAETrainer):
    def __init__(self,
                 dict_class=JumpReluAutoEncoder,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=5e-5,
                 target_l0=20.0,
                 warmup_steps=1000,
                 seed=None,
                 device=None,
                 layer=None,
                 lm_name=None,
                 wandb_name='JumpReluTrainer',
                 submodule_name=None,
                 set_linear_to_constant=False,
                 pre_encoder_bias=True,
    ):
        super().__init__(seed)
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.ae = dict_class(activation_dim, dict_size, pre_encoder_bias=pre_encoder_bias)
        self.ae.to(self.device)
        
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.set_linear_to_constant = set_linear_to_constant
        self.target_l0 = target_l0
        self.sparsity_warmup_steps = 3000
        
        self.sparsity_coefficient = 1.0 
        
        if not hasattr(self.ae, 'bandwidth'):
            self.ae.bandwidth = 0.001

        self.optimizer = t.optim.Adam(self.ae.parameters(), betas=(0.0, 0.999), eps=1e-8)
        
        def warmup_fn(step):
            return min(1, step / max(1, warmup_steps))
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_fn)

    def get_sparsity_scale(self, step):
        if step < self.sparsity_warmup_steps:
            return step / self.sparsity_warmup_steps
        return 1.0

    def loss(self, x, logging=False, **kwargs):
        step = kwargs.get('step', 0)
        x_hat, f = self.ae(x, output_features=True, set_linear_to_constant=self.set_linear_to_constant)
        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        current_l0 = StepFunction.apply(f, self.ae.threshold, self.ae.bandwidth).sum(dim=-1).mean()
        sparsity_scale = self.get_sparsity_scale(step)
        L_spars = self.sparsity_coefficient * ((current_l0 / self.target_l0) - 1).pow(2) * sparsity_scale
        loss = L_recon + L_spars

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                losses={
                    'mse_loss': L_recon.item(),
                    'sparsity_loss': L_spars.item(),
                    'loss': loss.item(),
                    'l0': current_l0.item()
                }
            )
        
    def update(self, step, x):
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        self.ae.W_dec.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.W_dec.T, self.ae.W_dec.grad.T, self.ae.activation_dim, self.ae.dict_size
        ).T
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.ae.W_dec.data = set_decoder_norm_to_unit_norm(
            self.ae.W_dec.T, self.ae.activation_dim, self.ae.dict_size
        ).T

        return loss.item()

    @property
    def config(self):
        return {
            'dict_class': 'JumpReluAutoEncoder',
            'trainer_class': 'JumpReluTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'warmup_steps': self.warmup_steps,
            'target_l0': self.target_l0, 
            'sparsity_coefficient': self.sparsity_coefficient,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'submodule_name': self.submodule_name,
            'wandb_name': self.wandb_name,
        }