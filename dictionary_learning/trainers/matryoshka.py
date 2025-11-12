import torch as t
from collections import namedtuple

from .trainer import SAETrainer
from ..dictionary import MatryoshkaAutoEncoder


class MatryoshkaTrainer(SAETrainer):
    def __init__(self,
                 dict_class=MatryoshkaAutoEncoder,
                 activation_dim=768,
                 dict_size=32*768,
                 total_dict_size=None,
                 lr=5e-5,
                 l1_penalty=1e-3,
                 warmup_steps=1000,
                 seed=None,
                 device=None,
                 layer=None,
                 lm_name=None,
                 wandb_name='MatryoshkaTrainer'):
        super().__init__(seed)
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # total_dict_size: full parameter tensor size. If omitted, fall back to dict_size
        if total_dict_size is None:
            total_dict_size = dict_size

        self.ae = dict_class(activation_dim=activation_dim, total_dict_size=total_dict_size, dict_size=dict_size)
        self.ae.to(self.device)

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.l1_penalty = l1_penalty

        self.optimizer = t.optim.Adam(self.ae.parameters(), betas=(0.0, 0.999), eps=1e-8)

        def warmup_fn(step):
            return min(1, step / warmup_steps)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_fn)

    def loss(self, x, logging=False, **kwargs):
        x_hat = self.ae(x)
        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        # l1 on active features
        with t.no_grad():
            f = self.ae.encode(x)
        L_l1 = f.abs().sum(dim=-1).mean()

        loss = L_recon + self.l1_penalty * L_l1

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'losses'])(
                x, x_hat,
                losses={
                    'mse_loss': L_recon.item(),
                    'l1_loss': L_l1.item(),
                    'loss': loss.item()
                }
            )

    def update(self, step, x):
        x = x.to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()

    @property
    def config(self):
        return {
            'dict_class': 'MatryoshkaAutoEncoder',
            'trainer_class': 'MatryoshkaTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'total_dict_size': self.ae.total_dict_size,
            'lr': self.lr,
            'warmup_steps': self.warmup_steps,
            'l1_penalty': self.l1_penalty,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'wandb_name': self.wandb_name,
        }
