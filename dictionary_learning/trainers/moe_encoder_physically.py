import einops
import torch as t
import torch.nn as nn
from collections import namedtuple
from ..kernels import TritonDecoder
from .trainer import SAETrainer
import torch.nn.init as init

@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, output_dim)
        self.encoder.bias.data.zero_()
    
    def forward(self, x):
        z = nn.functional.relu(self.encoder(x))
        return z

class MultiEncAutoEncoder(nn.Module):
    def __init__(self, activation_dim, dict_size, k, experts, e, heaviside=False):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k
        self.experts = experts
        self.e = e
        self.heaviside = heaviside
        self.expert_dict_size = self.dict_size // self.experts
        
        self.expert_modules = nn.ModuleList([
            Expert(activation_dim, self.expert_dict_size) 
            for _ in range(experts)
        ])
        self.decoder = nn.Parameter(t.empty(dict_size, activation_dim))
        init.kaiming_uniform_(self.decoder, a=0, mode='fan_in', nonlinearity='relu')
        self.set_decoder_norm_to_unit_norm()

        self.gate = nn.Linear(activation_dim, experts)
        self.gate.bias.data.zero_()
        
        self.b_gate = nn.Parameter(t.zeros(activation_dim))
        self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(self, x):
        gate_logits = self.gate(x - self.b_gate)
        gate_scores = t.softmax(gate_logits, dim=-1)
        
        top_values, top_indices = gate_scores.topk(self.e, dim=-1)
        
        if self.heaviside:
            expert_mask = t.zeros_like(gate_scores).scatter_(-1, top_indices, 1.0)
        else:
            sparse_weights = t.full_like(gate_scores, float('-inf'))
            sparse_weights.scatter_(-1, top_indices, top_values)
            expert_mask = t.softmax(sparse_weights, dim=-1)
        
        all_expert_outputs = []
        for expert in self.expert_modules:
            z_expert = expert.encoder(x - self.b_dec)
            z_expert = nn.functional.relu(z_expert)
            all_expert_outputs.append(z_expert)
        
        expert_stack = t.stack(all_expert_outputs, dim=0)
        
        mask_expanded = einops.repeat(expert_mask, 'b e -> b e d', d=self.expert_dict_size)
        
        weighted_experts = expert_stack.permute(1,0,2) * mask_expanded
        
        f_total = weighted_experts.reshape(x.size(0), self.dict_size)
        return f_total

    def decode(self, top_acts, top_indices):
        d = TritonDecoder.apply(top_indices, top_acts, self.decoder.mT)
        return d + self.b_dec

    def forward(self, x, output_features=False):
        f = self.encode(x.view(-1, x.shape[-1]))
        top_acts, top_indices = f.topk(self.k, sorted=False)
        x_hat = self.decode(top_acts, top_indices).view(x.shape)
        f = f.view(*x.shape[:-1], f.shape[-1])

        if not output_features:
            return x_hat
        else:
            return x_hat, f

    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = t.finfo(self.decoder.dtype).eps
        norm = t.norm(self.decoder.data, dim=1, keepdim=True)
        self.decoder.data /= norm + eps

    @t.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.grad is not None

        parallel_component = einops.einsum(
            self.decoder.grad,
            self.decoder.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.decoder.grad -= einops.einsum(
            parallel_component,
            self.decoder.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def from_pretrained(path, k=100, experts=16, e=1, heaviside=False, device=None
                        , activation_dim=768, dict_size=32*768):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        autoencoder = MultiEncAutoEncoder(activation_dim, dict_size, k, experts, e, heaviside)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder

class MoETrainer(SAETrainer):
    """
    MoE SAE training scheme.
    """
    def __init__(self,
                 dict_class=MultiEncAutoEncoder,
                 activation_dim=512,
                 dict_size=64*512,
                 k=100,
                 experts=16,
                 e=1,
                 heaviside=False,
                 auxk_alpha=1/32,
                 decay_start=24000,
                 steps=30000,
                 seed=None,
                 device=None,
                 layer=None,
                 lm_name=None,
                 wandb_name='MoEAutoEncoder',
                 submodule_name=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.activation_dim = activation_dim
        self.wandb_name = wandb_name
        self.steps = steps
        self.k = k
        self.experts = experts
        self.e = e
        self.heaviside = heaviside
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.ae = dict_class(activation_dim, dict_size, k, experts, e, heaviside)
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)
        
        scale = dict_size / (2 ** 14)
        self.lr = 2e-4 / scale ** 0.5
        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        def lr_fn(step):
            if step < decay_start:
                return 1.
            else:
                return (steps - step) / (steps - decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
    
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        
        self.logging_parameters = ["effective_l0", "dead_features"]
        self.effective_l0 = -1
        self.dead_features = -1
        
    def loss(self, x, step=None, logging=False):
        
        f = self.ae.encode(x)
        top_acts, top_indices = f.topk(self.k, sorted=False)
        x_hat = self.ae.decode(top_acts, top_indices)
        
        e = x_hat - x

        self.effective_l0 = sum([acts.nonzero().size(0) for acts in f])

        auxk_loss = x_hat.new_tensor(0.0)

        h = self.ae.gate(x - self.ae.b_gate)
        p = t.nn.functional.softmax(h, dim=-1)

        flb = t.argmax(p, dim=1)
        flb = t.nn.functional.one_hot(flb, num_classes=self.experts).float()
        flb = flb.mean(dim=0)

        P = p.mean(dim=0)

        lb_loss = self.experts * t.dot(flb, P)

        lb_loss_weight = 3
        
        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = auxk_loss.sum(dim=-1).mean()

        decoder_matrix = self.ae.decoder.detach()
        # decoder_normed = decoder_matrix / (decoder_matrix.norm(dim=1, keepdim=True) + 1e-8)

        # sim_matrix = decoder_normed @ decoder_normed.T

        # sim_matrix.fill_diagonal_(-float("-1e6"))

        # max_sim = sim_matrix.max(dim=1).values

        # sim_loss = max_sim.sum()

        # expert_dict_size = self.ae.expert_dict_size
        # experts = self.ae.experts

        # expert_centers = []
        # for expert in range(experts):
        #     start = expert * expert_dict_size
        #     end = (expert + 1) * expert_dict_size
        #     expert_features = decoder_matrix[start:end]
        #     center = expert_features.mean(dim=0, keepdim=True)
        #     expert_centers.append(center)
        # expert_centers = t.cat(expert_centers, dim=0)

        # centers_normed = expert_centers / expert_centers.norm(dim=1, keepdim=True)
        # sim_matrix = centers_normed @ centers_normed.T
        # sim_matrix.fill_diagonal_(float('-inf'))
        # sim_ext = sim_matrix.max(dim=1).values.mean()

        # sim_ints = []
        # for expert in range(experts):
        #     start = expert * expert_dict_size
        #     end = (expert + 1) * expert_dict_size
        #     expert_features = decoder_matrix[start:end]
        #     center = expert_centers[expert:expert+1]
        #     features_normed = expert_features / expert_features.norm(dim=1, keepdim=True)
        #     center_normed = center / center.norm(dim=1, keepdim=True)
        #     sim = (features_normed * center_normed).sum(dim=1).mean()
        #     sim_ints.append(sim)
        # sim_int = t.stack(sim_ints).mean()

        # sim_loss = (sim_ext + sim_int * 0.3) * self.ae.dict_size

        loss = l2_loss + lb_loss_weight * lb_loss * self.activation_dim 

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss': l2_loss.item(),
                    'auxk_loss': auxk_loss.item(),
                    'lb_loss': lb_loss.item(),
                    'loss' : loss.item()
                }
            )

    def update(self, step, x):

        x = x.to(self.device)

        # Initialise the decoder bias
        if step == 0:
            median = geometric_median(x)
            self.ae.b_dec.data = median
            self.ae.b_gate.data = median
            
        # Make sure the decoder is still unit-norm
        self.ae.set_decoder_norm_to_unit_norm()
        
        # compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.ae.remove_gradient_parallel_to_decoder_directions()

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        return loss.item()

    @property
    def config(self):
        return {
            'trainer_class' : 'MoETrainer',
            'dict_class' : 'MoEAutoEncoder',
            'lr' : self.lr,
            'steps' : self.steps,
            'seed' : self.seed,
            'activation_dim' : self.ae.activation_dim,
            'dict_size' : self.ae.dict_size,
            'k': self.ae.k,
            'experts': self.ae.experts,
            'e': self.ae.e,
            'heaviside': self.ae.heaviside,
            'device' : self.device,
            "layer" : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name' : self.wandb_name,
            'submodule_name' : self.submodule_name,
        }
