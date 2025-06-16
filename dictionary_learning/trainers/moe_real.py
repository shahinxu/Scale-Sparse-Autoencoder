import einops
import torch as t
import torch.nn as nn
from collections import namedtuple

from ..dictionary import Dictionary
from ..kernels import TritonDecoder
from .trainer import SAETrainer


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


class MoEAutoEncoder(Dictionary, nn.Module):
    """
    The MoE autoencoder architecture
    """
    def __init__(self, activation_dim, dict_size, k, experts, e, heaviside):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k // e
        if self.k < 1:
            self.k = 1
        self.experts = experts
        self.e = e
        self.heaviside = heaviside
        self.expert_dict_size = self.dict_size // self.experts
        
        self.encoders = nn.ModuleList([
            nn.Linear(activation_dim, self.expert_dict_size) for _ in range(experts)
        ])
        self.decoders = nn.ParameterList([
            nn.Parameter(self.encoders[expert].weight.data.clone()) for expert in range(experts)
        ])

        for expert in range(experts):
            self.encoders[expert].bias.data.zero_()
        
        self.set_decoder_norm_to_unit_norm()
        
        self.gater = nn.Linear(activation_dim, experts)
        self.gater.bias.data.zero_()

        self.b_dec = nn.Parameter(t.zeros(activation_dim))
        self.b_gate = nn.Parameter(t.zeros(activation_dim))

    def gate(self, x):
        gate_logits = self.gater(x - self.b_gate)
        probs = t.nn.functional.relu(gate_logits)
        _, top_indices = probs.topk(self.e, dim=-1)
        mask = t.zeros_like(probs)
        mask.scatter_(-1, top_indices, 1.0)
        top_e = probs * mask
        if self.heaviside:
            top_e = (top_e > 0).float()
        return top_e, top_indices
    
    def decode(self, top_acts_list, top_indices_list, top_e, top_indices_e):
        batch_size = top_e.shape[0]
        x_hat = t.zeros(batch_size, self.activation_dim, device=top_e.device)
        for i in range(self.e):
            expert_ids = top_indices_e[:, i]
            weights = top_e[:, i]
            for expert in range(self.experts):
                idx = (expert_ids == expert).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                acts = top_acts_list[expert][idx]
                indices = top_indices_list[expert][idx]
                x_hat_expert = TritonDecoder.apply(indices, acts, self.decoders[expert].mT)
                x_hat[idx] += weights[idx].unsqueeze(1) * x_hat_expert
        return x_hat + self.b_dec
    
    def encode(self, x):
        top_e, top_indices_e = self.gate(x)
        batch_size = x.shape[0]
        top_acts_list = [t.zeros(batch_size, self.k, device=x.device) for _ in range(self.experts)]
        top_indices_list = [t.zeros(batch_size, self.k, dtype=t.long, device=x.device) for _ in range(self.experts)]
        f = t.zeros(batch_size, self.dict_size, device=x.device)

        for i in range(self.e):
            expert_idx = top_indices_e[:, i]
            for expert in range(self.experts):
                idx = (expert_idx == expert).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                x_expert = x[idx]
                z = nn.functional.relu(self.encoders[expert](x_expert - self.b_dec))
                top_acts, top_indices_k = z.topk(self.k, sorted=False)
                top_acts_list[expert][idx] = top_acts
                top_indices_list[expert][idx] = top_indices_k
                offset = expert * self.expert_dict_size
                f[idx, offset:offset+self.expert_dict_size] = z
        return top_acts_list, top_indices_list, top_e, top_indices_e, f

    def forward(self, x, output_features=False):
        top_acts_list, top_indices_list, top_e, top_indices_e, f = self.encode(x.view(-1, x.shape[-1]))
        x_hat = self.decode(top_acts_list, top_indices_list, top_e, top_indices_e).view(x.shape)
        if not output_features:
            return x_hat
        else:
            return x_hat, f
        
    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        for expert in range(self.experts):
            eps = t.finfo(self.decoders[expert].dtype).eps
            norm = t.norm(self.decoders[expert].data, dim=1, keepdim=True)
            self.decoders[expert].data /= norm + eps

    @t.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        for expert in range(self.experts):
            if self.decoders[expert].grad is None:
                continue
            parallel_component = einops.einsum(
                self.decoders[expert].grad,
                self.decoders[expert].data,
                "d_sae d_in, d_sae d_in -> d_sae",
            )
            self.decoders[expert].grad -= einops.einsum(
                parallel_component,
                self.decoders[expert].data,
                "d_sae, d_sae d_in -> d_sae d_in",
            )
                   
    def from_pretrained(path, k=100, experts=16, e=1, heaviside=False, device=None
                        , activation_dim=768, dict_size=32*768):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        autoencoder = MoEAutoEncoder(activation_dim, dict_size, k, experts, e, heaviside)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder
    

class MoETrainer(SAETrainer):
    """
    MoE SAE training scheme.
    """
    def __init__(self,
                 dict_class=MoEAutoEncoder,
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
        
        top_acts_list, top_indices_list, top_e, top_indices_e, f = self.ae.encode(x)
        x_hat = self.ae.decode(top_acts_list, top_indices_list, top_e, top_indices_e)
        
        e = x_hat - x
        total_variance = (x - x.mean(0)).pow(2).sum(0)

        self.effective_l0 = sum([acts.nonzero().size(0) for acts in top_acts_list])
        
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        for expert in range(self.experts):
            did_fire[top_indices_list[expert].flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0
        
        dead_mask = (
            self.num_tokens_since_fired > self.dead_feature_threshold
            if self.auxk_alpha > 0
            else None
        ).to(f.device)
        self.dead_features = int(dead_mask.sum())
        
        # if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            
        #     k_aux = x.shape[-1] // 2

        #     scale = min(num_dead / k_aux, 1.0)
        #     k_aux = min(k_aux, num_dead)

        #     auxk_latents = t.where(dead_mask[None], f, -t.inf)

        #     auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

        #     e_hat = self.ae.decode(auxk_acts, auxk_indices)
        #     auxk_loss = (e_hat - e).pow(2)
        #     auxk_loss = scale * t.mean(auxk_loss / total_variance)
        # else:
        auxk_loss = x_hat.new_tensor(0.0)

        h = self.ae.gater(x - self.ae.b_gate)
        p = t.nn.functional.softmax(h, dim=-1)

        flb = t.argmax(p, dim=1)
        flb = t.nn.functional.one_hot(flb, num_classes=self.experts).float()
        flb = flb.mean(dim=0)

        P = p.mean(dim=0)

        lb_loss = self.experts * t.dot(flb, P)

        lb_loss_weight = 3
        
        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = auxk_loss.sum(dim=-1).mean()

        # decoder_matrix = t.cat([d for d in self.ae.decoders], dim=0)

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
        # print(f"l2_loss: {l2_loss.item():.4f}, lb_loss: {lb_loss.item():.4f}, loss: {loss.item():.4f}")
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
