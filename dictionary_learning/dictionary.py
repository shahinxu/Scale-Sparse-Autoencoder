"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.autograd as autograd

class Dictionary(ABC):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """
    dict_size : int # number of features in the dictionary
    activation_dim : int # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass
    
    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))
    
    def decode(self, f):
        return self.decoder(f) + self.bias
    
    def forward(self, x, output_features=False, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None: # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat
        
        else: # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(f_ghost) # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost
            
    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        autoencoder = AutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder
            
class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """
    def __init__(self, activation_dim=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x
    
    def decode(self, f):
        return f
    
    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x
        
class GatedAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with separate gating and magnitude networks.
    """
    def __init__(self, activation_dim, dict_size, initialization='default', device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(t.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        self.r_mag = nn.Parameter(t.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        if initialization == 'default':
            self._reset_parameters()
        else:
            initialization(self)

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x, return_gate=False):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).float()

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)

        f = f_gate * f_mag

        # W_dec norm is not kept constant, as per Anthropic's April 2024 Update
        # Normalizing after encode, and renormalizing before decode to enable comparability
        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def decode(self, f):
        # W_dec norm is not kept constant, as per Anthropic's April 2024 Update
        # Normalizing after encode, and renormalizing before decode to enable comparability
        f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias
    
    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)

        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        autoencoder = GatedAutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        rectangle = ((x - threshold).abs() < bandwidth / 2).float()
        x_grad = t.zeros_like(grad_output)
        threshold_grad = - (1.0 / bandwidth) * rectangle * grad_output
        return x_grad, threshold_grad, None

class JumpReLU(autograd.Function):
    """
    A jump ReLU activation function.
    """
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        # save context for backward pass
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

        # apply jump ReLU
        return x * (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Compute the gradient for x (standard backprop for ReLU part)
        x_grad = (x > threshold).float() * grad_output
        rectangle = ((x - threshold).abs() < bandwidth / 2).float()
        rectangle = rectangle.float()
        threshold_grad = - (threshold / bandwidth) * rectangle * grad_output
        return x_grad, threshold_grad, None
    
class JumpReluAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with jump ReLUs.
    """

    def __init__(self, activation_dim, dict_size, pre_encoder_bias=False):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(t.empty(activation_dim, dict_size))
        self.b_enc = nn.Parameter(t.zeros(dict_size))
        self.W_dec = nn.Parameter(t.empty(dict_size, activation_dim))
        self.b_dec = nn.Parameter(t.zeros(activation_dim))
        self.threshold = nn.Parameter(t.ones(dict_size) * 1e-3)
        self.threshold = nn.Parameter(t.ones(dict_size) * 0)
        self.bandwidth = 1e-3

        # rows of decoder weight matrix are initialized to unit vectors
        self.W_enc.data = t.randn_like(self.W_enc)
        self.W_enc.data = self.W_enc / self.W_enc.norm(dim=0, keepdim=True)
        self.W_dec.data = self.W_enc.data.clone().T

        self.pre_encoder_bias = pre_encoder_bias
        self.set_linear_to_constant = False

    def encode(self, x, output_pre_jump=False):
        if self.pre_encoder_bias:
            x = x - self.b_dec

        pre_jump = x @ self.W_enc + self.b_enc
        pre_jump = nn.ReLU()(pre_jump) # apply ReLU to pre-jump activations for stable training, see paper

        if not self.set_linear_to_constant:
            f = JumpReLU.apply(pre_jump, self.threshold, self.bandwidth)
        else:
            f = StepFunction.apply(pre_jump, self.threshold, self.bandwidth)

        # f = f * self.W_dec.norm(dim=1) # renormalize f as decoder weights are not normalized

        if output_pre_jump:
            return f, pre_jump
        else:
            return f
        

    def decode(self, f):
        # f = f / self.W_dec.norm(dim=1) # renormalize f as decoder weights are not normalized
        return f @ self.W_dec + self.b_dec
    
    def forward(self, x, output_features=False, set_linear_to_constant=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features (and their pre-jump version) as well as the decoded x
        """
        self.set_linear_to_constant = set_linear_to_constant

        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat
    
    def from_pretrained(
            path: str | None = None, 
            load_from_sae_lens: bool = False,
            device: t.device | None = None,
            **kwargs,
    ):
        """
        Load a pretrained autoencoder from a file.
        If sae_lens=True, then pass **kwargs to sae_lens's
        loading function.
        """
        if not load_from_sae_lens:
            state_dict = t.load(path)
            dict_size, activation_dim = state_dict['W_enc'].shape
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size)
            autoencoder.load_state_dict(state_dict)
        else:
            from sae_lens import SAE
            sae, cfg_dict, _ = SAE.from_pretrained(**kwargs, device=device)
            assert cfg_dict["finetuning_scaling_factor"] == False, "Finetuning scaling factor not supported"
            dict_size, activation_dim = cfg_dict["d_sae"], cfg_dict["d_in"]
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size)
            autoencoder.load_state_dict(sae.state_dict())
            autoencoder.apply_b_dec_to_input = cfg_dict["apply_b_dec_to_input"]

        if device is not None:
            autoencoder.to(device)
        return autoencoder

# TODO merge this with AutoEncoder
class AutoEncoderNew(Dictionary, nn.Module):
    """
    The autoencoder architecture and initialization used in https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        # initialize encoder and decoder weights
        w = t.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

        # initialize biases to zeros
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x))
    
    def decode(self, f):
        return self.decoder(f)
    
    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        """
        if not output_features:
            return self.decode(self.encode(x))
        else: # TODO rewrite so that x_hat depends on f
            f = self.encode(x)
            x_hat = self.decode(f)
            # multiply f by decoder column norms
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)
            return x_hat, f
            
    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        autoencoder = AutoEncoderNew(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class MatryoshkaAutoEncoder(Dictionary, nn.Module):
    """
    Matryoshka-style autoencoder.

    Holds a full parameterization for `total_dict_size` features but when
    encoding/decoding only uses the first `dict_size` features. Mirrors
    the other dictionary APIs in this file (encode/decode/forward/from_pretrained).
    """
    def __init__(self, activation_dim, total_dict_size, dict_size=None, bias=True):
        super().__init__()
        self.activation_dim = activation_dim
        self.total_dict_size = int(total_dict_size)
        self.dict_size = int(dict_size) if dict_size is not None else int(total_dict_size)

        # encoder produces scores for the full outer dictionary
        self.encoder = nn.Linear(activation_dim, self.total_dict_size, bias=bias)

        # decoder stores the full outer dictionary as columns; we'll only use
        # the first `dict_size` columns when decoding
        self.decoder = nn.Linear(self.total_dict_size, activation_dim, bias=False)

        # initialize decoder columns as unit vectors (consistent with other dicts)
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / (dec_weight.norm(dim=0, keepdim=True) + 1e-12)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x):
        """Encode x into the active feature space (size=self.dict_size).

        Returns tensor of shape (B, dict_size).
        """
        scores = self.encoder(x)
        f = nn.ReLU()(scores[:, : self.dict_size])
        return f

    def decode(self, f):
        """Decode features f (B, dict_size) using the first dict_size decoder columns."""
        if f.dim() == 1:
            f = f.unsqueeze(0)
        dec_cols = self.decoder.weight[:, : self.dict_size]  # (activation_dim, dict_size)
        x_hat = t.matmul(f, dec_cols.T)
        return x_hat

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            # scale features by decoder column norms to match other implementations
            col_norms = self.decoder.weight[:, : self.dict_size].norm(dim=0, keepdim=True)
            f_scaled = f * col_norms
            return x_hat, f_scaled
        return x_hat

    def from_pretrained(path, device=None, active_dict_size=None):
        """Load a MatryoshkaAutoEncoder from a state dict saved for a full-size autoencoder.

        The function is robust to a few different state-dict naming conventions used in this repo.
        """
        state = t.load(path, map_location='cpu')
        # try several possible keys to infer shapes
        if 'encoder.weight' in state:
            total_dict_size, activation_dim = state['encoder.weight'].shape
        elif 'W_enc' in state:
            total_dict_size, activation_dim = state['W_enc'].shape
        elif 'decoder.weight' in state:
            try:
                activation_dim, total_dict_size = state['decoder.weight'].shape
            except Exception:
                raise RuntimeError("Could not infer shapes from state dict for MatryoshkaAutoEncoder")
        else:
            raise RuntimeError("Could not infer shapes from state dict for MatryoshkaAutoEncoder")

        mat = MatryoshkaAutoEncoder(activation_dim=activation_dim, total_dict_size=total_dict_size, dict_size=active_dict_size or total_dict_size)

        # load encoder weights if present
        if 'encoder.weight' in state:
            try:
                mat.encoder.weight.data.copy_(state['encoder.weight'])
            except Exception:
                mat.encoder.weight.data.copy_(state['encoder.weight'].T)
        if 'encoder.bias' in state and getattr(mat.encoder, 'bias', None) is not None:
            mat.encoder.bias.data.copy_(state['encoder.bias'])

        # load decoder weight if present
        if 'decoder.weight' in state:
            try:
                mat.decoder.weight.data.copy_(state['decoder.weight'])
            except Exception:
                mat.decoder.weight.data.copy_(state['decoder.weight'].T)

        if device is not None:
            mat.to(device)
        return mat