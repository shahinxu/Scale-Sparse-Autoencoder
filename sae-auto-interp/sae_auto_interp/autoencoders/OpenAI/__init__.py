from functools import partial
from typing import List

import torch

from ..wrapper import AutoencoderLatents
import os
import sys
sys.path.append(os.path.abspath("/home/xuzhen/switch_sae"))
from dictionary_learning.trainers.switch import SwitchAutoEncoder
from dictionary_learning.trainers.moe_logically_scale_noise import ScaleNoiseAutoEncoder
from dictionary_learning.trainers.moe_logically_noise import NoiseAutoEncoder
from dictionary_learning.trainers.moe_logically import MoeAutoEncoder
from dictionary_learning.trainers.moe_logically_scale import ScaleAutoEncoder
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
DEVICE = "cuda:5"


def load_oai_autoencoders(model, ae_layers: List[int], weight_dir: str):
    submodules = {}

    for layer in ae_layers:
        path = f"{weight_dir}/{layer}.pt"
        state_dict = torch.load(path)
        # ae = ScaleNoiseAutoEncoder(activation_dim=768, dict_size=32*768, k=32, experts=64, e=8, heaviside=False)
        # ae = NoiseAutoEncoder(activation_dim=768, dict_size=32*768, k=32, experts=64, e=8, heaviside=False)
        ae = ScaleAutoEncoder(activation_dim=768, dict_size=32*768, k=32, experts=64, e=8, heaviside=False)
        # ae = MultiExpertAutoEncoder(activation_dim=768, dict_size=32*768, k=32, experts=64, e=8, heaviside=False)
        # ae = MoeAutoEncoder(activation_dim=768, dict_size=32*768, k=32, experts=64, e=8, heaviside=False)
        # ae = SwitchAutoEncoder(activation_dim=768, dict_size=32*768, k=32, experts=64, heaviside=False)

        ae.load_state_dict(state_dict)
        ae.to(DEVICE)

        def _forward(ae, x):
            latents = ae.encode(x.view(-1, x.shape[-1]))
            top_acts, top_indices = latents.topk(32, sorted=False)
            latents = torch.zeros_like(latents)
            latents.scatter_(1, top_indices, top_acts)
            latents = latents.view(x.shape[0], x.shape[1], -1)
            return latents

        submodule = model.transformer.h[layer]

        submodule.ae = AutoencoderLatents(ae, partial(_forward, ae), width=32*768)

        submodules[submodule._path] = submodule

    with model.edit(" "):
        for _, submodule in submodules.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

    return submodules
