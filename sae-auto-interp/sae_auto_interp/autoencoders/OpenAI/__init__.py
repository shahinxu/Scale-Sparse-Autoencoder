from functools import partial
from typing import List

import torch

from ..wrapper import AutoencoderLatents
import os
import sys
sys.path.append(os.path.abspath("/home/xuzhen/switch_sae"))
from dictionary_learning.trainers.moe_lb import MoEAutoEncoder
DEVICE = "cuda:0"


def load_oai_autoencoders(model, ae_layers: List[int], weight_dir: str):
    submodules = {}

    for layer in ae_layers:
        path = f"{weight_dir}/{layer}.pt"
        state_dict = torch.load(path)
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        ae = MoEAutoEncoder(activation_dim, dict_size, k=32, experts=64, e=2, heaviside=False)
        ae.load_state_dict(state_dict)
        ae.to(DEVICE)

        def _forward(ae, x):
            latents = ae.encode(x)
            return latents

        submodule = model.transformer.h[layer]

        submodule.ae = AutoencoderLatents(ae, partial(_forward, ae), width=131_072)

        submodules[submodule._path] = submodule

    with model.edit(" "):
        for _, submodule in submodules.items():
            acts = submodule.output[0]
            submodule.ae(acts, hook=True)

    return submodules
