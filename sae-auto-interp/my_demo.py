from nnsight import LanguageModel
PATH_TO_WEIGHTS = '/home/xuzhen/switch_sae/dictionaries/moe_lb_decoder_compose/'
from sae_auto_interp.autoencoders import load_oai_autoencoders
model = LanguageModel("/home/xuzhen/switch_sae/gpt2", device_map="auto", dispatch=True)
submodule_dict = load_oai_autoencoders(model, [8], PATH_TO_WEIGHTS)
import os
from sae_auto_interp.utils import load_tokenized_data
tokens = load_tokenized_data(
    ctx_len=256,
    tokenizer=model.tokenizer, 
    dataset_repo="openwebtext2",
    dataset_split="train[:15%]",
    )

from sae_auto_interp.features import FeatureCache
cache = FeatureCache(model, submodule_dict, batch_size=8)
cache.run(n_tokens=10_000_000, tokens=tokens)
cache.save_splits(n_splits=5, save_dir="raw_features")

from sae_auto_interp.features import FeatureDataset
from functools import partial
from sae_auto_interp.features import pool_max_activation_windows, sample
from sae_auto_interp.config import FeatureConfig, ExperimentConfig
import torch
from sae_auto_interp.clients import Local
from sae_auto_interp.explainers import SimpleExplainer
import orjson
from sae_auto_interp.pipeline import Pipe, Pipeline, process_wrapper
from sae_auto_interp.scorers import FuzzingScorer
import asyncio
fuzz_dir = "results/gpt2_fuzz"
explanation_dir = "results/gpt2_explanations"

cfg = FeatureConfig(
    width = 131_072,
    min_examples = 200,
    max_examples = 10_000,
    example_ctx_len = 20,
    n_splits = 5
)

sample_cfg = ExperimentConfig()

modules = [f".transformer.h.{8}"]

features = {mod: torch.arange(50) for mod in modules}

dataset = FeatureDataset(
    raw_dir='raw_features',
    cfg=cfg,
    features=features,
    modules=modules,
)

loader = partial(
    dataset.load,
    constructor=partial(
        pool_max_activation_windows, tokens=tokens, cfg=cfg
    ),
    sampler=partial(sample, cfg=sample_cfg),
)


client = Local("Llama-3.1")

### Build Explainer pipe ###

def preprocess(record):
    test = []
    extra_examples = []

    for examples in record.test:
        test.append(examples[:5])
        extra_examples.extend(examples[5:])

    record.test = test
    record.extra_examples = extra_examples
    return record

def explainer_postprocess(result):
    file = f"{explanation_dir}/{result.record.feature}.json"
    if not os.path.exists(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "wb") as f:
        f.write(orjson.dumps(result.explanation))

    return result

explainer_pipe = process_wrapper(
    SimpleExplainer(
        client,
        tokenizer=model.tokenizer,
        activations=True,
        max_tokens=500,
        temperature=0.0,
    ),
    preprocess=preprocess,
    postprocess=explainer_postprocess,
)

### Build Scorer pipe ###

def scorer_preprocess(result):
    record = result.record
    record.explanation = result.explanation
    return record

def scorer_postprocess(result, score_dir):
    file = f"{score_dir}/{result.record.feature}.json"
    if not os.path.exists(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "wb") as f:
        f.write(orjson.dumps(result.score))

scorer_pipe = Pipe(
    process_wrapper(
        FuzzingScorer(
            client,
            tokenizer=model.tokenizer,
            verbose=True,
            max_tokens=50,
            temperature=0.0,
            batch_size=10,
        ),
        preprocess=scorer_preprocess,
        postprocess=partial(scorer_postprocess, score_dir=fuzz_dir),
    ),
)

### Build the pipeline ###

pipeline = Pipeline(
    loader,
    explainer_pipe,
    scorer_pipe,
)

asyncio.run(pipeline.run(max_processes=5))