# run_interpretability.py

import pandas as pd
import matplotlib.pyplot as plt
import os
from nnsight import LanguageModel
from sae_auto_interp.autoencoders import load_oai_autoencoders
from sae_auto_interp.utils import load_tokenized_data
from sae_auto_interp.features import FeatureCache, FeatureDataset
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
import random
import argparse # 导入 argparse 模块

def main(advance_path_arg, loop_iteration): # 添加 loop_iteration 参数
    base_path = "/home/xuzhen/switch_sae/dictionaries"
    # advance_path = "ScaleEncoder" # This will now come from advance_path_arg
    PATH_TO_WEIGHTS = f"{base_path}/{advance_path_arg}/"

    # 根据 loop_iteration 修改 fuzz_dir 和 explanation_dir
    fuzz_dir = f"result/{advance_path_arg}/{advance_path_arg}_{loop_iteration}/gpt2_fuzz"
    explanation_dir = f"result/{advance_path_arg}/{advance_path_arg}_{loop_iteration}/gpt2_explanations"

    print(f"Running with advance_path: {advance_path_arg}, iteration: {loop_iteration}")
    print(f"Fuzzing results will be saved to: {fuzz_dir}")
    print(f"Explanations will be saved to: {explanation_dir}")

    # Ensure output directories exist
    os.makedirs(fuzz_dir, exist_ok=True)
    os.makedirs(explanation_dir, exist_ok=True)

    # Load model and SAEs
    print("Loading LanguageModel...")
    model = LanguageModel("/home/xuzhen/switch_sae/gpt2", device_map="auto", dispatch=True)
    print("Loading SAE autoencoders...")
    submodule_dict = load_oai_autoencoders(model, [8], PATH_TO_WEIGHTS)

    # Load tokenized data
    print("Loading tokenized data...")
    tokens = load_tokenized_data(
        ctx_len=256,
        tokenizer=model.tokenizer, 
        dataset_repo="openwebtext2",
        # dataset_repo="/home/xuzhen/switch_sae/openwebtext", # Uncomment if using local path
        dataset_split="train[:15%]",
    )

    # Feature Cache
    print("Running FeatureCache...")
    cache = FeatureCache(model, submodule_dict, batch_size=8)
    cache.run(n_tokens=10_000_000, tokens=tokens)
    cache.save_splits(n_splits=1, save_dir="raw_features")

    # Configuration for features
    cfg = FeatureConfig(
        width = 32*768,
        min_examples = 200,
        max_examples = 10_000,
        example_ctx_len = 20,
        n_splits = 1
    )

    sample_cfg = ExperimentConfig()

    modules = [f".transformer.h.{8}"]

    features = {mod: torch.tensor(random.sample(range(cfg.width), 10_000)) for mod in modules}

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
        # Use the dynamically created explanation_dir
        file = f"{explanation_dir}/{result.record.feature}.json"
        if not os.path.exists(os.path.dirname(file)): # Check parent directory
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

    def scorer_postprocess(result, score_dir_param): # Renamed to avoid conflict with outer fuzz_dir
        # Use the dynamically created fuzz_dir
        file = f"{score_dir_param}/{result.record.feature}.json"
        if not os.path.exists(os.path.dirname(file)): # Check parent directory
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
                batch_size=5,
            ),
            preprocess=scorer_preprocess,
            postprocess=partial(scorer_postprocess, score_dir_param=fuzz_dir),
        ),
    )

    ### Build the pipeline ###
    print("Building and running pipeline...")
    pipeline = Pipeline(
        loader,
        explainer_pipe,
        scorer_pipe,
    )

    asyncio.run(pipeline.run(max_processes=5))
    print("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM interpretability pipeline.")
    parser.add_argument("--advance_path", type=str, required=True,
                        help="The advance_path (e.g., 'ScaleEncoder').")
    parser.add_argument("--iteration", type=int, required=True,
                        help="The current loop iteration number (i).")
    
    args = parser.parse_args()
    main(args.advance_path, args.iteration)