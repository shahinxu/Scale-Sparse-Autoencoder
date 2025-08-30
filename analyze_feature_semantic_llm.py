#!/usr/bin/env python3
"""
Single-file tool: use a local LLM to label tokens for features then compute and plot
the discrete semantic-count distribution using the 80%-dominance rule.

This script is self-contained and does not depend on other repo modules.
It supports a sampling mode and two backends: ``llama_cpp`` (ggml) and ``transformers`` (HF).
"""

import argparse
import json
import os
import glob
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


def find_feature_files(features_root):
    pattern = os.path.join(features_root, '**', 'features', 'feature_*.txt')
    return sorted(glob.glob(pattern, recursive=True))


def write_csv(rows, out_csv):
    import csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = ['feature_path', 'expert', 'token_count', 'distinct_labels', 'dominant_proportion', 'semantic_count']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote CSV summary to {out_csv}")


def make_distribution(counts, max_x=5):
    dist = defaultdict(int)
    missing = 0
    for c in counts:
        if c is None:
            missing += 1
            continue
        if c >= max_x:
            dist[max_x] += 1
        else:
            dist[c] += 1
    return dist, missing


def plot_distribution(dist, missing, out_png, max_x=5):
    xs_labels = [str(x) for x in range(0, max_x)] + [f'>={max_x}']
    counts = [dist.get(i, 0) for i in range(0, max_x)] + [dist.get(max_x, 0)]
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(counts)), counts, color='C0')
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(xs_labels)
    ax.set_xlabel('semantic count (discrete)')
    ax.set_ylabel('number of features')
    ax.set_title('Feature semantic-count distribution')
    for i, b in enumerate(bars):
        h = b.get_height()
        if h > 0:
            pct = h / total * 100 if total > 0 else 0
            ax.text(b.get_x() + b.get_width() / 2, h + max(1, total * 0.01), f"{int(h)}\n{pct:.1f}%",
                    ha='center', va='bottom', fontsize=8)
    if missing:
        ax.text(0.99, 0.99, f"missing annotations: {missing}", transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='gray')
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Wrote distribution plot to {out_png}")


def build_prompt(tokens, exemplar=None):
    system = (
        "You are a helpful annotator that assigns a concise semantic label to each token.\n"
        "Output must be a JSON array of strings, one label per input token, preserving order.\n"
        "Labels should be short (single token or short phrase), consistent across tokens.\n"
        "If a token is truly meaningless, use the label \"UNK\".\n"
        "Do not output any extra text, commentary, or explanation. Only output the JSON array.\n"
    )
    few_shot = ''
    if exemplar is not None:
        few_shot = (
            'Example input tokens:\n' + '\n'.join(exemplar['tokens']) + '\n'
            'Example output (JSON array):\n' + json.dumps(exemplar['labels']) + '\n\n'
        )
    body = 'Input tokens (one per line):\n' + '\n'.join(tokens) + '\n\n'
    prompt = system + few_shot + body + 'Respond with JSON array now.'
    return prompt


def parse_response_text(resp_text, n_tokens):
    s = resp_text.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and len(obj) == n_tokens:
            return [str(x).strip() for x in obj]
    except Exception:
        pass
    lines = [l.strip() for l in s.splitlines() if l.strip()]
    if len(lines) == n_tokens:
        return lines
    import re
    m = re.search(r"\[.*\]", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, list) and len(obj) == n_tokens:
                return [str(x).strip() for x in obj]
        except Exception:
            pass
    return None


def llama_cpp_generate(model_path, prompt, max_tokens=256, temperature=0.0):
    try:
        import llama_cpp
    except Exception as e:
        raise RuntimeError('llama_cpp backend requested but `llama_cpp` is not importable: ' + str(e))
    Llama = getattr(llama_cpp, 'Llama', None)
    if Llama is None:
        # Some installations might expose a differently-named entry; fail with guidance
        raise RuntimeError('llama_cpp is installed but does not expose `Llama`. Please ensure you have `llama-cpp-python` (provides `llama_cpp.Llama`).')
    # instantiate and call
    llm = Llama(model_path=model_path)
    out = llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
    # llama_cpp returns a dict-like with choices/text on successful call
    if isinstance(out, dict):
        return out.get('choices', [{}])[0].get('text', '')
    # fallback: try to string-coerce
    return str(out)


def transformers_generate(model_path, prompt, max_new_tokens=256, temperature=0.0, force_cpu=False):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as e:
        raise RuntimeError('transformers backend requested but `transformers` is not available: ' + str(e))

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # Try to load model with device_map='auto' and fp16 to reduce GPU memory, unless force_cpu
    model = None
    load_kwargs = {'low_cpu_mem_usage': True}
    try:
        if force_cpu:
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map={'': 'cpu'}, torch_dtype=torch.float32, **load_kwargs)
        else:
            # prefer fp16 when CUDA is available
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16, **load_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        # fallback: try CPU-only load
        try:
            print('Warning: model load with device_map failed, retrying CPU load:', e)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map={'': 'cpu'}, torch_dtype=torch.float32, **load_kwargs)
        except Exception as e2:
            raise RuntimeError('Failed to load transformers model: ' + str(e2))

    model.eval()

    # prepare inputs
    inputs = tokenizer(prompt, return_tensors='pt')
    try:
        if not force_cpu and torch.cuda.is_available() and any(getattr(p, 'is_cuda', False) for p in model.parameters()):
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=temperature)
    except RuntimeError as e:
        # handle CUDA OOM by retrying on CPU
        if 'out of memory' in str(e).lower() or 'cuda' in str(e).lower():
            print('CUDA OOM during generate, retrying on CPU...')
            try:
                import gc
                del model
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map={'': 'cpu'}, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            model.eval()
            inputs = tokenizer(prompt, return_tensors='pt')
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=temperature)
        else:
            raise

    # decode generated text
    # note: token indices for newly generated tokens calculation
    generated = out[0]
    input_len = inputs['input_ids'].shape[1]
    text = tokenizer.decode(generated[input_len:], skip_special_tokens=True)
    return text


def save_labels(labels, feature_path, features_root, annotations_root=None):
    if annotations_root:
        rel = os.path.relpath(feature_path, start=features_root)
        out_path = os.path.join(annotations_root, rel).replace('.txt', '.labels')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        out_path = feature_path[:-4] + '.labels'
    with open(out_path, 'w', encoding='utf-8') as f:
        for l in (labels or []):
            f.write((l or '') + '\n')
    return out_path


def compute_semantic_from_labels(labels, dominance_threshold=0.8):
    if labels is None:
        return None, 0, 0.0
    n = len(labels)
    if n == 0:
        return 0, 0, 0.0
    ct = Counter(labels)
    most = ct.most_common(1)[0][1]
    dom = most / n
    if dom >= dominance_threshold:
        return 1, len(ct), dom
    return len(ct), len(ct), dom


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features-root', required=True)
    p.add_argument('--annotations-root', default=None)
    p.add_argument('--backend', choices=['llama_cpp', 'transformers'], default='llama_cpp')
    p.add_argument('--model-path', required=True, help='local model path (ggml bin for llama_cpp or HF dir for transformers)')
    p.add_argument('--out-prefix', default='results/feature_semantic_dist_llm')
    p.add_argument('--dominance-threshold', type=float, default=0.8)
    p.add_argument('--max-x', type=int, default=5)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--sample', type=int, default=0, help='process only a random sample of this many features')
    p.add_argument('--force-cpu', action='store_true', help='force transformers backend to run on CPU to avoid CUDA memory issues')
    p.add_argument('--max-new-tokens', type=int, default=32, help='max new tokens to generate per feature (keeps generation short)')
    args = p.parse_args()

    features = find_feature_files(args.features_root)
    if args.sample and args.sample > 0:
        import random
        features = random.sample(features, min(args.sample, len(features)))

    print(f'Found {len(features)} features')

    rows = []
    semantic_counts = []

    exemplar = None
    if features:
        try:
            with open(features[0], 'r', encoding='utf-8') as f:
                toks = [l.strip() for l in f if l.strip()][:20]
            exemplar = {'tokens': toks, 'labels': ['LABEL'] * len(toks)}
        except Exception:
            exemplar = None

    for i, fp in enumerate(features):
        ann_path = None
        if args.annotations_root:
            candidate = os.path.join(args.annotations_root, os.path.relpath(fp, args.features_root)).replace('.txt', '.labels')
            if os.path.exists(candidate) and not args.overwrite:
                ann_path = candidate
        else:
            sibling = fp[:-4] + '.labels'
            if os.path.exists(sibling) and not args.overwrite:
                ann_path = sibling

        if ann_path:
            with open(ann_path, 'r', encoding='utf-8') as f:
                labels = [l.strip() for l in f if l.strip()]
        else:
            with open(fp, 'r', encoding='utf-8') as f:
                tokens = [l.strip() for l in f if l.strip()]
            if not tokens:
                labels = []
            else:
                prompt = build_prompt(tokens, exemplar=exemplar)
                try:
                    max_toks = args.max_new_tokens
                    if args.backend == 'llama_cpp':
                        text = llama_cpp_generate(args.model_path, prompt, max_tokens=max_toks, temperature=0.0)
                    else:
                        text = transformers_generate(args.model_path, prompt, max_new_tokens=max_toks, temperature=0.0, force_cpu=args.force_cpu)
                except Exception as e:
                    print(f'LLM generation failed for {fp}: {e}')
                    labels = None
                    rows.append({'feature_path': fp, 'expert': os.path.basename(os.path.dirname(fp)),
                                 'token_count': len(tokens), 'distinct_labels': 0, 'dominant_proportion': 0.0, 'semantic_count': None})
                    semantic_counts.append(None)
                    continue

                parsed = parse_response_text(text, len(tokens))
                if parsed is None:
                    alt_prompt = prompt + '\n\nIf the previous response failed to match the expected format, output ONLY a JSON array of labels now.'
                    try:
                        if args.backend == 'llama_cpp':
                            text2 = llama_cpp_generate(args.model_path, alt_prompt, max_tokens=args.max_new_tokens, temperature=0.0)
                        else:
                            text2 = transformers_generate(args.model_path, alt_prompt, max_new_tokens=args.max_new_tokens, temperature=0.0, force_cpu=args.force_cpu)
                        parsed = parse_response_text(text2, len(tokens))
                    except Exception:
                        parsed = None

                if parsed is None:
                    print(f'Failed to parse LLM output for {fp}; skipping')
                    labels = None
                else:
                    labels = parsed
                try:
                    out_ann = save_labels(labels, fp, args.features_root, annotations_root=args.annotations_root)
                except Exception:
                    out_ann = None

        if labels is None:
            rows.append({'feature_path': fp, 'expert': os.path.basename(os.path.dirname(fp)),
                         'token_count': 0, 'distinct_labels': 0, 'dominant_proportion': 0.0, 'semantic_count': None})
            semantic_counts.append(None)
        else:
            token_count = len(labels)
            distinct = len(set(labels))
            most = Counter(labels).most_common(1)[0][1] if token_count > 0 else 0
            dom_prop = most / token_count if token_count > 0 else 0.0
            if dom_prop >= args.dominance_threshold:
                semantic_count = 1
            else:
                semantic_count = distinct
            rows.append({'feature_path': fp, 'expert': os.path.basename(os.path.dirname(fp)),
                         'token_count': token_count, 'distinct_labels': distinct,
                         'dominant_proportion': f'{dom_prop:.4f}', 'semantic_count': semantic_count})
            semantic_counts.append(semantic_count)

        if (i + 1) % 50 == 0:
            print(f'Processed {i + 1}/{len(features)} features')

    out_csv = args.out_prefix + '.csv'
    out_png = args.out_prefix + '.png'
    write_csv(rows, out_csv)
    dist, missing = make_distribution(semantic_counts, max_x=args.max_x)
    plot_distribution(dist, missing, out_png, max_x=args.max_x)


if __name__ == '__main__':
    main()
