import argparse
import os
import re
import csv
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch as t
try:
    from transformers import pipeline
    _HAVE_TRANSFORMERS = True
except Exception:
    _HAVE_TRANSFORMERS = False

# track limited LLM parse failures to avoid flooding the log
_LLM_PARSE_FAILURES = 0



def find_feature_token_sources(root: str) -> List[Tuple[str, str, List[str]]]:
    """Return list of tuples (expert_id, feature_id, tokens_list).
    Heuristics: look for .csv files first, then .txt lines under subdirs.
    """
    results = []
    # CSV files
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.csv'):
                path = os.path.join(dirpath, fn)
                with open(path, newline='') as f:
                    reader = csv.DictReader(f)
                    if {'expert', 'feature', 'tokens'}.issubset(reader.fieldnames or []):
                        for row in reader:
                            expert = row['expert']
                            feat = row['feature']
                            tokens = row['tokens'].split()
                            results.append((expert, feat, tokens))
    # txt files: treat each line as one feature; expert inferred from parent folder
    for dirpath, dirnames, filenames in os.walk(root):
        parent = os.path.basename(dirpath.rstrip('/'))
        for fn in filenames:
            if fn.endswith('.txt'):
                path = os.path.join(dirpath, fn)
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        tokens = line.split()
                        feat_id = f"{fn}:{i}"
                        results.append((parent, feat_id, tokens))
    return results


def heuristic_semantic_count(tokens: List[str], threshold: float = 0.7) -> int:
    if not tokens:
        return 0
    cnt = Counter(tokens)
    total = sum(cnt.values())
    cum = 0
    for i, (_, c) in enumerate(cnt.most_common(), start=1):
        cum += c
        # return the number of semantics needed to cover `threshold` of tokens
        # cap the reported value at 20 (meaning "20+")
        if cum / total >= threshold:
            return i if i <= 20 else 20
    return 20


def llm_semantic_count(llm, tokens: List[str], threshold: float = 0.7, print_raw: bool = False) -> int:
    # Build a concise, directive prompt that asks the LLM to return exactly "Answer: N"
    token_sample = ' '.join(tokens[:50]) + ('' if len(tokens) <= 50 else ' ...')
    prompt = (
        "Given the following tokens from a single feature:\n"
        f"{token_sample}\n\n"
        "Estimate the number of distinct semantic concepts represented in this feature using this rule: "
        "if one semantic covers >=70% of tokens, answer 1; if top two semantics together cover >=70%, answer 2; "
        "otherwise return the smallest integer that reaches the threshold.\n"
        "Return only and exactly a single line in this exact format: Answer: N\n"
        "where N is an integer between 0 and 20 (inclusive). Do NOT include any other text, code blocks, or explanation.\n"
        "Return format example (exact): Answer: 2\n\n"
    )

    def _text_from_llm(llm_obj, prm):
        # Use transformers pipeline output format where possible.
        try:
            # If llm_obj is a transformers pipeline, calling it returns a list of dicts
            out = llm_obj(prm, max_new_tokens=6, do_sample=False, temperature=0.0, return_full_text=False)
        except TypeError:
            # Some callables have signature llm(prompt) -> dict or str
            try:
                out = llm_obj(prm)
            except Exception:
                return ''
        except Exception:
            return ''

        if print_raw:
            try:
                print(f"[LLM raw output repr] type={type(out)} repr={repr(out)[:500]}")
            except Exception:
                pass

        # transformers pipeline returns list[{'generated_text': ...}]
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                for key in ('generated_text', 'text', 'output'):
                    if key in first and isinstance(first[key], str):
                        return first[key]
                # fallback to any string value
                for v in first.values():
                    if isinstance(v, str) and v.strip():
                        return v
                return str(first)
            if isinstance(first, str):
                return first
            return str(first)

        if isinstance(out, dict):
            for key in ('generated_text', 'text', 'output'):
                if key in out and isinstance(out[key], str):
                    return out[key]
            for v in out.values():
                if isinstance(v, str) and v.strip():
                    return v
            return str(out)

        if isinstance(out, str):
            return out

        return str(out)

    try:
        text = _text_from_llm(llm, prompt)
        # Prefer explicit 'Answer: N' format
        m = re.search(r"Answer\s*[:\-]?\s*(\d{1,2})", text, re.IGNORECASE)
        if not m:
            m = re.search(r"(\d{1,2})", text)
        if m:
            v = int(m.group(1))
            return min(max(v, 0), 20)

        # map written numbers if present (zero..twenty)
        words_to_num = {
            'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,
            'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,
            'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20
        }
        for w, n in words_to_num.items():
            if re.search(rf"\b{w}\b", text, re.IGNORECASE):
                return n
    except Exception:
        text = ''

    global _LLM_PARSE_FAILURES
    if _LLM_PARSE_FAILURES < 3:
        try:
            print(f"[LLM parse failed] sample tokens={token_sample}\nllm_text={text[:400]!r}")
        except Exception:
            print("[LLM parse failed] (couldn't print llm text)")
        _LLM_PARSE_FAILURES += 1

    return heuristic_semantic_count(tokens, threshold=threshold)


def aggregate_and_plot(results: List[Tuple[str, str, int]], out_csv: str, out_png: str):
    # results: list of (expert, feature, count)
    # write CSV
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['expert', 'feature', 'count'])
        for r in results:
            w.writerow([r[0], r[1], r[2]])

    # distribution 0..20 (20 means 20+)
    counts = [r[2] for r in results]
    bins = list(range(0, 21))  # 0..20 where 20 represents 20+
    hist = [counts.count(b) for b in bins]
    total = sum(hist)
    props = [h / total if total else 0.0 for h in hist]

    plt.figure(figsize=(10,4))
    xs = [str(i) for i in range(0, 20)] + ['20+']
    plt.bar(xs, props, color='C2')
    plt.xticks(rotation=45)
    plt.ylabel('Proportion')
    plt.xlabel('Estimated semantic count')
    plt.title('Distribution of semantic counts per feature')
    plt.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


folder_name = "expert_feature_analysis_MultiExpert_8_1_wikitext"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=folder_name)
    parser.add_argument('--mode', choices=['heuristic', 'llm'], default='llm')
    parser.add_argument('--out-csv', default='feature_semantic_counts.csv')
    parser.add_argument('--out-png', default=f'feature_semantic_counts_distribution_{folder_name}.png')
    parser.add_argument('--llm_id', default='/home/xuzhen/switch_sae/Llama-3.1')
    parser.add_argument('--device', default='cuda:0', help="Device to run the model on: 'cpu' or 'cuda:0'")
    parser.add_argument('--print-llm-raw', action='store_true', dest='print_llm_raw', help='Print raw LLM outputs for debugging')
    args = parser.parse_args()

    sources = find_feature_token_sources(args.root)
    if not sources:
        print(f'No token sources found under {args.root}. Place .txt/.csv files with tokens.')
        return

    before_n = len(sources)
    sources = [(e, f, toks) for (e, f, toks) in sources if len(toks) > 20]
    after_n = len(sources)
    print(f'Filtered features by >20 examples: kept {after_n} / {before_n}')
    if not sources:
        print(f'No features with >20 examples found under {args.root}. Exiting.')
        return

    results = []
    llm = None
    if args.mode == 'llm':
        if not _HAVE_TRANSFORMERS:
            print('transformers is not installed; falling back to heuristic mode')
            args.mode = 'heuristic'
        else:
            # initialize a text-generation pipeline
            device = -1 if args.device == 'cpu' else 0
            try:
                llm = pipeline('text-generation', model=args.llm_id, device=device)
            except Exception as e:
                print(f'Failed to initialize transformers pipeline: {e}; falling back to heuristic mode')
                llm = None
                args.mode = 'heuristic'

    for expert, feat, tokens in sources:
        if args.mode == 'llm' and llm is not None:
            c = llm_semantic_count(llm, tokens, threshold=0.7, print_raw=args.print_llm_raw)
        else:
            c = heuristic_semantic_count(tokens, threshold=0.7)
        results.append((expert, feat, int(c)))

    aggregate_and_plot(results, args.out_csv, args.out_png)
    print(f'Wrote {len(results)} features to {args.out_csv} and plot to {args.out_png}')


if __name__ == '__main__':
    main()
