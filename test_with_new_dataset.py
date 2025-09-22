from nnsight import LanguageModel
import torch as t
from dictionary_learning.utils import cfg_filename, str2bool
from dictionary_learning.trainers.moe_physically_scale import MultiExpertScaleAutoEncoder
import argparse
import itertools
from config import lm, activation_dim, layer
import os
import json
from typing import Dict, Any, Iterable, List
import pyarrow.parquet as pq


def compute_mse_dataset(
    ae: MultiExpertScaleAutoEncoder,
    model: LanguageModel,
    submodule,
    text_batch_iter: Iterable[List[str]],
    device: str,
    eval_batches: int,
    ctx_len: int,
    tracer_scan: bool = False,
    tracer_validate: bool = False,
) -> Dict[str, Any]:
    ae.eval()
    total_se = 0.0
    total_elems = 0
    total_samples = 0

    tracer_kwargs = {'scan': tracer_scan, 'validate': tracer_validate}

    with t.no_grad():
        for batch_idx, texts in enumerate(text_batch_iter):
            if batch_idx >= eval_batches:
                break

            tok = model.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=ctx_len,
            )

            if 'attention_mask' not in tok:
                attn_mask = (tok['input_ids'] != model.tokenizer.pad_token_id).long()
                tok['attention_mask'] = attn_mask

            with model.trace(tok, **tracer_kwargs):
                hidden_states_node = submodule.output.save()
                model_input_node = model.input.save()

            hidden_states = hidden_states_node.value
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            try:
                attn_mask = model_input_node.value[1]['attention_mask']
            except Exception:
                # Fallback to tokenizer-produced mask
                attn_mask = tok['attention_mask']

            # Filter padded positions
            mask = attn_mask.to(hidden_states.device).bool()
            # Align mask shape to [B, L, 1]
            mask_expanded = mask.unsqueeze(-1).expand_as(hidden_states)
            kept = hidden_states[mask_expanded].view(-1, hidden_states.size(-1))

            if kept.numel() == 0:
                continue

            x = kept.to(device)
            x_hat = ae(x)
            se = t.nn.functional.mse_loss(x_hat, x, reduction='sum')
            total_se += float(se.item())
            total_elems += x.numel()
            total_samples += x.shape[0]

    mse = total_se / total_elems if total_elems > 0 else float('nan')
    return {
        'mse': mse,
        'total_elements': int(total_elems),
        'total_samples': int(total_samples),
        'batches': int(min(eval_batches, batch_idx + 1) if total_elems > 0 else 0),
    }


def load_scale_ae_or_fallback(cfg: Dict[str, Any], device: str) -> MultiExpertScaleAutoEncoder:
    k = cfg['k']; experts = cfg['experts']; e = cfg['e']; heaviside = cfg['heaviside']

    # Try from_pretrained folder structure
    dir_path = os.path.join('dictionaries', cfg_filename(cfg))
    ckpt_path = os.path.join(dir_path, 'ae.pt')
    if os.path.exists(ckpt_path):
        return MultiExpertScaleAutoEncoder.from_pretrained(
            ckpt_path, k=k, experts=experts, e=e, heaviside=heaviside, device=device
        )

    # Fallback to direct state_dict checkpoint naming convention
    alt_ckpt = os.path.join('dictionaries', f"MultiExpert_Scale_{k}_{experts}_{e}", f"{layer}.pt")
    ae = MultiExpertScaleAutoEncoder(
        activation_dim=activation_dim,
        dict_size=cfg['dict_size'],
        k=k, experts=experts, e=e, heaviside=heaviside
    )
    if os.path.exists(alt_ckpt):
        state = t.load(alt_ckpt, map_location='cpu')
        ae.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} or {alt_ckpt}")
    ae.to(device)
    return ae


def qa_parquet_text_iter(parquet_path: str, fields=("question", "rationale")) -> Iterable[str]:
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=1024):
        tbl = batch.to_pylist()
        for row in tbl:
            parts = []
            for f in fields:
                if f in row and row[f]:
                    parts.append(str(row[f]))
            text = "\n".join(parts).strip()
            if text:
                yield text


def batchify(iterator: Iterable[str], batch_size: int, repeat: bool = True) -> Iterable[List[str]]:
    buf: List[str] = []
    while True:
        for item in iterator:
            buf.append(item)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf and not repeat:
            yield buf
        if not repeat:
            break
        # restart the underlying iterator
        iterator = iter(iterator)


def get_model_max_ctx_len(model: LanguageModel, default: int = 1024) -> int:
    # Prefer HF config if available
    try:
        n_pos = getattr(model.config, 'n_positions', None)
        if isinstance(n_pos, int) and n_pos > 0:
            return n_pos
    except Exception:
        pass
    # Fallback to tokenizer
    try:
        mlen = getattr(model.tokenizer, 'model_max_length', None)
        if isinstance(mlen, int) and 0 < mlen < 10_000_000:
            return mlen
    except Exception:
        pass
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument('--dict_ratio', type=int, default=32)
    parser.add_argument("--ks", nargs="+", type=int, required=True)
    parser.add_argument("--num_experts", nargs="+", type=int, required=True)
    parser.add_argument("--es", nargs="+", type=int, required=True)
    parser.add_argument("--heavisides", nargs="+", type=str2bool, required=True)
    # dataset args
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_length", type=int, default=128)
    # eval args
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batches", type=int, default=50)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    model = LanguageModel(lm, dispatch=True, device_map=device)
    submodule = model.transformer.h[layer]

    # Enforce safe max context length
    model_max = get_model_max_ctx_len(model, default=1024)
    if args.ctx_len > model_max:
        print(f"[warn] ctx_len {args.ctx_len} > model max {model_max}; clamping to {model_max}")
        args.ctx_len = model_max

    # Fixed QA dataset
    qa_path = os.path.join('QA', 'test_sampled_biology_medicine.parquet')
    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"Expected QA parquet at {qa_path}")
    text_iter = qa_parquet_text_iter(qa_path, fields=("question", "rationale"))
    text_batch_iter = batchify(text_iter, batch_size=args.batch_size, repeat=True)

    base_cfg = {
        'dict_class': MultiExpertScaleAutoEncoder,
        'activation_dim': activation_dim,
        'dict_size': args.dict_ratio * activation_dim,
        'device': device,
        'layer': layer,
        'lm_name': lm,
    }

    combos = itertools.product(args.ks, args.num_experts, args.es, args.heavisides)
    trainer_configs = [
        (base_cfg | {'k': k, 'experts': ex, 'e': e, 'heaviside': h})
        for (k, ex, e, h) in combos
    ]

    print("Evaluating MSE on new dataset...", flush=True)
    with open("metrics_log.jsonl", "a", encoding='utf-8') as f:
        for i, cfg in enumerate(trainer_configs):
            try:
                ae = load_scale_ae_or_fallback(cfg, device=device)
                metrics = compute_mse_dataset(
                    ae,
                    model,
                    submodule,
                    text_batch_iter,
                    device=device,
                    eval_batches=args.eval_batches,
                    ctx_len=args.ctx_len,
                )
            except Exception as e:
                metrics = {'error': str(e)}
            safe_cfg = {k: (str(v) if callable(v) or isinstance(v, type) else v) for k, v in cfg.items()}
            record = {"trainer_config": safe_cfg, "metrics": metrics}
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            print(record)


if __name__ == "__main__":
    main()