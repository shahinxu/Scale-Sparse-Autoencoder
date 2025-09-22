from nnsight import LanguageModel
import torch as t
from dictionary_learning.utils import cfg_filename, str2bool
from dictionary_learning.trainers.moe_physically_scale import MultiExpertScaleAutoEncoder
from dictionary_learning.trainers.moe_physically import MultiExpertAutoEncoder
from dictionary_learning.trainers.top_k import AutoEncoderTopK
from dictionary_learning.dictionary import GatedAutoEncoder
import argparse
import itertools
from config import lm, activation_dim, layer
import os
import json
from typing import Dict, Any, Iterable, List
import pyarrow.parquet as pq


def loss_recovered_specialized(
    text: List[str] | t.Tensor,
    model: LanguageModel,
    submodule,
    dictionary: MultiExpertScaleAutoEncoder,
    max_len: int | None = None,
    normalize_batch: bool = False,
    io: str = "out",
    tracer_args = {'use_cache': False, 'output_attentions': False},
):
    if isinstance(text, t.Tensor):
        invoker_args = {}
    else:
        if max_len is None:
            invoker_args = {}
        else:
            invoker_args = {"truncation": True, "max_length": max_len, "padding": True}

    # detect tuple output
    with model.trace("_"):
        temp_output = submodule.output.save()
    output_is_tuple = (type(temp_output) == tuple)

    # original
    with model.trace(text, invoker_args=invoker_args):
        logits_original = model.output.save()

    # capture x for reconstruction
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        elif io == 'out':
            x = submodule.output
            if output_is_tuple:
                x = x[0]
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        elif io == 'in_and_out':
            x = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / x.norm(dim=-1).mean()
                x = x * scale
        else:
            raise ValueError(f"Invalid value for io: {io}")
        x = x.save()

    assert len(x.shape) == 3, f"Expected x to have shape (B, L, D), got {x.shape}"
    x_hat = dictionary(x).to(model.dtype)

    # intervene with x_hat
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            xin = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / xin.norm(dim=-1).mean()
                x_hat = x_hat / scale
            submodule.input[:] = x_hat
        elif io == 'out':
            xout = submodule.output
            if output_is_tuple:
                xout = xout[0]
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / xout.norm(dim=-1).mean()
                x_hat = x_hat / scale
            if output_is_tuple:
                submodule.output[0][:] = x_hat
            else:
                submodule.output[:] = x_hat
        elif io == 'in_and_out':
            xin = submodule.input
            if normalize_batch:
                scale = (dictionary.activation_dim ** 0.5) / xin.norm(dim=-1).mean()
                x_hat = x_hat / scale
            if output_is_tuple:
                submodule.output[0][:] = x_hat
            else:
                submodule.output[:] = x_hat
        else:
            raise ValueError(f"Invalid value for io: {io}")
        logits_reconstructed = model.output.save()

    # intervene with zeros
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == 'in':
            xin = submodule.input
            submodule.input[:] = t.zeros_like(xin)
        elif io in ['out', 'in_and_out']:
            xout = submodule.output
            if output_is_tuple:
                submodule.output[0][:] = t.zeros_like(xout[0])
            else:
                submodule.output[:] = t.zeros_like(xout)
        else:
            raise ValueError(f"Invalid value for io: {io}")
        inputs_saved = model.inputs.save()
        logits_zero = model.output.save()

    # get logits tensors
    try:
        logits_original = logits_original.logits
        logits_reconstructed = logits_reconstructed.logits
        logits_zero = logits_zero.logits
    except Exception:
        pass

    # tokens
    if isinstance(text, t.Tensor):
        tokens = text
    else:
        try:
            tokens = inputs_saved[1]['input_ids']
        except Exception:
            tokens = inputs_saved[1]['input']

    # CE losses
    loss_kwargs = {'ignore_index': model.tokenizer.pad_token_id} if getattr(model, 'tokenizer', None) is not None else {}
    losses = []
    for logits in [logits_original, logits_reconstructed, logits_zero]:
        loss = t.nn.CrossEntropyLoss(**loss_kwargs)(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
        )
        losses.append(loss)
    return tuple(losses)


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
    sum_batch_mse = 0.0
    total_elems = 0
    total_samples = 0
    sum_frac_recovered = 0.0
    num_batches = 0

    tracer_kwargs = {'scan': tracer_scan, 'validate': tracer_validate}

    with t.no_grad():
        for batch_idx, texts in enumerate(text_batch_iter):
            if batch_idx >= eval_batches:
                break

            # Tokenize explicitly to enforce ctx_len and padding
            tok = model.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=ctx_len,
            )
            input_ids = tok['input_ids']
            attn_mask_tok = tok.get('attention_mask', None)

            # Trace forward using token IDs only to avoid kwarg collisions
            with model.trace(input_ids, **tracer_kwargs):
                hidden_states_node = submodule.output.save()

            # Resolve saved output to a real tensor, handling tuple outputs
            try:
                hidden_states = hidden_states_node.value
            except Exception:
                hidden_states = hidden_states_node
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            # Use tokenizer-produced attention mask if available; else treat all tokens as valid
            if attn_mask_tok is not None:
                attn_mask = attn_mask_tok.to(hidden_states.device)
            else:
                attn_mask = t.ones(hidden_states.shape[:2], dtype=t.long, device=hidden_states.device)

            # Filter padded positions
            mask = attn_mask.to(hidden_states.device).bool()
            # Align mask shape to [B, L, 1]
            mask_expanded = mask.unsqueeze(-1).expand_as(hidden_states)
            kept = hidden_states[mask_expanded].view(-1, hidden_states.size(-1))

            if kept.numel() == 0:
                continue

            x = kept.to(device)
            x_hat = ae(x)
            e = x - x_hat
            recon_mse = e.pow(2).sum(dim=-1).mean()
            sum_batch_mse += float(recon_mse.item())
            # compute frac_recovered for this batch
            try:
                # Pass pre-tokenized input_ids to ensure length safety
                loss_o, loss_r, loss_z = loss_recovered_specialized(
                    input_ids.to(model.device), model, submodule, ae, max_len=ctx_len,
                    normalize_batch=False, io='out',
                    tracer_args={'use_cache': False, 'output_attentions': False}
                )
                denom = (loss_o - loss_z).item()
                frac_rec = ((loss_r - loss_z) / (loss_o - loss_z)).item() if denom != 0 else float('nan')
            except Exception:
                frac_rec = float('nan')
            sum_frac_recovered += frac_rec
            num_batches += 1
            # 可选统计信息
            total_elems += x.numel()
            total_samples += x.shape[0]

    mse = (sum_batch_mse / num_batches) if num_batches > 0 else float('nan')
    return {
        'mse': mse,
        'frac_recovered': (sum_frac_recovered / num_batches) if num_batches > 0 else float('nan'),
        'total_elements': int(total_elems),
        'total_samples': int(total_samples),
        'batches': int(num_batches),
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


def load_moe_or_fallback(cfg: Dict[str, Any], device: str) -> MultiExpertAutoEncoder:
    k = cfg['k']; experts = cfg['experts']; e = cfg['e']; heaviside = cfg['heaviside']
    dir_path = os.path.join('dictionaries', cfg_filename(cfg))
    ckpt_path = os.path.join(dir_path, 'ae.pt')
    if os.path.exists(ckpt_path):
        return MultiExpertAutoEncoder.from_pretrained(
            ckpt_path, k=k, experts=experts, e=e, heaviside=heaviside, device=device,
            activation_dim=activation_dim, dict_size=cfg['dict_size']
        )
    # Fallback pattern used previously
    alt_ckpt = os.path.join('dictionaries', f"MultiExpert_{k}_{experts}_{e}", f"{layer}.pt")
    ae = MultiExpertAutoEncoder(
        activation_dim=activation_dim, dict_size=cfg['dict_size'],
        k=k, experts=experts, e=e, heaviside=heaviside
    )
    if os.path.exists(alt_ckpt):
        state = t.load(alt_ckpt, map_location='cpu')
        ae.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} or {alt_ckpt}")
    ae.to(device)
    return ae


def load_topk_or_fallback(cfg: Dict[str, Any], device: str) -> AutoEncoderTopK:
    k = cfg['k']
    dir_path = os.path.join('dictionaries', cfg_filename(cfg))
    ckpt_path = os.path.join(dir_path, 'ae.pt')
    if os.path.exists(ckpt_path):
        return AutoEncoderTopK.from_pretrained(ckpt_path, k=k, device=device)
    # Try a few common manual checkpoint patterns
    candidates = []
    dr = cfg.get('dict_ratio', None)
    if dr is not None:
        candidates.append(os.path.join('dictionaries', f"topk_{k}_{dr}", f"{layer}.pt"))
    # based on dict size multiple of activation_dim
    try:
        dr_guess = cfg['dict_size'] // activation_dim
        candidates.append(os.path.join('dictionaries', f"topk_{k}_{dr_guess}", f"{layer}.pt"))
    except Exception:
        pass
    for alt in candidates:
        if os.path.exists(alt):
            ae = AutoEncoderTopK(activation_dim=activation_dim, dict_size=cfg['dict_size'], k=k)
            ae.load_state_dict(t.load(alt, map_location='cpu'))
            ae.to(device)
            return ae
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} or any of: {candidates}")


def load_gated_or_fallback(cfg: Dict[str, Any], device: str) -> GatedAutoEncoder:
    dir_path = os.path.join('dictionaries', cfg_filename(cfg))
    ckpt_path = os.path.join(dir_path, 'ae.pt')
    if os.path.exists(ckpt_path):
        return GatedAutoEncoder.from_pretrained(ckpt_path, device=device)
    # Try a couple of fallback patterns
    candidates = []
    dr = cfg.get('dict_ratio', None)
    lr_label = cfg.get('lr', None)
    # First try LR-specific folder observed in this repo: dictionaries/gated_{ratio}_{lr}/{layer}.pt
    if dr is not None and lr_label is not None:
        candidates.append(os.path.join('dictionaries', f"gated_{dr}_{lr_label}", f"{layer}.pt"))
    if dr is not None:
        candidates.append(os.path.join('dictionaries', f"gated_{dr}_{layer}", f"{layer}.pt"))
    try:
        dr_guess = cfg['dict_size'] // activation_dim
        if lr_label is not None:
            candidates.append(os.path.join('dictionaries', f"gated_{dr_guess}_{lr_label}", f"{layer}.pt"))
        candidates.append(os.path.join('dictionaries', f"gated_{dr_guess}_{layer}", f"{layer}.pt"))
    except Exception:
        pass
    for alt in candidates:
        if os.path.exists(alt):
            ae = GatedAutoEncoder(activation_dim=activation_dim, dict_size=cfg['dict_size'], device=device)
            ae.load_state_dict(t.load(alt, map_location='cpu'))
            ae.to(device)
            return ae
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path} or any of: {candidates}")


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

def qa_parquet_text_stream(parquet_path: str, fields=("question", "rationale")) -> Iterable[str]:
    """Infinite stream that cycles over the QA parquet file repeatedly."""
    while True:
        for x in qa_parquet_text_iter(parquet_path, fields=fields):
            yield x


def batchify(iterator: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    """Group an iterator of strings into batches; works with endless iterators."""
    buf: List[str] = []
    for item in iterator:
        buf.append(item)
        if len(buf) >= batch_size:
            yield buf
            buf = []


def get_model_max_ctx_len(model: LanguageModel, default: int = 1024) -> int:
    try:
        n_pos = getattr(model.config, 'n_positions', None)
        if isinstance(n_pos, int) and n_pos > 0:
            return n_pos
    except Exception:
        pass
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
    # ks is only required for moe/moe_scale/topk; make optional and validate per-type
    parser.add_argument("--ks", nargs="+", type=int, required=False)
    parser.add_argument("--num_experts", nargs="+", type=int, required=False)
    parser.add_argument("--es", nargs="+", type=int, required=False)
    parser.add_argument("--heavisides", nargs="+", type=str2bool, required=False)
    parser.add_argument("--model_types", nargs="+", type=str, default=["moe_scale"],
                        help="Which models to evaluate: moe_scale, moe, topk, gated")
    parser.add_argument("--topk_dict_ratio", type=int, default=1, help="dict_ratio for TopK (default 1 -> 768)")
    # If omitted, gated falls back to --dict_ratio
    parser.add_argument("--gated_dict_ratio", type=int, default=None, help="dict_ratio for Gated (overrides --dict_ratio if set)")
    # Gated model variants identified by learning-rate labels in folder names, e.g., gated_32_10
    parser.add_argument("--learning_rates", "--lrs", nargs="+", type=int, default=None,
                        help="LR labels for gated models (e.g., 3 6 10 30 50 100)")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batches", type=int, default=50)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    model = LanguageModel(lm, dispatch=True, device_map=device)
    submodule = model.transformer.h[layer]

    model_max = get_model_max_ctx_len(model, default=1024)
    if args.ctx_len > model_max:
        print(f"[warn] ctx_len {args.ctx_len} > model max {model_max}; clamping to {model_max}")
        args.ctx_len = model_max

    # Ensure tokenizer has a pad token (GPT-2 often lacks one)
    try:
        tok = model.tokenizer
        if getattr(tok, 'pad_token_id', None) is None:
            if getattr(tok, 'eos_token_id', None) is not None:
                tok.pad_token = tok.eos_token
                tok.pad_token_id = tok.eos_token_id
    except Exception:
        pass

    qa_path = os.path.join('QA', 'test_sampled_biology_medicine.parquet')
    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"Expected QA parquet at {qa_path}")
    text_iter = qa_parquet_text_stream(qa_path, fields=("question", "rationale"))
    text_batch_iter = batchify(text_iter, batch_size=args.batch_size)

    print("Evaluating MSE on new dataset...", flush=True)
    with open("metrics_log.jsonl", "a", encoding='utf-8') as f:
        for model_type in args.model_types:
            if model_type == 'moe_scale':
                base_cfg = {
                    'model_type': 'moe_scale',
                    'dict_class': MultiExpertScaleAutoEncoder,
                    'activation_dim': activation_dim,
                    'dict_size': args.dict_ratio * activation_dim,
                    'device': device,
                    'layer': layer,
                    'lm_name': lm,
                }
                if args.ks is None:
                    raise ValueError("--ks is required for model_type 'moe_scale'")
                combos = itertools.product(args.ks, args.num_experts, args.es, args.heavisides)
                trainer_configs = [base_cfg | {'k': k, 'experts': ex, 'e': e, 'heaviside': h} for (k, ex, e, h) in combos]
                loader = load_scale_ae_or_fallback
            elif model_type == 'moe':
                base_cfg = {
                    'model_type': 'moe',
                    'dict_class': MultiExpertAutoEncoder,
                    'activation_dim': activation_dim,
                    'dict_size': args.dict_ratio * activation_dim,
                    'device': device,
                    'layer': layer,
                    'lm_name': lm,
                }
                if args.ks is None:
                    raise ValueError("--ks is required for model_type 'moe'")
                combos = itertools.product(args.ks, args.num_experts, args.es, args.heavisides)
                trainer_configs = [base_cfg | {'k': k, 'experts': ex, 'e': e, 'heaviside': h} for (k, ex, e, h) in combos]
                loader = load_moe_or_fallback
            elif model_type == 'topk':
                base_cfg = {
                    'model_type': 'topk',
                    'dict_class': AutoEncoderTopK,
                    'activation_dim': activation_dim,
                    'dict_size': args.topk_dict_ratio * activation_dim,
                    'dict_ratio': args.topk_dict_ratio,
                    'device': device,
                    'layer': layer,
                    'lm_name': lm,
                }
                if args.ks is None:
                    raise ValueError("--ks is required for model_type 'topk'")
                trainer_configs = [base_cfg | {'k': k} for k in args.ks]
                loader = load_topk_or_fallback
            elif model_type == 'gated':
                ratio = args.gated_dict_ratio if args.gated_dict_ratio is not None else args.dict_ratio
                base_cfg = {
                    'model_type': 'gated',
                    'dict_class': GatedAutoEncoder,
                    'activation_dim': activation_dim,
                    'dict_size': ratio * activation_dim,
                    'dict_ratio': ratio,
                    'device': device,
                    'layer': layer,
                    'lm_name': lm,
                }
                lrs = args.learning_rates if args.learning_rates is not None else [None]
                trainer_configs = [base_cfg | ({'lr': lr} if lr is not None else {}) for lr in lrs]
                loader = load_gated_or_fallback
            else:
                print(f"[warn] Unknown model_type '{model_type}', skipping.")
                continue

            for i, cfg in enumerate(trainer_configs):
                try:
                    ae = loader(cfg, device=device)
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