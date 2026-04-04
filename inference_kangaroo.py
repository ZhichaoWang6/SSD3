"""
Speculative decoding inference for Qwen2.5-VL with Kangaroo adapter.

This replaces model.generate() with a custom draft-verify loop:
1. Prefill: Run full model on all input tokens (text + visual) normally
2. Draft: Run early layers + adapter to generate candidate tokens
3. Verify: Run remaining layers to check draft tokens
4. Accept tokens until first mismatch (lossless for greedy decoding)

Adapted from Kangaroo's inference_kangaroo.py for Qwen2.5-VL.
"""

import argparse
import time

import torch
from transformers.cache_utils import DynamicCache


def _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens):
    """Build comprehensive timing and acceptance statistics."""
    decode_time = total_time - prefill_time
    avg_accept = sum(accept_length_list) / len(accept_length_list) if accept_length_list else 0
    tokens_per_second = num_new_tokens / total_time if total_time > 0 else 0
    decode_tokens_per_second = num_new_tokens / decode_time if decode_time > 0 else 0
    return {
        'accept_lengths': accept_length_list,
        'avg_accept_length': avg_accept,
        'total_rounds': len(accept_length_list),
        'total_tokens': num_new_tokens,
        'total_time': total_time,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'draft_times': draft_times,
        'verify_times': verify_times,
        'avg_draft_time': sum(draft_times) / len(draft_times) if draft_times else 0,
        'avg_verify_time': sum(verify_times) / len(verify_times) if verify_times else 0,
        'tokens_per_second': tokens_per_second,
        'decode_tokens_per_second': decode_tokens_per_second,
    }


@torch.no_grad()
def kangaroo_speculative_generate(
    model,
    inputs,
    processor,
    max_new_tokens: int = 512,
    early_exit_layer: int = 2,
    speculative_steps: int = 6,
    threshold: float = 0.6,
    do_sample: bool = False,
    past_key_values=None,
):
    assert not do_sample, "Only greedy decoding is supported for speculative decoding"

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    base_model = model.base_model
    adapter_model = model.adapter_model
    head_model = model.head_model
    device = inputs['input_ids'].device

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    token_eos = tokenizer.eos_token_id
    if isinstance(token_eos, list):
        token_eos_set = set(token_eos)
        token_eos = token_eos[0]
    else:
        token_eos_set = {token_eos}

    input_ids = inputs['input_ids']
    batch_size, context_length = input_ids.shape
    assert batch_size == 1, "Speculative decoding only supports batch_size=1"

    max_length = context_length + max_new_tokens

    global_tokens = torch.full((batch_size, max_length), token_eos, dtype=torch.long, device=device)
    global_tokens[:, :context_length] = input_ids

    accept_length_list = []
    start_index = context_length

    # ========== STEP 0: Prefill ==========
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_prefill_start = time.perf_counter()

    forward_kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs.get('attention_mask'),
        'use_cache': True,
        'output_hidden_states': True,
        'return_dict': True,
        'past_key_values': past_key_values,
        'pixel_values': inputs.get('pixel_values'),
        'pixel_values_videos': inputs.get('pixel_values_videos'),
        'image_grid_thw': inputs.get('image_grid_thw'),
        'video_grid_thw': inputs.get('video_grid_thw'),
        'second_per_grid_ts': inputs.get('second_per_grid_ts'),
        'drop_method': 'none',
        'drop_threshold': 1.0,
        'drop_absolute': True,
    }
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    output = base_model.model(**forward_kwargs)
    base_model.past_key_values = output.past_key_values

    first_token = torch.argmax(output.logits[:, -1, :], dim=-1)
    global_tokens[:, start_index] = first_token.item()

    hidden_state_early = output.hidden_states[early_exit_layer]
    _, adapter_past_key_values = adapter_model.forward_early_stop(
        inputs_embeds=hidden_state_early,
        use_cache=True,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    prefill_time = time.perf_counter() - t_prefill_start

    draft_times = []
    verify_times = []

    if first_token.item() in token_eos_set:
        output_ids = global_tokens[:, :start_index + 1]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - t_start
        stats = _build_stats([], prefill_time, [], [], total_time, 1)
        return output_ids, base_model.past_key_values, stats

    # ========== Draft-Verify Loop ==========
    max_infer_steps = min(max_length, start_index + max_new_tokens)
    stop = False
    round_idx = 0

    while start_index < max_infer_steps - 1:
        round_idx += 1
        start_index_copy = start_index
        end_index = start_index + 1
        remaining_budget = max_infer_steps - 1 - start_index
        round_speculative_steps = min(speculative_steps, remaining_budget)

        # ---- STEP 1: Draft ----
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_draft_start = time.perf_counter()
        exited_hidden_states = None
        draft_token_ids = []

        for step in range(1 + round_speculative_steps):
            in_token = global_tokens[:, end_index - 1:end_index]

            adapter_cache_len = adapter_past_key_values[0][0].shape[2] if adapter_past_key_values else 0
            if adapter_cache_len < end_index - 1:
                hidden_state_early_last = exited_hidden_states[:, -1:, :] if exited_hidden_states is not None else None
            else:
                hidden_state_early_last = None

            hidden_state_early = base_model.forward_draft_or_large_model(
                in_tokens_small=in_token,
            )

            if step == 0:
                exited_hidden_states = None

            exited_hidden_states = hidden_state_early if exited_hidden_states is None \
                else torch.cat([exited_hidden_states, hidden_state_early], dim=1)

            adapter_input = hidden_state_early
            if hidden_state_early_last is not None:
                adapter_input = torch.cat([hidden_state_early_last, hidden_state_early], dim=1)

            if step == round_speculative_steps:
                break
            if step > 0 and predict_score < threshold:
                break

            hidden_state, adapter_past_key_values = adapter_model.forward_early_stop(
                inputs_embeds=adapter_input,
                past_key_values=adapter_past_key_values,
                use_cache=True,
            )

            predict_logits = head_model(hidden_state[:, -1:, :]).float()
            predicted_token = torch.argmax(predict_logits[:, -1, :], dim=-1)
            predict_score = predict_logits.softmax(dim=-1).max().item()

            global_tokens[:, end_index] = predicted_token
            draft_token_ids.append(predicted_token.item())

            end_index += 1

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        draft_times.append(time.perf_counter() - t_draft_start)

        # ---- STEP 2+3: Verify and Accept ----
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_verify_start = time.perf_counter()

        output_length = end_index - start_index

        base_model.past_key_values._seen_tokens = start_index
        verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
        assert verify_cache_len == start_index, \
            f"Verify cache mismatch: {verify_cache_len} != {start_index}"

        _, hidden_state_normed = base_model.forward_draft_or_large_model(
            in_features_large=exited_hidden_states,
        )
        verify_logits = head_model(hidden_state_normed).float()
        verify_ids = torch.argmax(verify_logits, dim=-1)[0].tolist()

        for i, verify_id in enumerate(verify_ids):
            is_last = (i == output_length - 1)
            is_eos = (verify_id in token_eos_set)
            draft_id = global_tokens[0, start_index + 1 + i].item() if i < len(draft_token_ids) else None
            is_mismatch = (not is_last and draft_id is not None and verify_id != draft_id)

            if is_last or is_eos or is_mismatch:
                global_tokens[0, start_index + 1 + i] = verify_id
                start_index = start_index + 1 + i
                if is_eos:
                    stop = True
                break

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        verify_times.append(time.perf_counter() - t_verify_start)

        accept_len = start_index - start_index_copy
        accept_length_list.append(accept_len)

        # ---- STEP 4: Trim caches ----
        draft_cache_len = base_model._get_layer_cache_length(0)
        if draft_cache_len > start_index:
            base_model.trim_draft_layers_cache(start_index)

        verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
        if verify_cache_len > start_index:
            base_model.trim_verify_layers_cache(start_index)

        if adapter_past_key_values and adapter_past_key_values[0][0].shape[2] > start_index:
            adapter_past_key_values = [
                (k[:, :, :start_index, :], v[:, :, :start_index, :])
                for k, v in adapter_past_key_values
            ]

        base_model.past_key_values._seen_tokens = start_index

        if stop:
            break

    # Final output
    output_ids = global_tokens[:, :start_index + 1]
    num_new_tokens = start_index + 1 - context_length
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - t_start

    stats = _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens)
    return output_ids, base_model.past_key_values, stats


def speculative_generate_for_streaming(
    model,
    inputs,
    processor,
    past_key_values=None,
    max_new_tokens: int = 512,
    early_exit_layer: int = 2,
    speculative_steps: int = 6,
    threshold: float = 0.6,
):
    output_ids, past_key_values, stats = kangaroo_speculative_generate(
        model=model,
        inputs=inputs,
        processor=processor,
        max_new_tokens=max_new_tokens,
        early_exit_layer=early_exit_layer,
        speculative_steps=speculative_steps,
        threshold=threshold,
        do_sample=False,
        past_key_values=past_key_values,
    )

    input_length = inputs['input_ids'].shape[1]
    new_token_ids = output_ids[:, input_length:]

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    reply_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]

    return reply_text, past_key_values, stats


@torch.no_grad()
def autoregressive_generate_direct(
    model,
    inputs,
    processor,
    max_new_tokens: int = 512,
    past_key_values=None,
):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    full_model = model.base_model.model
    qwen_model = full_model.model
    lm_head = model.head_model
    device = inputs['input_ids'].device

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    token_eos = tokenizer.eos_token_id
    if isinstance(token_eos, list):
        token_eos_set = set(token_eos)
    else:
        token_eos_set = {token_eos}

    input_ids = inputs['input_ids']
    batch_size, context_length = input_ids.shape

    # Prefill
    forward_kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs.get('attention_mask'),
        'use_cache': True,
        'return_dict': True,
        'past_key_values': past_key_values,
        'pixel_values': inputs.get('pixel_values'),
        'pixel_values_videos': inputs.get('pixel_values_videos'),
        'image_grid_thw': inputs.get('image_grid_thw'),
        'video_grid_thw': inputs.get('video_grid_thw'),
        'second_per_grid_ts': inputs.get('second_per_grid_ts'),
        'drop_method': 'none', 'drop_threshold': 1.0, 'drop_absolute': True,
    }
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    output = full_model(**forward_kwargs)
    kv_cache = output.past_key_values
    rope_deltas = full_model.rope_deltas

    logits = output.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1).item()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    prefill_time = time.perf_counter() - t_start

    generated_tokens = [next_token]

    # Token-by-token decode
    for step in range(max_new_tokens - 1):
        if next_token in token_eos_set:
            break

        in_token = torch.tensor([[next_token]], device=device)
        pos = kv_cache.key_cache[0].shape[2]
        cache_pos = torch.tensor([pos], device=device)

        if rope_deltas is not None:
            delta = (pos + rope_deltas).to(device)
        else:
            delta = pos
        pos_ids = (torch.zeros(1, 1, device=device, dtype=torch.long) + delta)
        pos_ids = pos_ids.unsqueeze(0).expand(3, -1, -1)

        h = qwen_model.embed_tokens(in_token)
        pe = qwen_model.rotary_emb(h, pos_ids)
        attn_mask = torch.ones((1, pos + 1), dtype=torch.bool, device=device)
        cm = qwen_model._update_causal_mask(attn_mask, h, cache_pos, kv_cache, False)

        for layer in qwen_model.layers:
            lo = layer(h, attention_mask=cm, position_ids=pos_ids,
                       past_key_value=kv_cache, output_attentions=False,
                       use_cache=True, cache_position=cache_pos,
                       position_embeddings=pe)
            h = lo[0]

        h = qwen_model.norm(h)
        logits = lm_head(h).float()
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated_tokens.append(next_token)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - t_start
    num_tokens = len(generated_tokens)

    reply_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    stats = {
        'total_tokens': num_tokens,
        'total_time': total_time,
        'prefill_time': prefill_time,
        'decode_time': total_time - prefill_time,
        'tokens_per_second': num_tokens / total_time if total_time > 0 else 0,
        'decode_tokens_per_second': num_tokens / (total_time - prefill_time) if (total_time - prefill_time) > 0 else 0,
    }
    return reply_text, kv_cache, stats


def ar_generate_for_streaming(
    model,
    inputs,
    processor,
    past_key_values=None,
    max_new_tokens: int = 512,
):
    reply_text, past_key_values, stats = autoregressive_generate_direct(
        model=model,
        inputs=inputs,
        processor=processor,
        max_new_tokens=max_new_tokens,
        past_key_values=past_key_values,
    )
    return reply_text, past_key_values, stats


if __name__ == "__main__":
    from transformers import AutoProcessor
    import json
    from inference_example import build_inputs_from_conversation
    from kangaroo_model import KangarooQwenModel

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt')
    parser.add_argument('--adapter_path', type=str,
                        default='/data/wangzhichao/projects/SSD/SSD3/adapter_checkpoints/epochs/epoch007_acc0.4747_accept0.5387_loss2.9227')
    parser.add_argument('--exit_layer', type=int, default=2)
    parser.add_argument('--speculative_steps', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda:6')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data_path', type=str, default="/data/wangzhichao/projects/SSD/train_data_test.json")
    group.add_argument('--prompt', type=str, default=None)

    parser.add_argument('--sample_idx', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=1)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = KangarooQwenModel(
        base_model_path=args.model_path,
        adapter_model_path=args.adapter_path,
        early_exit_layer=args.exit_layer,
        dtype=torch.bfloat16,
    ).to(args.device)
    device = model.device
    processor = AutoProcessor.from_pretrained(args.model_path)

    if args.data_path is not None:
        with open(args.data_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {args.data_path}")

        if args.sample_idx is not None:
            samples = [data[args.sample_idx]]
            indices = [args.sample_idx]
        else:
            end = args.num_samples if args.num_samples else len(data)
            samples = data[:end]
            indices = list(range(end))

        for idx, sample in zip(indices, samples):
            conversation = sample['conversation']
            gen_conversation = list(conversation)
            if gen_conversation and gen_conversation[-1]['role'] == 'assistant':
                gen_conversation = gen_conversation[:-1]
            print(f"\n{'#' * 80}")
            inputs = build_inputs_from_conversation(processor, gen_conversation, device)
            kangaroo_speculative_generate(
                model=model, inputs=inputs, processor=processor,
                max_new_tokens=args.max_new_tokens,
                early_exit_layer=args.exit_layer,
                speculative_steps=args.speculative_steps,
                threshold=args.threshold,
            )
