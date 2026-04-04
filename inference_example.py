"""
Example inference script for Kangaroo self-speculative decoding.
Supports both single prompt and json file input.

Usage:
    # Single prompt
    python inference_example.py --prompt "Hello, what can you do?"

    # Json file, run all samples
    python inference_example.py --data_path ./data.json

    # Json file, run specific sample
    python inference_example.py --data_path ./data.json --sample_idx 0

    # Json file, streaming mode (turn-by-turn with KV cache reuse)
    python inference_example.py --data_path ./data.json --streaming
"""

import argparse
import json
import time
import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from kangaroo_model import KangarooQwenModel
from inference_kangaroo import kangaroo_speculative_generate, autoregressive_generate_direct, speculative_generate_for_streaming, ar_generate_for_streaming
from qwen_vl_utils import process_vision_info
import copy


def build_inputs_from_conversation(processor, conversation, device):
    """Build model inputs from a conversation list."""
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def _build_streaming_inputs(processor, history, device):
    """Build inputs for streaming mode: full text + all accumulated images."""
    text = processor.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(history)
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def run_one_sample_streaming(model, processor, conversation, args, device, sample_idx=0):
    """
    Streaming mode: process conversation turn-by-turn with KV cache reuse.
    Mirrors the ProactiveInferenceClient pattern in inference.py.
    """
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}  [Streaming]")
    print(f"{'='*70}")

    # Parse turns
    turns = []
    i = 0
    while i < len(conversation):
        turn = conversation[i]
        if turn['role'] == 'system':
            i += 1
            continue
        if turn['role'] == 'user':
            ref = None
            if i + 1 < len(conversation) and conversation[i + 1]['role'] == 'assistant':
                ref = conversation[i + 1]
                i += 2
            else:
                i += 1
            turns.append((turn, ref))
        else:
            i += 1

    if not turns:
        print("  No user turns found, skipping.")
        return {}

    history = []
    if conversation[0]['role'] == 'system':
        history.append(conversation[0])

    spec_past_kv = None
    ar_past_kv   = None

    def reset():
        model.base_model.past_key_values = None
        model.reset_status()
        if hasattr(model.base_model.model, 'rope_deltas'):
            model.base_model.model.rope_deltas = None

    # Warmup with first user turn
    history_warmup = list(history) + [turns[0][0]]
    warmup_inputs = _build_streaming_inputs(processor, history_warmup, device)
    reset()
    autoregressive_generate_direct(model=model, inputs=warmup_inputs, processor=processor, max_new_tokens=4)
    warmup_new_tokens = max(8, args.speculative_steps + 2)
    reset()
    kangaroo_speculative_generate(model=model, inputs=warmup_inputs, processor=processor,
                                  max_new_tokens=warmup_new_tokens, early_exit_layer=args.exit_layer,
                                  speculative_steps=args.speculative_steps,
                                  threshold=args.threshold)
    reset()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    all_spec_stats = []
    all_ar_stats   = []
    all_matches    = []

    for turn_idx, (user_turn, ref_turn) in enumerate(turns):
        history.append(user_turn)
        inputs = _build_streaming_inputs(processor, history, device)

        content = user_turn['content']
        if isinstance(content, list):
            texts  = [c['text'] for c in content if c.get('type') == 'text']
            imgs   = [c for c in content if c.get('type') == 'image']
            vids   = [c for c in content if c.get('type') == 'video']
            media  = (f"  ({len(imgs)} img)" if imgs else "") + (f"  ({len(vids)} vid)" if vids else "")
            text_s = ' '.join(texts)
        else:
            text_s, media = str(content), ""
        ref_text = ""
        if ref_turn:
            rc = ref_turn['content']
            ref_text = rc if isinstance(rc, str) else ' '.join(b['text'] for b in rc if b.get('type') == 'text')

        print(f"{'─'*70}")
        print(f"[Turn {turn_idx+1}/{len(turns)}]  input tokens={inputs['input_ids'].shape[1]}")
        print(f"  [user] {text_s}{media}")
        if ref_text:
            print(f"  [ref ] {ref_text}")

        # ---- Speculative ----
        spec_reply, spec_past_kv, spec_stats = speculative_generate_for_streaming(
            model=model,
            inputs=inputs,
            processor=processor,
            past_key_values=spec_past_kv,
            max_new_tokens=args.max_new_tokens,
            early_exit_layer=args.exit_layer,
            speculative_steps=args.speculative_steps,
            threshold=args.threshold,
        )
        spec_past_kv = model.base_model.past_key_values

        # ---- AR baseline ----
        model.base_model.past_key_values = ar_past_kv
        model.reset_status()
        ar_reply, ar_past_kv_new, ar_stats = ar_generate_for_streaming(
            model=model,
            inputs=inputs,
            processor=processor,
            past_key_values=ar_past_kv,
            max_new_tokens=args.max_new_tokens,
        )
        ar_past_kv = ar_past_kv_new

        # Restore spec KV cache for next round
        model.base_model.past_key_values = spec_past_kv
        model.reset_status()

        match = (spec_reply == ar_reply)
        all_matches.append(match)
        all_spec_stats.append(spec_stats)
        all_ar_stats.append(ar_stats)

        speedup_total  = ar_stats['total_time']  / spec_stats['total_time']  if spec_stats['total_time']  > 0 else 0
        speedup_decode = ar_stats['decode_time'] / spec_stats['decode_time'] if spec_stats['decode_time'] > 0 else 0

        print(f"  [Spec] {spec_reply}")
        print(f"  [AR  ] {ar_reply}")
        print(f"  {'MATCH' if match else 'MISMATCH'}  "
              f"accept_len={spec_stats['avg_accept_length']:.2f}  "
              f"speedup(total)={speedup_total:.2f}x  speedup(decode)={speedup_decode:.2f}x  "
              f"spec={spec_stats['total_time']*1000:.0f}ms  ar={ar_stats['total_time']*1000:.0f}ms")

        history.append({'role': 'assistant', 'content': spec_reply})

    # Summary
    print(f"\n{'='*70}")
    print(f"Streaming summary  ({len(turns)} turns)")
    print(f"{'='*70}")
    match_rate  = sum(all_matches) / len(all_matches)
    avg_accept  = sum(s['avg_accept_length'] for s in all_spec_stats) / len(all_spec_stats)
    total_spec  = sum(s['total_time'] for s in all_spec_stats)
    total_ar    = sum(s['total_time'] for s in all_ar_stats)
    total_spec_decode = sum(s.get('decode_time', 0) for s in all_spec_stats)
    total_ar_decode   = sum(s.get('decode_time', 0) for s in all_ar_stats)
    overall_speedup        = total_ar / total_spec if total_spec > 0 else 0
    overall_decode_speedup = total_ar_decode / total_spec_decode if total_spec_decode > 0 else 0
    print(f"  Output match rate:        {match_rate:.0%}  ({sum(all_matches)}/{len(all_matches)})")
    print(f"  Avg accept length:        {avg_accept:.2f}")
    print(f"  Overall speedup (total):  {overall_speedup:.2f}x")
    print(f"  Overall speedup (decode): {overall_decode_speedup:.2f}x")
    print(f"  Total spec time:          {total_spec:.3f}s")
    print(f"  Total AR   time:          {total_ar:.3f}s")

    return {
        'sample_idx': sample_idx,
        'match_rate': match_rate,
        'avg_accept_length': avg_accept,
        'overall_speedup': overall_speedup,
        'overall_decode_speedup': overall_decode_speedup,
    }


@torch.no_grad()
def evaluate_adapter_quality(model, processor, conversation, reference_answer, args, device):
    """
    Evaluate how well the adapter (using exit_layer hidden states) approximates
    the full model's output distribution on assistant tokens.
    """
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    if isinstance(reference_answer, list):
        asst_turn = {"role": "assistant", "content": reference_answer}
    else:
        asst_turn = {"role": "assistant", "content": [{"type": "text", "text": reference_answer}]}
    full_conversation = list(conversation) + [asst_turn]

    context_text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_conversation, tokenize=False, add_generation_prompt=False
    )

    image_inputs, video_inputs = process_vision_info(full_conversation)

    full_inputs = processor(
        text=[full_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)
    context_inputs = processor(
        text=[context_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    context_len = context_inputs['input_ids'].shape[1]
    full_len = full_inputs['input_ids'].shape[1]
    n_asst_tokens = full_len - context_len

    if n_asst_tokens <= 0:
        print("  [Adapter Eval] No assistant tokens to evaluate.")
        return

    forward_kwargs = {k: v for k, v in {
        'input_ids':           full_inputs['input_ids'],
        'attention_mask':      full_inputs.get('attention_mask'),
        'pixel_values':        full_inputs.get('pixel_values'),
        'pixel_values_videos': full_inputs.get('pixel_values_videos'),
        'image_grid_thw':      full_inputs.get('image_grid_thw'),
        'video_grid_thw':      full_inputs.get('video_grid_thw'),
        'second_per_grid_ts':  full_inputs.get('second_per_grid_ts'),
        'output_hidden_states': True,
        'return_dict':          True,
        'use_cache':            False,
        'drop_method':          'none',
        'drop_threshold':       1.0,
        'drop_absolute':        True,
    }.items() if v is not None}

    output = model.base_model.model(**forward_kwargs)

    early_hidden = output.hidden_states[args.exit_layer]
    final_hidden  = output.hidden_states[-1]

    adapter_hidden = model.adapter_model(inputs_embeds=early_hidden)

    head_dtype = next(model.head_model.parameters()).dtype
    full_logits    = model.head_model(final_hidden.to(head_dtype)).float()
    adapter_logits = model.head_model(adapter_hidden.to(head_dtype)).float()

    eval_slice = slice(context_len - 1, full_len - 1)
    full_logits_e    = full_logits[0, eval_slice, :]
    adapter_logits_e = adapter_logits[0, eval_slice, :]
    target_ids       = full_inputs['input_ids'][0, context_len:full_len]

    full_p     = F.softmax(full_logits_e,    dim=-1)
    adapter_p  = F.softmax(adapter_logits_e, dim=-1)
    adapter_lp = F.log_softmax(adapter_logits_e, dim=-1)

    full_argmax    = full_logits_e.argmax(dim=-1)
    adapter_argmax = adapter_logits_e.argmax(dim=-1)

    argmax_match   = (full_argmax == adapter_argmax)
    top1_acc       = argmax_match.float().mean().item()
    accept_prob    = torch.min(full_p, adapter_p).sum(dim=-1)
    avg_accept     = accept_prob.mean().item()
    kl_div         = (full_p * (full_p.clamp(min=1e-9).log() - adapter_lp)).sum(dim=-1)
    avg_kl         = kl_div.mean().item()
    full_conf      = full_p.max(dim=-1).values
    adapter_conf   = adapter_p.max(dim=-1).values
    N = full_argmax.shape[0]

    print(f"\n{'='*70}")
    print(f"ADAPTER QUALITY  (exit_layer={args.exit_layer},  tokens={N})")
    print(f"{'='*70}")
    print(f"  Top-1 accuracy (adapter argmax == full model argmax): {top1_acc*100:.1f}%")
    print(f"  Avg acceptance prob  Σmin(p_full, p_adapter):         {avg_accept:.4f}")
    print(f"  Avg KL divergence    KL(full || adapter):             {avg_kl:.4f}")
    print(f"  Full model avg confidence:                            {full_conf.mean().item():.4f}")
    print(f"  Adapter    avg confidence:                            {adapter_conf.mean().item():.4f}")

    print(f"\n  {'#':>4}  {'actual':>14}  {'full pred':>14}  {'adapt pred':>14}  "
          f"{'match':>5}  {'full conf':>9}  {'adapt conf':>10}  {'accept':>6}  {'KL':>6}")
    print(f"  {'─'*90}")
    for i in range(N):
        actual  = repr(tokenizer.decode([target_ids[i].item()],  skip_special_tokens=False))
        full_t  = repr(tokenizer.decode([full_argmax[i].item()], skip_special_tokens=False))
        adapt_t = repr(tokenizer.decode([adapter_argmax[i].item()], skip_special_tokens=False))
        match   = "Y" if argmax_match[i].item() else "N"
        print(f"  {i+1:>4}  {actual:>14}  {full_t:>14}  {adapt_t:>14}  {match:>5}  "
              f"{full_conf[i].item():>9.3f}  {adapter_conf[i].item():>10.3f}  "
              f"{accept_prob[i].item():>6.3f}  {kl_div[i].item():>6.3f}")
    print(f"{'='*70}\n")


def run_one_sample(model, processor, conversation, args, device, sample_idx=0):
    """Run speculative + AR on one conversation sample and print comparison."""
    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}")
    print(f"{'='*70}")

    for turn in conversation:
        role = turn['role']
        content = turn['content']
        if isinstance(content, list):
            texts = [c['text'] for c in content if c.get('type') == 'text']
            images = [c for c in content if c.get('type') == 'image']
            videos = [c for c in content if c.get('type') == 'video']
            media_info = []
            if images:
                media_info.append(f"{len(images)} image(s)")
            if videos:
                media_info.append(f"{len(videos)} video(s)")
            media_str = f"  ({', '.join(media_info)})" if media_info else ""
            print(f"  [{role}] {' '.join(texts)}{media_str}")
        else:
            print(f"  [{role}] {content}")

    # Remove last assistant turn for generation, keep as reference
    gen_conversation = list(conversation)
    reference_answer = None
    if gen_conversation and gen_conversation[-1]['role'] == 'assistant':
        reference_answer = gen_conversation[-1]['content']
        gen_conversation = gen_conversation[:-1]
        print(f"\n  [Reference] {reference_answer}")

    inputs = build_inputs_from_conversation(processor, gen_conversation, device)

    # ========== Adapter quality evaluation ==========
    if reference_answer is not None and getattr(args, 'eval_adapter', False):
        evaluate_adapter_quality(model, processor, gen_conversation, reference_answer, args, device)

    # ========== Warmup ==========
    model.base_model.past_key_values = None
    model.reset_status()
    if hasattr(model.base_model.model, 'rope_deltas'):
        model.base_model.model.rope_deltas = None

    autoregressive_generate_direct(
        model=model, inputs=inputs, processor=processor, max_new_tokens=4,
    )
    model.base_model.past_key_values = None
    model.reset_status()
    if hasattr(model.base_model.model, 'rope_deltas'):
        model.base_model.model.rope_deltas = None

    warmup_new_tokens = max(8, args.speculative_steps + 2)
    kangaroo_speculative_generate(
        model=model, inputs=inputs, processor=processor,
        max_new_tokens=warmup_new_tokens, early_exit_layer=args.exit_layer,
        speculative_steps=args.speculative_steps, threshold=args.threshold,
    )
    model.base_model.past_key_values = None
    model.reset_status()
    if hasattr(model.base_model.model, 'rope_deltas'):
        model.base_model.model.rope_deltas = None
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # ========== Speculative Decoding ==========
    print(f"\n  Running speculative decoding...")
    output_ids, _, stats = kangaroo_speculative_generate(
        model=model,
        inputs=inputs,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        early_exit_layer=args.exit_layer,
        speculative_steps=args.speculative_steps,
        threshold=args.threshold,
    )
    spec_new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
    spec_reply = processor.batch_decode(spec_new_tokens, skip_special_tokens=True)[0]
    spec_num_tokens = spec_new_tokens.shape[1]

    # ========== Autoregressive ==========
    print(f"  Running autoregressive decoding...")
    model.base_model.past_key_values = None
    model.reset_status()
    if hasattr(model.base_model.model, 'rope_deltas'):
        model.base_model.model.rope_deltas = None

    ar_reply, _, ar_stats = autoregressive_generate_direct(
        model=model,
        inputs=inputs,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
    )
    ar_num_tokens = ar_stats['total_tokens']

    # ========== Results ==========
    output_match = (spec_reply == ar_reply)
    length_match = (spec_num_tokens == ar_num_tokens)
    total_speedup = ar_stats['total_time'] / stats['total_time'] if stats['total_time'] > 0 else 0
    decode_speedup = ar_stats['decode_time'] / stats['decode_time'] if stats['decode_time'] > 0 else 0

    print(f"\n  {'─'*60}")
    print(f"  [Speculative]  {spec_reply}")
    print(f"  [AR baseline]  {ar_reply}")
    if reference_answer:
        print(f"  [Reference  ]  {reference_answer}")
    print(f"  {'─'*60}")
    print(f"  Output match:      {'MATCH' if output_match else 'MISMATCH'}")
    print(f"  Length match:      {'MATCH' if length_match else 'MISMATCH'}")
    print(f"  Speedup (total):   {total_speedup:.2f}x")
    print(f"  Speedup (decode):  {decode_speedup:.2f}x")
    print(f"  Spec  tok/s:       {stats['tokens_per_second']:.1f}  "
          f"(decode: {stats['decode_tokens_per_second']:.1f})")
    print(f"  AR    tok/s:       {ar_stats['tokens_per_second']:.1f}  "
          f"(decode: {ar_stats['decode_tokens_per_second']:.1f})")
    print(f"  Avg accept length: {stats['avg_accept_length']:.2f}")
    print(f"  Spec time:         {stats['total_time']:.3f}s  "
          f"(prefill={stats['prefill_time']:.3f}s, decode={stats['decode_time']:.3f}s)")
    print(f"  AR   time:         {ar_stats['total_time']:.3f}s  "
          f"(prefill={ar_stats['prefill_time']:.3f}s, decode={ar_stats['decode_time']:.3f}s)")
    print(f"  Accept lengths:    {stats['accept_lengths']}")

    if not output_match:
        print(f"\n  WARNING: outputs differ.")
        print(f"    Spec: {repr(spec_reply[:200])}")
        print(f"    AR:   {repr(ar_reply[:200])}")

    return {
        'sample_idx': sample_idx,
        'spec_reply': spec_reply,
        'ar_reply': ar_reply,
        'match': output_match,
        'length_match': length_match,
        'speedup_total': total_speedup,
        'speedup_decode': decode_speedup,
        'avg_accept_length': stats['avg_accept_length'],
        'spec_tps': stats['tokens_per_second'],
        'spec_decode_tps': stats['decode_tokens_per_second'],
        'ar_tps': ar_stats['tokens_per_second'],
        'ar_decode_tps': ar_stats['decode_tokens_per_second'],
        'spec_tokens': spec_num_tokens,
        'ar_tokens': ar_num_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt')
    parser.add_argument('--adapter_path', type=str,
                        default='/data/wangzhichao/projects/SSD/SSD3/adapter_checkpoints/epochs/epoch000_acc0.8219_accept0.8507_loss0.8485')
    parser.add_argument('--exit_layer', type=int, default=2)
    parser.add_argument('--speculative_steps', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--device', type=str, default='cuda:3')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data_path', type=str, default="/data/wangzhichao/projects/SSD/train_data_test.json")
    group.add_argument('--prompt', type=str, default=None)

    parser.add_argument('--eval_adapter', default=False, action='store_true',
                        help='Evaluate adapter quality against full model before inference')
    parser.add_argument('--streaming', default=False, action='store_true',
                        help='Streaming mode: process turns one-by-one with KV cache reuse')
    parser.add_argument('--sample_idx', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=3)
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

        all_results = []
        for idx, sample in zip(indices, samples):
            conversation = sample['conversation']
            if args.streaming:
                result = run_one_sample_streaming(model, processor, conversation, args, device, idx)
            else:
                result = run_one_sample(model, processor, conversation, args, device, idx)
            all_results.append(result)

        if len(all_results) > 1:
            print(f"\n{'='*70}")
            print(f"SUMMARY  ({len(all_results)} samples)")
            print(f"{'='*70}")
            match_count = sum(r['match'] for r in all_results)
            length_match_count = sum(r['length_match'] for r in all_results)
            avg_speedup_total = sum(r['speedup_total'] for r in all_results) / len(all_results)
            avg_speedup_decode = sum(r['speedup_decode'] for r in all_results) / len(all_results)
            avg_accept = sum(r['avg_accept_length'] for r in all_results) / len(all_results)
            avg_spec_tps = sum(r['spec_decode_tps'] for r in all_results) / len(all_results)
            avg_ar_tps = sum(r['ar_decode_tps'] for r in all_results) / len(all_results)
            print(f"  Output match:        {match_count}/{len(all_results)}")
            print(f"  Length match:        {length_match_count}/{len(all_results)}")
            print(f"  Avg speedup (total): {avg_speedup_total:.2f}x")
            print(f"  Avg speedup (decode):{avg_speedup_decode:.2f}x")
            print(f"  Avg accept length:   {avg_accept:.2f}")
            print(f"  Avg spec decode tok/s: {avg_spec_tps:.1f}")
            print(f"  Avg AR   decode tok/s: {avg_ar_tps:.1f}")

    else:
        prompt = args.prompt or 'Hello, what can you do?'
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        run_one_sample(model, processor, conversation, args, device, sample_idx=0)


if __name__ == '__main__':
    main()
