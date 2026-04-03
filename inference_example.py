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

    # Json file, run first N samples, show draft/verify details
    python inference_example.py --data_path ./data.json --num_samples 3 --verbose
"""

import argparse
import json
import time
from tkinter import N
import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from kangaroo_model import KangarooQwenModel
from inference_kangaroo import kangaroo_speculative_generate, autoregressive_generate_direct
from qwen_vl_utils import process_vision_info


def build_inputs_from_conversation(processor, conversation, device):
    """Build model inputs from a conversation list."""
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    # print(f"Processed conversation text:\n{text}\n")
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def warmup_model(model, processor, conversation, args, device):
    """Run one short forward pass to warm up CUDA kernels and memory allocators."""
    print("  Warming up CUDA kernels...")
    inputs = build_inputs_from_conversation(processor, conversation, device)

    # Warmup spec path
    kangaroo_speculative_generate(
        model=model, inputs=inputs, processor=processor,
        max_new_tokens=8, early_exit_layer=args.exit_layer,
        speculative_steps=args.speculative_steps, threshold=args.threshold,
        block_verify=False,
    )

    # Reset state
    model.base_model.past_key_values = None
    model.reset_status()
    if hasattr(model.base_model.model, 'rope_deltas'):
        model.base_model.model.rope_deltas = None

    # Warmup AR path
    autoregressive_generate_direct(
        model=model, inputs=inputs, processor=processor, max_new_tokens=8,
    )

    # Reset state
    model.base_model.past_key_values = None
    model.reset_status()
    if hasattr(model.base_model.model, 'rope_deltas'):
        model.base_model.model.rope_deltas = None

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print("  Warmup done.\n")


@torch.no_grad()
def evaluate_adapter_quality(model, processor, conversation, reference_answer, args, device):
    """
    评估 adapter 用第 exit_layer 层 hidden state 模仿完整大模型输出分布的能力。
    在 (prompt + reference_answer) 的 assistant token 位置上对比:
      - 完整大模型的预测分布
      - adapter 的预测分布
    """
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    # 构造完整对话（含 reference answer）与仅 context 部分
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
        print("  [Adapter Eval] 没有 assistant token 可以评估。")
        return

    # 完整前向传播（含所有层 hidden states）
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

    early_hidden = output.hidden_states[args.exit_layer]  # [1, L, D]
    final_hidden  = output.hidden_states[-1]               # [1, L, D]

    # Adapter 用 early hidden state 预测
    adapter_hidden = model.adapter_model(inputs_embeds=early_hidden)  # [1, L, D]

    # LM head 映射到 vocab
    full_logits    = model.head_model(final_hidden.float())    # [1, L, V]
    adapter_logits = model.head_model(adapter_hidden.float())  # [1, L, V]

    # 取 assistant token 对应的预测位置：
    #   hidden_state[i] 预测 token[i+1]
    #   → 预测 token[context_len..full_len-1] 用位置 context_len-1..full_len-2
    eval_slice = slice(context_len - 1, full_len - 1)
    full_logits_e    = full_logits[0, eval_slice, :]    # [N, V]
    adapter_logits_e = adapter_logits[0, eval_slice, :] # [N, V]
    target_ids       = full_inputs['input_ids'][0, context_len:full_len]  # [N]

    full_p     = F.softmax(full_logits_e,    dim=-1)
    adapter_p  = F.softmax(adapter_logits_e, dim=-1)
    adapter_lp = F.log_softmax(adapter_logits_e, dim=-1)

    full_argmax    = full_logits_e.argmax(dim=-1)
    adapter_argmax = adapter_logits_e.argmax(dim=-1)

    argmax_match   = (full_argmax == adapter_argmax)
    top1_acc       = argmax_match.float().mean().item()
    accept_prob    = torch.min(full_p, adapter_p).sum(dim=-1)         # [N]
    avg_accept     = accept_prob.mean().item()
    kl_div         = (full_p * (full_p.clamp(min=1e-9).log() - adapter_lp)).sum(dim=-1)  # [N]
    avg_kl         = kl_div.mean().item()
    full_conf      = full_p.max(dim=-1).values
    adapter_conf   = adapter_p.max(dim=-1).values
    N = full_argmax.shape[0]

    print(f"\n{'='*70}")
    print(f"ADAPTER 质量评估  (exit_layer={args.exit_layer},  评估 token 数={N})")
    print(f"{'='*70}")
    print(f"  Top-1 准确率  (adapter argmax == full model argmax) : {top1_acc*100:.1f}%")
    print(f"  平均接受概率  Σmin(p_full, p_adapter)               : {avg_accept:.4f}")
    print(f"  平均 KL 散度  KL(full || adapter)                   : {avg_kl:.4f}")
    print(f"  Full model 平均置信度                               : {full_conf.mean().item():.4f}")
    print(f"  Adapter    平均置信度                               : {adapter_conf.mean().item():.4f}")

    print(f"\n  {'步':>4}  {'实际token':>14}  {'Full预测':>14}  {'Adapter预测':>14}  "
          f"{'匹配':>4}  {'Full置信':>8}  {'Adapt置信':>9}  {'接受率':>6}  {'KL':>6}")
    print(f"  {'─'*90}")
    for i in range(N):
        actual  = repr(tokenizer.decode([target_ids[i].item()],  skip_special_tokens=False))
        full_t  = repr(tokenizer.decode([full_argmax[i].item()], skip_special_tokens=False))
        adapt_t = repr(tokenizer.decode([adapter_argmax[i].item()], skip_special_tokens=False))
        match   = "✓" if argmax_match[i].item() else "❌"
        print(f"  {i+1:>4}  {actual:>14}  {full_t:>14}  {adapt_t:>14}  {match:>4}  "
              f"{full_conf[i].item():>8.3f}  {adapter_conf[i].item():>9.3f}  "
              f"{accept_prob[i].item():>6.3f}  {kl_div[i].item():>6.3f}")
    print(f"{'='*70}\n")


def run_one_sample(model, processor, conversation, args, device, sample_idx=0):
    """Run speculative + AR on one conversation sample and print comparison."""
    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}")
    print(f"{'='*70}")

    # Print conversation summary
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
    # print(f"  Input tokens: {inputs['input_ids'].shape[1]}")

    # ========== Adapter 质量评估 ==========
    if reference_answer is not None and getattr(args, 'eval_adapter', False):
        evaluate_adapter_quality(model, processor, gen_conversation, reference_answer, args, device)

    verify_mode = args.verify_mode
    block_verify = (verify_mode == 'fast_length')

    # ========== Warmup ==========
    print(f"\n  Running warmup (short forward to warm CUDA kernels)...")
    model.base_model.past_key_values = None
    model.reset_status()
    if hasattr(model.base_model.model, 'rope_deltas'):
        model.base_model.model.rope_deltas = None

    # Warmup AR first (so spec doesn't eat the cold-start cost)
    autoregressive_generate_direct(
        model=model, inputs=inputs, processor=processor, max_new_tokens=4,
    )
    model.base_model.past_key_values = None
    model.reset_status()
    if hasattr(model.base_model.model, 'rope_deltas'):
        model.base_model.model.rope_deltas = None

    kangaroo_speculative_generate(
        model=model, inputs=inputs, processor=processor,
        max_new_tokens=4, early_exit_layer=args.exit_layer,
        speculative_steps=args.speculative_steps, threshold=args.threshold,
        block_verify=block_verify,
    )
    model.base_model.past_key_values = None
    model.reset_status()
    if hasattr(model.base_model.model, 'rope_deltas'):
        model.base_model.model.rope_deltas = None
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"  Warmup done.")

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
        verbose=args.verbose,
        block_verify=block_verify,
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

    # ========== Print Results ==========
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
    print(f"  Verify mode:       {verify_mode}")
    print(f"  Output match:      {'✓ MATCH' if output_match else '❌ MISMATCH'}")
    print(f"  Length match:      {'✓ MATCH' if length_match else '❌ MISMATCH'}")
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
        if verify_mode == 'strict':
            print(f"\n  ⚠ WARNING: outputs differ! Greedy decoding should be identical in strict mode.")
        else:
            print(f"\n  ⚠ WARNING: outputs differ in fast_length mode. This mode only targets similar length / faster decode.")
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
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='Show draft/verify details for each round')

    # Input source
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--data_path', type=str, default="/data/wangzhichao/projects/SSD/train_data_test.json",
                       help='Path to json file with conversations')
    group.add_argument('--prompt', type=str, default=None,
                       help='Single text prompt (no image)')

    parser.add_argument('--eval_adapter', default=False, action='store_true',
                        help='在 inference 前评估 adapter 模仿完整大模型的能力（需要 reference answer）')
    parser.add_argument('--verify_mode', type=str, default='strict', choices=['strict', 'fast_length'],
                        help='strict: lossless sequential verify; fast_length: block verify for faster approximate decoding with length match metric')
    parser.add_argument('--sample_idx', type=int, default=1,
                        help='Run only this sample index (default: run all)')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Max number of samples to run')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = KangarooQwenModel(
        base_model_path=args.model_path,
        adapter_model_path=args.adapter_path,
        early_exit_layer=args.exit_layer,
        dtype=torch.bfloat16,
    ).to(args.device)
    device = model.device
    processor = AutoProcessor.from_pretrained(args.model_path)

    # ========== Json file input ==========
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
            result = run_one_sample(model, processor, conversation, args, device, idx)
            all_results.append(result)

        # Summary across all samples
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
            print(f"  Verify mode:          {args.verify_mode}")
            print(f"  Output match:        {match_count}/{len(all_results)}")
            print(f"  Length match:        {length_match_count}/{len(all_results)}")
            print(f"  Avg speedup (total): {avg_speedup_total:.2f}x")
            print(f"  Avg speedup (decode):{avg_speedup_decode:.2f}x")
            print(f"  Avg accept length:   {avg_accept:.2f}")
            print(f"  Avg spec decode tok/s: {avg_spec_tps:.1f}")
            print(f"  Avg AR   decode tok/s: {avg_ar_tps:.1f}")

    # ========== Single prompt input ==========
    else:
        prompt = args.prompt or 'Hello, what can you do?'
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        run_one_sample(model, processor, conversation, args, device, sample_idx=0)


if __name__ == '__main__':
    main()