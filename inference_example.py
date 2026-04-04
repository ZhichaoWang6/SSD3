"""
Example inference script for Kangaroo self-speculative decoding.

Usage:
    # Run specific sample
    python inference_example.py --data_path ./data.json --sample_idx 0

    # Run first N samples
    python inference_example.py --data_path ./data.json --num_samples 3
"""

import argparse
import json
import torch
from transformers import AutoProcessor

from kangaroo_model import KangarooQwenModel
from inference_kangaroo import kangaroo_speculative_generate, autoregressive_generate_direct
from qwen_vl_utils import process_vision_info


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
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--data_path', type=str, default="/data/wangzhichao/projects/SSD/train_data_test.json")
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
        result = run_one_sample(model, processor, sample['conversation'], args, device, idx)
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


if __name__ == '__main__':
    main()
