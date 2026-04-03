"""
Generate training data for the Kangaroo adapter by collecting hidden states
from the full Qwen2.5-VL model on MMDuet2's multimodal data.

Key change vs original: instead of processing a full conversation (88 images)
as one sample, we slice at each assistant turn and produce one ckpt per turn.

For each assistant turn at index k, the input is:
    system + user_0 + asst_0 + ... + user_k
and the target is:
    asst_k

For each sample, saves:
- input_ids:            tokenized input sequence (up to and including asst_k)
- loss_mask:            1 only on asst_k tokens
- hidden_state_layerN:  hidden states from early exit layer N
- hidden_state:         hidden states from the final layer

Output file naming:
    data_{global_sample_idx}_turn{turn_idx}.ckpt

Usage:
    python generate_training_data.py \
        --model_path /path/to/model \
        --data_path  ./datasets/egoexolearn_train.json \
        --output_dir ./datasets/training_data/ \
        --exit_layers 2,3 \
        --start 0 --end 100
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoProcessor

from model import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description='Generate adapter training data (per-turn)')
    parser.add_argument('--model_path', type=str,
                        default='/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt')
    parser.add_argument('--data_path', type=str,
                        default='/data/wangzhichao/projects/SSD/SSD3/datasets/egoexolearn_train.json')
    parser.add_argument('--output_dir', type=str, default='./datasets/training_data/')
    parser.add_argument('--exit_layers', type=str, default='2',
                        help='Comma-separated list of exit layers to save hidden states for')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end',   type=int, default=None)
    parser.add_argument('--max_seq_len', type=int, default=4096,
                        help='Maximum sequence length; turns exceeding this are skipped')
    parser.add_argument('--skip_no_reply', action='store_true',
                        help='Skip assistant turns whose content is exactly "NO REPLY"')
    parser.add_argument('--gpu', type=str, default='cuda:6')
    return parser.parse_args()


def load_data(data_path):
    if data_path.endswith('.jsonl'):
        data = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(data_path) as f:
            data = json.load(f)
    return data


def get_assistant_text(turn: dict) -> str:
    """Extract plain text from an assistant turn regardless of content format."""
    content = turn.get('content', '')
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [b['text'] for b in content if isinstance(b, dict) and b.get('type') == 'text']
        return ' '.join(texts).strip()
    return ''


def build_loss_mask(full_input_ids, context_input_ids, tokenizer):
    """
    Mark only the current assistant reply *content* tokens with 1.

    We first locate the appended assistant turn via the length difference between
    (context) and (context + assistant). Then we trim chat-template control tokens
    from the leading assistant header and the trailing <|im_end|> marker.
    """
    loss_mask = torch.zeros_like(full_input_ids[0], dtype=torch.float32)

    full_ids = full_input_ids[0]
    full_len = full_ids.shape[0]
    context_len = context_input_ids.shape[1]
    if context_len >= full_len:
        return loss_mask

    ids = full_ids.tolist()
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

    start = context_len
    end = full_len

    # Skip assistant header tokens (typically "assistant\n").
    content_start = start
    while content_start < end:
        prefix = tokenizer.decode(ids[start:content_start + 1], skip_special_tokens=False)
        if '\n' in prefix:
            content_start += 1
            break
        content_start += 1

    # Drop trailing template terminator(s).
    while end > content_start and ids[end - 1] == im_end_id:
        end -= 1

    for i in range(content_start, end):
        loss_mask[i] = 1.0

    return loss_mask


def slice_turns(messages: list):
    """
    Split a message list into (context, assistant_turn) pairs.
    Each turn is treated independently: context = [system, user_k], target = asst_k.
    """
    prefix = [messages[0]] if messages and messages[0]['role'] == 'system' else []
    rest = messages[len(prefix):]

    turn_idx = 0
    i = 0
    while i < len(rest) - 1:
        user_turn = rest[i]
        asst_turn = rest[i + 1]
        if user_turn['role'] != 'user' or asst_turn['role'] != 'assistant':
            i += 1
            continue

        context = prefix + [user_turn]
        yield context, asst_turn, turn_idx

        turn_idx += 1
        i += 2


@torch.no_grad()
def process_turn(model, processor, context: list, asst_turn: dict,
                 exit_layers: list, max_seq_len: int):
    """
    Run one (context + assistant_turn) slice through the model.
    Returns a result dict or None if the turn should be skipped.
    """
    full_history = context + [asst_turn]

    context_text = processor.apply_chat_template(
        context, tokenize=False, add_generation_prompt=False,
    )
    full_text = processor.apply_chat_template(
        full_history, tokenize=False, add_generation_prompt=False,
    )

    image_inputs, video_inputs = process_vision_info(full_history)

    inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )
    context_inputs = processor(
        text=[context_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
    )
    inputs = inputs.to(model.device)

    seq_len = inputs.input_ids.shape[1]
    if seq_len > max_seq_len:
        return None

    forward_kwargs = {
        'input_ids':           inputs['input_ids'],
        'attention_mask':      inputs.get('attention_mask'),
        'pixel_values':        inputs.get('pixel_values'),
        'pixel_values_videos': inputs.get('pixel_values_videos'),
        'image_grid_thw':      inputs.get('image_grid_thw'),
        'video_grid_thw':      inputs.get('video_grid_thw'),
        'second_per_grid_ts':  inputs.get('second_per_grid_ts'),
        'output_hidden_states': True,
        'return_dict':          True,
        'use_cache':            False,
        'drop_method':          'none',
        'drop_threshold':       1.0,
        'drop_absolute':        True,
    }
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    try:
        outputs = model(**forward_kwargs)
    except Exception as e:
        print(f"  [ERROR] forward pass failed: {e}")
        return None

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    loss_mask = build_loss_mask(inputs['input_ids'], context_inputs['input_ids'], tokenizer)

    result = {
        'input_ids': inputs['input_ids'].cpu()[0],
        'loss_mask': loss_mask.cpu(),
    }

    # 存储时转 float32，避免 bf16 极端值问题
    for layer in exit_layers:
        if layer < len(outputs.hidden_states):
            hs = outputs.hidden_states[layer].float().cpu()[0]
            result[f'hidden_state_layer{layer}'] = hs

    result['hidden_state'] = outputs.hidden_states[-1].float().cpu()[0]

    # 检查是否含 NaN/Inf，有则跳过
    for key, val in result.items():
        if torch.is_tensor(val) and torch.is_floating_point(val):
            if torch.isnan(val).any() or torch.isinf(val).any():
                print(f"  [SKIP] {key} contains NaN/Inf, dropping this turn")
                return None

    return result


def main():
    args = parse_args()
    exit_layers = [int(x) for x in args.exit_layers.split(',')]

    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    ).eval().to(args.gpu)

    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)

    end  = args.end if args.end is not None else len(data)
    data = data[args.start:end]
    print(f"Processing {len(data)} samples (index {args.start}..{end})")

    os.makedirs(args.output_dir, exist_ok=True)

    total_turns   = 0
    total_saved   = 0
    total_skipped = 0

    for sample_i, example in enumerate(tqdm(data)):
        global_idx = args.start + sample_i
        messages   = example.get('messages', example.get('conversation', []))
        if not messages:
            continue

        for context, asst_turn, turn_idx in slice_turns(messages):
            total_turns += 1

            if args.skip_no_reply and get_assistant_text(asst_turn) == 'NO REPLY':
                total_skipped += 1
                continue

            result = process_turn(
                model, processor, context, asst_turn,
                exit_layers, args.max_seq_len,
            )

            if result is None:
                total_skipped += 1
                continue

            save_path = os.path.join(
                args.output_dir, f'data_{global_idx}_turn{turn_idx}.ckpt'
            )
            torch.save(result, save_path)
            total_saved += 1

    print(f"\nDone.")
    print(f"  Total turns : {total_turns}")
    print(f"  Saved       : {total_saved}")
    print(f"  Skipped     : {total_skipped}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Exit layers : {exit_layers}")


if __name__ == '__main__':
    main()