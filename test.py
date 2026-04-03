"""
Minimal test: compare full-layers token-by-token decode vs split-layers decode.
No generate() involved - both paths use identical prefill and manual decode loop.
"""

import copy
import torch
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

from kangaroo_model import KangarooQwenModel


def test_split_decode(model_path='Qwen/Qwen2.5-VL-3B-Instruct', exit_layer=2, num_tokens=10):
    print(f"Loading model from {model_path}, exit_layer={exit_layer}...")
    model = KangarooQwenModel(
        base_model_path=model_path,
        adapter_model_path=None,
        early_exit_layer=exit_layer,
        dtype=torch.bfloat16,
    )
    model.to("cuda:3")
    device = model.device
    processor = AutoProcessor.from_pretrained(model_path)

    base_model = model.base_model  # EarlyExitQwen2_5_VLForConditionalGeneration
    full_model = base_model.model  # Qwen2_5_VLForConditionalGeneration
    qwen_model = full_model.model  # Qwen2_5_VLModel
    lm_head = model.head_model
    num_layers = len(qwen_model.layers)

    # Build input
    prompt = "Hello, what can you do?"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)
    context_len = inputs['input_ids'].shape[1]
    print(f"Input tokens: {context_len}, num_layers: {num_layers}")

    # ========== SHARED PREFILL ==========
    print("\n[Prefill]")
    with torch.no_grad():
        output = full_model(
            **{k: v for k, v in inputs.items() if v is not None},
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            drop_method='none', drop_threshold=1.0, drop_absolute=True,
        )

    first_token = torch.argmax(output.logits[:, -1, :], dim=-1).item()
    rope_deltas = full_model.rope_deltas
    print(f"  First token: {first_token} = '{processor.decode([first_token])}'")
    print(f"  rope_deltas: {rope_deltas}")

    # Deep copy KV cache so both paths start from identical state
    prefill_cache = output.past_key_values
    cache_A = DynamicCache()  # Path A: full layers
    cache_B = DynamicCache()  # Path B: split layers
    for layer_idx in range(num_layers):
        cache_A.update(
            prefill_cache.key_cache[layer_idx].clone(),
            prefill_cache.value_cache[layer_idx].clone(),
            layer_idx,
        )
        cache_B.update(
            prefill_cache.key_cache[layer_idx].clone(),
            prefill_cache.value_cache[layer_idx].clone(),
            layer_idx,
        )
    # Fix _seen_tokens (each update at layer 0 increments it, but we only want context_len)
    cache_A._seen_tokens = context_len
    cache_B._seen_tokens = context_len

    print(f"  cache_A._seen_tokens: {cache_A._seen_tokens}, cache_B._seen_tokens: {cache_B._seen_tokens}")
    print(f"  cache_A layer 0 shape: {cache_A.key_cache[0].shape}, layer {exit_layer} shape: {cache_A.key_cache[exit_layer].shape}")

    # Verify caches are identical
    for li in range(num_layers):
        diff = (cache_A.key_cache[li] - cache_B.key_cache[li]).abs().max().item()
        if diff > 0:
            print(f"  WARNING: cache diff at layer {li}: {diff}")

    # ========== DECODE: Path A (full layers) vs Path B (split layers) ==========
    print(f"\n[Decode {num_tokens} tokens]")
    tokens_A = [first_token]
    tokens_B = [first_token]

    for step in range(num_tokens):
        pos = context_len + step  # position of previous token (what we're processing)

        # ----- Path A: Full model, all layers -----
        in_A = torch.tensor([[tokens_A[-1]]], device=device)
        cache_pos_A = torch.tensor([pos], device=device)
        if rope_deltas is not None:
            delta_A = (pos + rope_deltas).to(device)
        else:
            delta_A = pos
        pos_ids_A = (torch.zeros(1, 1, device=device, dtype=torch.long) + delta_A).unsqueeze(0).expand(3, -1, -1)

        with torch.no_grad():
            h_A = qwen_model.embed_tokens(in_A)
            pe_A = qwen_model.rotary_emb(h_A, pos_ids_A)
            attn_A = torch.ones((1, pos + 1), dtype=torch.bool, device=device)
            cm_A = qwen_model._update_causal_mask(attn_A, h_A, cache_pos_A, cache_A, False)
            for layer in qwen_model.layers:
                lo = layer(h_A, attention_mask=cm_A, position_ids=pos_ids_A,
                           past_key_value=cache_A, output_attentions=False,
                           use_cache=True, cache_position=cache_pos_A,
                           position_embeddings=pe_A)
                h_A = lo[0]
            h_A_normed = qwen_model.norm(h_A)
            logits_A = lm_head(h_A_normed).float().squeeze(0).squeeze(0)
            next_A = torch.argmax(logits_A).item()
        tokens_A.append(next_A)

        # ----- Path B: Split layers (draft + verify) -----
        in_B = torch.tensor([[tokens_B[-1]]], device=device)

        with torch.no_grad():
            # Draft: layers 0 to exit_layer-1
            cache_pos_draft = torch.tensor([cache_B.key_cache[0].shape[2]], device=device)
            if rope_deltas is not None:
                delta_B_draft = (cache_pos_draft[0].item() + rope_deltas).to(device)
            else:
                delta_B_draft = cache_pos_draft[0].item()
            pos_ids_draft = (torch.zeros(1, 1, device=device, dtype=torch.long) + delta_B_draft).unsqueeze(0).expand(3, -1, -1)
            h_B = qwen_model.embed_tokens(in_B)
            pe_draft = qwen_model.rotary_emb(h_B, pos_ids_draft)
            attn_draft = torch.ones((1, cache_pos_draft[-1].item() + 1), dtype=torch.bool, device=device)
            cm_draft = qwen_model._update_causal_mask(attn_draft, h_B, cache_pos_draft, cache_B, False)
            for layer in qwen_model.layers[:exit_layer]:
                lo = layer(h_B, attention_mask=cm_draft, position_ids=pos_ids_draft,
                           past_key_value=cache_B, output_attentions=False,
                           use_cache=True, cache_position=cache_pos_draft,
                           position_embeddings=pe_draft)
                h_B = lo[0]
            draft_h = h_B

            # Verify: layers exit_layer to end
            cache_pos_verify = torch.tensor([cache_B.key_cache[exit_layer].shape[2]], device=device)
            if rope_deltas is not None:
                delta_B_verify = (cache_pos_verify[0].item() + rope_deltas).to(device)
            else:
                delta_B_verify = cache_pos_verify[0].item()
            pos_ids_verify = (torch.zeros(1, 1, device=device, dtype=torch.long) + delta_B_verify).unsqueeze(0).expand(3, -1, -1)
            pe_verify = qwen_model.rotary_emb(draft_h, pos_ids_verify)
            attn_verify = torch.ones((1, cache_pos_verify[-1].item() + 1), dtype=torch.bool, device=device)
            cm_verify = qwen_model._update_causal_mask(attn_verify, draft_h, cache_pos_verify, cache_B, False)
            for layer in qwen_model.layers[exit_layer:]:
                lo = layer(draft_h, attention_mask=cm_verify, position_ids=pos_ids_verify,
                           past_key_value=cache_B, output_attentions=False,
                           use_cache=True, cache_position=cache_pos_verify,
                           position_embeddings=pe_verify)
                draft_h = lo[0]
            h_B_normed = qwen_model.norm(draft_h)
            logits_B = lm_head(h_B_normed).float().squeeze(0).squeeze(0)
            next_B = torch.argmax(logits_B).item()
        tokens_B.append(next_B)

        # Compare
        match = next_A == next_B
        logit_diff = (logits_A - logits_B).abs().max().item()
        logit_A_top = torch.topk(logits_A, 3)
        logit_B_top = torch.topk(logits_B, 3)

        # Check cache alignment
        ca_len = cache_A.key_cache[0].shape[2]
        cb0_len = cache_B.key_cache[0].shape[2]
        cb_ex_len = cache_B.key_cache[exit_layer].shape[2]

        tag = "OK" if match else "MISMATCH"
        print(f"  Step {step+1}: [{tag}] "
              f"A={next_A}('{processor.decode([next_A])}') "
              f"B={next_B}('{processor.decode([next_B])}') "
              f"logit_diff={logit_diff:.4f} "
              f"cache_A={ca_len} cache_B_L0={cb0_len} cache_B_L{exit_layer}={cb_ex_len}")
        if not match:
            print(f"    A_top3: tokens={logit_A_top.indices.tolist()}, vals={logit_A_top.values.tolist()}")
            print(f"    B_top3: tokens={logit_B_top.indices.tolist()}, vals={logit_B_values.tolist()}")

            # Check KV cache divergence
            for li in [0, 1, exit_layer, exit_layer+1, num_layers-1]:
                if li < num_layers:
                    kd = (cache_A.key_cache[li] - cache_B.key_cache[li]).abs().max().item()
                    vd = (cache_A.value_cache[li] - cache_B.value_cache[li]).abs().max().item()
                    print(f"    Cache diff layer {li}: key={kd:.6f}, val={vd:.6f}")

    text_A = processor.decode(tokens_A, skip_special_tokens=True)
    text_B = processor.decode(tokens_B, skip_special_tokens=True)
    print(f"\n  Path A (full):  '{text_A}'")
    print(f"  Path B (split): '{text_B}'")
    print(f"  Match: {tokens_A == tokens_B}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt')
    parser.add_argument('--exit_layer', type=int, default=2)
    parser.add_argument('--num_tokens', type=int, default=10)
    args = parser.parse_args()
    test_split_decode(args.model_path, args.exit_layer, args.num_tokens)