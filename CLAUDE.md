# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SSD3 implements **self-speculative decoding** (Kangaroo method) for **Qwen2.5-VL**, a multimodal LLM used in the MMDuet2 proactive video QA system. The adapter learns to mimic the full model's output distribution using only early-layer hidden states, enabling a draft-verify loop that accelerates greedy decoding.

## Common Commands

```bash
# Step 1: Generate training data (hidden states from full model)
bash scripts/generate_data.sh

# Step 2: Train the adapter
bash scripts/train_adapter.sh

# Step 3: Run speculative inference + AR comparison
python inference_example.py --device cuda:6 --data_path ./data/annotations/ego-frame_input_format.json --sample_idx 0

# Step 4: Full-scale inference via inference.py
bash scripts/inference_speculative.sh   # speculative decoding
bash scripts/inference.sh               # baseline AR

# Step 5: Evaluate outputs
bash scripts/evaluate.sh
```

Single-command train example with custom settings:
```bash
CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes 1 --mixed_precision bf16 \
    train_adapter.py --basepath /path/to/model --datadir ./datasets/training_data \
    --outdir ./adapter_checkpoints --exit_layer 2 --num_adapter_layers 1 --lr 1e-4
```

## Architecture

### Speculative Decoding Pipeline

The core idea (Kangaroo): use the first `early_exit_layer` (default: 2) transformer layers as a cheap draft model, then verify multiple draft tokens with the remaining layers in one pass.

**Data flow per decoding round:**
1. **Draft**: `EarlyExitQwen.forward_draft_or_large_model(in_tokens_small=...)` runs layers `[0, exit_layer)` → early hidden states → `AdapterModel` → `lm_head` → draft tokens (up to `speculative_steps`, stops early if confidence < `threshold`)
2. **Verify**: same function called with `in_features_large=exited_hidden_states` runs layers `[exit_layer, end)` → normed hidden states → `lm_head` → verify logits
3. **Accept/reject**: accept up to first mismatch or EOS; trim KV caches to accepted length

**Critical KV cache detail**: draft layers and verify layers accumulate KV cache independently. `_seen_tokens` on the `DynamicCache` is manually managed; `trim_draft_layers_cache()` and `trim_verify_layers_cache()` trim them separately to avoid cross-contamination.

### Key Files

| File | Role |
|------|------|
| `earlyexit_qwen.py` | `EarlyExitQwen2_5_VLForConditionalGeneration` — wraps the base model, splits forward pass into draft (layers 0..exit-1) and verify (layers exit..end), manages KV cache trimming |
| `adapter.py` | `AdapterModel` — lightweight transformer (1 decoder layer: attention + RMSNorm, no MLP) that maps early-exit hidden states toward full model's distribution; uses tuple-based KV cache |
| `kangaroo_model.py` | `KangarooQwenModel` — assembles base model + adapter + shared `lm_head`; `to(device)` moves base and adapter but `lm_head` is a reference into base model weights |
| `inference_kangaroo.py` | `kangaroo_speculative_generate()` — the main draft-verify loop; also `autoregressive_generate_direct()` for baseline AR |
| `inference.py` | `ProactiveInferenceClient` — turn-by-turn streaming inference with KV cache reuse across conversation turns (used by MMDuet2 evaluation pipeline) |
| `inference_example.py` | Script for quick local testing: runs spec + AR on JSON data, prints speedup/match stats |
| `train_adapter.py` | Accelerate-based training loop; loss = KL divergence between `softmax(lm_head(final_hidden))` and `log_softmax(lm_head(adapter_out))`, masked to assistant tokens only |
| `generate_training_data.py` | Runs full model on MMDuet2 data, saves per-turn `.ckpt` files with `input_ids`, `loss_mask`, `hidden_state_layerN`, `hidden_state` |
| `model/` | Local copy of Qwen2.5-VL model code with token-drop modifications (DTD variant) |

### Training Data Format

Each `.ckpt` file (one per assistant turn) contains:
- `input_ids`: full token sequence up to and including the assistant turn
- `loss_mask`: 1 only on assistant token positions
- `hidden_state_layerN`: hidden states at exit layer N (adapter input)
- `hidden_state`: hidden states at final layer (training target)

### AdapterModel Design

- Architecture: N layers of `(RMSNorm → attention → residual)` — no MLP, intentionally cheap
- Uses 1D RoPE (not 3D mRoPE used by the base Qwen model)
- KV cache is a plain list of `(key, value)` tuples, not `DynamicCache`
- Shares `lm_head` weights with the base model (no extra parameters for output projection)

### inference.py vs inference_example.py

- `inference.py` (`ProactiveInferenceClient`) is the production path: handles multi-turn video conversations, token-drop masks, comparison mode (`compare_with_baseline=True`)
- `inference_example.py` is for offline testing on JSON samples: runs spec + AR back-to-back, prints per-sample stats and aggregate summary

## Important Notes

- **Batch size**: speculative decoding only supports `batch_size=1`
- **Greedy only**: sampling (`do_sample=True`) is not supported
- **`block_verify=True` (fast mode) only**: the strict sequential verify mode has been removed; fast mode runs verify layers on the full batch of draft hidden states in one forward pass (approximate, may differ slightly from pure AR)
- **`rope_deltas`**: must be reset (`model.base_model.model.rope_deltas = None`) when reusing the model across unrelated inputs, otherwise position IDs will be wrong
- **Memory**: model is ~7GB for 3B param variant on bfloat16; adapter adds ~50MB
