"""
Early exit mechanism for Qwen2.5-VL, enabling self-speculative decoding.

Splits the model's forward pass into two phases:
1. Draft: layers 0 to early_exit_layer-1 (fast, for generating draft tokens)
2. Verify: layers early_exit_layer to end (completes the full forward pass)

Adapted from Kangaroo's EarlyExitLlamaForCausalLM for Qwen2.5-VL architecture.
"""

import torch
from typing import Optional, Tuple, List
from transformers.cache_utils import DynamicCache


class EarlyExitQwen2_5_VLForConditionalGeneration:
    """
    Wraps a Qwen2_5_VLForConditionalGeneration model to support early exit.
    This is a mixin-style wrapper that adds forward_draft_or_large_model() method.

    Unlike Kangaroo's approach of subclassing, we wrap the existing model to avoid
    modifying the complex Qwen2.5-VL class hierarchy.
    """

    def __init__(self, model, early_exit_layer=2):
        self.model = model  # Qwen2_5_VLForConditionalGeneration instance
        self.early_exit_layer = early_exit_layer
        self.past_key_values = None  # DynamicCache

    @property
    def device(self):
        return self.model.device

    @property
    def config(self):
        return self.model.config

    @property
    def lm_head(self):
        return self.model.lm_head

    def full_forward(self, **kwargs):
        """Run a normal full forward pass (used for prefill)."""
        return self.model(**kwargs)

    @torch.no_grad()
    def forward_draft_or_large_model(
        self,
        in_tokens_small: Optional[torch.LongTensor] = None,
        in_features_large: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """
        Forward pass through either draft layers or remaining layers.

        Args:
            in_tokens_small: Token IDs for draft mode (runs layers 0 to early_exit_layer-1).
            in_features_large: Hidden states for verify mode (runs layers early_exit_layer to end).
            position_ids: Position IDs, shape [3, batch, seq_len] for mRoPE.
            cache_position: Cache position indices.

        Returns:
            Draft mode: hidden_states at early exit layer (un-normed)
            Verify mode: (hidden_states_unnormed, hidden_states_normed)
        """
        assert self.past_key_values is not None, "Must initialize KV cache first via prefill"

        if in_tokens_small is not None and in_features_large is not None:
            raise ValueError("Cannot specify both in_tokens_small and in_features_large")
        if in_tokens_small is None and in_features_large is None:
            raise ValueError("Must specify either in_tokens_small or in_features_large")

        qwen_model = self.model.model  # Qwen2_5_VLModel

        if in_tokens_small is not None:
            # Draft mode: embed tokens and run through early layers
            batch_size, seq_length = in_tokens_small.shape
            hidden_states = qwen_model.embed_tokens(in_tokens_small)
            layers = qwen_model.layers[:self.early_exit_layer]
        else:
            # Verify mode: run through remaining layers
            batch_size, seq_length, _ = in_features_large.shape
            hidden_states = in_features_large
            layers = qwen_model.layers[self.early_exit_layer:]

        # Determine the correct past_length based on which layers we're running
        # IMPORTANT: Cannot use get_seq_length() (_seen_tokens) because draft layers
        # increment it, making it wrong for verify layers. Use per-layer cache length.
        if in_tokens_small is not None:
            focu_layer = 0  # Draft layers use cache length at layer 0
        else:
            focu_layer = self.early_exit_layer  # Verify layers use cache length at exit layer
        layer_past_length = self._get_layer_cache_length(focu_layer)

        # Compute cache_position (where to write in KV cache)
        if cache_position is None:
            cache_position = torch.arange(
                layer_past_length, layer_past_length + seq_length,
                device=hidden_states.device,
            )

        # Compute position embeddings (mRoPE)
        if position_ids is None:
            # For text-only decode tokens, all 3 dims of mRoPE use the same value
            rope_deltas = self.model.rope_deltas
            if rope_deltas is not None:
                delta = (layer_past_length + rope_deltas).to(hidden_states.device)
            else:
                delta = layer_past_length
            position_ids = torch.arange(seq_length, device=hidden_states.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1) + delta
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Compute rotary embeddings
        position_embeddings = qwen_model.rotary_emb(hidden_states, position_ids)

        # Compute causal mask
        attention_mask = torch.ones(
            (batch_size, cache_position[-1].item() + 1),
            dtype=torch.bool, device=hidden_states.device,
        )
        causal_mask = qwen_model._update_causal_mask(
            attention_mask, hidden_states, cache_position,
            self.past_key_values, output_attentions=False,
        )

        # Run through layers
        for decoder_layer in layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=self.past_key_values,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        if in_features_large is not None:
            # Verify mode: return both un-normed and normed hidden states
            return hidden_states, qwen_model.norm(hidden_states)

        # Draft mode: return un-normed hidden states
        return hidden_states

    def _get_layer_cache_length(self, layer_idx):
        """Get the cache length for a specific layer."""
        if self.past_key_values is None:
            return 0
        if layer_idx < len(self.past_key_values.key_cache) and \
           len(self.past_key_values.key_cache[layer_idx]) > 0:
            return self.past_key_values.key_cache[layer_idx].shape[2]
        return 0

    def trim_kv_cache(self, max_length):
        """Trim the KV cache to max_length for all layers."""
        if self.past_key_values is None:
            return
        for layer_idx in range(len(self.past_key_values.key_cache)):
            if len(self.past_key_values.key_cache[layer_idx]) > 0:
                self.past_key_values.key_cache[layer_idx] = \
                    self.past_key_values.key_cache[layer_idx][:, :, :max_length, :]
                self.past_key_values.value_cache[layer_idx] = \
                    self.past_key_values.value_cache[layer_idx][:, :, :max_length, :]
        # Update the seen_tokens counter
        self.past_key_values._seen_tokens = max_length

    def trim_draft_layers_cache(self, max_length):
        """Trim only the draft layers' KV cache (layers 0 to early_exit_layer-1)."""
        if self.past_key_values is None:
            return
        for layer_idx in range(self.early_exit_layer):
            if layer_idx < len(self.past_key_values.key_cache) and \
               len(self.past_key_values.key_cache[layer_idx]) > 0:
                self.past_key_values.key_cache[layer_idx] = \
                    self.past_key_values.key_cache[layer_idx][:, :, :max_length, :]
                self.past_key_values.value_cache[layer_idx] = \
                    self.past_key_values.value_cache[layer_idx][:, :, :max_length, :]

    def trim_verify_layers_cache(self, max_length):
        """Trim only the verify layers' KV cache (layers early_exit_layer to end)."""
        if self.past_key_values is None:
            return
        for layer_idx in range(self.early_exit_layer, len(self.past_key_values.key_cache)):
            if len(self.past_key_values.key_cache[layer_idx]) > 0:
                self.past_key_values.key_cache[layer_idx] = \
                    self.past_key_values.key_cache[layer_idx][:, :, :max_length, :]
                self.past_key_values.value_cache[layer_idx] = \
                    self.past_key_values.value_cache[layer_idx][:, :, :max_length, :]