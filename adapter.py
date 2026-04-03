"""
Adapter model for self-speculative decoding on Qwen2.5-VL.
A lightweight single-layer transformer decoder that bridges early-exit hidden states
to the full model's output distribution.

Adapted from Kangaroo (https://github.com/Equationliu/Kangaroo) for Qwen2.5 architecture.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Standard 1D rotary position embeddings for the adapter."""

    def __init__(self, dim, max_position_embeddings=32768, base=1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        # position_ids: (batch, seq_len)
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1).float()
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=position_ids.dtype if position_ids.is_floating_point() else torch.float32), \
               sin.to(dtype=position_ids.dtype if position_ids.is_floating_point() else torch.float32)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # (batch, 1, seq_len, dim)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class AdapterAttention(nn.Module):
    """Multi-head attention with GQA support and tuple-based KV cache."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        rope_theta = getattr(config, 'rope_theta', 1000000.0)
        max_pos = getattr(config, 'max_position_embeddings', 32768)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=max_pos, base=rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        target_dtype = hidden_states.dtype  # 记录输入dtype，后续统一用这个

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 统一dtype，防止RoPE引入的float32和其他张量不一致
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
        value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_expanded.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_expanded)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


class AdapterDecoderLayer(nn.Module):
    """A single transformer decoder layer for the adapter (attention + layernorm, no MLP for speed)."""

    def __init__(self, config):
        super().__init__()
        self.self_attn = AdapterAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


def _make_causal_mask(input_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class AdapterModel(nn.Module):
    """
    Lightweight adapter that maps early-exit hidden states to full model output space.
    Architecture: N decoder layers (typically 1) + final RMSNorm.
    Uses tuple-based KV cache for compatibility with Kangaroo-style speculative decoding.
    """

    def __init__(self, config):
        super().__init__()
        self.gradient_checkpointing = False
        self.padding_idx = getattr(config, 'pad_token_id', 0)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.norm = RMSNorm(config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-6))
        self.layers = nn.ModuleList([
            AdapterDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])  # 解码层，通常为1层 num_hidden_layers层decoder层

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        dtype = inputs_embeds.dtype  # 跟随输入dtype，不写死float32
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, dtype, device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        return self.forward_early_stop(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def forward_early_stop(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_length, _ = inputs_embeds.shape
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_length,
                dtype=torch.long, device=device,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            # Flatten 3D mRoPE position_ids to 1D if needed (take first channel)
            if position_ids.dim() == 3:
                position_ids = position_ids[0]  # (batch, seq_len) - use temporal dim
            position_ids = position_ids.view(batch_size, seq_length).long()

        seq_length_with_past = seq_length + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,
        )

        hidden_states = inputs_embeds
        next_decoder_cache = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, None, False)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states, attention_mask, position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, next_decoder_cache
        return hidden_states


def create_adapter_config(base_model_path, num_adapter_layers=1):
    """Create adapter config from base model config, with num_hidden_layers overridden."""
    base_config = AutoConfig.from_pretrained(base_model_path)
    adapter_config = type(base_config)(
        hidden_size=base_config.hidden_size,
        num_attention_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
        intermediate_size=base_config.intermediate_size,
        num_hidden_layers=num_adapter_layers,
        rms_norm_eps=base_config.rms_norm_eps,
        vocab_size=base_config.vocab_size,
        max_position_embeddings=base_config.max_position_embeddings,
        rope_theta=getattr(base_config, 'rope_theta', 1000000.0),
        pad_token_id=getattr(base_config, 'pad_token_id', 0),
    )
    return adapter_config


if __name__ == "__main__":
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    import torch

    DEVICE     = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "/data/wangzhichao/models/Qwen2.5-VL-3B-Instruct"
    CKPT_PATH  = "/data/wangzhichao/projects/SSD/SSD2/kangaroo/training_data/data_0.ckpt"
    EXIT_LAYER = 2

    def section(title): print(f"\n{'='*60}\n  {title}\n{'='*60}")
    def ok(msg):        print(f"  ✓  {msg}")
    def info(msg):      print(f"  →  {msg}")

    section("1. 读取 ckpt")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    ok(f"加载成功: {CKPT_PATH}")
    info(f"包含的键: {list(ckpt.keys())}")

    input_ids    = ckpt["input_ids"]
    loss_mask    = ckpt["loss_mask"]
    early_hidden = ckpt[f"hidden_state_layer{EXIT_LAYER}"]
    final_hidden = ckpt["hidden_state"]

    seq_len, hidden_size = early_hidden.shape
    info(f"seq_len: {seq_len},  hidden_size: {hidden_size}")
    info(f"loss_mask 有效 token 数: {int(loss_mask.sum().item())}")

    section("2. 加载 Qwen 模型")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map=DEVICE
    )
    qwen.eval()
    lm_head = qwen.lm_head
    ok(f"lm_head: {lm_head.weight.shape}")

    section("3. 创建 AdapterModel")
    adapter_config = create_adapter_config(MODEL_PATH, num_adapter_layers=1)
    adapter = AdapterModel(adapter_config).to(DEVICE).to(torch.float32)
    adapter.eval()
    n_params = sum(p.numel() for p in adapter.parameters())
    ok(f"参数量: {n_params/1e6:.2f}M")

    section("4. Early hidden → AdapterModel")
    inputs_embeds = early_hidden.unsqueeze(0).to(DEVICE).to(torch.float32)
    with torch.no_grad():
        adapter_out = adapter(inputs_embeds=inputs_embeds)
    ok(f"adapter 输出: {adapter_out.shape}  dtype={adapter_out.dtype}")
    info(f"含 NaN: {torch.isnan(adapter_out).any().item()}  (期望 False)")

    section("5. AdapterModel → lm_head → logits")
    with torch.no_grad():
        adapter_logits = lm_head(adapter_out)
        final_logits   = lm_head(final_hidden.unsqueeze(0).to(DEVICE).to(torch.float32))
    ok(f"adapter logits: {adapter_logits.shape}")
    ok(f"full    logits: {final_logits.shape}")

    section("6. 对比预测（assistant token 位置）")
    assistant_pos = loss_mask.nonzero(as_tuple=True)[0].tolist()
    adapter_pred  = adapter_logits[0].argmax(dim=-1).cpu()
    full_pred     = final_logits[0].argmax(dim=-1).cpu()

    print(f"\n  {'位置':>6}  {'真实token':>12}  {'完整模型':>12}  {'adapter':>12}  {'一致':>4}")
    print(f"  {'-'*56}")
    match = 0
    for pos in assistant_pos[:20]:
        t  = repr(tokenizer.decode([input_ids[pos].item()]))
        fp = repr(tokenizer.decode([full_pred[pos].item()]))
        ap = repr(tokenizer.decode([adapter_pred[pos].item()]))
        eq = "✓" if full_pred[pos] == adapter_pred[pos] else "✗"
        if full_pred[pos] == adapter_pred[pos]:
            match += 1
        print(f"  {pos:>6}  {t:>12}  {fp:>12}  {ap:>12}  {eq:>4}")

    section("7. MSE Loss（adapter_out vs final_hidden）")
    target = final_hidden.unsqueeze(0).to(DEVICE).to(torch.float32)
    mse_all = F.mse_loss(adapter_out, target).item()
    info(f"全序列 MSE: {mse_all:.6f}  （未训练时偏大，训练目标是让此值降低）")

    if assistant_pos:
        pos_t = torch.tensor(assistant_pos)
        mse_asst = F.mse_loss(adapter_out[0, pos_t], target[0, pos_t]).item()
        info(f"assistant 位置 MSE: {mse_asst:.6f}")

    print(f"\n{'='*60}")
    print("  ALL STEPS PASSED ✓")
    print(f"  seq_len:      {seq_len}")
    print(f"  hidden_size:  {hidden_size}")
    print(f"  adapter 参数: {n_params/1e6:.2f}M")
    print(f"  token 一致率: {match}/{min(20, len(assistant_pos))}  （未训练，低正常）")
    print(f"  全序列 MSE:   {mse_all:.6f}")
    print('='*60)