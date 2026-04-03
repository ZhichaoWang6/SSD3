"""
KangarooQwenModel: Wraps Qwen2.5-VL base model + adapter + LM head for speculative decoding.

Usage:
    model = KangarooQwenModel(
        base_model_path='Qwen/Qwen2.5-VL-3B-Instruct',
        adapter_model_path='path/to/adapter/checkpoint',
        early_exit_layer=2,
        dtype=torch.bfloat16,
    )
"""

import os
import json

import torch
import torch.nn as nn
from transformers import AutoConfig

from adapter import AdapterModel, create_adapter_config
from earlyexit_qwen import EarlyExitQwen2_5_VLForConditionalGeneration


class KangarooQwenModel(nn.Module):

    def __init__(
        self,
        base_model_path: str,
        adapter_model_path: str = None,
        early_exit_layer: int = 2,
        dtype=torch.bfloat16,
        attn_implementation: str = 'flash_attention_2',
        num_adapter_layers: int = 1,
    ):
        super().__init__()
        self.early_exit_layer = early_exit_layer

        from model import Qwen2_5_VLForConditionalGeneration

        # Load base model
        raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).eval()

        # Wrap with early exit
        self.base_model = EarlyExitQwen2_5_VLForConditionalGeneration(
            raw_model, early_exit_layer=early_exit_layer,
        )

        # Create adapter
        base_config = AutoConfig.from_pretrained(base_model_path)
        adapter_config = create_adapter_config(base_model_path, num_adapter_layers=num_adapter_layers)
        self.adapter_model = AdapterModel(adapter_config)

        # Load adapter weights if provided
        if adapter_model_path is not None:
            adapter_ckpt = os.path.join(adapter_model_path, 'adapter_model.bin')
            if os.path.exists(adapter_ckpt):
                state_dict = torch.load(adapter_ckpt, map_location='cpu')

                # Strip 'module.' prefix added by Accelerate/DDP wrapping
                cleaned = {}
                for k, v in state_dict.items():
                    new_key = k.replace('module.', '', 1) if k.startswith('module.') else k
                    cleaned[new_key] = v

                missing, unexpected = self.adapter_model.load_state_dict(cleaned, strict=False)
                if missing:
                    print(f"WARNING: Adapter missing keys ({len(missing)}): {missing[:5]}")
                if unexpected:
                    print(f"WARNING: Adapter unexpected keys ({len(unexpected)}): {unexpected[:5]}")
                if not missing and not unexpected:
                    print(f"Loaded adapter weights from {adapter_ckpt} (all {len(cleaned)} keys matched)")
                else:
                    print(f"Loaded adapter weights from {adapter_ckpt}")
            else:
                print(f"Warning: adapter checkpoint not found at {adapter_ckpt}, using random weights")

        self.adapter_model = self.adapter_model.eval().to(raw_model.device).to(dtype)

        # Reuse lm_head from base model (shared weights, no duplication)
        self.head_model = raw_model.lm_head

    @property
    def device(self):
        return self.base_model.device

    @property
    def config(self):
        return self.base_model.config

    def to(self, device):
        self.base_model.model.to(device)
        self.adapter_model.to(device)
        return self

    def forward(self):
        raise NotImplementedError("Use speculative decoding inference loop instead of direct forward")

    def reset_status(self):
        """Reset model status for a new inference session."""
        self.base_model.past_key_values = None
        if hasattr(self.base_model.model, 'reset_status'):
            self.base_model.model.reset_status()



if __name__ == "__main__":
    # Example usage
    model = KangarooQwenModel(
        base_model_path='/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt',
        adapter_model_path='/data/wangzhichao/projects/SSD/SSD2/adapter_checkpoints/epoch/adapter_epoch_19',
        early_exit_layer=2,
        dtype=torch.bfloat16,
    )
    print("KangarooQwenModel initialized successfully")
    print(model)