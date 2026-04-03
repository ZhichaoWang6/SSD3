"""
Train the Kangaroo adapter for Qwen2.5-VL self-speculative decoding.
"""

import argparse

parser = argparse.ArgumentParser(description='Train Kangaroo adapter for Qwen2.5-VL')
parser.add_argument('--basepath', type=str, default='/data/wangzhichao/projects/MMDuet2/ckpt/MMDuet2_ckpt')
parser.add_argument('--datadir', type=str, required=True, default="/data/wangzhichao/projects/SSD/SSD2/datasets/training_data/")
parser.add_argument('--outdir', type=str, required=True, default="/data/wangzhichao/projects/SSD/SSD2/adapter_checkpoints/")
parser.add_argument('--exit_layer', type=int, default=2)
parser.add_argument('--num_adapter_layers', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--num_warmup_steps', type=int, default=2000)
parser.add_argument('--total_steps', type=int, default=800000)
parser.add_argument('--max_len', type=int, default=4096)
parser.add_argument('--grad_clip', type=float, default=0.5)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--log_steps', type=int, default=10)
args = parser.parse_args()

import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoConfig, get_linear_schedule_with_warmup

torch.backends.cuda.matmul.allow_tf32 = True

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

set_seed(0)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    mixed_precision='bf16',
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    kwargs_handlers=[ddp_kwargs],
)

from adapter import AdapterModel, create_adapter_config

try:
    from torch.utils.tensorboard import SummaryWriter
    if accelerator.is_main_process:
        writer = SummaryWriter(os.path.join(args.outdir, "tensorboard"))
    else:
        writer = None
except ImportError:
    writer = None

# ========== Load LM Head (frozen, always float32) ==========
base_config = AutoConfig.from_pretrained(args.basepath)
head = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)

try:
    from safetensors import safe_open
    index_path = os.path.join(args.basepath, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path), framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except Exception:
    try:
        index_path = os.path.join(args.basepath, "pytorch_model.bin.index.json")
        with open(index_path, "r") as f:
            index_json = json.loads(f.read())
            head_path = index_json["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(args.basepath, head_path), map_location='cpu')
        tensor = weights["lm_head.weight"].float()
    except Exception:
        model_path = os.path.join(args.basepath, "model.safetensors")
        if os.path.exists(model_path):
            with safe_open(model_path, framework="pt", device="cpu") as f:
                tensor = f.get_tensor("lm_head.weight").float()
        else:
            raise RuntimeError(f"Cannot find lm_head weights in {args.basepath}")

head.weight.data = tensor
head.eval()
for param in head.parameters():
    param.requires_grad = False


# ========== Dataset ==========
def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            if file.endswith('.ckpt'):
                datapath.append(os.path.join(root, file))
    return sorted(datapath)


class AdapterDataset(Dataset):
    def __init__(self, datapath, exit_layer, max_len):
        self.data = datapath
        self.exit_layer = exit_layer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index], map_location='cpu', weights_only=False)

        # Truncate from the RIGHT (keep tail) so the assistant response
        # at the end of the sequence is preserved.  The loss_mask marks
        # assistant tokens which always sit at the tail.
        seq_len = data['hidden_state'].shape[0]
        if seq_len > self.max_len:
            start = seq_len - self.max_len
        else:
            start = 0
        hidden_state = data['hidden_state'][start:start + self.max_len][None, :]
        input_ids = data['input_ids'][start:start + self.max_len][None, :]
        loss_mask = data['loss_mask'][start:start + self.max_len]
        hidden_state_early = data[f'hidden_state_layer{self.exit_layer}'][start:start + self.max_len][None, :]

        # ---- Alignment fix ----
        # The adapter is trained to map position-t early hidden → position-t final hidden,
        # which the lm_head then uses to predict token t+1.
        # So the supervision at position t should fire when token t+1 belongs to the
        # assistant reply — i.e. we left-shift the loss_mask by one position.
        # hidden_state / hidden_state_early stay at their original positions (NO shift).
        #
        # loss_mask layout (original, from generate_training_data.py):
        #   loss_mask[i] = 1  ⟺  token i is part of the assistant reply content
        # After left-shift:
        #   loss_mask_shifted[i] = 1  ⟺  token i+1 is assistant content
        #                             ⟺  position i should predict an assistant token
        loss_mask_shifted = torch.zeros(self.max_len, dtype=torch.float32)
        orig_len = loss_mask.shape[0]
        if orig_len > 1:
            # left-shift: mask[i] = original mask[i+1]
            loss_mask_shifted[:orig_len - 1] = loss_mask[1:orig_len].float()
        # last position has no next token → stays 0

        length = hidden_state.shape[1]
        attention_mask = [1] * length

        # target: same-position final hidden (no shift)
        target = hidden_state  # [1, L, D]

        return {
            "attention_mask": attention_mask,
            "loss_mask": loss_mask_shifted[:length].tolist(),
            "target": target,
            "hidden_state_big": hidden_state,
            "hidden_state_early": hidden_state_early,
        }


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        return torch.cat((intensors, padding_tensor), dim=1)

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        return torch.cat((intensors, padding_tensor), dim=1)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        return {
            "hidden_states":        torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features]),
            "hidden_states_early":  torch.cat([self.paddingtensor(item['hidden_state_early'], max_length) for item in features]),
            "target":               torch.cat([self.paddingtensor(item['target'], max_length) for item in features]),
            "loss_mask":            torch.tensor([item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features]),
            "attention_mask":       torch.tensor([item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features]),
        }


def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True) for k in topk]


def save_adapter(model, adapter_config, args, tag):
    """Save adapter weights and config to outdir/tag/."""
    unwrapped_model = accelerator.unwrap_model(model)
    save_dir = os.path.join(args.outdir, tag)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "adapter_model.bin"))
    adapter_config_dict = {
        'hidden_size':             adapter_config.hidden_size,
        'num_attention_heads':     adapter_config.num_attention_heads,
        'num_key_value_heads':     adapter_config.num_key_value_heads,
        'intermediate_size':       adapter_config.intermediate_size,
        'num_hidden_layers':       adapter_config.num_hidden_layers,
        'rms_norm_eps':            adapter_config.rms_norm_eps,
        'vocab_size':              adapter_config.vocab_size,
        'max_position_embeddings': adapter_config.max_position_embeddings,
        'exit_layer':              args.exit_layer,
    }
    with open(os.path.join(save_dir, "adapter_config.json"), 'w') as f:
        json.dump(adapter_config_dict, f, indent=2)
    print(f"  → Saved [{tag}] to {save_dir}")


# ========== Setup data ==========
datapath = list_files(args.datadir)
if len(datapath) == 0:
    raise ValueError(f"No .ckpt files found in {args.datadir}")

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath  = datapath[int(len(datapath) * 0.95):]
print(f"Training: {len(traindatapath)} samples, Testing: {len(testdatapath)} samples")

traindataset = AdapterDataset(traindatapath, args.exit_layer, args.max_len)
testdataset  = AdapterDataset(testdatapath,  args.exit_layer, args.max_len)

train_loader = DataLoader(traindataset, batch_size=args.bs, shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=4, pin_memory=True)
test_loader  = DataLoader(testdataset,  batch_size=args.bs, shuffle=False,
                          collate_fn=DataCollatorWithPadding(), num_workers=4, pin_memory=True)

# ========== Setup model ==========
if accelerator.is_main_process:
    os.makedirs(args.outdir, exist_ok=True)

adapter_config = create_adapter_config(args.basepath, num_adapter_layers=args.num_adapter_layers)
model = AdapterModel(adapter_config)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.total_steps,
)

model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    model, head, optimizer, train_loader, test_loader, scheduler,
)

if args.start_epoch > 0:
    state_dir = os.path.join(args.outdir, "state", f"state_{args.start_epoch - 1}")
    if os.path.exists(state_dir):
        accelerator.load_state(state_dir)
        print(f"Resumed from {state_dir}")

# ========== 历史最优记录 ==========
best_metrics = {
    "accuracy": 0.0,
    "accept":   0.0,
}

# ========== Training Loop ==========
for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
    print(f"=== Epoch {epoch} ===")
    correct = 0
    total = 0
    epoch_loss = 0.0
    epoch_accept = 0.0
    num_batches = 0
    nan_detected = False
    model.train()

    for batch_idx, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        predict = model(inputs_embeds=data["hidden_states_early"], attention_mask=data["attention_mask"])

        with torch.no_grad():
            target_head = head(data["target"].float())
            target_p = F.softmax(target_head, dim=2).detach()

        out_head = head(predict.float())
        prob_exit = F.softmax(out_head, dim=2)
        prob_last = F.softmax(target_head, dim=2)
        prob_acc  = torch.min(prob_last, prob_exit).sum(dim=2)

        out_logp  = F.log_softmax(out_head, dim=2)
        loss_mask = data["loss_mask"][:, :, None]
        plogp     = target_p * out_logp
        loss      = -torch.sum(torch.sum(loss_mask * plogp, 2)) / loss_mask.sum().clamp(min=1)
        prob_acc  = torch.sum(data["loss_mask"] * prob_acc) / data["loss_mask"].sum().clamp(min=1)

        nan_flag = torch.tensor(1.0 if (torch.isnan(loss) or torch.isinf(loss)) else 0.0,
                                device=accelerator.device)
        nan_flag = accelerator.reduce(nan_flag, reduction="sum")
        if nan_flag.item() > 0:
            if accelerator.is_main_process:
                print(f"\nNaN/Inf loss at epoch {epoch}, batch {batch_idx} — skipping")
            nan_detected = True
            optimizer.zero_grad()
            continue

        if accelerator.is_main_process and batch_idx % args.log_steps == 0:
            print(f"\nStep: {batch_idx}\tLR: {optimizer.optimizer.param_groups[0]['lr']:.6f}"
                  f"\tAccept: {prob_acc.item():.4f}\tLoss: {loss.item():.4f}")

        accelerator.backward(loss)
        accelerator.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target    = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            total   += ct
            correct += cc

        if accelerator.is_main_process and writer is not None and ct != 0:
            global_step = batch_idx + len(train_loader) * epoch
            writer.add_scalar("train/lr",           optimizer.optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/loss",          loss.item(),       global_step)
            writer.add_scalar("train/prob_accept",   prob_acc.item(),   global_step)
            writer.add_scalar("train/accuracy",      cc / ct,           global_step)

        epoch_loss   += loss.item()
        epoch_accept += prob_acc.item()
        num_batches  += 1

    # ---- Epoch 汇总 ----
    correct_t = torch.tensor(correct, dtype=torch.float32, device=accelerator.device)
    total_t   = torch.tensor(total,   dtype=torch.float32, device=accelerator.device)
    correct_t, total_t = accelerator.gather_for_metrics((correct_t, total_t))
    correct_val = correct_t.sum().item()
    total_val   = total_t.sum().item()

    epoch_loss   /= max(num_batches, 1)
    epoch_accept /= max(num_batches, 1)
    epoch_acc     = correct_val / max(total_val, 1)

    if accelerator.is_main_process:
        print(f"Epoch [{epoch + 1}/{args.start_epoch + args.num_epochs}]"
              f"  Loss: {epoch_loss:.4f}"
              f"  Acc: {100 * epoch_acc:.2f}%"
              f"  Accept: {epoch_accept:.4f}")
        if nan_detected:
            print("  (部分 NaN batch 已跳过)")

        epoch_tag = (
            f"epochs/"
            f"epoch{epoch:03d}"
            f"_acc{epoch_acc:.4f}"
            f"_accept{epoch_accept:.4f}"
            f"_loss{epoch_loss:.4f}"
        )
        save_adapter(model, adapter_config, args, epoch_tag)

    if epoch % args.save_freq == 0 or epoch == args.start_epoch + args.num_epochs - 1:
        accelerator.save_state(output_dir=os.path.join(args.outdir, "state", f"state_{epoch}"))
        if accelerator.is_main_process:
            print(f"  → Saved full state for epoch {epoch}")

if accelerator.is_main_process and writer is not None:
    writer.close()

print("Training complete!")