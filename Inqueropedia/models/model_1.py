import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from Inqueropedia.custom_tokenizers import load_BPETokenizerV2
from Inqueropedia.models import DecoderOnlyTransformer, Block  # We may use a single sa-block later

from datasets import load_dataset
from pathlib import Path
import os
from typing import List, Tuple
import numpy as np

# ---------------- Hyper-params ---------------- #
# These params are model specific
batch_size = 32
block_size = 32
n_embed = 32
num_heads = 4
head_size = 16
num_blocks = 4
vocab_size = None  # Load tokenizer...

# ---------------- Setup stuff ---------------- #
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"[INFO] Loading the data and prep-ing for training")
# Load the dataset(s) - train and val
ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
train_dataset = ds["train"]
validation_dataset = ds["validation"]
# Load tokenizer
current_dir = Path(os.getcwd())
path_to_tokenizer = current_dir.parent / "custom_tokenizers" / "bpe_tokenizer_v1_train_dataset.pth"
tokenizer = load_BPETokenizerV2(file_path=path_to_tokenizer)

vocab_size = len(tokenizer._vocab)

# ---------------- Helper functions ---------------- #
def get_batch(dataset,
              block_size: int,
              batch_size: int = batch_size) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a single batch of data from 'split'.
    NOTE: this function moves the batches to device.
    NOTE: this function assumes that the device is set globally.
    """
    data = dataset
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in idx]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idx]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module,
                  dataset,
                  iters: int) -> float:
    losses = []
    model.eval()
    for _ in range(iters):
        xb, yb = get_batch(dataset, block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        losses.append(loss.item())
    out = np.mean(losses)
    return float(out)  # Cast from np.floating to float

@torch.no_grad()
def generate_from_model(model, tokenizer, start_string):
    print("[INFO] generating from the model")
    _temp = torch.tensor([tokenizer.encode(start_string)]).to(device)
    with torch.no_grad():
        _temp = model.generate(_temp, max_new_tokens=100)
    print(f"[INFO] Model generated: {tokenizer.decode(_temp.tolist()[0])}")


# ---------------- Setup model ---------------- #
class Model2(nn.Module):
    def __init__(self):
        """
        Its assumed that the model's args are set globally.
        """
        super().__init__()
        self.transformer = DecoderOnlyTransformer(
            num_blocks=num_blocks,
            num_heads=num_heads,
            vocab_size=vocab_size,
            block_size=block_size,
            n_embed=n_embed,
            head_size=head_size,
        )

    def forward(self, idx, targets=None):
        logits = self.transformer(idx)  # (B,T) -> (B,T,vocab_size) | logits

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx = idx[:, -block_size:]
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            pred_probs = F.softmax(logits, dim=-1)
            _idx = torch.multinomial(pred_probs, num_samples=1)
            idx = torch.cat((idx, _idx), dim=1)  # (B,T) concat (T) -> (B,T+1)

        return idx