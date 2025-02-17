import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

import os
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from timeit import default_timer as timer
from typing import List, Tuple

from Inqueropedia.custom_tokenizers import load_BPETokenizerV2

# ---------------- Hyper-params ---------------- #
batch_size = 32
block_size = 8
train_iters = 1000
eval_iters = 100
compile_backend = "aot_eager"  # For mps devices
device = torch.device("mps" if torch.backends.mps.is_available() else "cpe")

# ---------------- Load and prep data ---------------- #
print(f"Loading the data and prep-ing for training")

# Load the dataset(s) - train and val
from datasets import load_dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
train_dataset = ds["train"]
validation_dataset = ds["validation"]


# Load tokenizer
current_dir = Path(os.getcwd())
path_to_tokenizer = current_dir.parent / "custom_tokenizers" / "bpe_tokenizer_v1_train_dataset.pth"
tokenizer = load_BPETokenizerV2(file_path=path_to_tokenizer)
vocab_size = len(tokenizer._vocab)


# Creating the dataset
class BiGramDataset(Dataset):
    """
    Thin wrapper object for a list of ids.
    """

    def __init__(self, texts: List[str], tokenizer, verbose: int = 0):
        """
        Creates a torch-Dataset instance for the given text data.
        Args:
            texts (List[str]): List of strings for dataset.
            tokenizer: Tokenizer to encode the strings.
        """
        if verbose > 0:
            print(f"[INFO] Building dataset...")
            st = timer()

        # Build the dataset here
        self.tokenizer = tokenizer
        tokenized_texts = []  # Concat all the samples in 'texts' and tokenize them
        for text in tqdm(texts, desc="Tokenizing texts", leave=False):
            tokenized_texts.extend(tokenizer.encode(text))

        self.data = torch.tensor(tokenized_texts, dtype=torch.long)

        if verbose > 0:
            et = timer()
            print(f"{(et - st):.5f} sec to build the dataset.")
            print(f"[INFO] Dataset built")
            print("---" * 5)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create dataset(s)
train_dataset_1 = BiGramDataset(train_dataset["text"], tokenizer, verbose=1)
validation_dataset_1 = BiGramDataset(validation_dataset["text"], tokenizer, verbose=1)
print(f"[INFO] Created the datasets | train-length: {len(train_dataset_1)} | validation-length: {len(validation_dataset_1)}")

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

# ---------------- Model class ---------------- #

class BiGramV1(nn.Module):
    """BiGram model for sequence generation"""
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, X: torch.Tensor, targets: torch.Tensor or None = None):
        """
        Performs a forward pass.
        Expected tensor of shape: (B,1) | (batch_size, 1)
        """
        logits = self.emb(X)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Args:
            idx (torch.Tensor): start index from vocab.
            max_new_tokens (int): max number of tokens to generate.
        Returns:
            torch.Tensor with new (generated) tokens.
        """
        for _ in range(max_new_tokens):
            logits, loss = self(idx)   # logits: (B,T,C)
            logits = logits[:, -1, :]  # (B,C)
            pred_probs = F.softmax(logits, dim=-1)
            _idx = torch.multinomial(pred_probs, num_samples=1)
            idx = torch.cat((idx, _idx), dim=1)  # (B,T+1)
        return idx


# ---------------- Train the model ---------------- #
model_1 = BiGramV1(vocab_size)
model_1.to(device)
model_1.compile(backend="aot_eager")
optimizer = optim.AdamW(model_1.parameters(), lr=5e-2)
scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

# Train loop
for i in range(train_iters):
    st = timer()

    xb, yb = get_batch(train_dataset_1, block_size=block_size, batch_size=batch_size)
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model_1(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    scheduler.step()

    val_loss = estimate_loss(model_1, validation_dataset_1, eval_iters)

    et = timer()
    print(f"Epoch: {i}/{train_iters} | time: {(et-st):.4f}| val-loss: {val_loss:.4f} | current-train-loss: {loss.item():.4f} | lr: {scheduler.get_last_lr()[0]:6f}")

    if (i+1) % 100 == 0:
        generate_from_model(model_1, tokenizer, " between")  # ' between' takes only 1 token in the vocab

# ---------------- Generate some text from the model ---------------- #
