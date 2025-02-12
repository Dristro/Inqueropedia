import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List

### ------------- Create 'BigramDataset' ------------- ###
class BigramDataset(Dataset):
    def __init__(self, data: List[str], tokenizer, block_size: int):
        """
        Creates a torch-dataset for a Bigram model.

        Args:
            data (List[str]): List of text strings.
            tokenizer: Instance of a tokenizer class (must have a .encode method implemented)
            block_size (int): Context window of the model
        """
        tokenized_data = []
        for text in data:
            tokenized_data.extend(tokenizer.encode(text=text, return_tensor=False, skip_special_tokens=True))
        self.data = torch.tensor(tokenized_data, dtype=torch.long)  # Convert arr into tensor.
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Returns the current window and target
        input_seq = self.data[idx: idx + self.block_size]
        target_seq = self.data[idx + 1: idx + self.block_size + 1]
        return input_seq, target_seq


### ------------- Create Model ------------- ###
