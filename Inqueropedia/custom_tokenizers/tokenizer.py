from typing import List
import regex as re
import pickle
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """
    Parent class for all tokenizers.
    Implements common methods like stats calculation, merging, saving/loading tokenizer states.
    """
    def __init__(self):
        self._vocab = None
        self._merges = None
        self._built = False
    
    def _get_stats(self, tokens: List[int]):
        """
        Counts occurences of byte-pairs in the tokenized list.
        """
        pairs = {}
        for split in tokens:
            for pair in zip(split, split[1:]):
                pair = tuple(pair)  # Ensure the pair is hashable
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _merge(self, tokens, pair, idx):
        """
        Merges a given byte pair in each split separately.
        """
        new_tokens = []
        for split in tokens:
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i+1]) == pair:
                    new_split.append(idx)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_tokens.append(new_split)
        return new_tokens

    def save(self, file_path: str):
        """
        Save the tokenizer's vocab and merges to a file.
        Args:
            file_path: Path to save tokenizer instance.
        """
        with open(file_path, 'wb') as f:
            pickle.dump({'vocab': self._vocab, 'merges': self._merges}, f)
        print(f"[INFO] Tokenizer saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str):
        """
        Load the tokenizer from a saved file.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Create an uninitialized instance
        tokenizer = object.__new__(cls)
        tokenizer._vocab = data['vocab']
        tokenizer._merges = data['merges']
        tokenizer._built = True
        print(f"[INFO] Tokenizer loaded from {file_path}")
        return tokenizer
    
    @abstractmethod
    def _prepare_texts(self, texts):
        """
        Must be implemented by child classes to define how text preprocessing works.
        """
        pass

    @abstractmethod
    def _build_tokenizer(self, vocab_size):
        """
        Must be implemented by child classes to define vocabulary construction.
        """
        pass

    @abstractmethod
    def encode(self, text: str):
        """
        Convert text into token IDs. Must be implemented by child classes.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: List[int]):
        """
        Convert token IDs back to text. Must be implemented by child classes.
        """
        raise NotImplementedError
