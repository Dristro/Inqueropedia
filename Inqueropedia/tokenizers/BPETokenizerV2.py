import regex as re
import warnings
from tqdm.auto import tqdm
import pickle
from typing import List

class BPETokenizerV2:
    def __init__(self, texts: List[str]):
        """
        Creates a BPETokenizerV1 instance using regex-based tokenization.
        Args:
            texts (List[str]): List of input strings.
        """
        self.gpt2_pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

        _text = " ".join(texts)
        self.splits = self.gpt2_pat.findall(_text)

        self.split_tokens = [list(tok.encode("utf-8")) for tok in self.splits]

        self.__built = False
        self._vocab = None
        self._merges = None

    def _get_stats(self, tokens):
        """
        Counts occurrences of byte pairs in the tokenized list.
        """
        pairs = {}
        for split in tokens:
            for pair in zip(split, split[1:]):
                pair = tuple(pair)  # FIX: was getting a type error otherwise
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

    def __build_tokenizer(self, vocab_size):
        """
        Builds the BPE tokenizer's vocabulary.
        """
        assert vocab_size >= 256, "Vocabulary size must be at least 256 for byte-level tokens."
        
        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}

        n_merges = vocab_size - 256
        ids = self.split_tokens.copy()

        initial_token_count = sum(len(split) for split in ids)

        for i in tqdm(range(n_merges), leave=False, desc="Merging"):
            stats = self._get_stats(ids)
            if not stats:
                self.vocab_size = 256 + i
                break
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            merges[top_pair] = idx
            ids = self._merge(ids, top_pair, idx)
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]

        final_token_count = sum(len(split) for split in ids)

        # Print some info after tokenizer is built
        print(f"Before length: {initial_token_count}")
        print(f"After length: {final_token_count}")
        print(f"Compression ratio: {(initial_token_count / final_token_count):.3f}")

        self._vocab = vocab
        self._merges = merges
        self.__built = True

    def fit(self, vocab_size: int, texts: List[str] = None):
        """
        Builds the tokenizer's vocabulary using the given texts.
        """
        if texts:
            warnings.warn("Using .fit with new texts is discouraged. Pass texts during initialization.")
            _text = " ".join(texts)
            self.splits = self.gpt2_pat.findall(_text)
            self.split_tokens = [list(tok.encode("utf-8")) for tok in self.splits]

        self.__build_tokenizer(vocab_size)

    def encode(self, text: str):
        """
        Encodes a given text into a sequence of token IDs.
        """
        assert self.__built, "Tokenizer must be built using `fit` before encoding."

        # Step 1: Split and encode text using regex and bytes
        splits = self.gpt2_pat.findall(text)
        split_tokens = [list(tok.encode("utf-8")) for tok in splits]

        encoded_ids = []
        for tokens in split_tokens:
            while len(tokens) >= 2:
                stats = self._get_stats([tokens])  # Compute within-split stats
                pair = min(stats, key=lambda p: self._merges.get(p, float('inf')), default=None)
                if pair is None or pair not in self._merges:
                    break
                idx = self._merges[pair]
                tokens = self._merge([tokens], pair, idx)[0]  # Apply merge
            encoded_ids.extend(tokens)  # Append the final tokens to the result

        return encoded_ids

    def decode(self, ids: List[int]):
        """
        Decodes a list of token IDs back into a string.
        """
        assert self.__built, "Tokenizer must be built using `fit` before decoding."

        tokens = b"".join(self._vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    def save(self, file_path: str):
        """
        Save the tokenizer's vocab and merges to a file.
        Args:
            file_path: Path to save at.
        """
        with open(file_path, 'wb') as f:
            pickle.dump({'vocab': self._vocab, 'merges': self._merges}, f)
        print(f"[INFO] Tokenizer saved to {file_path}")



def load_BPETokenizerV2(file_path: str = "./bpe_tokenizer_v1_train_dataset.pth"):
    """
    Load the BPE (V2) tokenizer from a saved file without requiring texts in the constructor.
    Args:
        file_path (str): Path to the saved tokenizer file. (default = ./bpe_tokenizer_v1_train_dataset.pth)
    Returns:
        BPETokenizerV2 instance: A loaded tokenizer instance with vocab and merges.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Uninitialized instance of the tokenizer
    tokenizer = object.__new__(BPETokenizerV2)

    # Set vocab and merges directly
    tokenizer._vocab = data['vocab']
    tokenizer._merges = data['merges']
    setattr(tokenizer, '_BPETokenizerV2__built', True)  # FIX for name mangling

    # Initialize necessary attributes that are otherwise set in '__init__'
    tokenizer.gpt2_pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
    print(f"[INFO] Tokenizer loaded from: {file_path}")
    return tokenizer

# Testing the load process.
# You may use code similar to this to load the tokenizer

def _run_test():
    """
    Use this method to check if the tokenizer-path exists in dir.
    
    The output should look something like:
    Loading tokenizer from: ./bpe_tokenizer_v1_train_dataset.pth
    [INFO] Tokenizer loaded from: bpe_tokenizer_v1_train_dataset.pth
    ------
    Original string: some random string of words
    String encoded: [115, 521, 402, 383, 315, 1175, 290, 279, 267, 1965]
    Encoded-decoded: some random string of words
    """
    import os
    from pathlib import Path
    
    current_dir = Path(os.getcwd())
    tokenizer_path = "./bpe_tokenizer_v1_train_dataset.pth"
    path = None
    for file in current_dir.rglob(tokenizer_path): 
        path = file
    if path is None:
        print(f"Unable to find the tokenizer at: {current_dir} | Looking for the file w/name: {tokenizer_path}")
        return
    
    print(f"Loading tokenizer from: {path}")
    loaded_tokenizer = load_BPETokenizerV2(file_path=path)
    
    _temp_sample = "some random string of words"
    _temp_enc = loaded_tokenizer.encode(_temp_sample)
    _temp_dec = loaded_tokenizer.decode(_temp_enc)
    
    print("------")
    print(f"Original string: {_temp_sample}")
    print(f"String encoded: {_temp_enc}")
    print(f"Encoded-decoded: {_temp_dec}")

_run_test()