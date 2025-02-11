import torch
from typing import List

class TokenizerV0():
    def __init__(self,
                 token_size: int,
                 strip_punctuation: bool = False,
                 data: List[str] = [],
                 build_tokenizer: bool = False):
        """
        Creates an instance of a tokenizer.
        You can use this object to convert raw data into tokens.

        Args:
            token_size (int): size of each token
            strip_puntuation (bool): strips punctuation if True (default False)
            data (List[str]): dataset used to build the tokenzier (default None)
            build_tokenizer (bool): builds_tokenizer if True (default False)

        NOTE: do read the doc-string for `.build_tokenizer` before building the tokenizer in the constructor (here).
        """
        self.token_size = token_size

        self._strip_punctuation = strip_punctuation
        self._built = False

        # Stuff to encode/decode some text
        self._vocab = None     # vocab that tokenizer understands
        self._encode = None    # string -> index (dict)
        self._decode = None    # index -> string (dict)

        # Build tokenizer if needed
        self._unk_token = "<UNK>"
        if build_tokenizer:
            assert len(data) > 0, f"Must pass data to build the tokenizer. If you don't want to build the tokenizer yet, set `build_tokenizer = False`."
            self.build_tokenizer(data, unk_token=self._unk_token)

    def _preprocess(self,
                    text: str) -> str:
        """
        Preprocesses a string to handle punctuation and normalize tokens.
        str -> str (preprocessed)
        
        Args:
            text (str): a string to preprocess
        Returns:
            str
        """
        import re
        import string

        # Strip punctuation if required
        if self._strip_punctuation:
            text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)
        else:
            # Add spaces around punctuation
            text = re.sub(r"([.,!?;:])", r" \1 ", text)
            text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text

    def _token_split(self,
                     text: str) -> List[str]:
        """
        Converts a single string input into token_size number of words and returns them as a List[str].

        Args:
            text (str): a string to tokenize
        Returns:
            List[str]
        """
        tokens = []
        
        # Preprocess the string
        text = self._preprocess(text=text)

        # Append into tokens
        words = text.split()
        for i in range(0, len(words), self.token_size):
            tokens.append(" ".join(words[i:i + self.token_size]))

        return tokens

    def build_tokenizer(self,
                        data: List[str],
                        return_vocab: bool = False,
                        unk_token: str = "<UNK>"):
        """
        Initializes the tokenizer's vocab, encoder-dict and decoder-dict.
        If return_vocab = True, then the vocab-list is returned.

        Args:
            data (List[str]): list of sentences to build the vocab on
            return_vocab (bool): returns the vocab if True (default False)
            unk_token (str): the token used for unkown words (default "<UNK>")
        Returns:
            List[str]: list of strings (vocab), if return_vocab = True
            Prints on completion: if return_vocab = False
        """
        # Update priv - unk_token
        self._unk_token = unk_token
        
        # Get all tokens from data
        all_tokens = []
        for sentence in data:
            all_tokens.extend(self._token_split(sentence))
        all_tokens.append(unk_token)

        # Init _vocab, _encode, _decode
        self._vocab = list(sorted(set(all_tokens)))
        self._encode = {token: idx for idx, token in enumerate(self._vocab)}
        self._decode = {idx: token for token, idx in self._encode.items()}
        assert len(self._vocab) > 0, f"No data passed to build vocab. Got ({data})"
        self._built = True

        # Return based on user requirement
        if return_vocab:
            return self._vocab
        else:
            print(f"[INFO] Vocab, encoder, decoder are initialized and ready to use.")
            print(f"[INFO] Vocab size: {len(self._vocab)}")
    
    def encode(self,
               text: str,
               return_tensor: bool = False,
               skip_special_tokens = True):
        """
        Encodes the text (str) into numerical tokens on the built vocab.

        Args:
            text (str): string that you want to encode into indices
            return_tensor (bool): reutrns a PyTorch tensor if True, a Python list if False (default False)
            skip_special_tokens (bool): skips tokens outside vocab if True, assigns special index if False (default True)
            unk_token (str): the token used for unkown words (default "<UNK>")

        Returns:
            List[int]: a list of token indices (return_tensors = False).
            Tensor[int]: a PyTorch Tensor of token indices (return_tensors = True).
        """
        ### Error handling
        # Check if tokenizer is built (using _vocab and _encode)
        assert self._built == True, f"Tokenizer not built yet, run `.build_tokenizer()` first."
        
        #  Tokenize the text
        tokens = self._token_split(text=text)
        
        # Encode tokens into indices
        if skip_special_tokens:
            token_indices = [self._encode[token] for token in tokens if token in self._encode]
        else:
            token_indices = [self._encode[token] if token in self._encode else self._encode[self._unk_token] for token in tokens]
        
        # Setup tensor support
        if return_tensor:
            return torch.tensor(data=token_indices, dtype=torch.int32)
        return token_indices

    def decode(self,
               token_ids: List[int],
               skip_special_tokens: bool = True,
               return_type: str = "str"):
        """
        Decodes a list/tensor of tokens into a readable list of string(s).

        Args:
            tokens_ids (List[int] | Tensor[int]): a Python list or PyTorch tensor with indices to map into strings
            skip_special_tokens (bool): skips a token that isnt a part of the vocab if True (default True)
            return_type (str): returns a string if 'str', returns a list of 'list' (choose from: str, list)
        Returns:
            List[str]: a list of strings, if return_type = 'list'
            str: a string if return_type = 'str'
        """
        assert self._built == True, f"Tokenizer not built yet, run `.build_tokenizer()` first."

        tokens = []
        for idx in token_ids:
            if idx in self._decode:
                token = self._decode[idx]
                if not skip_special_tokens and token == self._unk_token:
                    continue
                tokens.append(token)
            else:
                if not skip_special_tokens:
                    tokens.append(self._unk_token)

        return tokens if return_type=="list" else " ".join(tokens)
