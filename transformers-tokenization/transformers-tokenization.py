import numpy as np
from typing import List, Dict

from pygments.lexer import words


class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id[self.pad_token] = 0
        self.id_to_word[0] = self.pad_token
        self.word_to_id[self.unk_token] = 1
        self.id_to_word[1] = self.unk_token
        self.word_to_id[self.bos_token] = 2
        self.id_to_word[2] = self.bos_token
        self.word_to_id[self.eos_token] = 3
        self.id_to_word[3] = self.eos_token
        self.vocab_size = 4
        word_set = set()
        for text in texts:
            for word in text.split():
                word_set.add(word)
        word_list = list(word_set)
        word_list.sort()

        # alphabetically
        for word in word_list:
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1
        pass

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        return [self.word_to_id.get(word, self.word_to_id[self.unk_token]) for word in text.split()]
        pass

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        return " ".join([self.id_to_word.get(id) for id in ids])
        pass