import string
from typing import List
import string
import numpy as np

from wordle.constants import IN_PLACE, IN_WORD, NOT_IN_WORD
from wordle.policy.common import Policy
from wordle import utils


class VocabElimination:
    def __init__(self, vocab: List[str]):
        self.v = np.char.array([[char for char in v] for v in vocab])
        self.eliminated = np.ones(len(vocab)).astype(bool)

    def process_in_place_result(self, place, char):
        self.eliminated &= self.v[:, place] == char

    def process_in_word_result(self, place, char):
        # Remaining words must contain char in any position
        self.eliminated &= (self.v == char).any(axis=1)
        # Remaining words cannot contain `char` at position `place` (otherwise
        # it would be a GREEN/IN_PLACE result)
        self.eliminated &= (self.v[:, place] != char)

    def process_not_in_word_result(self, char):
        self.eliminated &= ~(self.v == char).any(axis=1)

    def update(self, output):
        for i, (char, result) in enumerate(output):
            if result == IN_PLACE:
                self.process_in_place_result(i, char)
            elif result == IN_WORD:
                self.process_in_word_result(i, char)
            elif result == NOT_IN_WORD:
                self.process_not_in_word_result(char)


class RandomVocabElimination(VocabElimination, Policy):
    """Eliminates vocabulary based on observations and guesses randomly
    based on what remains."""

    def guess(self) -> str:
        v = self.v[self.eliminated]
        i = np.random.randint(len(v))
        return "".join(v[i])


class CharacterFrequencyVocabElimination(VocabElimination, Policy):
    """Guesses words based on which ones contain common characters"""
    def __init__(self, vocab: List[str], deterministic=True):
        super().__init__(vocab)
        self.deterministic = deterministic

        # Calculate character frequencies
        char_frequencies = {}
        for char in string.ascii_lowercase:
            char_frequencies[char] = (self.v == char).sum()

        # Replaces each character in the vocab array with its frequency in the vocab
        chars_to_freqs = utils.map_array(self.v, char_frequencies)
        self.freqs = chars_to_freqs.sum(axis=1)
        self.probs = self.freqs / chars_to_freqs.sum()

    def guess(self) -> str:
        v = self.v[self.eliminated]
        if self.deterministic:
            i = self.freqs[self.eliminated].argmax()
        else:
            i = np.random.choice(np.arange(len(v)), 10, p=self.probs)
        return "".join(v[i])


class RandomUniqueVocabElimination(VocabElimination, Policy):
    """Eliminates vocabulary based on observations and guesses randomly
    based on what remains, prioritising words with more unique characters."""
    def __init__(self, vocab: List[str]):
        super().__init__(vocab)
        self.n_unique = np.array([len(set([char for char in v])) for v in vocab])

    def guess(self) -> str:
        n_unique = self.n_unique.copy()
        n_unique[~self.eliminated] = 0
        mask_plus_unique = self.eliminated & (n_unique == n_unique.max())
        v = self.v[mask_plus_unique]
        i = np.random.randint(len(v))
        return "".join(v[i])


class RandomCharFreqUniqueVocabElimination(VocabElimination, Policy):
    def __init__(self, vocab: List[str]):
        super().__init__(vocab)
        self.n_unique = np.array([len(set([char for char in v])) for v in vocab])

        # Calculate character frequencies
        char_frequencies = {}
        for char in string.ascii_lowercase:
            char_frequencies[char] = (self.v == char).sum()

        u, inv = np.unique(self.v, return_inverse=True)
        # Replaces each character in the vocab array with its frequency in the vocab
        chars_to_freqs = np.array([char_frequencies[x] for x in u])[inv].reshape(self.v.shape)
        self.freqs = chars_to_freqs.sum(axis=1)
        self.probs = self.freqs / chars_to_freqs.sum()

    def guess(self) -> str:
        n_unique = self.n_unique.copy()
        n_unique[~self.eliminated] = 0
        mask_plus_unique = self.eliminated & (n_unique == n_unique.max())
        v = self.v[mask_plus_unique]
        i = self.freqs[mask_plus_unique].argmax()
        return "".join(v[i])


class InfoSeekingVocabElimination(VocabElimination, Policy):
    def __init__(self, vocab: List[str]):
        super().__init__(vocab)
        self.n_unique = np.array([len(set([char for char in v])) for v in vocab])
        self.tried_letters = set()

    def guess(self) -> str:
        n_unique = self.n_unique.copy()
        n_unique[~self.eliminated] = 0
        mask_plus_unique = self.eliminated & (n_unique == n_unique.max())
        v = self.v[mask_plus_unique]
        # But now don't guess based only remaining possible answers

        guess = ""
        for char in guess:
            self.tried_letters.add(char)
        return guess
