import string
from typing import List
import string
import numpy as np

from wordle.constants import IN_PLACE, IN_WORD, NOT_IN_WORD, NUM_LETTERS
from wordle.policy.common import Policy
from wordle import utils


class VocabElimination:
    def __init__(self, vocab: List[str]):
        self.v = np.char.array([[char for char in v] for v in vocab])
        self.reset()

    def reset(self):
        self.remaining = np.ones(len(self.v)).astype(bool)
        self.untried_letters = {char: 1 for char in string.ascii_lowercase}
        self.in_word_but_not_placed = {char: 0 for char in string.ascii_lowercase}

    def process_in_place_result(self, place, char):
        self.remaining &= self.v[:, place] == char
        self.in_word_but_not_placed[char] = 0

    def process_in_word_result(self, place, char):
        # Remaining words must contain char in any position
        self.remaining &= (self.v == char).any(axis=1)
        # Remaining words cannot contain `char` at position `place` (otherwise
        # it would be a GREEN/IN_PLACE result)
        self.remaining &= (self.v[:, place] != char)
        self.in_word_but_not_placed[char] = 1

    def process_not_in_word_result(self, char):
        self.remaining &= ~(self.v == char).any(axis=1)

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
        v = self.v[self.remaining]
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
        v = self.v[self.remaining]
        if self.deterministic:
            i = self.freqs[self.remaining].argmax()
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
        n_unique[~self.remaining] = 0
        mask_plus_unique = self.remaining & (n_unique == n_unique.max())
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
        n_unique[~self.remaining] = 0
        mask_plus_unique = self.remaining & (n_unique == n_unique.max())
        v = self.v[mask_plus_unique]
        i = self.freqs[mask_plus_unique].argmax()
        return "".join(v[i])


class InfoSeekingVocabElimination(VocabElimination, Policy):
    def __init__(self, vocab: List[str]):
        super().__init__(vocab)
        self.n_unique = np.array([len(set([char for char in v])) for v in vocab])
        self.is_untried = np.ones(self.v.shape)
        self.is_in_word_but_not_placed = np.zeros(self.v.shape)

        self.vocab = np.char.array(vocab)
        self.v_int = utils.map_array(self.v, {char: i for i, char in enumerate(string.ascii_lowercase)})

        self.v_onehot = np.zeros((len(self.v), 5, len(string.ascii_lowercase)))
        for i in range(NUM_LETTERS):
            self.v_onehot[np.arange(len(self.v)), i, self.v_int[:, i]] = 1

    def guess(self) -> str:
        self.is_untried = utils.map_array(self.v, self.untried_letters)
        untried_letters_arr = np.array(list(self.untried_letters.values()))
        n_unique_untried = (self.v_onehot.sum(axis=1) * untried_letters_arr).clip(0, 1).sum(axis=1).astype(int)

        self.is_in_word_but_not_placed = utils.map_array(self.v, self.in_word_but_not_placed)
        first_occurences = (self.v != np.roll(self.v, 1, axis=1)).astype(int)
        # Multiply by first occurences to prevent the following edge case:
        # Say the letter "a" is in the word but unplaced
        # The candidate guess "aaabc" would get a score of 3
        unplaced_letters_arr = untried_letters_arr = np.array(list(self.untried_letters.values()))
        # n_notplaced = self.is_in_word_but_not_placed.sum(axis=1)
        n_notplaced = (self.v_onehot.sum(axis=1) * unplaced_letters_arr).clip(0, 1).sum(axis=1).astype(int)

        scores = n_unique_untried + n_notplaced

        candidates = self.v[scores == scores.max()]
        # print("Non-eliminated: ", len(self.v[self.eliminated]))
        # print("Untried: ", {k for k, v in self.untried_letters.items() if v == 1})
        # print("Tried: ", {k for k, v in self.untried_letters.items() if v == 0})
        # print("Unplaced: ", {k for k, v in self.in_word_but_not_placed.items() if v == 1})
        # print("Best scores. Unplaced={}, Untried={}, Total={}".format(
        #     n_notplaced.max(), n_unique_untried.max(), scores.max()))
        # print("gorge" in self.vocab[self.eliminated].tolist())
        # import pdb; pdb.set_trace()

        if len(self.v[self.remaining]) == 1:
            return "".join(self.v[self.remaining][0])
        else:
            i = np.random.randint(len(candidates))
            guess = "".join(candidates[i])
            for char in guess:
                self.untried_letters[char] = 0
            return guess


class ExploreExploitVocabElimination(VocabElimination, Policy):
    """"""
    def __init__(self, vocab: List[str], c_explore: float = 1.0, temperature: float = 0.1):
        super().__init__(vocab)
        self.n_unique = np.array([len(set([char for char in v])) for v in vocab])
        self.is_untried = np.ones(self.v.shape)
        self.is_in_word_but_not_placed = np.zeros(self.v.shape)

        self.vocab = np.char.array(vocab)
        v_int = utils.map_array(self.v, {char: i for i, char in enumerate(string.ascii_lowercase)})
        self.v_onehot = np.zeros((len(self.v), 5, len(string.ascii_lowercase)))
        for i in range(NUM_LETTERS):
            self.v_onehot[np.arange(len(self.v)), i, v_int[:, i]] = 1

        self.c_explore = c_explore
        self.temperature = temperature

    def guess(self) -> str:
        self.is_untried = utils.map_array(self.v, self.untried_letters)
        untried_letters_arr = np.array(list(self.untried_letters.values()))
        n_unique_untried = (self.v_onehot.sum(axis=1) * untried_letters_arr).clip(0, 1).sum(axis=1).astype(int)

        scores = 5 * self.remaining + self.c_explore * n_unique_untried

        if len(self.v[self.remaining]) == 1:
            return "".join(self.v[self.remaining][0])
        else:
            i = np.random.choice(np.arange(len(self.v)), 1, p=utils.softmax(scores, self.temperature)).item()
            guess = "".join(self.v[i])
            for char in guess:
                self.untried_letters[char] = 0
            return guess
