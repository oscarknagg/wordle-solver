from typing import List

import numpy as np

from wordle.common import IN_PLACE, IN_WORD, NOT_IN_WORD
from wordle.policy.common import Policy


class RandomVocabElimination(Policy):
    """Eliminates vocabulary based on observations and guesses randomly
    based on what remains."""

    def __init__(self, vocab: List[str]):
        self.v = np.char.array([[char for char in v] for v in vocab])

    def process_in_place_result(self, place, char):
        mask = self.v[:, place] == char
        self.v = self.v[mask]

    def process_in_word_result(self, char):
        mask = (self.v == char).any(axis=1)
        self.v = self.v[mask]

    def process_not_in_word_result(self, char):
        mask = ~(self.v == char).any(axis=1)
        self.v = self.v[mask]

    def update(self, output):
        for i, (char, result) in enumerate(output):
            if result == IN_PLACE:
                self.process_in_place_result(i, char)
            elif result == IN_WORD:
                self.process_in_word_result(char)
            elif result == NOT_IN_WORD:
                self.process_not_in_word_result(char)

    def guess(self) -> str:
        i = np.random.randint(len(self.v))
        return "".join(self.v[i])
