from nltk.corpus import words
from tqdm import tqdm
import numpy as np
from wordle.wordle import Wordle
from wordle.common import NUM_LETTERS
from wordle import policy as policies
from wordle.game import Game

if __name__ == '__main__':

    vocab = [w for w in words.words() if len(w) == NUM_LETTERS and w.islower()]

    results = []
    iterator = tqdm(vocab)
    for v in iterator:
        wordle = Wordle(set(vocab), v)
        # import pdb; pdb.set_trace()
        # policy = policies.RandomVocabElimination(vocab)
        # policy = policies.CharacterFrequencyVocabElimination(vocab)
        policy = policies.RandomUniqueVocabElimination(vocab)

        game = Game(wordle, policy, display=False)
        success = game.play()
        results.append(success)
        iterator.set_description("{:.2f}".format(np.mean(results)))
