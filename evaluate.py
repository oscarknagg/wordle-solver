from nltk.corpus import words
from tqdm import tqdm
import numpy as np
from wordle.wordle import Wordle
from wordle.common import NUM_LETTERS
from wordle.policy.vocab_elimination import RandomVocabElimination
from wordle.game import Game

if __name__ == '__main__':

    vocab = [w for w in words.words() if len(w) == NUM_LETTERS and w.islower()]

    results = []
    iterator = tqdm(vocab)
    for v in iterator:
        wordle = Wordle(set(vocab), v)
        policy = RandomVocabElimination(vocab)
        game = Game(wordle, policy)
        success = game.play()
        results.append(success)
        iterator.set_description("{:.2f}".format(np.mean(results)))
