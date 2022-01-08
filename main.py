import random
import argparse

from wordle.game import Game
from wordle.wordle import Wordle
from wordle.constants import NUM_LETTERS
import wordle.policy as policies

if __name__ == '__main__':
    from nltk.corpus import words
    from colorama import init
    init(autoreset=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="Human")
    args = parser.parse_args()

    vocab = [w for w in words.words() if len(w) == NUM_LETTERS and w.islower()]
    wordle = Wordle(set(vocab), random.choice(vocab))

    # policy = policies.RandomVocabElimination(vocab)
    policy = policies.RandomUniqueVocabElimination(vocab)

    game = Game(wordle, policy, display=True)
    success = game.play()
    print(success, wordle.target)


