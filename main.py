import random

from wordle.game import Game
from wordle.policy.vocab_elimination import RandomVocabElimination
from wordle.wordle import Wordle
from wordle.common import NUM_LETTERS


if __name__ == '__main__':
    from nltk.corpus import words
    from colorama import init
    init(autoreset=True)

    vocab = [w for w in words.words() if len(w) == NUM_LETTERS and w.islower()]
    wordle = Wordle(set(vocab), random.choice(vocab))

    # policy = HumanPlayer()
    policy = RandomVocabElimination(vocab)

    game = Game(wordle, policy)
    success = game.play()
    print(success, wordle.target)


