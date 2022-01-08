import random

from wordle.game import Game
from wordle.policy.vocab_elimination import RandomVocabElimination
from wordle.wordle import Wordle


if __name__ == '__main__':
    from nltk.corpus import words
    from colorama import init
    init(autoreset=True)

    vocab = [w for w in words.words() if len(w) == 5 and w.islower()]
    wordle = Wordle(set(vocab), random.choice(vocab))

    # policy = HumanPlayer()
    policy = RandomVocabElimination(vocab)

    game = Game(wordle, policy)
    success = game.play()
    print(success, wordle.target)


