import colorama
import numpy as np
from colorama import init

from wordle.constants import NUM_ROUNDS
from wordle.rl.env import WordleEnv

if __name__ == '__main__':
    from nltk.corpus import words
    from wordle.constants import NUM_LETTERS
    colorama.init(autoreset=True)
    vocab = [w for w in words.words() if len(w) == NUM_LETTERS and w.islower()]
    vocab = sorted(list(set(vocab)))

    env = WordleEnv(vocab, display=True)
    env.reset()

    for i in range(5*NUM_ROUNDS):
        guess = np.random.randint(0, len(vocab))
        obs, reward, done, info = env.step(guess)

        if done:
            env.reset()
            print()
