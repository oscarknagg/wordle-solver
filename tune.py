import ray
from ray import tune
import numpy as np
from nltk.corpus import words

from wordle.wordle import Wordle
from wordle.game import Game
import wordle.policy as policies
from wordle.constants import NUM_LETTERS

REPORT_EVERY = 1000


def evaluate(config):
    c_explore, temperature = config["c_explore"], config["temperature"]
    results = []
    for i, v in enumerate(vocab):
        wordle = Wordle(set(vocab), v)
        policy = policies.ExploreExploitVocabElimination(vocab, c_explore, temperature)

        game = Game(wordle, policy, display=False)
        success = game.play()
        results.append(success)

        if (i + 1) % REPORT_EVERY == 0:
            tune.report(iterations=i, success_rate=np.mean(results))


if __name__ == '__main__':
    ray.init()

    vocab = [w for w in words.words() if len(w) == NUM_LETTERS and w.islower()]
    vocab = sorted(list(set(vocab)))

    scheduler = tune.schedulers.AsyncHyperBandScheduler()
    analysis = tune.run(
        evaluate,
        metric="success_rate",
        mode="max",
        scheduler=scheduler,
        num_samples=1000,
        config={
            "c_explore": tune.uniform(0, 10),
            "temperature": tune.loguniform(0.1, 5)
        }
    )

