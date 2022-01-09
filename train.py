import colorama
import numpy as np
from colorama import init

import ray
from ray.rllib.models import ModelCatalog
from ray import tune
from ray.rllib.models.repeated_values import RepeatedValues
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.agents import ppo

from wordle.constants import NUM_ROUNDS
from wordle.rl.env import WordleEnv
from wordle.rl.model import WordleModel


def _minimal():
    env = WordleEnv({"vocab": vocab, "display": True})
    initial_obs = env.reset()
    from pprint import pprint
    env.observation_space["feedback"]["feedback"].contains(initial_obs["feedback"]["feedback"])
    check_obs = env.observation_space.contains(initial_obs)
    assert check_obs

    for i in range(100*NUM_ROUNDS):
        guess = np.random.randint(0, len(vocab))
        obs, reward, done, info = env.step(guess)

        check_obs = env.observation_space.contains(obs)
        assert check_obs

        if done:
            env.reset()
            print()
    exit()


if __name__ == '__main__':
    from nltk.corpus import words
    from wordle.constants import NUM_LETTERS
    colorama.init(autoreset=True)

    vocab = [w for w in words.words() if len(w) == NUM_LETTERS and w.islower()]
    vocab = sorted(list(set(vocab)))

    # _minimal()

    ModelCatalog.register_custom_model("WordleModel", WordleModel)

    ray.init()
    # ray.init(local_mode=True)

    stop = {
        "training_iteration": 1000
    }
    rllib_config = {
        "env": WordleEnv,
        "env_config": {
            "vocab": vocab,
        },
        "model": {
            "custom_model": "WordleModel",
            "custom_model_config": {
                "embedding_dim": 128
            }
        },
        "num_workers": 2,
        # "rollout_fragment_length": NUM_ROUNDS,
        # "train_batch_size": 6,
        "framework": "torch",
        "lr": 1e-3,
        "num_gpus": 1
    }

    tune.run(
        "PPO",
        config=rllib_config,
        stop=stop
    )

    # print("Running manual train loop without Ray Tune.")
    # ppo_config = ppo.DEFAULT_CONFIG.copy()
    # ppo_config.update(rllib_config)
    # # use fixed learning rate instead of grid search (needs tune)
    # ppo_config["lr"] = 1e-3
    # trainer = ppo.PPOTrainer(config=ppo_config, env=WordleEnv)
    # # run manual training loop and print results after each iteration
    # for _ in range(stop["training_iteration"]):
    #     result = trainer.train()
    #     print(result)
