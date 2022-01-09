from typing import List, Optional
import string
import gym
from ray.rllib.utils.spaces.repeated import Repeated
import numpy as np

from wordle import Wordle
from wordle.constants import NUM_LETTERS, NUM_ROUNDS, CHARSET, CHAR_TO_INDEX
from wordle.policy.vocab_elimination import VocabElimination
from wordle.game import print_observation


Char = gym.spaces.Discrete(len(CHARSET) + 1)
Word = gym.spaces.Tuple([Char, ] * NUM_LETTERS)
Result = gym.spaces.Discrete(4)
Feedback = gym.spaces.Tuple([Result, ] * NUM_LETTERS)


class WordleEnv(gym.Env):
    observation_space = gym.spaces.Dict({
        "vocab": Repeated(
            gym.spaces.Dict({
                "word": Word
            }),
            max_len=10000
        ),
        "feedback": Repeated(
            gym.spaces.Dict({
                "word": Word,
                "feedback": Feedback
            }),
            max_len=NUM_ROUNDS
        )
    })

    def __init__(self, vocab: List[str], display: bool = False):
        self.vocab = vocab
        self.vocab_observation = [{
            "word": tuple(CHAR_TO_INDEX[char] for char in word)
        } for word in vocab]
        self.action_space = gym.spaces.Discrete(len(vocab))
        self.vocab_tracker = VocabElimination(vocab)
        self.display = display

        self.wordle: Wordle = None
        self.target: str = None
        self.initial_feedback = [{
            "word": tuple([0 for _ in range(NUM_LETTERS)]),
            "feedback": tuple([0 for _ in range(NUM_LETTERS)])
        } for _ in range(NUM_ROUNDS)]
        self.feedback = None
        self.i_round = None

    def reset(self):
        self.i_round = 0
        self.target = np.random.choice(self.vocab)
        self.wordle = Wordle(self.vocab, self.target)
        self.vocab_tracker.reset()
        self.feedback = self.initial_feedback.copy()
        return self.feedback.copy()

    def _update_feedback(self, raw_observation):
        word = [0, ] * NUM_LETTERS
        feedback = [0, ] * NUM_LETTERS
        for i, (char, result) in enumerate(raw_observation):
            word[i] = CHAR_TO_INDEX[char]
            feedback[i] = result

        self.feedback[self.i_round]["word"] = tuple(word)
        self.feedback[self.i_round]["feedback"] = tuple(feedback)

    def _reward(self):
        num_remaining = self.vocab_tracker.remaining.sum()
        assert num_remaining > 0
        return - np.log(num_remaining)

    def step(self, action: int):
        """"""
        guess = self.vocab[action]
        observation = self.wordle.step(guess)
        if self.display:
            print_observation(observation)

        self._update_feedback(observation)
        self.vocab_tracker.update(observation)

        self.i_round += 1
        reward = self._reward()
        done = self.wordle.won or self.i_round == NUM_ROUNDS
        observation = {
            "vocab": self.vocab_observation,
            "feedback": self.feedback
        }
        return observation, reward, done, {}
