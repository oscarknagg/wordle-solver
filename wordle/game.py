from wordle.common import NUM_ROUNDS, IN_PLACE, IN_WORD
from colorama import Back


def print_observation(obs):
    """Aims to mimic website output"""
    to_print = []
    for character, x in obs:
        if x == IN_PLACE:
            to_print.append(Back.GREEN + character)
        elif x == IN_WORD:
            to_print.append(Back.YELLOW + character)
        else:
            to_print.append(character)

    print(*to_print)


class Game:
    def __init__(self, wordle, policy, display=False):
        self.wordle = wordle
        self.policy = policy
        self.display = display

    def play(self):
        i_round = 0
        while i_round < NUM_ROUNDS:
            guess = self.policy.guess()
            observation = self.wordle.step(guess)
            if observation is None:
                continue

            self._print(observation)
            self.policy.update(observation)

            if self.wordle.won:
                return True

            i_round += 1

        return False

    def _print(self, obs):
        if self.display:
            print(obs)