import random
from colorama import Back

NUM_ROUNDS = 6
NOT_IN_WORD = 0
IN_WORD = 1
IN_PLACE = 2


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


class Wordle:
    def __init__(self, vocabulary, target):
        self.vocabulary = vocabulary
        self.target = target

        self.won = False

        # print("Debug: target = {}".format(self.target))

    def step(self, guess: str):
        if guess not in self.vocabulary:
            print("Guess not in word list.")
            return None

        if len(guess) != len(self.target):
            print("Guess is not length of target")
            return None

        self.won = guess == self.target

        output = []
        for target_char, guess_char in zip(self.target, guess):
            if target_char == guess_char:
                output.append((guess_char, IN_PLACE))
            elif guess_char in self.target:
                output.append((guess_char, IN_WORD))
            else:
                output.append((guess_char, NOT_IN_WORD))

        return output


class Game:
    def __init__(self, wordle, policy):
        self.wordle = wordle
        self.policy = policy

    def play(self):
        i_round = 0
        while i_round < NUM_ROUNDS:
            guess = self.policy.guess()
            observation = self.wordle.step(guess)
            if observation is None:
                continue

            print_observation(observation)
            self.policy.update(observation)

            if self.wordle.won:
                return True

            i_round += 1

        return False


class Policy:
    def guess(self) -> str:
        raise NotImplementedError

    def update(self, output):
        raise NotImplementedError


class HumanPlayer(Policy):
    def guess(self) -> str:
        prompt = "Enter your guess: "
        while True:
            player_input = input(prompt)
            if len(player_input) == 5 and player_input.isalpha() and player_input.islower():
                return player_input
            else:
                reasons = []
                if len(player_input) != 5:
                    reasons.append("wrong length")
                if not player_input.isalpha():
                    reasons.append("must be letters only")

                reasons = "({})".format(", ".join(reasons))
                prompt = "Bad guess {}, try again: ".format(reasons)

    def update(self, output):
        pass


if __name__ == '__main__':
    from nltk.corpus import words
    from colorama import init
    init(autoreset=True)

    vocab = [w for w in words.words() if len(w) == 5 and w.islower()]
    wordle = Wordle(set(vocab), random.choice(vocab))
    player = HumanPlayer()
    game = Game(wordle, player)
    success = game.play()
    print(wordle.target)


