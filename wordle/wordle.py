from wordle.common import IN_PLACE, IN_WORD, NOT_IN_WORD


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
