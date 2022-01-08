from wordle.policy.common import Policy


class Human(Policy):
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
