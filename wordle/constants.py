import string

NUM_ROUNDS = 6

# result enumeration
NULL = 0
IN_PLACE = 1
IN_WORD = 2
NOT_IN_WORD = 3

NUM_LETTERS = 5
CHARSET = string.ascii_lowercase
CHAR_TO_INDEX = {char: i for i, char in enumerate(CHARSET)}
