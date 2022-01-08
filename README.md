# wordle-solver

```python
import nltk
nltk.download("words")
```

| Corpus                | Algorithm                            | Win rate |
|-----------------------|--------------------------------------|----------|
| NLTK words            | RandomVocabElimination               | 0.72     |
| NLTK words            | CharacterFrequencyVocabElimination   | 0.14     |
| NLTK words            | RandomUniqueCharVocabElimination     | 0.83     |
| NLTK words            | RandomCharFreqUniqueVocabElimination | 0.80     |
| NLTK words            | RandomVocabElimination+              | 0.91     |
| NLTK words            | RandomUniqueCharVocabElimination+    | 0.94     |
| NLTK words            | InfoSeekingVocabElimination (v1)     | 0.40     |

CharacterFrequencyVocabElimination picks "arara" first in NLTK words corpus which 

New strategy: pick the word which you think will eliminate the most options
- This includes words that we know can't be the target but might give us lots of information

Consider this failure case:
```
Answer: bulky
a w f u l
c l o u d
s l u i t
g l u m p
l u r k y
h u l k y
```

Heuristics for what words are "valuable":
- Contains all unique letters
- Doesn't repeat any (char, position) that we're used before and we know are wrong
- Contains lots of letters we haven't tried yet
- Contains letters that we know are in the word but haven't positioned yet

Other heuristics based on this idea:
- If there's only one letter remaining to guess
  - e.g. * u s h y, could be "bushy", "cushy" or "mushy"
  - then it might be worth submitting "climb" as it will find out which of c, m or b is in the remaining slot


More failure cases:
```
Answer: douse
b i l g e
c o p s e
h o r s e
y o u s e
t o u s e
m o u s e

Answer: varve
m o i s e
a r u k e
c a r t e
p a r g e
l a r v e
w a r v e
```