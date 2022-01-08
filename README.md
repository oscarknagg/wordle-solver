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
| NLTK words            | RandomUniqueCharVocabElimination+    | 0.XX     |

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
- 

Other heuristics based on this idea:
- If there's only one letter remaining to guess
  - e.g. * u s h y, could be "bushy", "cushy" or "mushy"
  - then it might be worth submitting "climb" as it will find out which of c, m or b is in the remaining slot
