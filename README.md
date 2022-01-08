# wordle-solver

```python
import nltk
nltk.download("words")
```

| Corpus                | Algorithm                           | Win rate |
|-----------------------|-------------------------------------|----------|
| NLTK words            | RandomVocabElimination              | 0.72     |
| NLTK words            | CharacterFrequencyVocabElimination  | 0.14     |
| NLTK words            | RandomUniqueCharVocabElimination    | 0.83     |

CharacterFrequencyVocabElimination picks "arara" first in NLTK words corpus which 