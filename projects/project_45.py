import re
import collections

"""
Project 45: Spelling Correction Model
Description:
Implement a basic spelling correction model using edit distance and word probability ranking, inspired by Peter Norvig’s classic approach, in TensorFlow-friendly Python.
"""

# Sample corpus text to build word frequency (can be replaced with large corpus)
corpus = """
machine learning is fun and powerful. machine learning algorithms are used everywhere.
deep learning is a subset of machine learning. spelling mistakes can be corrected.
"""
 
# Tokenize words
def words(text): 
    return re.findall(r'\w+', text.lower())
 
# Build frequency dictionary
word_counts = collections.Counter(words(corpus))
word_probs = {w: count / sum(word_counts.values()) for w, count in word_counts.items()}
 
# Vocabulary
WORDS = set(word_counts)
 
# Generate possible edits (1 edit distance away)
def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)
 
# Known words from candidates
def known(words): 
    return set(w for w in words if w in WORDS)
 
# Generate candidates and rank by probability
def correct(word):
    candidates = (known([word]) or 
                  known(edits1(word)) or 
                  [word])
    return max(candidates, key=lambda w: word_probs.get(w, 0))
 
# Test the model
misspelled = ["machin", "lerning", "deeep", "corected"]
for w in misspelled:
    print(f"{w} → {correct(w)}")