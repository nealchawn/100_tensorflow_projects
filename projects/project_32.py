import tensorflow_datasets as tfds
import tensorflow as tf

"""
Project 32: Tokenization with SubwordTextEncoder
Description:
Use TensorFlow Datasets’ SubwordTextEncoder to build a subword-level tokenizer, then encode and decode sentences for NLP tasks like translation or classification.
"""

# Sample sentences
corpus = [
    "TensorFlow is an end-to-end open-source platform for machine learning.",
    "Natural Language Processing is a fascinating field.",
    "Tokenization is the first step in NLP pipelines.",
    "Subword tokenization helps with rare words."
]
 
# Build SubwordTextEncoder from corpus
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    corpus, target_vocab_size=1000
)
 
# Print vocabulary size
print("Subword vocabulary size:", tokenizer.vocab_size)
 
# Encode and decode a test sentence
test_sentence = "Subword tokenization is powerful for text models."
encoded = tokenizer.encode(test_sentence)
decoded = tokenizer.decode(encoded)
 
# Display results
print("\nOriginal Sentence:\n", test_sentence)
print("\nEncoded Tokens:\n", encoded)
print("\nDecoded Sentence:\n", decoded)
 
# Optional: visualize subwords
print("\nSubword Tokens:")
print([tokenizer.decode([token]) for token in encoded])