import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

"""
Project 44: Topic Modeling using LDA + TensorFlow
Description:
Use tfp.distributions.OneHotCategorical and tensorflow_probability to simulate basic Latent Dirichlet Allocation (LDA) behavior on synthetic text data for topic modeling.

⚠️ For production use, prefer libraries like gensim or scikit-learn. This project illustrates LDA concepts in TensorFlow using TensorFlow Probability (TFP).
"""

# Simulated vocabulary and documents
vocab = ["economy", "money", "market", "football", "goal", "team", "python", "code", "model"]
vocab_size = len(vocab)
 
# Simulate 3 topics
topics = [
    [0.4, 0.3, 0.3, 0, 0, 0, 0, 0, 0],  # Topic 0: economy
    [0, 0, 0, 0.3, 0.4, 0.3, 0, 0, 0],  # Topic 1: sports
    [0, 0, 0, 0, 0, 0, 0.3, 0.3, 0.4]   # Topic 2: tech
]
 
# Sample a topic distribution per document (Dirichlet prior)
doc_topic_dist = tfp.distributions.Dirichlet(concentration=[0.5, 0.5, 0.5])
sampled_topic_dists = doc_topic_dist.sample(5)  # 5 documents
 
# Sample words for each document based on topic mixture
for i, topic_dist in enumerate(sampled_topic_dists):
    topic_idx = tf.random.categorical(tf.math.log([topic_dist]), num_samples=1)[0][0]
    topic_word_probs = topics[topic_idx]
    
    word_dist = tfp.distributions.OneHotCategorical(probs=topic_word_probs)
    sampled_words = word_dist.sample(6).numpy()  # 6 words per doc
    
    print(f"\nDocument {i+1} (Topic {topic_idx.numpy()}):")
    for word_vec in sampled_words:
        word_idx = np.argmax(word_vec)
        print(vocab[word_idx], end=' ')