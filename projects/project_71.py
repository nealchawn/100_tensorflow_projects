from sentence_transformers import SentenceTransformer, util

"""
Project 71: Text Embeddings with Sentence Transformers
Description:
Generate semantic vector embeddings for sentences using Sentence Transformers (e.g., all-MiniLM-L6-v2) for tasks like similarity, clustering, or retrieval.
"""

# Load pretrained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
 
# Define sample sentences
sentences = [
    "I love machine learning.",
    "Deep learning is a branch of AI.",
    "Let's grab a cup of coffee.",
    "Artificial intelligence is transforming the world."
]
 
# Generate embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)
 
# Compute cosine similarity between sentence pairs
cos_sim = util.pytorch_cos_sim(embeddings, embeddings)
 
# Display similarity matrix
print("🔢 Cosine Similarity Matrix:")
for i, s in enumerate(sentences):
    similarities = ["{:.2f}".format(score) for score in cos_sim[i]]
    print(f"{s}\n → {similarities}\n")