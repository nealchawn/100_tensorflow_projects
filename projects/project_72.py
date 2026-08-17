from sentence_transformers import SentenceTransformer, util

"""
Project 72: Semantic Search with Sentence Embeddings
Description:
Build a simple semantic search engine using Sentence Transformers to find the most relevant document to a user’s query based on embedding similarity.
"""

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
 
# Sample document corpus
documents = [
    "Machine learning models require a lot of data to perform well.",
    "The Eiffel Tower is located in Paris, France.",
    "Neural networks are a powerful tool in deep learning.",
    "I love visiting historical places during vacations.",
    "Transformers have revolutionized natural language processing."
]
 
# Encode documents to embeddings
doc_embeddings = model.encode(documents, convert_to_tensor=True)
 
# User query
query = "How do neural networks work in AI?"
 
# Encode query
query_embedding = model.encode(query, convert_to_tensor=True)
 
# Compute cosine similarities
cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
 
# Find top match
top_result = int(cos_scores.argmax())
print("🔍 Query:", query)
print("\n📄 Top Matching Document:\n", documents[top_result])
print("\n🔢 Similarity Score: {:.2f}".format(cos_scores[top_result].item()))