from sentence_transformers import SentenceTransformer, util

"""
Project 74: Multilingual Text Similarity with LaBSE
Description:
Use the LaBSE (Language-agnostic BERT Sentence Embedding) model to compute semantic similarity between texts across different languages.
"""

# Load LaBSE model (supports 100+ languages)
model = SentenceTransformer("sentence-transformers/LaBSE")
 
# Sentences in different languages with similar meanings
sentences = [
    "Where is the nearest pharmacy?",                # English
    "¿Dónde está la farmacia más cercana?",          # Spanish
    "Où se trouve la pharmacie la plus proche ?",     # French
    "Wo ist die nächste Apotheke?",                  # German
    "最近の薬局はどこですか？"                           # Japanese
]
 
# Encode sentences
embeddings = model.encode(sentences, convert_to_tensor=True)
 
# Compute similarity matrix
similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
 
# Display similarity results
print("🌐 Multilingual Text Similarity Matrix:")
for i, s in enumerate(sentences):
    sim_scores = ["{:.2f}".format(score) for score in similarity_matrix[i]]
    print(f"{s}\n → {sim_scores}\n")