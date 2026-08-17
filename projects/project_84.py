import tensorflow as tf
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

"""
Project 84: Contrastive Learning for Sentence Embeddings
Description:
Train sentence embeddings using a contrastive learning approach (e.g., SimCSE) where positive pairs are augmented versions of the same sentence and negative pairs are randomly selected.
"""

# Define training data: sentence pairs (same sentence twice for unsupervised SimCSE)
sentences = [
    "The sky is blue today.",
    "Dogs are loyal animals.",
    "Transformers perform well on NLP tasks.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language."
]
train_examples = [InputExample(texts=[s, s]) for s in sentences]
 
# Load a SentenceTransformer model
model = SentenceTransformer("distilbert-base-uncased")
 
# Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
 
# Use contrastive loss (e.g., CosineSimilarityLoss for SimCSE-style training)
train_loss = losses.CosineSimilarityLoss(model)
 
# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=10,
    show_progress_bar=False
)
 
# Test embeddings
embeddings = model.encode(["The Eiffel Tower is in Paris.", "Paris is home to the Eiffel Tower."])
similarity = tf.keras.losses.cosine_similarity(embeddings[0], embeddings[1]).numpy()
 
print("🧪 Cosine similarity between semantically similar sentences: {:.2f}".format(1 - similarity))
