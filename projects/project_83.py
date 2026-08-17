from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
)
from datasets import Dataset
import numpy as np

"""
Project 83: Custom Text Classifier with HuggingFace Trainer API
Description:
Train a custom transformer-based text classifier using the Hugging Face Trainer API with your own dataset — ideal for fine-tuning on small- to mid-sized text tasks.
"""

# Sample dataset for binary classification
data = {
    "text": [
        "The new iPhone looks amazing!",
        "The game was boring and unwatchable.",
        "I absolutely loved the movie!",
        "This laptop performs very poorly.",
        "Fantastic service and great food!"
    ],
    "label": [1, 0, 1, 0, 1]  # 1 = positive, 0 = negative
}
 
# Convert to Hugging Face dataset
dataset = Dataset.from_dict(data).train_test_split(test_size=0.2)
 
# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
 
# Tokenize dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)
 
tokenized_ds = dataset.map(tokenize, batched=True)
tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
 
# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    save_strategy="epoch"
)
 
# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"]
)
 
# Train model
trainer.train()
 
# Inference
sample = "I hated the user interface of this app."
tokens = tokenizer(sample, return_tensors="pt", truncation=True, padding=True)
output = model(**tokens)
pred = int(np.argmax(output.logits.detach().numpy()))
print("📝 Sample:", sample)
print("🔖 Predicted Sentiment:", "Positive" if pred == 1 else "Negative")