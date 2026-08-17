from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from transformers import TextClassificationPipeline
import tensorflow as tf

"""
Project 81: News Topic Classification with DistilBERT
Description:
Use the lightweight DistilBERT transformer model for classifying news headlines or articles into topics like politics, tech, sports, etc.
"""

# Load pretrained DistilBERT model fine-tuned on AG News dataset
model_name = "distilbert-base-uncased-finetuned-ag-news"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
 
# Create classification pipeline
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, framework='tf')
 
# Sample news headlines
headlines = [
    "Apple announces the launch of its new iPhone.",
    "The government passed a new climate change bill.",
    "Manchester United wins the championship match.",
    "NASA prepares a new mission to Mars."
]
 
# Classify each headline
print("📰 News Headline Classification:\n")
for headline in headlines:
    result = classifier(headline)[0]
    label = max(result, key=lambda x: x['score'])['label']
    print(f"📄 {headline}\n → Topic: {label}\n")