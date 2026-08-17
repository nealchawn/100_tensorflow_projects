from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

"""
Project 61: Question Generation from Text (T5)
Description:
Use a pretrained T5 (Text-to-Text Transfer Transformer) model to generate questions from a given context sentence using TensorFlow and Hugging Face's Transformers.
"""

# Load T5 model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
 
# Input context (e.g., sentence from a passage)
context = "Albert Einstein was a physicist who developed the theory of relativity."
 
# Format input for T5
input_text = "generate question: " + context
input_ids = tokenizer.encode(input_text, return_tensors="tf")
 
# Generate output
outputs = model.generate(input_ids, max_length=32, num_beams=4, early_stopping=True)
question = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
print("📄 Context:\n", context)
print("\n❓Generated Question:\n", question)