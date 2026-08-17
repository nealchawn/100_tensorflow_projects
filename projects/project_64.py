from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

"""
Project 64: Grammar Correction with T5
Description:
Use the T5 model with a grammar correction prompt to fix grammatical errors in a sentence. This works well with fine-tuned versions or instructional prompts like "correct grammar:".
"""

# Load the T5 model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
 
# Input sentence with grammar issues
sentence = "She no went to the store because it raining."
 
# Format the input with a grammar correction instruction
input_text = "correct grammar: " + sentence
input_ids = tokenizer.encode(input_text, return_tensors="tf", max_length=128, truncation=True)
 
# Generate the corrected sentence
outputs = model.generate(
    input_ids,
    max_length=64,
    num_beams=5,
    early_stopping=True
)
corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
# Display result
print("❌ Original:\n", sentence)
print("\n✅ Corrected:\n", corrected)