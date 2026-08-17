from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

"""
Project 65: Text Simplification using T5
Description:
Use a T5 model to simplify complex sentences into easier-to-read language with the prompt "simplify:".
"""

# Load T5 model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
 
# Input complex sentence
text = "The precipitation will persist throughout the afternoon, primarily impacting regions with lower atmospheric pressure."
 
# Add simplification prompt
input_text = "simplify: " + text
input_ids = tokenizer.encode(input_text, return_tensors="tf", max_length=128, truncation=True)
 
# Generate simplified output
outputs = model.generate(
    input_ids,
    max_length=50,
    num_beams=4,
    early_stopping=True
)
simplified = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
# Display result
print("📚 Original Text:\n", text)
print("\n🧾 Simplified Version:\n", simplified)