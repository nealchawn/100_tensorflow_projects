from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

"""
Project 86: Table-to-Text Generation with T5
Description:
Generate natural language summaries from tabular data using a T5 model by converting table rows into serialized text format (e.g., column: value) as input.
"""

# Load pretrained T5 model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
 
# Sample table row: person info
table_row = {
    "Name": "Alice Johnson",
    "Age": "29",
    "Occupation": "Software Engineer",
    "Location": "San Francisco"
}
 
# Convert table row to serialized string input
input_text = "table to text: " + " | ".join(f"{k}: {v}" for k, v in table_row.items())
 
# Tokenize input
input_ids = tokenizer.encode(input_text, return_tensors="tf")
 
# Generate summary
outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
# Display result
print("📊 Table Row:\n", table_row)
print("\n📝 Generated Summary:\n", summary)