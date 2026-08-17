from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

"""
Project 62: Automatic Text Summarization with T5
Description:
Use the pretrained t5-small model to generate concise summaries of longer texts using the "summarize:" instruction format.
"""

# Load T5 model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
 
# Input text to summarize
text = """
Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.
"""
 
# Prepare input for T5
input_text = "summarize: " + text
input_ids = tokenizer.encode(input_text, return_tensors="tf", max_length=512, truncation=True)
 
# Generate summary
outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
# Display result
print("📝 Original Text:\n", text.strip())
print("\n🔍 Generated Summary:\n", summary)