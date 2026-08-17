from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

"""
Project 66: Headline Generation with T5
Description:
Use the T5 model to generate short and catchy headlines from longer news paragraphs using the prompt "headline:".
"""

# Load pretrained T5 model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
 
# News paragraph input
article = """
Scientists have discovered a new species of dinosaur in Argentina. The fossilized remains suggest the dinosaur was 
one of the largest to have ever walked the Earth, measuring over 120 feet long and weighing up to 70 tons.
"""
 
# Format with headline prompt
input_text = "headline: " + article
input_ids = tokenizer.encode(input_text, return_tensors="tf", max_length=256, truncation=True)
 
# Generate headline
outputs = model.generate(input_ids, max_length=20, num_beams=4, early_stopping=True)
headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
# Display result
print("📰 Article:\n", article.strip())
print("\n🗞️ Generated Headline:\n", headline)