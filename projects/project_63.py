from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

"""
Project 63: Paraphrase Generation with T5
Description:
Generate paraphrased versions of input sentences using the T5 model and the instruction prefix "paraphrase:".
"""

# Load pretrained T5 model and tokenizer
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
 
# Input sentence to paraphrase
sentence = "The weather today is beautiful with clear skies and sunshine."
 
# Prepare input for T5
input_text = "paraphrase: " + sentence
input_ids = tokenizer.encode(input_text, return_tensors="tf", max_length=128, truncation=True)
 
# Generate paraphrase
outputs = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    early_stopping=True
)
paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
# Display result
print("💬 Original Sentence:\n", sentence)
print("\n🔁 Paraphrased Sentence:\n", paraphrased)