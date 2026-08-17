from transformers import pipeline

"""
Project 70: Few-shot Classification with GPT-style Prompting
Description:
Use GPT-style few-shot prompting to classify input text into categories by including examples directly in the prompt — no model fine-tuning required.
"""

# Load a text generation pipeline (e.g., GPT-3-like model)
generator = pipeline("text-generation", model="gpt2", max_length=100)
 
# Define prompt with a few examples for few-shot learning
prompt = """
Classify the following text into categories: [greeting, booking, question, complaint]
 
Example 1:
Text: Hello, how are you?
Category: greeting
 
Example 2:
Text: I want to book a flight to Toronto.
Category: booking
 
Example 3:
Text: Can you tell me the train schedule?
Category: question
 
Example 4:
Text: My hotel room was dirty and not cleaned.
Category: complaint
 
Now classify:
Text: I need help canceling my reservation.
Category:"""
 
# Generate prediction
output = generator(prompt, do_sample=False)[0]['generated_text']
 
# Extract model-generated category
predicted_line = output.strip().split('\n')[-1]
print("📝 Prompted Classification:\n", predicted_line)