from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

"""
Project 87: Text-to-Image Prompt Generation
Description:
Use a language model like GPT or T5 to generate prompts that can be used for text-to-image generation models (e.g., DALL·E or Stable Diffusion).
"""

# Load a T5 model fine-tuned for generating descriptive prompts (or use GPT-3-like models)
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
 
# Input: user request for a scene description
user_input = "Generate a prompt for a landscape painting."
 
# Format the prompt generation task for the model
input_text = f"generate image prompt: {user_input}"
 
# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors="tf")
 
# Generate the image prompt
outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
generated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
# Display result
print("📝 User Input:", user_input)
print("\n🖼️ Generated Image Prompt:", generated_prompt)