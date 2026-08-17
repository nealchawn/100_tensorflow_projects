from transformers import DALL_E
import torch
from PIL import Image
import requests
from io import BytesIO

"""
Project 100: Text-to-Image Generation with DALL·E
Description:
Use OpenAI’s DALL·E model to generate images from text descriptions, enabling the creation of unique visuals from natural language prompts.
"""

# Load DALL·E model and processor (hypothetical, this could be an API call to OpenAI's DALL·E model)
model = DALL_E.from_pretrained("openai/dall-e")
 
# Input text prompt
prompt = "A futuristic city skyline at sunset, with flying cars and neon lights"
 
# Generate the image based on the prompt
generated_image = model.generate(prompt)
 
# Convert generated tensor to image
image = Image.fromarray(generated_image[0].numpy())
 
# Display the generated image
image.show()
 
# Optionally, save the generated image
image.save("generated_image.jpg")