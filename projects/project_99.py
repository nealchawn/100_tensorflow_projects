from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

"""
Project 99: Image Captioning with Transformers
Description:
Use a transformer-based model like BLIP to generate natural language descriptions for images, combining vision and language processing for automatic caption generation.
"""

# Load BLIP model and processor
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)
 
# Load an image (URL or local path)
image_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"  # Example image
image = Image.open(requests.get(image_url, stream=True).raw)
 
# Preprocess the image and prepare it for captioning
inputs = processor(images=image, return_tensors="pt")
 
# Generate caption for the image
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
 
# Display the image and caption
image.show()
print("\n📝 Generated Caption:", caption)