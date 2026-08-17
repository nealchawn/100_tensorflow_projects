import tensorflow as tf
import tensorflow_hub as hub

"""
Project 40: Text Summarization with Transformers
Description:
Use a pretrained transformer model from TensorFlow Hub to generate abstractive summaries of long text passages using the PEGASUS model.
"""

# Load PEGASUS summarization model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/pegasus/xsum/1")   # XSum fine-tuned PEGASUS
 
# Example long document
document = """
Artificial Intelligence (AI) is transforming the world around us. From voice assistants and self-driving cars
to medical diagnostics and financial predictions, AI systems are now integral to modern life. At its core, AI
involves creating machines that can mimic human intelligence and improve themselves through data-driven learning.
"""
 
# Preprocess text (the model expects UTF-8 input)
def generate_summary(text):
    inputs = [text]
    outputs = model.signatures['serving_default'](tf.constant(inputs))['outputs'].numpy()
    return outputs[0].decode('utf-8')
 
# Generate and print summary
summary = generate_summary(document)
print("Original Document:\n", document.strip())
print("\nGenerated Summary:\n", summary)