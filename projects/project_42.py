import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Required for BERT tokenizer
import numpy as np

"""
Project 42: Question Answering with BERT
Description:
Use a pretrained BERT QA model from TensorFlow Hub to answer questions given a context paragraph using extractive question answering.
"""

# Load BERT QA model and tokenizer
qa_model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
bert_model_name = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
preprocessor = hub.load(bert_model_name)
 
# Define context and question
context = """
TensorFlow is an end-to-end open-source platform for machine learning. It has a comprehensive,
flexible ecosystem of tools, libraries and community resources that lets researchers innovate
with machine learning and productionize AI easily.
"""
 
question = "What is TensorFlow used for?"
 
# Preprocess inputs for BERT QA
def preprocess_qa(context, question):
    inputs = preprocessor.bert_pack_inputs([question], [context], seq_length=256)
    return inputs
 
inputs = preprocess_qa(context, question)
 
# Run model and extract answer
outputs = qa_model(inputs)
start_logits = outputs['start_logits'][0].numpy()
end_logits = outputs['end_logits'][0].numpy()
input_word_ids = inputs['input_word_ids'][0].numpy()
 
# Find the answer span
start = np.argmax(start_logits)
end = np.argmax(end_logits)
 
# Convert token IDs back to string tokens
vocab = preprocessor.tokenize.get_vocabulary()
tokens = [vocab[i] if i < len(vocab) else "[UNK]" for i in input_word_ids]
 
# Reconstruct answer from tokens
answer = " ".join(tokens[start:end + 1]).replace(" ##", "")
print("Question:", question)
print("Answer:", answer)