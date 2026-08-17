import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # Required to load BERT tokenizer

"""
Project 60: Emotion Classification from Text (BERT)
Description:
Use a pretrained BERT model from TensorFlow Hub to classify the emotional tone (e.g., joy, anger, sadness) of a given text input.
"""

# Load pretrained BERT model and preprocessor
bert_preprocess = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.load("https://tfhub.dev/google/experts/bert/wiki_books/sst2/2")
 
# Example emotion-labeled dataset (text, label)
texts = [
    "I am so happy today!",         # joy
    "This is absolutely terrible",  # anger
    "I'm feeling really down",      # sadness
    "You did a great job!",         # joy
    "Why would you say that?"       # anger
]
labels = [0, 1, 2, 0, 1]  # 0 = joy, 1 = anger, 2 = sadness
 
# Preprocess input texts
text_inputs = tf.constant(texts)
encoder_inputs = bert_preprocess(text_inputs)
 
# Extract BERT embeddings
outputs = bert_encoder(encoder_inputs)['pooled_output']
 
# Build emotion classifier on top of BERT
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(outputs.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 emotion classes
])
 
# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(outputs, labels, epochs=20, verbose=0)
 
# Inference on new text
def classify_emotion(text):
    processed = bert_preprocess([text])
    embedded = bert_encoder(processed)['pooled_output']
    pred = model.predict(embedded)[0]
    emotion = ["Joy", "Anger", "Sadness"][tf.argmax(pred).numpy()]
    return emotion
 
# Test the classifier
print("Text: 'I can’t believe how amazing this is!'")
print("Predicted Emotion:", classify_emotion("I can’t believe how amazing this is!"))