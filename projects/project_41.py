import tensorflow as tf
import numpy as np

"""
Project 41: Chatbot using Seq2Seq + Attention
Description:
Build a simple chatbot using a sequence-to-sequence architecture with attention to generate context-aware responses from user input.
"""

# Sample conversation dataset
questions = ["hi", "how are you", "what's your name", "bye"]
answers = ["hello", "i'm fine", "i'm a chatbot", "goodbye"]
 
# Add <start> and <end> to target sentences
answers = [f"<start> {a} <end>" for a in answers]
 
# Tokenize input and output
q_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
a_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
 
q_tokenizer.fit_on_texts(questions)
a_tokenizer.fit_on_texts(answers)
 
q_sequences = q_tokenizer.texts_to_sequences(questions)
a_sequences = a_tokenizer.texts_to_sequences(answers)
 
max_q_len = max(len(q) for q in q_sequences)
max_a_len = max(len(a) for a in a_sequences)
 
q_padded = tf.keras.preprocessing.sequence.pad_sequences(q_sequences, maxlen=max_q_len, padding='post')
a_padded = tf.keras.preprocessing.sequence.pad_sequences(a_sequences, maxlen=max_a_len, padding='post')
 
# Split decoder input and output
decoder_input = a_padded[:, :-1]
decoder_target = tf.keras.utils.to_categorical(a_padded[:, 1:], num_classes=len(a_tokenizer.word_index) + 1)
 
# Define attention mechanism
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
 
    def call(self, enc_output, dec_hidden):
        dec_hidden = tf.expand_dims(dec_hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(dec_hidden)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context = attention_weights * enc_output
        context_vector = tf.reduce_sum(context, axis=1)
        return context_vector, attention_weights
 
# Encoder
encoder_inputs = tf.keras.Input(shape=(max_q_len,))
x = tf.keras.layers.Embedding(len(q_tokenizer.word_index) + 1, 64)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)(x)
 
# Decoder with attention
decoder_inputs = tf.keras.Input(shape=(max_a_len - 1,))
y = tf.keras.layers.Embedding(len(a_tokenizer.word_index) + 1, 64)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(y, initial_state=[state_h, state_c])
 
# Apply attention
attention = BahdanauAttention(64)
context_vector, _ = attention(encoder_outputs, state_h)
context_vector = tf.expand_dims(context_vector, 1)
context_vector = tf.repeat(context_vector, tf.shape(decoder_outputs)[1], axis=1)
 
concat = tf.concat([decoder_outputs, context_vector], axis=-1)
final_output = tf.keras.layers.Dense(len(a_tokenizer.word_index) + 1, activation='softmax')(concat)
 
# Compile and train model
model = tf.keras.Model([encoder_inputs, decoder_inputs], final_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([q_padded, decoder_input], decoder_target, epochs=500, verbose=0)
 
# Basic inference function
def chat(input_text):
    seq = q_tokenizer.texts_to_sequences([input_text])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_q_len, padding='post')
    enc_out, h, c = model.layers[3](model.layers[2](seq))
    dec_input = tf.constant([[a_tokenizer.word_index['<start>']]])
    result = []
 
    for _ in range(max_a_len):
        y = model.layers[6](model.layers[5](dec_input), initial_state=[h, c])[0]
        context_vec, _ = attention(enc_out, h)
        context_vec = tf.expand_dims(context_vec, 1)
        context_vec = tf.repeat(context_vec, tf.shape(y)[1], axis=1)
        concat = tf.concat([y, context_vec], axis=-1)
        pred = model.layers[-1](concat)
        token = tf.argmax(pred[:, -1, :], axis=-1).numpy()[0]
        if a_tokenizer.index_word[token] == '<end>':
            break
        result.append(a_tokenizer.index_word[token])
        dec_input = tf.constant([[token]])
 
    return ' '.join(result)
 
# Try it out
print("User: hi")
print("Bot:", chat("hi"))