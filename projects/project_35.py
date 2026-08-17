import tensorflow as tf
import numpy as np

"""
Project 35: Sequence-to-Sequence Translator (English–French)
Description:
Build a basic Seq2Seq (Encoder–Decoder) model using LSTM layers in TensorFlow 2 to translate short English sentences into French.
"""

# Sample parallel corpus (tiny for demo)
english_sentences = ["hello", "how are you", "thank you", "good night"]
french_sentences = ["bonjour", "comment ça va", "merci", "bonne nuit"]
 
# Tokenize source (English)
src_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
src_tokenizer.fit_on_texts(english_sentences)
src_sequences = src_tokenizer.texts_to_sequences(english_sentences)
src_word_index = src_tokenizer.word_index
src_vocab_size = len(src_word_index) + 1
 
# Tokenize target (French) and add <start>, <end> tokens
french_sentences = [f"<start> {s} <end>" for s in french_sentences]
tgt_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tgt_tokenizer.fit_on_texts(french_sentences)
tgt_sequences = tgt_tokenizer.texts_to_sequences(french_sentences)
tgt_word_index = tgt_tokenizer.word_index
tgt_vocab_size = len(tgt_word_index) + 1
 
# Pad sequences
src_padded = tf.keras.preprocessing.sequence.pad_sequences(src_sequences, padding='post')
tgt_padded = tf.keras.preprocessing.sequence.pad_sequences(tgt_sequences, padding='post')
 
# Split target into decoder input and output
decoder_input = tgt_padded[:, :-1]
decoder_target = tf.keras.utils.to_categorical(tgt_padded[:, 1:], num_classes=tgt_vocab_size)
 
# Define the Seq2Seq model
embedding_dim = 64
encoder_inputs = tf.keras.Input(shape=(None,))
x = tf.keras.layers.Embedding(src_vocab_size, embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(64, return_state=True)(x)
 
decoder_inputs = tf.keras.Input(shape=(None,))
y = tf.keras.layers.Embedding(tgt_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(64, return_sequences=True)
decoder_outputs = decoder_lstm(y, initial_state=[state_h, state_c])
decoder_dense = tf.keras.layers.Dense(tgt_vocab_size, activation='softmax')(decoder_outputs)
 
# Compile and train
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_dense)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([src_padded, decoder_input], decoder_target, epochs=300, verbose=0)
 
# Inference: simple translation (just index decoding here)
def translate(input_text):
    seq = src_tokenizer.texts_to_sequences([input_text])
    seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=src_padded.shape[1], padding='post')
    enc_out, h, c = model.layers[3](model.layers[2](seq))
    dec_input = np.array([[tgt_word_index['<start>']]])
    translated = []
 
    for _ in range(10):
        y = model.layers[6](model.layers[5](dec_input), initial_state=[h, c])
        token_probs = model.layers[7](y)
        token = np.argmax(token_probs[0, -1, :])
        if tgt_tokenizer.index_word[token] == '<end>':
            break
        translated.append(tgt_tokenizer.index_word[token])
        dec_input = np.array([[token]])
    return ' '.join(translated)
 
# Test translation
print("Translate 'thank you':", translate("thank you"))