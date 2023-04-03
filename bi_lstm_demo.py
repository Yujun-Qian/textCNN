import tensorflow as tf
import numpy as np
import random
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MAX_SEQUENCE_LENGTH = 4
EMBEDDING_CHAR_DIM = 8
LSTM_DIM = 4

def switch_layer(inputs):

    inp, emb = inputs
    zeros = tf.zeros_like(inp)
    ones = tf.ones_like(inp)

    inp = tf.keras.backend.switch(inp > 0, ones, zeros)
    inp = tf.expand_dims(inp, -1)

    return inp * emb

#main_char_input = tf.zeros([1, 4, 16], tf.float32)
main_char_input = np.array([
  [
    [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ]
], dtype=float)

print(main_char_input)
embedding_char_layer = tf.keras.layers.Embedding(5,
                            EMBEDDING_CHAR_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
embed_char = embedding_char_layer(main_char_input)
print(embed_char)

embed_char = tf.keras.layers.Lambda(switch_layer)([main_char_input, embed_char])
print(embed_char)

char_lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
          LSTM_DIM,
          input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_CHAR_DIM),
          return_sequences=True,
          return_state=True),
        backward_layer = tf.keras.layers.LSTM(
          LSTM_DIM,
          input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_CHAR_DIM),
          return_sequences=True,
          go_backwards=True,
          return_state=True),
name='Bi-LSTM_Char')

print(embed_char.shape)
print(embed_char[0].shape)


masking_layer = tf.keras.layers.Masking(mask_value=0., input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_CHAR_DIM))
embed_char = masking_layer(embed_char[0])
print(embed_char._keras_mask)
char_lstm_output, state1, cell_1, state2, cell_2 = char_lstm_layer(embed_char)

print(char_lstm_output.shape)
print(char_lstm_output)
print(state1)
print(state2)
