#!/usr/bin/python3

"""decoder.py Contains an implementation of the decoder for the image caption
              task

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM

from typing import List


class Decoder(Model):
    """Subclass of tf.keras.Model: Decoder"""
    def __init__(self, embedding_dim: int, units: int, vocab_size: int):
        """Class constructor

        Args:
            embedding_dim: Dimensionality of the image embedding
            units:
            vocab_size:
        """
        super(Decoder, self).__init__()

        self.units = units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(units=units, return_sequences=True, return_state=True,
                name="lstm")
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x, img_feats: tf.Tensor, lstm_state: List[tf.Tensor]=None,
            training: bool=True, time_step: int=0):
        """Call function of the encoder model

        Args:
            x: Input tensor
            img_feats: Feature embedding of the input image
            lstm_state: Initital state of the LSTM based on previous time step
            training: Whether the current phase is training
            it: Specify the current timestep

        Returns:
            x: Output tensor
        """
        x = self.embedding(x)

        # Only input the image features to the LSTM during the first pass
        if time_step == 0:
            _, h_state, c_state = self.lstm(img_feats)
            output, h_state, c_state = self.lstm(x,
                    initial_state=[h_state, c_state])
        else:
            output, h_state, c_state = self.lstm(x, initial_state=lstm_state)

        # Store the LSTM's hidden state and cell state in a list
        lstm_state = [h_state, c_state]

        # Because of return_sequences = True outut has three dimensions
        # (batch_size, time_step, output_units), which is needed since we are
        # stacking LSTMs. However, we want to return a 2D array of shape
        # (batch_Size, output_units)
        x = tf.reshape(output, (-1, output.shape[2]))

        # Pass the reshaped
        x = self.dense(x)

        return x, lstm_state

