#!/usr/bin/python3

"""encoder.py Contains an implementation of the encoder for the image caption
              task

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout


class Encoder(Model):
    """Subclass of tf.keras.Model: Encoder"""
    def __init__(self, embedding_dim: int):
        """Class constructor

        Args:
            embedding_dim: Dimensionality of the image embedding
        """
        super(Encoder, self).__init__()

        self.dense = Dense(embedding_dim, activation="relu")
        self.dropout = Dropout(0.5)
        self.b_norm = BatchNormalization()

    def call(self, x: tf.Tensor, training: bool=True):
        """Call function of the encoder model

        Args:
            x: Input tensor

        Returns:
            x: Output tensor
        """
        if training:
            x = self.dropout(x)

        x = self.dense(x)
        x = self.b_norm(x)

        # After the dense layer, the shape of x is: # (batch_size, 64,
        # embedding_dim). It should be of the same dimension as the text
        # embedding, which is (1, 1, embedding_dim)
        x = tf.expand_dims(tf.reduce_sum(x, axis=1), 1)

        return x

