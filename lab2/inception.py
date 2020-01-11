#!/usr/bin/python3

"""inception.py Contains an implementation of the Inception V1 model using the
                functional Keras API in TensorFlow 2.0.

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


import tensorflow as tf
from tensorflow.keras import Input, Model, layers, utils
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Dropout, \
        Flatten, MaxPool2D, concatenate


def auxiliary_network(inputs: tf.Tensor, name: str=None) -> tf.Tensor:
    """Define an Inception auxiliary network

    Args:
        inputs: Input tensor
        name: Name of the network

    Returns:
        x: Output tensor
    """
    # Average pooling layer
    x = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding="valid",
        name=f"{name}_ap_1")(inputs)

    # Convolutional layer
    x = Conv2D(filters=128, kernel_size=(1, 1), padding="same",
            activation="relu", name=f"{name}_conv_1x1")(x)

    # Flatten the input
    x = Flatten()(x)

    # Fully connected layer
    x = Dense(units=1024, activation="relu",  name=f"{name}_fc")(x)

    # Dropout layer
    x = Dropout(rate=0.7, name=f"{name}_dropout")(x)

    # Linear layer with softmax activation
    x = Dense(units=1000, activation="softmax", name=f"{name}_lin_softmax")(x)

    return x


def inception_module(x: tf.Tensor, f_1x1: int, f_3x3_red: int, f_3x3: int,
        f_5x5_red: int, f_5x5: int, f_pool_proj: int,
        name: str=None) -> tf.Tensor:
    """Define an Inception module

    Args:
        x:           Input tensor
        f_1x1:       Nr. of filter maps for the 1x1 convolution
        f_3x3_red:   Nr. of filter maps for the 3x3 reduced convolution
        f_3x3:       Nr. of filter maps for the 3x3 convolution
        f_5x5_red:   Nr. of filter maps for the 5x5 reduced convolution
        f_5x5:       Nr. of filter maps for the 5x5 convolution
        f_pool_proj: Nr. of filter maps for the pooling projection

    Returns:
        x: Output tensor
    """
    # 1x1 convolution
    conv_1x1 = Conv2D(filters=f_1x1, kernel_size=(1, 1), padding="same",
            activation="relu", name=f"{name}_conv_1x1")(x)

    # 3x3 convolution (with reduction)
    conv_3x3_red = Conv2D(filters=f_3x3_red, kernel_size=(1, 1), padding="same",
            activation="relu", name=f"{name}_conv_3x3_red")(x)
    conv_3x3 = Conv2D(filters=f_3x3, kernel_size=(3, 3), padding="same",
            activation="relu", name=f"{name}_conv_3x3")(conv_3x3_red)

    # 5x5 convolution (with reduction)
    conv_5x5_red = Conv2D(filters=f_5x5_red, kernel_size=(1, 1), padding="same",
            activation="relu", name=f"{name}_conv_5x5_red")(x)
    conv_5x5 = Conv2D(filters=f_5x5, kernel_size=(5, 5), padding="same",
            activation="relu", name=f"{name}_conv_5x5")(conv_5x5_red)

    # Max pooling (with reduction)
    pool_proj = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same",
            name=f"{name}_mp")(x)
    pool_proj_red = Conv2D(filters=f_pool_proj, kernel_size=(1, 1),
            padding="same", activation="relu", name=f"{name}_mp_red")(pool_proj)

    # Ouput concatenation
    outputs = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj_red], axis=3,
            name=name)

    return outputs



def construct_inception_model(inputs):
    """Constructor of the Inception v1 model

    Args:
        inputs: Input tensor

    Returns:
        outputs: Output tensor
    """
    # 7x7.2 convolutional layer
    x = Conv2D(filters=64, kernel_size=(7, 7), padding="same", strides=(2, 2),
            activation="relu", name="inc_conv_1_7x7")(inputs)

    # 3x3/2 max pooling layer
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same",
            name="inc_mp_1")(x)

    # Local Response Normalization layer
    x = tf.nn.local_response_normalization(x, name="inc_lrn_1")

    # 3x3/1 convolutional layer with reduction
    x = Conv2D(filters=64, kernel_size=(1, 1), padding="same", strides=(1, 1),
            activation="relu", name="inc_conv_2_3x3_red")(x)
    x = Conv2D(filters=192, kernel_size=(3, 3), padding="same", strides=(1, 1),
            activation="relu", name="inc_conv_2_3x3")(x)

    # Local Response Normalization layer
    x = tf.nn.local_response_normalization(x, name="inc_lrn_2")

    # 3x3/2 max pooling layer
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same",
            name="inc_mp_2")(x)

    # Inception module 3a
    x = inception_module(x, 64, 96, 128, 16, 32, 32, "inc_mod_3a")

    # Inception module 3b
    x = inception_module(x, 128, 128, 192, 32, 96, 64, "inc_mod_3b")

    # 3x3/2 max pooling layer
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same",
            name="inc_mp_3")(x)

    # Inception module 4a
    x = inception_module(x, 192, 96, 208, 16, 48, 64, "inc_mod_4a")

    # First auxiliary network
    a1 = auxiliary_network(x, name="inc_aux_1")

    # Inception module 4b
    x = inception_module(x, 160, 112, 224, 24, 64, 64, "inc_mod_4b")

    # Inception module 4c
    x = inception_module(x, 128, 128, 256, 24, 64, 64, "inc_mod_4c")

    # Inception module 4d
    x = inception_module(x, 112, 144, 288, 32, 64, 64, "inc_mod_4d")

    # Second auxiliary network
    a2 = auxiliary_network(x, name="inc_aux_2")

    # Inception module 4e
    x = inception_module(x, 256, 160, 320, 32, 128, 128, "inc_mod_4e")

    # 3x3/2 max pooling layer
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same",
            name="inc_mp_4")(x)

    # Inception module 5a
    x = inception_module(x, 256, 160, 320, 32, 128, 128, "inc_mod_5a")

    # Inception module 5b
    x = inception_module(x, 384, 192, 384, 48, 128, 128, "inc_mod_5b")

    # 7x7/1 average pooling layer
    x = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding="valid",
        name="inc_ap_1")(x)

    # Flatten the input
    x = Flatten()(x)

    # 40% dropout layer
    x = Dropout(rate=0.4, name="inc_dropout_1")(x)

    # Linear layer (dense)
    x = Dense(units=1000, activation="softmax", name=f"inc_lin_softmax_1")(x)

    return x, a1, a2


def main():
    """Create an input, build the Inception v1 model, print the network summary
       and plot the network topology"""
    input_shape = (224, 224, 3)
    inputs = Input(shape=input_shape, name='input_layer')
    outputs = construct_inception_model(inputs)

    model = Model(inputs=inputs, outputs=outputs, name='inception_model')

    model.summary()

    utils.plot_model(model, 'model_plot.png')


if __name__ == "__main__":
    main()

