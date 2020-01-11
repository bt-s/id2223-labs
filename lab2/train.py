#!/usr/bin/python3

"""train.py Contains an implementation for training the image captioning
            model

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.data.ops.dataset_ops import PrefetchDataset

import os
import numpy as np
from typing import List, Tuple

from encoder import Encoder
from decoder import Decoder
from helpers import get_image_names, get_all_captions

from cap_processing import preprocess_captions
from img_processing import extract_img_features, \
        init_feat_extract_inception_v3


def load_img_features(img_name: str, cap: np.ndarray) -> Tuple[tf.Tensor,
        np.ndarray]:
    """Loads Numpy image features based on file name

    Args:
        img_name: Path to image file
        cap: Caption corresponding to image

    Returns:
        img: Tensor of image feature
        cap: Caption corresponding to image

    Note: We are passing cap, since this function is part of a lambda mapping
          function
    """
    img = np.load(img_name.decode("utf-8") + '.npy')

    return img, cap


def create_tf_dataset(img_names: List, seqs: np.ndarray, batch_size: int,
        buffer_size: int=8000):
    """Create a TensorFlow data set from a list of image names

    Args:
        img_names: List of image names
        seqs: Array of tokenized sequences corresponding to captions
        batch_size: Batch size
        buffer_size: Buffer size for data set shuffling

    Returns:
        dataset: TensorFlow data set partioned in batches of batch_size
    """
    # Create dataset form tensors
    dataset = tf.data.Dataset.from_tensor_slices((img_names, seqs))

    # Load the numpy files into the dataset
    dataset = dataset.map(lambda img, cap: tf.numpy_function(
        load_img_features, [img, cap], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle the data set and create batches of size batch_size
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def loss_function(real: tf.Tensor, pred: tf.Tensor) -> tf.Tensor:
    """Define the loss function

    Args:
        real: Real caption
        pred: Predicted caption

    Returns:
        loss: Loss
    """
    objective = SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = objective(real, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_mean(loss)

    return loss


@tf.function
def train_step(img, cap, encoder, decoder, optimizer, tokenizer):
    """Perform a single training step

    Args:
        img: Input image
        cap: Input (target) caption
        encoder: Encoder model
        decoder: Decoder model
        optimizer: Optimizer
        tokenizer: Tokenizer

    Returns:
        batch_loss: Batch loss
        total_loss: Total loss
    """
    batch_loss = 0

    # Define the decoder input -- tokenizer target caption
    dec_input = tf.expand_dims([tokenizer.word_index["<start>"]] *
            cap.shape[0], 1)

    with tf.GradientTape() as tape:
        # Encode the image
        img_feats = encoder(img, training=True)

        # At the start the LSTM doesn't have a state yet
        lstm_state = None

        for i in range(1, cap.shape[1]):
            preds, lstm_state = decoder(dec_input, img_feats, lstm_state,
                    training=True, time_step=i-1)

            batch_loss += loss_function(cap[:, i], preds)

            dec_input = tf.expand_dims(cap[:, i], axis=1)

    total_loss = (batch_loss / int(cap.shape[1]))

    # Identify all trainable variables
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables

    # Compute the gradients of the trainable variables
    gradients = tape.gradient(batch_loss, trainable_vars)

    # Apply the gradients to the optimizer
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    return batch_loss, total_loss


def train(dataset: PrefetchDataset, epochs: int, encoder: Encoder,
        decoder: Decoder, optimizer: Adam, tokenizer: Tokenizer,
        num_steps: float):
    """Train the image captioning model

    Args:
        dataset: Training data set
        epochs: Number of training epochs
        encoder: Encoder model
        decoder: Decoder model
        optimizer: Optimizer
        tokenizer: Tokenizer
        num_steps: Number of training steps
    """
    ckpt_path = "./checkpoints"

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    # Create a checkpoint
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder,
            optimizer=optimizer)

    # Pass the checkpoint to the checkpoint manager
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        print("Loading checkpoint...")
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    print(f"Start epoch: {start_epoch}")
    for epoch in range(start_epoch, epochs):
        total_loss = 0

        for (batch, (img, cap)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img, cap, encoder, decoder,
                    optimizer, tokenizer)
            total_loss += t_loss

        print("Saving a checkpoint...")
        ckpt_manager.save(checkpoint_number=epoch)

        print("Epoch {} - Loss {:.6f}".format(epoch + 1, total_loss/num_steps))


if __name__ == "__main__":
    batch_size = 64

    # Get the train image names
    train_image_names = get_image_names(
            "./flickr8k/text/Flickr_8k.trainImages.txt")

    # Initialize the feature extraction model (InceptionV3)
    feat_extract_model = init_feat_extract_inception_v3()

    # Extract features from the images
    extract_img_features(train_image_names, feat_extractor=feat_extract_model)

    # Get captions in the whole data set
    all_captions = get_all_captions("./flickr8k/text/Flickr8k.token.txt")

    # Fit a tokenizer on the training data with vocabulary of length k and
    # find a tokenized sequences for each caption in the training data set
    tokenizer, train_seqs = preprocess_captions(train_image_names,
            all_captions, k=5000)

    # Duplicate each image name 5 times
    train_image_names = [name for name in train_image_names
            for _ in range(5)]

    # Create TensorFlow dataset
    dataset = create_tf_dataset(train_image_names, train_seqs,
            batch_size=batch_size)

    # Initialize the encoder and decoder
    encoder = Encoder(embedding_dim=256)
    decoder = Decoder(embedding_dim=256, units=512,
            vocab_size=len(tokenizer.word_index)+1)

    # Instantiate the optimizer
    optimizer = Adam()

    # Calculate number of training steps
    num_steps = len(train_image_names) / batch_size

    # Train the model
    train(dataset, 50, encoder, decoder, optimizer, tokenizer, num_steps)


