#!/usr/bin/python3

"""img_processing.py Contains the image preprocessing and feature extraction
                     functions for the image captioning task

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, \
        preprocess_input
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.keras.engine import training

import os
import numpy as np
from tqdm import tqdm
from typing import List, Tuple


def preprocess_image(img_path: str) -> Tuple[EagerTensor, str]:
    """Load and preprocess an image from path and preprocess

    Args:
        img_path: Image path

    Returns:
        img: The preprocessed image
    """
    # Read image from path as TensorFlow tensor
    img = tf.io.read_file(img_path)

    # Tell TensorFlow that the image is RGB
    img = tf.image.decode_jpeg(img, channels=3)

    # The default input shape of InceptionV3 is (299, 299, 3)
    img = tf.image.resize(img, (299, 299))

    # Default InceptionV3 image processing
    img = preprocess_input(img)

    return img, img_path


def extract_img_features(img_names: List, feat_extractor):
    """Extract features from list of images and store them in .npy matrices

    Args:
        img_names: List of image names
    """
    print("Starting image preprocessing...")
    # Check whether image needs preprocessing and feature extraction
    unprocessed_imgs = []
    for img in img_names:
        # The image might have been processed already
        if not os.path.isfile(img + ".npy"):
            unprocessed_imgs.append(img)

    # Only start preprocessing and feature extraction process if
    # unprocessed_imgs is not empty
    if unprocessed_imgs:
        # Create a data set from the images
        imgs = tf.data.Dataset.from_tensor_slices(unprocessed_imgs)

        # Preprocess each image in batches of 8
        img_batches = imgs.map(preprocess_image, num_parallel_calls=
                tf.data.experimental.AUTOTUNE).batch(8)

        for img, path in tqdm(img_batches):
            # Extract features for each batch of images
            batch_feat = feat_extractor(img)
            batch_feat = tf.reshape(batch_feat, (batch_feat.shape[0], -1,
            batch_feat.shape[3]))

            # Store the feature matrices in ./flickr8k/imgs/
            for bf, p in zip(batch_feat, path):
                feat_path = p.numpy().decode("utf-8")
                np.save(feat_path, bf.numpy())
    print(("Finished image preprocessing! "
        f"Processed {len(unprocessed_imgs)} images."))


def init_feat_extract_inception_v3() -> training.Model:
    """Initialize InceptionV3 as feature extraction model

    Returns:
        feat_extract_model: The feature extraction model based on weights
                            learnt on ImageNet
    """
    # Load the pretrained InceptionV3 network with weights from trianing on
    # ImageNet
    incv3 = InceptionV3(include_top=False, weights='imagenet')

    # Create a default model input
    model_input = incv3.input

    # We want the penultimate layer to be the last layer of our feature
    # extraction CNN. Note that the size of this layer is 8*8*2048.
    model_output = incv3.layers[-1].output

    # Define the feature extraction model
    feat_extract_model = Model(model_input, model_output)

    return feat_extract_model

