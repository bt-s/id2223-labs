#!/usr/bin/python3

"""train.py Contains an implementation for training the image captioning
            model

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.engine import training

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle

from tensorflow.keras.optimizers import Adam

from img_processing import preprocess_image
from encoder import Encoder
from decoder import Decoder
from helpers import get_image_names
from img_processing import init_feat_extract_inception_v3


def evaluate(img: str, feat_extract_model: training.Model,
        tokenizer: Tokenizer, encoder: Encoder, decoder: Decoder,
        max_length: int=20, eval_type: str="beam", beam_size=2):
    """Predict a caption for a specific image using naive sequence sampling

    Args:
        img: Name of input image for evaluation
        feat_extract_model: Image feature extraction model
        tokenizer: Tokenizer
        encoder: Encoder model
        decoder: Decoder model
        max_length: Maximum length of the resulting caption
        eval_type: Either 'beam' or 'greedy' to specify the type of evaluation
        beam_size: Size of the beam if eval_type=='beam"

    Return:
        caption: Final caption
    """
    img = tf.expand_dims(preprocess_image(img)[0], 0)
    img = feat_extract_model(img)
    img = tf.reshape(img, (img.shape[0], -1,
        img.shape[3]))

    # Encode the image to extract features
    img_feats = encoder(img)

    if eval_type == "greedy":
        print("Starting greedy search...")
        # Define the initial decoder input
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

        # The LSTM does not have a state at first
        lstm_state = None

        caption = []
        for i in range(max_length):
            # Predict next word id using :STM
            predictions, lstm_state = decoder(dec_input, img_feats, lstm_state,
                    time_step=i)

            # Predcict the id
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            caption.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return " ".join(caption[:-1])

            dec_input = tf.expand_dims([predicted_id], 0)

        caption = " ".join(caption[:-1])

    elif eval_type == "beam":
        print(f"Starting beam search with beam size {beam_size}...")
        # Initialize the result (ids, decoder input, probability, lstm state)
        cap = [tokenizer.word_index["<start>"]]
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        lstm_state = None
        prob = 0.0

        result = [[cap, dec_input, prob, lstm_state]]

        for i in range(max_length):
            # Track a temporary result
            temp_res = []

            # Iterate over each word in the current result
            for r in result:
                dec_input, lstm_state = r[1], r[3]
                predictions, lstm_state = decoder(dec_input, img_feats,
                        lstm_state, time_step=i)

                # Obtain the indices of the top-beamsize predictions
                predicted_ids = predictions[0].numpy().argsort() \
                        [-beam_size:][::-1]

                # Loop over all predicted ids
                for i in predicted_ids:
                    # Obtain the current partial caption and probability
                    cap, prob = r[0][:], r[2]

                    # Add the index to the current partial capption
                    cap.append(i)
                    # Add the probability of i to prob
                    prob += predictions[0].numpy()[i]

                    # Add the updated partial caption and probability to the
                    # temporary result
                    dec_input = tf.expand_dims([i], 0)
                    temp_res.append([cap, dec_input, prob, lstm_state])

            # Update result by the temporary result
            result = temp_res

            # Sort the result based on the probabilities and only keep the
            # top-beamsize partial captions
            result = sorted(result, reverse=True, key=lambda l: l[2])
            result = result[:beam_size]

        # Obtain the final best result
        caption = result[0][0]

        caption = [tokenizer.index_word[i] for i in caption][1:]
        caption = " ".join(caption)

        try:
            caption = caption.split("<end>", 1)[:-1][0][:-1]
        except:
            pass

    else:
        raise ValueError("Input argument <eval_type> has to be either " +
                "'beam' or 'greedy'")

    return caption


if __name__ == "__main__":
    # Get the test image names
    test_image_names = get_image_names(
            "./flickr8k/text/Flickr_8k.testImages.txt")

    # Initialize the feature extraction model (InceptionV3)
    feat_extract_model = init_feat_extract_inception_v3()

    print("Loading tokenizer...")
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("Loaded tokenizer!")

    # Initialize the encoder and decoder
    encoder = Encoder(embedding_dim=256)
    decoder = Decoder(embedding_dim=256, units=512,
            vocab_size=len(tokenizer.word_index)+1)

    # Instantiate the optimizer
    optimizer = Adam()

    # Load the latest checkpoint
    ckpt_path = "./checkpoints"

    # Create a checkpoint
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder,
            optimizer=optimizer)

    # Pass the checkpoint to the checkpoint manager
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        print("Loading checkpoint...")
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!")

    # Evaluate the model
    caption = evaluate(test_image_names[0], feat_extract_model, tokenizer,
            encoder, decoder)

    # Print the predicted caption
    print(f"Caption: {caption}")

    img=mpimg.imread(test_image_names[0])
    imgplot = plt.imshow(img)
    plt.show()

