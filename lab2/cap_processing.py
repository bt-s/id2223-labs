#!/usr/bin/python3

"""cap_processing.py Contains functions for preprocessing and feature extraction
                     of captions for the image captioning task

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
import pickle
import numpy as np
from typing import List, Tuple


def preprocess_captions(img_names_training: List, all_captions:
        dict, k: int) -> Tuple[Tokenizer, np.ndarray]:
    """Preprocess all captions

    Args:
        img_names: Image names from the training data set
        all_captions: Captions of the complete data set
        k: Maximum number of words in our vocabulary

    Returns:
        tokenizer: The tokenizer fit on the captions of the training data set
        seqs: All tokenized captions in the training data set
    """
    # Split image names to remove everything from path except file name
    img_names = [img.split("/")[-1] for img in img_names_training]

    # Filter all_captions for the training set
    captions = {k: all_captions[k] for k in img_names if k in
            all_captions.keys()}

    # Extract all individual captions for the training set
    captions = [cap for img in img_names for cap in captions[img]]

    if os.path.isfile("./tokenizer.pkl"):
        # Load the tokenizer
        print("Loading tokenizer...")
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        # Define the tokenizer
        tokenizer = Tokenizer(num_words=k, oov_token="<unk>",
                filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

        # Specificy word indexing
        tokenizer.word_index["<pad>"] = 0
        tokenizer.index_word[0] = "<pad>"

        # Fit the tokenizer on all captions in the training set
        print("Fitting tokenizer...")
        tokenizer.fit_on_texts(captions)

        # Save the tokenizer
        with open('tokenizer.pkl', 'wb') as f:
            print("Saving tokenizer...")
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Create tokenized sequences for all captions
    seqs = tokenizer.texts_to_sequences(captions)

    # Pad the sequences s.t. they become of the same size
    seqs = pad_sequences(seqs, padding="post")

    return tokenizer, seqs

