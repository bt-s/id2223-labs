#!/usr/bin/python3

"""helpers.py Contains helper functions for the image captioning task

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


from typing import List
import re


def get_image_names(f_name: str) -> List:
    """Gets a list of image names

    Args:
        f_name: File name/Path

    Returns:
        img_names: List of image names
    """
    with open(f_name, 'r') as f:
        img_names = f.read().splitlines()

    img_names = ["./flickr8k/imgs/" + name for name in img_names]

    # Sort the images and make sure that the list is unique
    img_names = sorted(set(img_names))

    return img_names


def get_all_captions(f_name: str) -> dict:
    """ Store all captions in a dictionary

    Args:
        f_name: Path of file containing all captions

    Returns:
        all_captions: Dictionary where keys are image names and values are
                      captions
    """
    with open(f_name, 'r') as f:
        caption_file = f.read().splitlines()

    # Store captions in a dictionary
    all_captions = {}

    for line in caption_file:
        name = re.search("^.*?jpg", line).group(0)
        caption = "<start> " +  re.search("\t(.*)", line).group(0)[1:] + " <end>"

        if name in all_captions.keys():
            all_captions[name].append(caption)
        else:
            all_captions[name] = [caption]

    return all_captions

