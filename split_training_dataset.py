"""
This script is designed to preprocess a dataset for machine learning tasks. It loads data from a JSONL file (where each line is a JSON object), shuffles the data to ensure randomness, splits the data into training and validation sets based on a specified ratio, and finally saves these sets into separate JSONL files. This preprocessing step is crucial for preparing data for model training and evaluation.
"""

import json
import random


def load_and_shuffle_data(filepath):
    """
    Loads data from a JSONL file and shuffles it.

    Parameters:
    - filepath (str): The path to the JSONL file containing the data.

    Returns:
    - list: A list of shuffled data loaded from the file.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
    random.shuffle(data)
    return data


def split_data(data, train_ratio=0.8):
    """
    Splits the data into training and validation sets based on a specified ratio.

    Parameters:
    - data (list): The list of data to be split.
    - train_ratio (float, optional): The proportion of the data to be used for training. Default is 0.8.

    Returns:
    - tuple: A tuple containing two lists, (train_data, valid_data), where `train_data` is the training set and `valid_data` is the validation set.
    """
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    valid_data = data[train_size:]
    return train_data, valid_data


def save_data(
    train_data,
    valid_data,
    train_filepath="train_data/train.jsonl",
    valid_filepath="train_data/valid.jsonl",
):
    """
    Saves the training and validation data to separate JSONL files.

    Parameters:
    - train_data (list): The training data to save.
    - valid_data (list): The validation data to save.
    - train_filepath (str, optional): The filepath for the training data file. Default is 'train.jsonl'.
    - valid_filepath (str, optional): The filepath for the validation data file. Default is 'valid.jsonl'.
    """
    with open(train_filepath, "w", encoding="utf-8") as file:
        for item in train_data:
            file.write(json.dumps(item) + "\n")

    with open(valid_filepath, "w", encoding="utf-8") as file:
        for item in valid_data:
            file.write(json.dumps(item) + "\n")


def main():
    """
    Main function to execute the data preprocessing steps.
    """
    filepath = "train_data/training_data.jsonl"
    data = load_and_shuffle_data(filepath)
    train_data, valid_data = split_data(data)
    save_data(train_data, valid_data)


if __name__ == "__main__":
    main()
