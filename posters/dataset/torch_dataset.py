import os
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from posters.util.dataset import load_dataset
from posters.util.io import write_json_to_file

class MoviePosters(Dataset):
    def __init__(self):
        pass

def split_dataset(train_percent: float = 0.8):
    dataset_dir = os.path.dirname(__file__)
    dataset_file = os.path.join(dataset_dir, 'metadata.csv')
    dataset = load_dataset(dataset_file)
    ids = dataset['imdbId'].values.tolist()
    random.shuffle(ids)

    split_index = int(len(ids) * train_percent)
    train_split = ids[:split_index]
    val_split = ids[split_index:]

    write_json_to_file(train_split, os.path.join(dataset_dir, 'train_ids.json'))
    write_json_to_file(val_split, os.path.join(dataset_dir, 'val_ids.json'))

if __name__ == "__main__":
    split_dataset()