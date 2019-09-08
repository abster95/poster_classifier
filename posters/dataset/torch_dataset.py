import os
import random

import pandas as pd
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset

from posters.dataset.summary import acceptable_genres_list
from posters.util.dataset import load_dataset, get_image_path
from posters.util.io import write_json_to_file, read_json_from_file

IMG_SHAPE = (268,182,3)
class MoviePosters(Dataset):
    def __init__(self, dataset_type: str, cutoff: int = 1000):
        dataset_dir = os.path.dirname(__file__)
        dataset_file = os.path.join(dataset_dir, 'metadata.csv')
        self.image_root = os.path.join(dataset_dir, 'images')
        self.dataset_type = dataset_type
        self.dataset = load_dataset(dataset_file)
        self.ids = read_json_from_file(os.path.join(dataset_dir, f'{dataset_type}_ids.json'))
        self.genres = acceptable_genres_list(self.dataset['Genre'], cutoff)
        self.genre_to_id = {genre: i for i, genre in enumerate(self.genres)}

    def __len__(self):
        return len(self.ids)

    def _one_hot_genre(self, labels):
        # Acceptable genres + unknown
        one_hot = torch.zeros(len(self.genres)) # pylint: disable=no-member
        if not isinstance(labels, str):
            one_hot[self.genre_to_id['Unknown']] = 1
            return one_hot
        labels = labels.split('|')
        for label in labels:
            if label in self.genre_to_id:
                one_hot[self.genre_to_id[label]] = 1
            else:
                one_hot[self.genre_to_id['Unknown']] = 1
        return one_hot


    def __getitem__(self, index):
        # pylint: disable=no-member
        imdb_id = self.ids[index]
        img_path = get_image_path(imdb_id, self.image_root)
        if not os.path.exists(img_path):
            return torch.zeros(IMG_SHAPE), torch.zeros(len(acceptable_genres_list))
        img = imageio.imread(img_path)
        if len(img.shape) < 3:
            img = np.stack((img,img,img), axis=-1)
        labels = np.squeeze(self.dataset.loc[self.dataset['imdbId'] == imdb_id, ['Genre']])
        labels = self._one_hot_genre(labels)
        return img, labels


def split_dataset(train_percent: float = 0.8):
    dataset_dir = os.path.dirname(__file__)
    images_dir = os.path.join(dataset_dir, 'images')
    ids = os.listdir(images_dir)
    ids = [os.path.splitext(img)[0] for img in ids]
    random.shuffle(ids)

    split_index = int(len(ids) * train_percent)
    train_split = ids[:split_index]
    val_split = ids[split_index:]

    write_json_to_file(train_split, os.path.join(dataset_dir, 'train_ids.json'))
    write_json_to_file(val_split, os.path.join(dataset_dir, 'val_ids.json'))

if __name__ == "__main__":
    split_dataset()