import os
import typing
from collections import defaultdict

import pandas as pd

from posters.util.dataset import load_dataset


def genre_frequency(genres_column) -> typing.Dict[str, int]:
    genre_to_freq = defaultdict(int)
    for row in genres_column:
        genres = '|'.split(row)
        for genre in genres:
            genre_to_freq[genre] = genre_to_freq[genre] + 1
    return genre_to_freq

if __name__ == "__main__":
    dataset_dir = os.path.dirname(__file__)
    dataset_file = os.path.join(dataset_dir, 'metadata.csv')
    dataset = load_dataset(dataset_file)
    print(genre_frequency(dataset['Genre']))
