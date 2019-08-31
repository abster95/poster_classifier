import os
import typing
from collections import defaultdict

import pandas as pd

from posters.util.dataset import load_dataset


def genre_frequency(genres_column) -> typing.Dict[str, int]:
    genre_to_freq = defaultdict(int)
    for row in genres_column:
        if not isinstance(row, str):
            genre_to_freq['Unknown'] = genre_to_freq['Unknown'] + 1
        else:
            genres = row.split('|')
            for genre in genres:
                genre_to_freq[genre] = genre_to_freq[genre] + 1
    return genre_to_freq

if __name__ == "__main__":
    dataset_dir = os.path.dirname(__file__)
    dataset_file = os.path.join(dataset_dir, 'metadata.csv')
    dataset = load_dataset(dataset_file)
    freq = genre_frequency(dataset['Genre'])
    df = pd.DataFrame.from_dict(freq, orient='index')
    df.to_csv('summary.csv')
