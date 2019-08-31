import os
import pandas as pd

def get_image_path(img_id: str, images_dir: str):
    return os.path.join(images_dir, f'{img_id}.jpg')

def load_dataset(dataset_file: str) -> pd.DataFrame:
    return pd.read_csv(dataset_file)

def get_already_downloaded(images_dir):
    images = os.listdir(images_dir)
    image_ids = [int(os.path.splitext(image)[0]) for image in images]
    return set(image_ids)