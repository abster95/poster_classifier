import os
import requests
from PIL import Image
import logging
import typing
import pandas as pd
from threading import Thread

def get_image(link):
    try:
        response = requests.get(link)
    except Exception:
        return None
    if response.status_code == 200:
        return response.content
    else:
        return None

def get_image_path(img_id: str, images_dir: str):
    # TODO: Move to utils
    return os.path.join(images_dir, f'{img_id}.jpg')

def download_image(link: str, img_id: str, images_dir: str):
    image_data = get_image(link)
    if not image_data:
        logging.error(f'Download failed for image:{img_id} with link {link}')
    else:
        path = get_image_path(img_id, images_dir)
        with open(path, 'wb') as fp:
            fp.write(image_data)

def load_dataset(dataset_file: str) -> pd.DataFrame:
    return pd.read_csv(dataset_file)

def get_already_downloaded(images_dir):
    images = os.listdir(images_dir)
    image_ids = [int(os.path.splitext(image)[0]) for image in images]
    return set(image_ids)

class Downloader(Thread):
    def __init__(self, link: str, img_id: str, images_dir: str):
        super(Downloader, self).__init__()
        self.link = link
        self.img_id = img_id
        self.images_dir = images_dir

    def run(self):
        download_image(self.link, self.img_id, self.images_dir)

if __name__ == "__main__":
    dataset_dir = os.path.dirname(__file__)
    logname = os.path.join(dataset_dir,'dataset_download.log')
    logging.basicConfig(filename=logname,
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    
    images_dir = os.path.join(dataset_dir, 'images')
    dataset_file = os.path.join(dataset_dir, 'metadata.csv')

    dataset = load_dataset(dataset_file)
    logging.info(f'Number of all images in dataset: {dataset.shape[0]}')
    already_downloaded = get_already_downloaded(images_dir)
    filtered_ds = dataset.imdbId.isin(already_downloaded)
    dataset = dataset[~filtered_ds]
    img_ids = dataset['imdbId'].values.tolist()
    urls = dataset['Poster'].values.tolist()
    logging.info(f'Number of images to download:{dataset.shape[0]}')
    started = []
    for index, row in dataset.iterrows():
        img_id = row['imdbId']
        if img_id in already_downloaded:
            logging.info(f'Skipping {img_id} since it already exists')
            continue
        link = row['Poster']
        logging.info(f'Trying to download {img_id} on index {index}...')
        downloader = Downloader(link, img_id, images_dir)
        downloader.start()
        started.append(downloader)
    for thread in started:
        thread.join()
    logging.info('All done!')