from posters.model.classifier import Classifier
from flask import Flask, request, render_template
import os
import posters
from torchvision.models import resnet50
from posters.dataset.torch_dataset import MoviePosters, pad_to_fixed_shape, resnet_preprocess
import torch
import imageio
import numpy as np
import cv2
import base64

IMG_H = 268
IMG_W = 182

app = Flask(__name__)
basedir = posters.__path__[0]
ckpt = os.path.join(basedir, 'ckpt', 'best.ckpt')
val_data = MoviePosters('val')

model = Classifier(backbone=resnet50, imagenet_weights=False, num_classes=len(val_data.genres))
model.cpu()
model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    poster = request.files['poster']
    if not poster:
        return render_template('error.html', message="Seems like you didn't upload a file. Try again.")
    try:
        image = imageio.imread(poster)
    except Exception:
        return render_template('error.html', message="We could not read that file as image. Try again.")
    image = cv2.resize(image, (IMG_W, IMG_H))
    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = np.stack((image,image,image), axis=-1)
    image = pad_to_fixed_shape(image)
    image = torch.from_numpy(resnet_preprocess(image))
    preds = model.forward(image.unsqueeze_(0))
    preds = torch.sigmoid(preds)
    preds = list(preds.detach().numpy()[0])
    preds = zip(val_data.genres, preds)
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    poster.seek(0)
    poster_encoded = base64.b64encode(poster.read()).decode('utf8')
    return render_template('prediction.html', preds=preds, poster=f'data:image/jpeg;base64,{poster_encoded}')

def genre_by_pid():
    pid = request.args.get('pid')
    image_pth = os.path.join(basedir,'dataset','images', f'{pid}.jpg')
    image = imageio.imread(image_pth)
    image = pad_to_fixed_shape(image)
    image = torch.from_numpy(resnet_preprocess(image))
    preds = model.forward(image.unsqueeze_(0))
    preds = torch.sigmoid(preds)
    preds = list(preds.detach().numpy()[0])
    preds = zip(val_data.genres[:-1], preds[:-1])
    preds = sorted(preds, key=lambda x: x[1], reverse=True)

    return f'{pid} : {preds}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)