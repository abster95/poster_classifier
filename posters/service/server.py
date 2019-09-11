from posters.model.classifier import Classifier
from flask import Flask, request
import os
import posters
from torchvision.models import resnet50
from posters.dataset.torch_dataset import MoviePosters, pad_to_fixed_shape, resnet_preprocess
import torch
import imageio
import numpy as np

app = Flask(__name__)
basedir = posters.__path__[0]
ckpt = os.path.join(basedir, 'ckpt', 'best.ckpt')
val_data = MoviePosters('val')

model = Classifier(backbone=resnet50, imagenet_weights=False, num_classes=len(val_data.genres))
model.load_state_dict(torch.load(ckpt))
model.cpu()
model.eval()

@app.route('/')
def hello_world():
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
    app.run(host='0.0.0.0', port=8080, debug=True)