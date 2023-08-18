import onnx
from onnxruntime import InferenceSession
import numpy as np

import argparse
import datetime
import json
import numpy as np
import os
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.io import read_image

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import matplotlib.pyplot as plt

from PIL import Image

def show_img(img):
    img = np.asarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def show_img2(img1, img2, alpha=0.8):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    plt.figure(figsize=(10, 10))
    plt.imshow(img1)
    plt.imshow(img2, alpha=alpha)
    plt.axis('off')
    plt.show()

model_quant = './output_dir/model.quant.onnx'

onnx_model = onnx.load(model_quant)
sess = InferenceSession(onnx_model.SerializeToString(), providers=['CPUExecutionProvider'])
print(sess.run(['output'], {'input': np.zeros((1, 3, 224, 224), dtype=np.float32)}))

img_directory = './slan/'

files = Path(img_directory).glob("*")
img_list = []
for file in files:
    # print(file)
    # img_list.append(read_image(str(file)))
    with open(str(file), 'rb') as f:
        with Image.open(f) as img:
            img_list.append(img.convert('RGB'))

transforms = torch.nn.Sequential(
        T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    )

toTensor = T.Compose(
        [T.ToTensor()]
    )

for image in enumerate(img_list):
        im = toTensor(image[1])
        im = transforms(im)
        y = sess.run(['output'], {'input': im.unsqueeze(0).numpy()})
        print(y)
        show_img(im.permute(1,2,0))

