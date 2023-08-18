import onnx
from onnxruntime import InferenceSession
import numpy as np
from onnx import onnx_pb
from onnx_coreml import convert

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



model_quant = './output_dir/model.quant.onnx'

model_file = open(model_quant, 'rb')
model_proto = onnx_pb.ModelProto()
model_proto.ParseFromString(model_file.read())
coreml_model = convert(model_proto, image_input_names=['0'])
coreml_model.save('./output_dir/coreml.mlmodel')