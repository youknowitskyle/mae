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
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2" # version check

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, build_transform
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit

from engine_finetune import train_one_epoch, evaluate

from dataset import *

import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # * Visualization params
    parser.add_argument('--checkpoint', required=True, default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument('--num_vis', default=1, type=int, help='number of examples to visualize')

    # Dataset parameters
    parser.add_argument('--image_dir', default='test_imgs', type=str, help='path of directory with test images')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    return parser

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

def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 2:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward

AU_names = ['Left Eye Close', 
            'Right Eye Close', 
            'Left Lid Raise', 
            'Right Lid Raise', 
            'Left Brow Lower', 
            'Right Brow Lower', 
            'Left Brow Raise', 
            'Right Brow Raise', 
            'Jaw Slide Left', 
            'Jaw Slide Right', 
            'Left Lip Corner Pull', 
            'Right Lip Corner Pull', 
            'Left Lip Corner Stretch', 
            'Right Lip Corner Stretch']

def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

     # fix the seed for reproducibility
    # seed = args.seed + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    cudnn.benchmark = True

    files = Path(args.image_dir).glob("*")
    img_list = []
    for file in files:
        # print(file)
        # img_list.append(read_image(str(file)))
        with open(str(file), 'rb') as f:
            with Image.open(f) as img:
                img_list.append(img.convert('RGB'))

    print(img_list)

    transforms = T.Compose(
        [T.ToTensor(), T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)]
    )
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % args.checkpoint)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params (M): %.2f' % (n_parameters / 1.e6))


    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, 0,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=0
    )
    optimizer = torch.optim.AdamW(param_groups, lr=0)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    attention_layer = -1

    model.blocks[attention_layer].attn.forward = my_forward_wrapper(model.blocks[attention_layer].attn)
    model.eval()

    print(model)

    for i, image in enumerate(img_list):
        im = transforms(image)
        y = model(im.unsqueeze(0).to(device)).detach().cpu().numpy()
        print('-------------------------------------')
        print(f'Inference results for image {i + 1}')
        print('-------------------------------------')
        for j in range(len(y[0])):
            print(f'{AU_names[j]}: {y[0][j]}')

        attn_map = model.blocks[attention_layer].attn.attn_map.mean(dim=1).squeeze(0).detach()
        attn = model.blocks[attention_layer].attn.cls_attn_map.mean(dim=1).detach().cpu()
        pad = torch.zeros((1, 1), dtype=torch.float32).cpu()
        cls_weight = torch.cat((pad, attn), 1).view(14, 14)
        
        img_resized = im.to(device).permute(1, 2, 0) * 0.5 + 0.5
        cls_resized = F.interpolate(cls_weight.view(1, 1, 14, 14), (224, 224), mode='bilinear').view(224, 224, 1)
        show_img2(img_resized.cpu(), cls_resized.cpu(), alpha=0.8)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)