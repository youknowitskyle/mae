# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        t = []
        # t.append(transforms.RandomRotation(30))
        t.append(transforms.RandomAffine(30, (0.2, 0.2), (0.5, 1.5)))
        t.append(transforms.RandomResizedCrop(224, (0.5, 2.5)))
        t.append(transforms.RandomPerspective(0.4))
        # t.append(transforms.ElasticTransform(25.0))
        t.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        t.append(transforms.GaussianBlur(9, (0.1, 2.0)))
        t.append(transforms.RandomPosterize(6))
        t.append(transforms.RandomAdjustSharpness(1.25, 0.25))
        t.append(transforms.RandomAdjustSharpness(0.75, 0.25))
        t.append(transforms.RandomEqualize())
        t.append(transforms.RandomInvert(0.1))
        t.append(transforms.RandomGrayscale(0.05))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))

        return transforms.Compose(t)
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
