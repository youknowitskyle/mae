import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os


def make_dataset(image_list, label_list, au_relation=None):
    len_ = len(image_list)
    if au_relation is not None:
        images = [(image_list[i].strip(),  label_list[i, :],au_relation[i,:]) for i in range(len_)]
    else:
        images = [(image_list[i].strip(),  label_list[i, :]) for i in range(len_)]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class DATA(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, unilateral=False, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader

        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'train_img_path_fold' + str(fold) + '.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path).astype('float32')
            # only use unilateral labels for FEAFA
            if unilateral:
                train_label_list = np.concatenate((train_label_list[:, :8], train_label_list[:, 9:15]), axis=1)
            self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path).astype('float32')
            # only use uilateral labels for FEAFA
            if unilateral:
                test_label_list = np.concatenate((test_label_list[:, :8], test_label_list[:, 9:15]), axis=1)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        img, label = self.data_list[index]
        img = self.loader(img)

        if self._transform is not None:
            img = self._transform(img)
        return img, label

    def __len__(self):
        return len(self.data_list)


class DISFA(Dataset):
    def __init__(self, root_path, train=True, fold = 1, transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert fold>0 and fold <=3, 'The fold num must be restricted from 1 to 3'
        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = '/home/kyle/school/farapy/DISFA/processed/left_cam_256'
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'DISFA_train_img_path_fold' + str(fold) + '.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'DISFA_train_label_fold' + str(fold) + '.txt')
            train_label_list = np.loadtxt(train_label_list_path).astype('float32')
            self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'DISFA_test_img_path_fold' + str(fold) + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'DISFA_test_label_fold' + str(fold) + '.txt')
            test_label_list = np.loadtxt(test_label_list_path).astype('float32') 
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        img, label = self.data_list[index]
        img = self.loader(os.path.join(self.img_folder_path, img))

        if self._transform is not None:
            img = self._transform(img)
        return img, label

    def __len__(self):
        return len(self.data_list)
