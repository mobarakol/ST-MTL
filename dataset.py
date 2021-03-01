'''
@author: Mobarakol Islam
'''
from PIL import Image
import random
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
from albumentations import Compose, Normalize

class SurgicalDataset(Dataset):
    def __init__(self, data_root, seq_set, is_train=None):
        self.transform = Compose([Normalize(p=1)], p=1)
        self.is_train = is_train
        self.list = seq_set
        self.dir_root_gt = data_root + '/instrument_dataset_'
        self.img_dir_list = []
        for i in self.list:
            dir_sal = self.dir_root_gt + str(i) + '/salmap/'
            self.img_dir_list = self.img_dir_list + glob(dir_sal + '/*.png')
            random.shuffle(self.img_dir_list)

    def __len__(self):
        return len(self.img_dir_list)
    
    def __getitem__(self, index):
        _target_sal = Image.open(self.img_dir_list[index]).convert('L')
        _img = Image.open(os.path.dirname(os.path.dirname(self.img_dir_list[index])) + '/images/' + os.path.basename(
            self.img_dir_list[index][:-4]) + '.jpg').convert('RGB')
        _target = Image.open(os.path.dirname(os.path.dirname(self.img_dir_list[index])) + '/instruments_masks/'
                             + os.path.basename(self.img_dir_list[index][:-4]) + '.png')
        if self.is_train:
            isAugment = random.random() < 0.5
            if isAugment:
                isHflip = random.random() < 0.5
                if isHflip:
                    _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                    _target = _target.transpose(Image.FLIP_LEFT_RIGHT)
                    _target_sal = _target_sal.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    _img = _img.transpose(Image.FLIP_TOP_BOTTOM)
                    _target = _target.transpose(Image.FLIP_TOP_BOTTOM)
                    _target_sal = _target_sal.transpose(Image.FLIP_TOP_BOTTOM)

        data = {"image": np.array(_img), "mask": np.array(_target)}
        augmented = self.transform(**data)
        _img, _target = augmented["image"], augmented["mask"]
        _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1)).float()
        _target = torch.from_numpy(np.array(_target)).long()
        _target_sal = np.array(_target_sal)
        _target_sal = _target_sal * 1.0 / 255
        _target_sal = torch.from_numpy(np.array(_target_sal)).float()

        if self.is_train:
            return _img, _target, _target_sal
        return _img, _target, _target_sal, self.img_dir_list[index]