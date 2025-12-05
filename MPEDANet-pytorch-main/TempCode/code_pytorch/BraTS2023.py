import os
import torch
import random
import numpy as np
import SimpleITK as sitk
import pathlib

from torchvision import transforms
from torch.utils.data import Dataset

# Mapping modalities for BraTS 2023 Part 2
# case folder contains files like:
# BraTS-GLI-XXXX-YYY-seg.nii
# BraTS-GLI-XXXX-YYY-t1c.nii (T1ce)
# BraTS-GLI-XXXX-YYY-t1n.nii (T1 native)
# BraTS-GLI-XXXX-YYY-t2f.nii (T2 FLAIR)
# BraTS-GLI-XXXX-YYY-t2w.nii (T2 weighted)
modalities_suffix = {
    '_flair': '-t2f.nii',   # FLAIR
    '_t1ce':  '-t1c.nii',   # T1 contrast enhanced
    '_t1':    '-t1n.nii',   # T1 native
    '_t2':    '-t2w.nii',   # T2 weighted
}

START = 45
END = 109

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (c, w, h, d) = image.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))
        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], START:END]
        image = image[:, w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], START:END]
        return {'image': image, 'label': label}

def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
    return data_sample

class RandomRotFlip(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.stack([np.rot90(x, k) for x in image], axis=0)
        label = np.rot90(label, k)
        axis = np.random.randint(1, 4)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis - 1).copy()
        return {'image': image, 'label': label}

class GaussianNoise(object):
    def __init__(self, noise_variance=(0, 0.1), p=0.5):
        self.prob = p
        self.noise_variance = noise_variance

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if np.random.uniform() < self.prob:
            image = augment_gaussian_noise(image, self.noise_variance)
        return {'image': image, 'label': label}

class ToTensor(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        return {'image': image, 'label': label}

class BraTS2023(Dataset):
    def __init__(self, data_path, file_path=None, transform=None):
        # if file list provided: read case names; else glob all folders
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                names = [x.strip() for x in f.readlines() if x.strip()]
            self.paths = [os.path.join(data_path, name) for name in names]
        else:
            self.paths = [str(p) for p in pathlib.Path(data_path).glob('BraTS-GLI-*')]
        self.transform = transform

    def __getitem__(self, item):
        case_dir = self.paths[item]
        base = os.path.basename(case_dir)  # e.g., BraTS-GLI-00740-000
        # read label (.nii, not .nii.gz)
        label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_dir, f"{base}-seg.nii"))).transpose(1, 2, 0)
        # read images for four modalities
        images = []
        for key, suf in modalities_suffix.items():
            img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_dir, f"{base}{suf}"))).transpose(1, 2, 0)
            images.append(img)
        images = np.stack(images, 0).astype(np.float32)
        label = label.astype(np.uint8)

        # normalize non-zero region per channel
        mask = images.sum(0) > 0
        for k in range(4):
            x = images[k, ...]
            y = x[mask]
            if y.size > 0 and y.std() > 0:
                x[mask] -= y.mean()
                x[mask] /= y.std()
            images[k, ...] = x

        # map labels 4 -> 3 (consistent with existing code)
        label[label == 4] = 3
        sample = {'image': images, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.paths)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
