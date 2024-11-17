import numpy as np
import os
import pandas as pd
import random
import tifffile
import torch
import torchvision as thv
from torch import Tensor
from torch.utils.data import Dataset

from typing import Union, Tuple, Callable, Optional


class Identity(object):
    def __init__(self) -> None:
        pass

    def __call__(self, img: Tensor) -> Tensor:
        return img


class RandomChannelScale(object):
    def __init__(self, int_max: Optional[float] = None) -> None:
        if int_max == None:
            all_int_max = np.linspace(0, 100, 20)
            self.int_max = all_int_max[random.randint(0, len(all_int_max) - 1)]
        else:
            self.int_max = int_max

    def __call__(self, img: Tensor) -> Tensor:
        '''
        Input
          img - shape (N, C, H, W) or (N, C)
        '''
        ch_select = random.randint(0, img.shape[1])
        img[ch_select] = img[ch_select]*self.int_max
        return img
    
        
class RandomGammaScale(object):
    def __init__(self, gamma: Optional[float] = None) -> None:
        if gamma == None:
            gammas = np.linspace(0.1, 2, 20)
            self.gamma = gammas[random.randint(0, len(gammas)-1)]
        else:
            self.gamma = gamma

    def __call__(self, img: Tensor) -> Tensor:
        return img**self.gamma


class RandomLinearScale(object):
    def __init__(self, int_max: Optional[float] = None) -> None:
        if int_max == None:
            all_int_max = np.linspace(0, 500, 100)
            self.int_max = all_int_max[random.randint(0, len(all_int_max) - 1)]
        else:
            self.int_max = int_max

    def __call__(self, img: Tensor) -> Tensor:
        min_img = torch.min(img)
        max_img = torch.max(img)
        img = (img - min_img)/(max_img - min_img)
        img = img*self.int_max + min_img
        return img


class RandomGammaPlusLinearScale(object):
    def __init__(self, gamma: Optional[float] = None, int_max: Optional[float] = None) -> None:
        self.rng_gamma_scale = RandomGammaScale(gamma)
        self.rng_linear_scale = RandomLinearScale(int_max)

    def __call__(self, img: Tensor) -> Tensor:
        img = self.rng_gamma_scale(img)
        img = self.rng_linear_scale(img)
        return img


class RandomLinearPlusGammaScale(object):
    def __init__(self, gamma: Optional[float] = None, int_max: Optional[float] = None) -> None:
        self.rng_gamma_scale = RandomGammaScale(gamma)
        self.rng_linear_scale = RandomLinearScale(int_max)

    def __call__(self, img: Tensor) -> Tensor:
        img = self.rng_linear_scale(img)
        img = self.rng_gamma_scale(img)
        return img


TRANSFORMS_VERSION = {
    'all': [
        thv.transforms.RandomRotation(degrees=180),
        thv.transforms.RandomResizedCrop(size=(64, 64)),
        RandomGammaScale(),
        RandomLinearScale(),
        Identity()
    ],
    'all_v2':[
        thv.transforms.RandomHorizontalFlip(),
        thv.transforms.RandomVerticalFlip(),
        thv.transforms.RandomResizedCrop(size=(64, 64)),
        RandomGammaScale(),
        RandomLinearScale(),
        RandomGammaPlusLinearScale(),
        RandomLinearPlusGammaScale(),
        Identity(),
    ],
    'all_v3':[
        RandomGammaScale(),
        RandomLinearScale(),
        RandomGammaPlusLinearScale(),
        RandomLinearPlusGammaScale(),
        Identity(),
    ]
}

class CellData(Dataset):
    def __init__(self, uniq_sample_img_names, all_centers, regions_per_image):
        super().__init__()
        
        self.BASE_PATH = '/scratch3/schaudhary/SenescenceProject2022'
        self.IMG_PATH = os.path.join(BASE_PATH, 'med_gauss_filtered_med3_sigma3')
        self.MASK_PATH = os.path.join(BASE_PATH, 'masks_n2v_denoised_dilation_20_notissue')
        self.uniq_sample_img_names = uniq_sample_img_names
        self.all_centers = all_centers
        self.regions_per_image = regions_per_image

        self.transforms = [
            thv.transforms.RandomRotation(degrees=180),
            thv.transforms.RandomResizedCrop(size=(64, 64)),
            RandomGammaScale(),
            RandomLinearScale()
        ]

    def read_img(self, sample_img_name):
        return tifffile.imread(os.path.join(self.IMG_PATH, sample_img_name))
        
    def read_mask(self, sample_img_name):
        return tifffile.imread(os.path.join(self.MASK_PATH, f'{sample_img_name}_mask.tiff'))
    
    def get_cell(self, img, mask, centers=None, cell_id=None):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
    
        if type(centers) == type(None):
            centers = get_centers(mask)
        if type(cell_id) == type(None):
            cell_id = random.randint(1, len(centers) - 1)
        # mask = mask == select
        mask_crop = thv.transforms.functional.crop(torch.tensor(mask, dtype=torch.float), int(centers[cell_id][0])-32, int(centers[cell_id][1])-32, 64, 64)
        img_crop = thv.transforms.functional.crop(torch.tensor(img, dtype=torch.float), int(centers[cell_id][0])-32, int(centers[cell_id][1])-32, 64, 64)
        return mask_crop, img_crop

    def get_transforms(self):
        t = [i for i in range(len(self.transforms))]
        random.shuffle(t)
        return self.transforms[t[0]], self.transforms[t[1]]
    
    def __len__(self):
        return len(self.uniq_sample_img_names)

    def __getitem__(self, idx):
        img_name = self.uniq_sample_img_names[idx]
        img = self.read_img(img_name)
        mask = self.read_mask(img_name)

        all_img_crop = []
        all_i1 = []
        all_i2 = []
        for n in range(self.regions_per_image):
            mask_crop, img_crop = self.get_cell(img, mask, self.all_centers[img_name])
            t1, t2 = self.get_transforms()
            i1, i2 = t1(img_crop), t2(img_crop)
            all_img_crop.append(img_crop)
            all_i1.append(i1)
            all_i2.append(i2)

        all_img_crop = torch.stack(all_img_crop, dim=0)
        all_i1 = torch.stack(all_i1, dim=0)
        all_i2 = torch.stack(all_i2, dim=0)
        return (all_img_crop, all_i1, all_i2)


class CellDataV2(Dataset):
    def __init__(self, img_crop_path, mask_crop_path, transforms='all', mask_cell=False, data_size=0, mode='train', **kwargs):
        super().__init__()

        self.img_crop_path = img_crop_path
        self.mask_crop_path = mask_crop_path
        self.num_img = len(os.listdir(self.img_crop_path))
        if data_size != 0:
            self.num_img = min(self.num_img, data_size)
        self.mask_cell = mask_cell
        self.mode = mode
        self.metadata_df = pd.read_pickle(os.path.join(os.path.dirname(img_crop_path), 'metadata_df.pkl'))

        if transforms not in TRANSFORMS_VERSION:
            self.transforms = [
                Identity(),
                Identity()
            ]
        else:
            self.transforms = TRANSFORMS_VERSION[transforms]
    
    def get_transforms(self):
        t = [i for i in range(len(self.transforms))]
        random.shuffle(t)
        return self.transforms[t[0]], self.transforms[t[1]]
    
    def __len__(self):
        return self.num_img

    def __getitem__(self, idx):
        img_crop = tifffile.imread(os.path.join(self.img_crop_path, f'img_{idx + 1}.tiff'))
        mask_crop = tifffile.imread(os.path.join(self.mask_crop_path, f'mask_{idx + 1}.tiff'))
        img_crop = torch.tensor(img_crop, dtype=torch.float)
        mask_crop = torch.tensor(mask_crop, dtype=torch.float)

        if self.mask_cell:
            mask_crop = mask_crop.expand(img_crop.shape[0], -1, -1)
            region = self.metadata_df.iloc[idx]['region_id'].item()
            img_crop = torch.mul(img_crop, mask_crop == region)
        
        t1, t2 = self.get_transforms()
        i1, i2 = t1(img_crop), t2(img_crop)
        if self.mode == 'train':
            return (i1, i2)
        else:
            return {'data': (i1, i2), 'metadata': self.metadata_df.iloc[idx].to_dict()}


class CellDataPixel(Dataset):
    def __init__(self,
                 img_crop_path: str,
                 mask_crop_path: str,
                 transforms: str = 'all_v3',
                 mask_cell: bool = False,
                 data_mode: str = 'mean_pixel',
                 data_size: int = 0,
                 mode: str = 'train',
                 **kwargs) -> None:
        
        super().__init__()

        self.img_crop_path = img_crop_path
        self.mask_crop_path = mask_crop_path
        self.mask_cell = mask_cell
        self.data_mode = data_mode
        self.num_img = len(os.listdir(self.img_crop_path))
        if data_size != 0:
            self.num_img = min(self.num_img, data_size)
        self.mode = mode
        self.metadata_df = pd.read_pickle(os.path.join(os.path.dirname(img_crop_path), 'metadata_df.pkl'))

        if transforms not in TRANSFORMS_VERSION:
            self.transforms = [
                Identity(),
                Identity()
            ]
        else:
            self.transforms = TRANSFORMS_VERSION[transforms]

        # needed to load img crops belonging to a specific image
        self.SUBSET_CTR = 0 
    
    def get_transforms(self) -> Tuple[Callable, Callable]:
        t = [i for i in range(len(self.transforms))]
        random.shuffle(t)
        return self.transforms[t[0]], self.transforms[t[1]]

    def get_data(self, idx: int) -> Tensor:
        img_crop = tifffile.imread(os.path.join(self.img_crop_path, f'img_{idx + 1}.tiff'))
        mask_crop = tifffile.imread(os.path.join(self.mask_crop_path, f'mask_{idx + 1}.tiff'))
        img_crop = torch.tensor(img_crop, dtype=torch.float)
        mask_crop = torch.tensor(mask_crop, dtype=torch.float)
        
        if self.data_mode == 'mean_cell':
            region = self.metadata_df.iloc[idx]['region_id'].item()
            data = torch.mean(img_crop[:, mask_crop[0] == region], dim=1)
        elif self.data_mode == 'pixel':
            fg_pixels = torch.where(torch.sum(img_crop, dim=0) > 0)
            pixel_select = random.sample(range(fg_pixels[0].shape[0]), 1)[0]
            x_coord = fg_pixels[0][pixel_select]
            y_coord = fg_pixels[1][pixel_select]
            data = torch.squeeze(thv.transforms.functional.crop(img_crop, x_coord, y_coord, 1, 1))
        elif self.data_mode == 'mean_pixel':
            fg_pixels = torch.where(torch.sum(img_crop, dim=0) > 0)
            pixel_select = random.sample(range(fg_pixels[0].shape[0]), 1)[0]
            x_coord = fg_pixels[0][pixel_select]
            y_coord = fg_pixels[1][pixel_select]
            data = torch.mean(thv.transforms.functional.crop(img_crop, x_coord, y_coord, 3, 3), dim=(1, 2))
        elif self.data_mode == 'patch':
            if self.mask_cell:
                mask_crop = mask_crop.expand(img_crop.shape[0], -1, -1)
                region = self.metadata_df.iloc[idx]['region_id'].item()
                data = torch.mul(img_crop, mask_crop == region)
            else:
                data = img_crop
        else:
            pass
        return data    
    
    def __len__(self) -> int:
        return self.num_img

    def __getitem__(self, idx:int) -> Union[Tuple[Tensor, Tensor], dict]:
        data = self.get_data(self.SUBSET_CTR + idx)
        t1, t2 = self.get_transforms()
        i1, i2 = t1(data), t2(data)
        return {'data': (i1, i2), 'metadata': self.metadata_df.iloc[self.SUBSET_CTR + idx].to_dict()}
    