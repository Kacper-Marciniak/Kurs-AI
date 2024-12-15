import os
import tarfile
from PIL import Image
from tqdm import tqdm
import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


RESIZE = 256
CROPSIZE = 224
TRANSFORM = T.Compose([
    T.Resize(RESIZE, Image.Resampling.BICUBIC),
    T.CenterCrop(CROPSIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
TRANSFORM_MASK = T.Compose([
    T.Resize(RESIZE, Image.Resampling.NEAREST),
    T.CenterCrop(CROPSIZE),
    T.ToTensor()
])

URL_FULL = 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'
URL_CLASS = {
    'bottle': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz',
    'cable': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz',
    'capsule': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz',
    'carpet': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz',
    'grid': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz',
    'hazelnut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz',
    'leather': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz',
    'metal_nut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz',
    'pill': 'https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz', 
    'screw': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz',
    'tile' : 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz',
    'toothbrush': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz',
    'transistor': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz',
    'wood' : 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz',
    'zipper': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz',
}

CLASS_NAMES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

class MVTecSingleClassDataset(Dataset):
    def __init__(
            self,
            dataset_path,
            class_name = 'bottle',
            subset = 'train'
        ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        assert subset in ['train', 'test'], 'subset should be train or test'
        
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.subset = subset

        # download dataset if not exist
        self.download()

        # load dataset
        self.x, self.y, self.mask, self.anomaly_names = self.load_dataset_folder()

        # set transforms
        self.transform_x = TRANSFORM
        self.transform_mask = TRANSFORM_MASK

    def __getitem__(self, idx):
        x, y, mask, anomaly_name = self.x[idx], self.y[idx], self.mask[idx], self.anomaly_names[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, CROPSIZE, CROPSIZE])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask, anomaly_name

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        x, y, mask, anomaly_name = [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, self.subset)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            anomaly_name.extend([img_type] * len(img_fpath_list))

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask), list(anomaly_name)

    def download(self):
        """Download dataset if not exist"""

        url = URL_CLASS[self.class_name]            
        if os.path.exists(os.path.join(self.dataset_path, self.class_name)): 
            return

        print(f'Downloading dataset from {url}')
        
        os.makedirs(os.path.join(self.dataset_path), exist_ok=True)

        tar_file_path = self.dataset_path + '.tar.xz'
        if not os.path.exists(tar_file_path):
            download_url(url, tar_file_path)      

        print(f'Unzipping downloaded dataset from {tar_file_path}')
        tar = tarfile.open(tar_file_path, 'r:xz')
        tar.extractall(self.dataset_path)
        tar.close()

        os.remove(tar_file_path)

        print('Download completed!')

        return


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)