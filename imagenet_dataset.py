import os.path as osp
import json
import requests
import time
import numpy as np
import io
from PIL import Image
import logging
from torch.utils.data import Dataset

logger = logging.getLogger('global')

import random
import torch
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
augmentation = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    normalize,
]

def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert('RGB')
    except IOError:
        logger.info('Failed in loading {}'.format(filepath))
    return img


class ImageNetDataset(Dataset):
    """
    ImageNet Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_type (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - server_cfg (list): server configurations

    Metafile example::
        "n01440764/n01440764_10026.JPEG 0\n"
    """
    def __init__(self, root_dir, meta_file, transform=transforms.Compose(augmentation),
                 read_from='mc', evaluator=None, image_reader_type='pil',
                 server_cfg={}):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.read_from = read_from
        self.transform = transform
        self.evaluator = evaluator
        self.image_reader = pil_loader
        self.initialized = False
        self.use_server = False

        # read from local file
        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            items = line.rstrip().split(' ')
            filename = items[0]
            label = items[1:]
            self.metas.append({'filename': filename, 'label': label})


    def __len__(self):
        return self.num

    def _load_meta(self, idx):
        return self.metas[idx]

    def __getitem__(self, idx):
        curr_meta = self._load_meta(idx)
        filename = osp.join(self.root_dir, curr_meta['filename'])
        label = [int(x) for x in curr_meta['label']]
        # add root_dir to filename
        curr_meta['filename'] = filename
        img_bytes = np.fromfile(curr_meta['filename'], dtype=np.uint8)
        img = self.image_reader(img_bytes, filename)

        if self.transform is not None:
            img = self.transform(img)

        item = {
            'image': img,
            'label': label,
            'image_id': idx,
            'filename': filename
        }
        return item

    def dump(self, writer, output):
        prediction = self.tensor2numpy(output['prediction'])
        label = self.tensor2numpy(output['label'])
        score = self.tensor2numpy(output['score'])

        if 'filename' in output:
            # pytorch type: {'image', 'label', 'filename', 'image_id'}
            filename = output['filename']
            image_id = output['image_id']
            for _idx in range(prediction.shape[0]):
                res = {
                    'filename': filename[_idx],
                    'image_id': int(image_id[_idx]),
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        else:
            # dali type: {'image', 'label'}
            for _idx in range(prediction.shape[0]):
                res = {
                    'prediction': int(prediction[_idx]),
                    'label': int(label[_idx]),
                    'score': [float('%.8f' % s) for s in score[_idx]],
                }
                writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()


