import torch
import numpy as np
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def get_data_path() -> Path:
    proj_root = get_project_root()
    return proj_root / 'data'

def get_shapenet_path() -> Path:
    data_path = get_data_path()
    return data_path / 'ShapeNet' / 'raw'


def get_partnet_path() -> Path:
    data_path = get_data_path()
    return data_path / 'PartNet' / 'sem_seg_h5'

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def remap_labels(y_true):
    y_remap = torch.zeros_like(y_true)
    for i, l in enumerate(torch.unique(y_true)):
        y_remap[y_true==l] = i
    return y_remap

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc
