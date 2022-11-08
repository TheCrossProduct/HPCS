import unittest
import torch
import numpy as np
import os.path as osp
from hpcs.nn.dgcnn import VN_DGCNN_partseg
from data.ShapeNet.ShapeNetDataLoader import PartNormalDataset

def get_rot(theta):
    return np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])

def rotate_sample(x, R):
    z = x.T
    return (R@z).T

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

class TestRotationInvariance(unittest.TestCase):
    def setUp(self) -> None:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data', 'ShapeNet/raw')
        self.num_classes = 16
        self.nn = VN_DGCNN_partseg(in_channels=3, out_features=2, k=10, dropout=0.5, pooling='mean')

        train_dataset = PartNormalDataset(root=path, npoints=256, split='train', class_choice='Airplane')
        airplane_sample = train_dataset.__getitem__(0)

        self.points, self.label, self.target = airplane_sample  ## 3D coordinates of the airplane
        print(self.points.size())

    def test_invariance(self):
        R = get_rot(np.random.rand())
        R = torch.from_numpy(R)
        label = torch.from_numpy(self.label)
        label = to_categorical(label, self.num_classes)
        label = label[None, ...]
        points = torch.from_numpy(self.points)
        points = points[None, ...] ## moves from [n_pts, 3] -> [1, n_, 3pts, 3]

        y_1 = self.nn(points, label)
        y_2 = self.nn((R @ points.T).T, label)  # need to verify the dimensions
        print(y_1)

        self.assertAlmostEqual(torch.linalg.norm(y_1-y_2), 0.0)