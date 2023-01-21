import unittest
import torch
import os.path as osp
from torch.utils.data import DataLoader
from torch.nn import functional as F
from hpcs.nn.dgcnn import VN_DGCNN_partseg
from hpcs.utils.math import rot_3D
from hpcs.hyp_hc import remap_labels
from data.ShapeNet.ShapeNetDataLoader import PartNormalDataset


class TestRotationInvariance(unittest.TestCase):
    def setUp(self) -> None:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', 'ShapeNet/raw')
        self.num_classes = 16
        self.nn = VN_DGCNN_partseg(in_channels=3, out_features=2, k=10, dropout=0.5, pooling='mean', num_class=self.num_classes)

        train_dataset = PartNormalDataset(root=path, npoints=256, split='train', class_choice='Airplane')
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, drop_last=True)

        # getting one sample from training set
        for airplane_sample in train_loader:
            points, label, target = airplane_sample  ## 3D coordinates of the airplane
            break
        self.points = points.float()
        self.label = label.long()
        self.target = target.long()

        num_parts = self.num_classes
        batch_class_vector = []
        for object in self.target:
            parts = F.one_hot(remap_labels(torch.unique(object)), num_parts)
            class_vector = parts.sum(dim=0).float()
            batch_class_vector.append(class_vector)

        self.decode_vector = torch.stack(batch_class_vector)

    def test_invariance(self):
        angles = 2 * torch.pi * torch.rand([3])
        R = rot_3D(angles[0], angles[1], angles[2])
        print(f"Point size: {self.points.size}")
        self.assertEqual(self.points.size(), (2, 256, 3))
        rot_points = torch.empty_like(self.points)
        for i in range(self.points.size(0)):
            rot_points[i] = (R@self.points[i].T).T

        y_1 = self.nn(self.points.transpose(2, 1), self.decode_vector)
        y_2 = self.nn(rot_points.transpose(2, 1), self.decode_vector)

        self.assertAlmostEqual(torch.linalg.norm(y_1-y_2).detach().numpy(), 0.0)