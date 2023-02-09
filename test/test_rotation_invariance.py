import unittest
import torch
import os.path as osp
from torch.utils.data import DataLoader
from torch.nn import functional as F
from hpcs.nn.dgcnn import VN_DGCNN_partseg
from hpcs.utils.math import rot_3D
from hpcs.utils.data import remap_labels, to_categorical
from hpcs.data import ShapeNetDataset
import pyvista as pv

class TestRotationInvariance(unittest.TestCase):
    def setUp(self) -> None:
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../data', 'ShapeNet/raw')
        self.num_classes = 16
        self.batch_size = 1

        self.nn = VN_DGCNN_partseg(in_channels=3, out_features=2, k=10, dropout=0.5, pooling='mean', num_class=self.num_classes)
        self.nn.eval()
        train_dataset = ShapeNetDataset(root=path, npoints=256, split='train', class_choice='Airplane')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1, drop_last=True)

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
        # checking input points shape
        self.assertEqual(self.points.size(), (self.batch_size, 256, 3))

        # generating 3D rotation
        angles = 2 * torch.pi * torch.rand([3])
        yaw, pitch, roll = angles[0], angles[1], angles[2]
        R = rot_3D(yaw, pitch, roll)

        # rotating point cloud
        rot_points = torch.empty_like(self.points)
        for i in range(self.points.size(0)):
            rot_points[i] = (R@self.points[i].T).T

        # plotting point cloud and its rotated version
        plotter = pv.Plotter(shape=(1,2))
        plotter.subplot(0,0)
        data0 = pv.PolyData(self.points[0].detach().numpy())
        plotter.add_mesh(data0,scalars=self.target[0].detach().numpy(), render_points_as_spheres=True, point_size=5.0)
        plotter.camera_position = 'xy'
        plotter.add_text(f'Input Sample')
        plotter.subplot(0, 1)
        data1 = pv.PolyData(rot_points[0].detach().numpy())
        plotter.add_mesh(data1, scalars=self.target[0].detach().numpy(), render_points_as_spheres=True, point_size=5.0)
        plotter.camera_position = 'xy'
        plotter.add_text(f'Rotation angles: (yaw: {torch.rad2deg(yaw):.1f}, '
                         f'pitch: {torch.rad2deg(pitch):.1f} roll: {torch.rad2deg(roll):.1f}) degrees')
        plotter.show_axes_all()
        plotter.show()

        # inference
        y_1 = self.nn(self.points.transpose(2, 1), to_categorical(self.label, self.num_classes))
        y_2 = self.nn(rot_points.transpose(2, 1), to_categorical(self.label, self.num_classes))

        self.assertAlmostEqual(torch.linalg.norm(y_1-y_2).detach().numpy(), 0.0)