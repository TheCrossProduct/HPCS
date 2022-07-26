import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader


category = 'Airplane'  # Pass in `None` to train on all categories.
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ShapeNet')

pre_transform, transform = T.NormalizeScale(), T.FixedPoints(256)
train_dataset = ShapeNet(path, category, split='train', transform=transform, pre_transform=pre_transform)
valid_dataset = ShapeNet(path, category, split='val', transform=transform, pre_transform=pre_transform)
test_dataset = ShapeNet(path, category, split='test', transform=transform, pre_transform=pre_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=6)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=6)
