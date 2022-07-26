import os.path as osp

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torchmetrics.functional import jaccard_index

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv

category = 'Airplane'  # Pass in `None` to train on all categories.
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ShapeNet')
transform = T.Compose([
    T.RandomRotate(15, axis=0),
    T.RandomRotate(15, axis=1),
    T.RandomRotate(15, axis=2)
])
pre_transform = T.NormalizeScale()
train_dataset = ShapeNet(path, category, split='train', transform=transform, pre_transform=pre_transform)
valid_dataset = ShapeNet(path, category, split='val', transform=transform, pre_transform=pre_transform)
test_dataset = ShapeNet(path, category, split='test', pre_transform=pre_transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=6)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6)