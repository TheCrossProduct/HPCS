import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from hpcs.utils.data import pc_normalize


class PartNetDataset(Dataset):
    def __init__(self, filelist, npoints):
        points = []
        point_nums = []
        labels_seg = []
        folder = os.path.dirname(filelist)
        for line in open(filelist):
            data = h5py.File(os.path.join(folder, line.strip()))
            points.append(data['data'][...].astype(np.float32))
            point_nums.append(data['data_num'][...].astype(np.int32))
            labels_seg.append(data['label_seg'][...].astype(np.int64))
        points = np.concatenate(points, axis=0)
        point_nums = np.concatenate(point_nums, axis=0)
        labels_seg = np.concatenate(labels_seg, axis=0)

        self.points = points
        self.data_num = point_nums
        self.label_seg = labels_seg
        self.npoints = npoints

    def __getitem__(self, index):
        points, data_num, label_seg = self.points[index], self.data_num[index], self.label_seg[index]
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        choice = np.random.choice(len(label_seg), self.npoints, replace=True)
        point_set = points[choice, :]
        label_seg_set = label_seg[choice]
        return point_set.astype(np.float32), label_seg_set.astype(np.int64)

    def __len__(self):
        return self.points.shape[0]


class PartNetDatasetHierarchical(Dataset):
    def __init__(self, filelist, npoints, level):
        points = []
        point_nums = []
        folder = os.path.dirname(filelist[0])
        for line in open(filelist[0]):
            data = h5py.File(os.path.join(folder, line.strip()))
            points.append(data['data'][...].astype(np.float32))
            point_nums.append(data['data_num'][...].astype(np.int32))
        points = np.concatenate(points, axis=0)
        point_nums = np.concatenate(point_nums, axis=0)

        all_labels_seg = []
        for level, file in enumerate(filelist):
            labels_seg = []
            folder = os.path.dirname(file)
            for line in open(file):
                data = h5py.File(os.path.join(folder, line.strip()))
                labels_seg.append(data['label_seg'][...].astype(np.int64))
            labels_seg = np.concatenate(labels_seg, axis=0)
            all_labels_seg.append(labels_seg)

        self.points = points
        self.data_num = point_nums
        self.label_seg = all_labels_seg
        self.npoints = npoints

    def __getitem__(self, index):
        points, data_num = self.points[index], self.data_num[index]
        points[:, 0:3] = pc_normalize(points[:, 0:3])
        choice = np.random.choice(len(points), self.npoints, replace=True)
        point_set = points[choice, :]
        label_choice = []
        for level, item in enumerate(self.label_seg):
            label_seg_set = item[level][choice]
            label_choice.append(label_seg_set)
        return point_set.astype(np.float32), label_choice

    def __len__(self):
        return self.points.shape[0]