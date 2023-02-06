import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class H5Dataset_hierarchical(Dataset):
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